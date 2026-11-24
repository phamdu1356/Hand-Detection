"""Tier-1 static benchmark: hand detection, 21-keypoint & 14-control-kpt PCK, Gate14 & latency.

Evaluates YOLO Pose (.pt / .onnx) weights on a deduplicated, leak-free validation split
of a YOLO pose hand-keypoints dataset (default: ../datasets/hand-keypoints).

Highlights:
- Cryptographic SHA-256 deduplication removing intra-val duplicates and train∩val leaks.
- Perceptual 64-bit dHash near-duplicate scan (Hamming <= 4).
- Single-pass inference (conf >= 0.01) producing both operating-point metrics (conf >= 0.50)
  and exact all-point interpolated AP@0.5 across 100% of clean val images.
- Dual-denominator PCK@0.2 (Box on 100% images; Palm 21-kpt & 14-control-kpt on palm-valid
  images), dual-denominator Gate14 pass rate + per-joint drop histogram, and pixel size buckets.
- Optional scaled small-hand stress test (e.g. 36 px, 64 px; explicitly tagged diagnostic)
  and optional background-only negative folder for False Positive Rate.
- True GPU (CUDA) & CPU latency (20 warmup + 200 timed iterations) and visual error audit PNGs.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from benchmark.schema import (
    CONTROL_KPT_INDICES,
    KPT_NAMES,
    PALM_INDICES,
    SIZE_BUCKETS,
    dhash64,
    file_sha256,
    git_commit_info,
    size_bucket,
    wilson_ci,
)
from dronectl.body_frame import hand_size

REPO_ROOT = Path(__file__).resolve().parent.parent

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)

_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ------------------------------------------------------------------ data types

@dataclass(frozen=True)
class SampleGT:
    image_path: str
    sha256: str
    width: int
    height: int
    box_xyxy: tuple[float, float, float, float]
    size_px: float
    size_bucket: str
    kps_xy: np.ndarray           # (21, 2) float64 pixels
    kps_vis: np.ndarray          # (21,) bool: vis > 0
    kps_in_frame: np.ndarray     # (21,) bool: vis > 0 and inside [0, W] x [0, H]
    palm_px: float | None
    all14_in_frame: bool
    oob_box: bool
    oob_kp: bool
    oob_kp_count: int
    extra_hands_in_label: int = 0


@dataclass(frozen=True)
class PredDet:
    box_xyxy: tuple[float, float, float, float]
    conf: float
    kps_xy: np.ndarray           # (21, 2) float64 pixels
    kps_conf: np.ndarray         # (21,) float64 in [0, 1]


@dataclass
class SampleOutcome:
    gt: SampleGT
    matched_pred: PredDet | None
    top_unmatched_pred: PredDet | None
    tp: bool
    fp_count: int
    extra_fp_count: int
    iou: float
    pck_box_hits: int
    pck_box_total: int
    pck_palm21_hits: int
    pck_palm21_total: int
    pck_palm14_hits: int
    pck_palm14_total: int
    joint_err_px: np.ndarray       # (21,) float64, nan when not applicable
    joint_err_palm: np.ndarray     # (21,) float64, nan when not applicable
    gate14_pass: bool
    gate14_failed_joints: list[int] = field(default_factory=list)
    gate14_min_conf: float = 0.0
    top1_conf_tp: bool = False


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    path: str
    sha256: str
    imgsz: int
    is_onnx: bool
    params: int | None
    aliases: tuple[str, ...] = ()


# ------------------------------------------------------------------ pure math

def box_iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over Union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-12 else 0.0


def compute_ap50(pr_records: list[tuple[float, bool]], total_gt: int) -> float:
    """All-point interpolated Average Precision at IoU 0.50 (VOC2010 / COCO standard)."""
    if total_gt <= 0 or not pr_records:
        return 0.0
    ordered = sorted(pr_records, key=lambda r: r[0], reverse=True)
    tp = np.array([1.0 if is_tp else 0.0 for _, is_tp in ordered], dtype=np.float64)
    fp = 1.0 - tp
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(mpre.size - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def check_gate14(pred: PredDet, conf_min: float = 0.5) -> tuple[bool, list[int], float]:
    """Mirror dronectl.gestures.hand_command's 14-keypoint validity gate verbatim."""
    failed: list[int] = []
    min_c = 1.0
    for idx in CONTROL_KPT_INDICES:
        c = float(pred.kps_conf[idx])
        pt = pred.kps_xy[idx]
        min_c = min(min_c, c)
        valid_pt = bool(np.isfinite(pt).all() and (pt[0] > 0.0 or pt[1] > 0.0))
        if c < conf_min or not valid_pt:
            failed.append(idx)
    return (len(failed) == 0, failed, min_c)


def evaluate_sample_metrics(
    gt: SampleGT,
    preds: list[PredDet],
    *,
    op_conf: float = 0.5,
    iou_thres: float = 0.5,
    pck_thresh: float = 0.2,
    gate_conf: float = 0.5,
) -> tuple[SampleOutcome, list[tuple[float, bool]]]:
    """Evaluate one single-hand ground-truth sample against all model predictions (conf >= 0.01)."""
    ordered = sorted(preds, key=lambda d: d.conf, reverse=True)

    # Full PR-curve records across all predictions (at most 1 TP per single-hand image)
    pr_records: list[tuple[float, bool]] = []
    gt_claimed = False
    for det in ordered:
        iou = box_iou_xyxy(det.box_xyxy, gt.box_xyxy)
        if not gt_claimed and iou >= iou_thres:
            pr_records.append((float(det.conf), True))
            gt_claimed = True
        else:
            pr_records.append((float(det.conf), False))

    # Operating-point evaluation (conf >= op_conf) — same greedy-by-conf order as PR-curve
    op_preds = [d for d in ordered if d.conf >= op_conf]
    matched: PredDet | None = None
    top_unmatched: PredDet | None = None
    best_iou = 0.0
    top1_conf_tp = False
    if op_preds:
        top1_iou = box_iou_xyxy(op_preds[0].box_xyxy, gt.box_xyxy)
        top1_conf_tp = top1_iou >= iou_thres
        for d in op_preds:
            iou = box_iou_xyxy(d.box_xyxy, gt.box_xyxy)
            if iou >= iou_thres:
                matched = d
                best_iou = iou
                break
        if matched is None:
            top_unmatched = op_preds[0]

    tp = matched is not None
    if tp:
        extra_fp = max(0, len(op_preds) - 1)
        fp_count = extra_fp
    else:
        extra_fp = 0
        fp_count = len(op_preds)

    # Keypoint metrics on GT keypoints physically inside the image frame
    box_total = int(gt.kps_in_frame.sum())
    palm_ok = gt.palm_px is not None and gt.palm_px > 1e-6
    palm21_total = box_total if palm_ok else 0
    palm14_mask = np.zeros(21, dtype=bool)
    palm14_mask[list(CONTROL_KPT_INDICES)] = True
    palm14_eval_mask = gt.kps_in_frame & palm14_mask
    palm14_total = int(palm14_eval_mask.sum()) if palm_ok else 0

    err_px = np.full(21, np.nan, dtype=np.float64)
    err_palm = np.full(21, np.nan, dtype=np.float64)
    box_hits = 0
    palm21_hits = 0
    palm14_hits = 0
    gate_pass = False
    gate_failed: list[int] = []
    gate_min_c = 0.0

    if matched is not None:
        dists = np.linalg.norm(matched.kps_xy - gt.kps_xy, axis=1)
        in_f = gt.kps_in_frame
        err_px[in_f] = dists[in_f]
        box_hits = int((dists[in_f] <= pck_thresh * gt.size_px).sum())
        if palm_ok:
            assert gt.palm_px is not None
            err_palm[in_f] = dists[in_f] / gt.palm_px
            palm21_hits = int((dists[in_f] <= pck_thresh * gt.palm_px).sum())
            palm14_hits = int((dists[palm14_eval_mask] <= pck_thresh * gt.palm_px).sum())
        gate_pass, gate_failed, gate_min_c = check_gate14(matched, conf_min=gate_conf)

    outcome = SampleOutcome(
        gt=gt,
        matched_pred=matched,
        top_unmatched_pred=top_unmatched,
        tp=tp,
        fp_count=fp_count,
        extra_fp_count=extra_fp,
        iou=float(best_iou) if tp else 0.0,
        pck_box_hits=box_hits,
        pck_box_total=box_total,
        pck_palm21_hits=palm21_hits,
        pck_palm21_total=palm21_total,
        pck_palm14_hits=palm14_hits,
        pck_palm14_total=palm14_total,
        joint_err_px=err_px,
        joint_err_palm=err_palm,
        gate14_pass=gate_pass,
        gate14_failed_joints=gate_failed,
        gate14_min_conf=gate_min_c,
        top1_conf_tp=top1_conf_tp,
    )
    return outcome, pr_records


def _prop_dict(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_ci(k, n)
    return {
        "k": int(k),
        "n": int(n),
        "rate": (k / n) if n > 0 else None,
        "ci95": [round(lo, 6), round(hi, 6)] if n > 0 else None,
    }


def aggregate_outcomes(
    outcomes: list[SampleOutcome],
    pr_records: list[tuple[float, bool]],
    *,
    op_conf: float = 0.5,
) -> dict[str, Any]:
    """Aggregate sample outcomes into detection, PCK, Gate14, per-joint and per-bucket stats."""
    n = len(outcomes)
    tp = sum(1 for o in outcomes if o.tp)
    top1_tp = sum(1 for o in outcomes if o.top1_conf_tp)
    fp = sum(o.fp_count for o in outcomes)
    extra_fp = sum(o.extra_fp_count for o in outcomes)
    fn = n - tp

    rec = tp / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    ap50 = compute_ap50(pr_records, n)

    tp_ious = [o.iou for o in outcomes if o.tp]
    mean_iou_overall = sum(o.iou for o in outcomes) / n if n > 0 else 0.0
    mean_iou_tp = (sum(tp_ious) / len(tp_ious)) if tp_ious else 0.0

    # Keypoint PCK@0.2 (Overall: FN counts as 0 hits; on_tp: conditional on TP)
    box_hits = sum(o.pck_box_hits for o in outcomes)
    box_tot_all = sum(o.pck_box_total for o in outcomes)
    box_tot_tp = sum(o.pck_box_total for o in outcomes if o.tp)

    palm21_hits = sum(o.pck_palm21_hits for o in outcomes)
    palm21_tot_all = sum(o.pck_palm21_total for o in outcomes)
    palm21_tot_tp = sum(o.pck_palm21_total for o in outcomes if o.tp)
    n_palm_imgs = sum(1 for o in outcomes if o.gt.palm_px is not None)
    n_palm_tp_imgs = sum(1 for o in outcomes if o.tp and o.gt.palm_px is not None)

    palm14_hits = sum(o.pck_palm14_hits for o in outcomes)
    palm14_tot_all = sum(o.pck_palm14_total for o in outcomes)
    palm14_tot_tp = sum(o.pck_palm14_total for o in outcomes if o.tp)

    # Gate14 dual denominators
    gate_pass_all_tp = sum(1 for o in outcomes if o.tp and o.gate14_pass)
    tp_gt14 = [o for o in outcomes if o.tp and o.gt.all14_in_frame]
    gate_pass_gt14_tp = sum(1 for o in tp_gt14 if o.gate14_pass)

    gate14_drop_counts = {KPT_NAMES[i]: 0 for i in CONTROL_KPT_INDICES}
    gate14_drop_counts_gt14 = {KPT_NAMES[i]: 0 for i in CONTROL_KPT_INDICES}
    for o in outcomes:
        if o.tp and not o.gate14_pass:
            for j in o.gate14_failed_joints:
                gate14_drop_counts[KPT_NAMES[j]] += 1
                if o.gt.all14_in_frame:
                    gate14_drop_counts_gt14[KPT_NAMES[j]] += 1

    # Per-joint pixel & normalized errors (across TP images where joint is in frame)
    per_joint: dict[str, dict[str, Any]] = {}
    if outcomes:
        px_mat = np.stack([o.joint_err_px for o in outcomes], axis=0)
        palm_mat = np.stack([o.joint_err_palm for o in outcomes], axis=0)
    else:
        px_mat = np.zeros((0, 21))
        palm_mat = np.zeros((0, 21))

    for j, name in enumerate(KPT_NAMES):
        valid_px = px_mat[:, j][np.isfinite(px_mat[:, j])]
        valid_pm = palm_mat[:, j][np.isfinite(palm_mat[:, j])]
        per_joint[name] = {
            "index": j,
            "is_control14": j in CONTROL_KPT_INDICES,
            "n_tp": int(valid_px.size),
            "mean_err_px": float(np.mean(valid_px)) if valid_px.size > 0 else None,
            "p95_err_px": float(np.percentile(valid_px, 95)) if valid_px.size > 0 else None,
            "mean_err_palm_pct": float(np.mean(valid_pm) * 100.0) if valid_pm.size > 0 else None,
        }

    top5_worst_joints = sorted(
        [v | {"name": k} for k, v in per_joint.items() if v["mean_err_palm_pct"] is not None],
        key=lambda x: x["mean_err_palm_pct"],
        reverse=True,
    )[:5]

    top5_gate14_drops = sorted(
        [{"name": k, "drops_all_tp": v, "drops_gt14_tp": gate14_drop_counts_gt14[k]}
         for k, v in gate14_drop_counts.items() if v > 0],
        key=lambda x: x["drops_gt14_tp"],
        reverse=True,
    )[:5]

    # Size-bucket breakdown
    by_bucket: dict[str, dict[str, Any]] = {}
    for bname, _, _ in SIZE_BUCKETS:
        sub = [o for o in outcomes if o.gt.size_bucket == bname]
        sn = len(sub)
        stp = sum(1 for o in sub if o.tp)
        sb_hits = sum(o.pck_box_hits for o in sub)
        sb_tot = sum(o.pck_box_total for o in sub)
        sp14_hits = sum(o.pck_palm14_hits for o in sub)
        sp14_tot = sum(o.pck_palm14_total for o in sub)
        sg_pass = sum(1 for o in sub if o.tp and o.gate14_pass)
        sub_gt14 = [o for o in sub if o.tp and o.gt.all14_in_frame]
        sg_gt14_pass = sum(1 for o in sub_gt14 if o.gate14_pass)
        by_bucket[bname] = {
            "n": sn,
            "recall_05": _prop_dict(stp, sn),
            "pck_box_overall": (sb_hits / sb_tot) if sb_tot > 0 else None,
            "pck_palm14_overall": (sp14_hits / sp14_tot) if sp14_tot > 0 else None,
            "gate14_all_tp": _prop_dict(sg_pass, stp),
            "gate14_gt14_tp": _prop_dict(sg_gt14_pass, len(sub_gt14)),
        }

    oob_kpts_excluded = sum(o.gt.oob_kp_count for o in outcomes)

    return {
        "n_images": n,
        "op_conf": op_conf,
        "oob_gt_kpts_excluded": int(oob_kpts_excluded),
        "detection": {
            "tp": tp,
            "fp": fp,
            "extra_fp_boxes": extra_fp,
            "fn": fn,
            "recall_05": _prop_dict(tp, n),
            "top1_conf_recall_05": _prop_dict(top1_tp, n),
            "precision_05": _prop_dict(tp, tp + fp),
            "f1_05": f1,
            "ap50": ap50,
            "mean_iou_overall": mean_iou_overall,
            "mean_iou_tp": mean_iou_tp,
        },
        "keypoints": {
            "pck_box_overall": {"hits": box_hits, "total": box_tot_all,
                                "rate": (box_hits / box_tot_all) if box_tot_all > 0 else None},
            "pck_box_on_tp": {"hits": box_hits, "total": box_tot_tp,
                              "rate": (box_hits / box_tot_tp) if box_tot_tp > 0 else None},
            "palm_valid_images": {"k": n_palm_imgs, "n": n, "tp_k": n_palm_tp_imgs},
            "pck_palm21_overall": {"hits": palm21_hits, "total": palm21_tot_all,
                                   "rate": (palm21_hits / palm21_tot_all) if palm21_tot_all > 0 else None},
            "pck_palm21_on_tp": {"hits": palm21_hits, "total": palm21_tot_tp,
                                 "rate": (palm21_hits / palm21_tot_tp) if palm21_tot_tp > 0 else None},
            "pck_palm14_overall": {"hits": palm14_hits, "total": palm14_tot_all,
                                   "rate": (palm14_hits / palm14_tot_all) if palm14_tot_all > 0 else None},
            "pck_palm14_on_tp": {"hits": palm14_hits, "total": palm14_tot_tp,
                                 "rate": (palm14_hits / palm14_tot_tp) if palm14_tot_tp > 0 else None},
            "top5_worst_joints": top5_worst_joints,
            "per_joint": per_joint,
        },
        "gate14": {
            "pass_all_tp": _prop_dict(gate_pass_all_tp, tp),
            "pass_gt14_tp": _prop_dict(gate_pass_gt14_tp, len(tp_gt14)),
            "drop_counts_all_tp": gate14_drop_counts,
            "drop_counts_gt14_tp": gate14_drop_counts_gt14,
            "top5_drop_joints": top5_gate14_drops,
        },
        "by_size_bucket": by_bucket,
    }


# ------------------------------------------------------------------ dataset audit

def parse_yolo_pose_label(
    image_path: str,
    label_path: str,
    width: int,
    height: int,
    sha256: str,
) -> SampleGT | None:
    """Parse one single-hand 21-keypoint YOLO pose label into a SampleGT."""
    if not os.path.exists(label_path):
        return None
    with open(label_path, encoding="utf-8") as fh:
        lines = [ln.strip().split() for ln in fh if ln.strip()]
    if not lines:
        return None
    extra_hands = max(0, len(lines) - 1)
    parts = lines[0]
    if len(parts) != 5 + 21 * 3:
        return None
    vals = [float(x) for x in parts[1:]]
    cx, cy, bw, bh = vals[:4]
    oob_box = (
        cx - bw / 2.0 < -1e-3 or cy - bh / 2.0 < -1e-3
        or cx + bw / 2.0 > 1.0 + 1e-3 or cy + bh / 2.0 > 1.0 + 1e-3
        or bw <= 0.0 or bh <= 0.0
    )
    x1 = float(np.clip((cx - bw / 2.0) * width, 0.0, float(width)))
    y1 = float(np.clip((cy - bh / 2.0) * height, 0.0, float(height)))
    x2 = float(np.clip((cx + bw / 2.0) * width, 0.0, float(width)))
    y2 = float(np.clip((cy + bh / 2.0) * height, 0.0, float(height)))
    size_px = max(x2 - x1, y2 - y1)
    if size_px <= 1e-6:
        return None

    raw_kps = np.asarray(vals[4:], dtype=np.float64).reshape(21, 3)
    kps_xy = np.column_stack([raw_kps[:, 0] * width, raw_kps[:, 1] * height])
    kps_vis = raw_kps[:, 2] > 0.0
    in_bounds = (
        (kps_xy[:, 0] >= 0.0) & (kps_xy[:, 0] <= float(width))
        & (kps_xy[:, 1] >= 0.0) & (kps_xy[:, 1] <= float(height))
    )
    kps_in_frame = kps_vis & in_bounds
    oob_kp_count = int((kps_vis & ~in_bounds).sum())
    oob_kp = oob_kp_count > 0

    palm_px: float | None = None
    if all(bool(kps_in_frame[i]) for i in PALM_INDICES):
        palm_px = hand_size(kps_xy)

    all14 = all(bool(kps_in_frame[i]) for i in CONTROL_KPT_INDICES)
    return SampleGT(
        image_path=str(Path(image_path).resolve()),
        sha256=sha256,
        width=int(width),
        height=int(height),
        box_xyxy=(x1, y1, x2, y2),
        size_px=float(size_px),
        size_bucket=size_bucket(size_px),
        kps_xy=kps_xy,
        kps_vis=kps_vis,
        kps_in_frame=kps_in_frame,
        palm_px=palm_px,
        all14_in_frame=all14,
        oob_box=oob_box,
        oob_kp=oob_kp,
        oob_kp_count=oob_kp_count,
        extra_hands_in_label=extra_hands,
    )


def find_near_duplicates_dhash(
    hashes: list[int],
    names: list[str],
    max_hamming: int = 4,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Vectorized all-pairs Hamming scan over 64-bit dHashes (chunked for low memory)."""
    n = len(hashes)
    if n < 2:
        return {"max_hamming": max_hamming, "pair_count": 0, "unique_images_involved": 0, "examples": []}
    arr = np.asarray(hashes, dtype=np.uint64)
    pair_count = 0
    involved: set[int] = set()
    examples: list[dict[str, Any]] = []
    chunk = 256
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        xor = arr[i0:i1, None] ^ arr[None, :]
        xor_bytes = xor.view(np.uint8).reshape(i1 - i0, n, 8)
        dist = _POPCOUNT_LUT[xor_bytes].sum(axis=2)
        rows, cols = np.where(dist <= max_hamming)
        for r, c in zip(rows.tolist(), cols.tolist()):
            gi = i0 + r
            if c <= gi:
                continue
            pair_count += 1
            involved.add(gi)
            involved.add(c)
            if len(examples) < max_examples:
                examples.append({"a": names[gi], "b": names[c], "hamming": int(dist[r, c])})
    return {
        "max_hamming": max_hamming,
        "pair_count": pair_count,
        "unique_images_involved": len(involved),
        "examples": examples,
    }


def find_cross_split_dhash(
    val_hashes: list[int],
    val_names: list[str],
    train_hashes: list[int],
    train_names: list[str],
    max_hamming: int = 2,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Find val images that have a perceptually near-identical counterpart in train."""
    if not val_hashes or not train_hashes:
        return {"max_hamming": max_hamming, "val_images_matched": 0, "pair_count": 0, "examples": []}
    va = np.asarray(val_hashes, dtype=np.uint64)
    ta = np.asarray(train_hashes, dtype=np.uint64)
    matched_val: set[int] = set()
    pair_count = 0
    examples: list[dict[str, Any]] = []
    chunk = 128
    for i0 in range(0, len(va), chunk):
        i1 = min(len(va), i0 + chunk)
        xor = va[i0:i1, None] ^ ta[None, :]
        xor_bytes = xor.view(np.uint8).reshape(i1 - i0, len(ta), 8)
        dist = _POPCOUNT_LUT[xor_bytes].sum(axis=2)
        rows, cols = np.where(dist <= max_hamming)
        for r, c in zip(rows.tolist(), cols.tolist()):
            gi = i0 + r
            pair_count += 1
            matched_val.add(gi)
            if len(examples) < max_examples:
                examples.append({"val": val_names[gi], "train": train_names[c], "hamming": int(dist[r, c])})
    return {
        "max_hamming": max_hamming,
        "val_images_matched": len(matched_val),
        "pair_count": pair_count,
        "examples": examples,
    }


def _resolve_label_path(img_path: Path, val_dir: Path, root: Path, val_subdir: str) -> Path:
    """Resolve YOLO label .txt path relative to dataset root (safe even if parent dirs contain 'images')."""
    rel = img_path.relative_to(val_dir)
    parts = Path(val_subdir).parts
    if parts and parts[0] == "images":
        lbl_dir = root.joinpath("labels", *parts[1:])
    else:
        lbl_dir = root / "labels" / Path(val_subdir)
    return (lbl_dir / rel).with_suffix(".txt")


def _collect_images(directory: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    out: list[Path] = []
    for ext in exts:
        out.extend(directory.glob(ext))
    return sorted(out)


def audit_and_clean_split(
    data_dir: str | Path,
    *,
    val_subdir: str = "images/val",
    train_subdir: str = "images/train",
    max_images: int | None = None,
    seed: int = 42,
    check_dhash: bool = True,
    dhash_max_hamming: int = 4,
) -> tuple[list[SampleGT], dict[str, Any]]:
    """Deduplicate val by SHA-256 (removing intra-val dups and train∩val leaks) and parse GT."""
    root = Path(data_dir).resolve()
    train_dir = root / train_subdir
    val_dir = root / val_subdir

    train_imgs = _collect_images(train_dir) if train_dir.is_dir() else []
    train_hashes: set[str] = set()
    train_unique_paths: list[Path] = []
    train_intra_dups = 0
    for p in train_imgs:
        h = file_sha256(p)
        if h in train_hashes:
            train_intra_dups += 1
        else:
            train_hashes.add(h)
            train_unique_paths.append(p)

    val_imgs = _collect_images(val_dir)
    seen_val: set[str] = set()
    leaked_train_val: list[str] = []
    val_intra_dups: list[str] = []
    clean_entries: list[tuple[Path, str]] = []
    for p in val_imgs:
        h = file_sha256(p)
        if h in train_hashes:
            leaked_train_val.append(p.name)
            continue
        if h in seen_val:
            val_intra_dups.append(p.name)
            continue
        seen_val.add(h)
        clean_entries.append((p, h))

    if max_images is not None and max_images < len(clean_entries):
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(clean_entries), size=max_images, replace=False).tolist())
        clean_entries = [clean_entries[i] for i in idx]

    samples: list[SampleGT] = []
    dhashes: list[int] = []
    dhash_names: list[str] = []
    skipped_bad = 0

    for img_path, h in clean_entries:
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_bad += 1
            continue
        H, W = img.shape[:2]
        lbl_path = str(_resolve_label_path(img_path, val_dir, root, val_subdir))
        gt = parse_yolo_pose_label(str(img_path), lbl_path, W, H, h)
        if gt is None:
            skipped_bad += 1
            continue
        samples.append(gt)
        if check_dhash:
            dhashes.append(dhash64(img))
            dhash_names.append(img_path.name)

    dhash_report = (
        find_near_duplicates_dhash(dhashes, dhash_names, max_hamming=dhash_max_hamming)
        if check_dhash
        else None
    )
    cross_dhash_report = None
    if check_dhash and train_unique_paths and dhashes:
        tr_dh: list[int] = []
        tr_nm: list[str] = []
        for tp in train_unique_paths:
            gray = cv2.imread(str(tp), cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                tr_dh.append(dhash64(gray))
                tr_nm.append(tp.name)
        cross_dhash_report = find_cross_split_dhash(dhashes, dhash_names, tr_dh, tr_nm, max_hamming=2)

    bucket_counts = {b: 0 for b, _, _ in SIZE_BUCKETS}
    for s in samples:
        bucket_counts[s.size_bucket] += 1
    sizes = np.array([s.size_px for s in samples], dtype=np.float64) if samples else np.zeros(0)

    audit = {
        "data_dir": str(root),
        "train_total_files": len(train_imgs),
        "train_intra_dups": train_intra_dups,
        "train_unique_files": len(train_hashes),
        "val_total_files": len(val_imgs),
        "val_leaked_from_train_count": len(leaked_train_val),
        "val_leaked_from_train_sample": leaked_train_val[:20],
        "val_intra_dups_count": len(val_intra_dups),
        "val_intra_dups_sample": val_intra_dups[:20],
        "val_clean_available": len(val_imgs) - len(leaked_train_val) - len(val_intra_dups),
        "val_evaluated": len(samples),
        "skipped_unreadable_or_bad_label": skipped_bad,
        "multi_hand_images_count": sum(1 for s in samples if s.extra_hands_in_label > 0),
        "multi_hand_extra_instances": sum(s.extra_hands_in_label for s in samples),
        "oob_box_count": sum(1 for s in samples if s.oob_box),
        "oob_kp_count": sum(1 for s in samples if s.oob_kp),
        "missing_palm5_count": sum(1 for s in samples if s.palm_px is None),
        "all14_in_frame_count": sum(1 for s in samples if s.all14_in_frame),
        "size_buckets": bucket_counts,
        "size_px_percentiles": (
            {str(p): round(float(v), 1) for p, v in zip((10, 25, 50, 75, 90), np.percentile(sizes, [10, 25, 50, 75, 90]))}
            if sizes.size > 0 else {}
        ),
        "dhash_near_duplicates": dhash_report,
        "dhash_cross_train_val": cross_dhash_report,
    }
    return samples, audit


# ------------------------------------------------------------------ model loading

def _inspect_weight(path_str: str, forced_imgsz: int | None) -> tuple[int, bool, int | None]:
    p = Path(path_str)
    is_onnx = p.suffix.lower() == ".onnx"
    if is_onnx:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        shape = sess.get_inputs()[0].shape
        native = int(shape[-1]) if len(shape) >= 4 and isinstance(shape[-1], int) else 640
        if forced_imgsz is not None and forced_imgsz != native:
            raise ValueError(f"ONNX model {path_str} has static imgsz={native}, cannot force {forced_imgsz}")
        return native, True, None

    import torch
    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
    ta = ckpt.get("train_args", {}) if isinstance(ckpt, dict) else {}
    raw_sz = ta.get("imgsz", 640)
    native = int(raw_sz[0] if isinstance(raw_sz, (list, tuple)) else raw_sz)
    model_obj = ckpt.get("model") if isinstance(ckpt, dict) else None
    nparams = int(sum(param.numel() for param in model_obj.parameters())) if model_obj is not None else None
    return int(forced_imgsz or native), False, nparams


def resolve_models(specs: list[str]) -> list[ModelSpec]:
    """Resolve model[@imgsz] specs and collapse byte-identical models at the same imgsz into aliases."""
    groups: dict[tuple[str, int], dict[str, Any]] = {}
    order: list[tuple[str, int]] = []
    for raw in specs:
        if "@" in raw and raw.rsplit("@", 1)[1].isdigit():
            path_part, sz_part = raw.rsplit("@", 1)
            forced_imgsz = int(sz_part)
        else:
            path_part, forced_imgsz = raw, None
        p = Path(path_part)
        if not p.exists():
            raise FileNotFoundError(f"model weight not found: {path_part}")
        sha = file_sha256(p)
        imgsz, is_onnx, nparams = _inspect_weight(str(p), forced_imgsz)
        key = (sha, imgsz)
        label = f"{path_part}@{imgsz}" if forced_imgsz is not None else path_part
        if key not in groups:
            groups[key] = {
                "display_name": label,
                "path": str(p),
                "sha256": sha,
                "imgsz": imgsz,
                "is_onnx": is_onnx,
                "params": nparams,
                "aliases": [],
            }
            order.append(key)
        else:
            groups[key]["aliases"].append(label)
    return [
        ModelSpec(
            display_name=groups[k]["display_name"],
            path=groups[k]["path"],
            sha256=groups[k]["sha256"],
            imgsz=groups[k]["imgsz"],
            is_onnx=groups[k]["is_onnx"],
            params=groups[k]["params"],
            aliases=tuple(groups[k]["aliases"]),
        )
        for k in order
    ]


def _predict_single(yolo_model, img: np.ndarray, imgsz: int, device: str, conf_floor: float = 0.01) -> list[PredDet]:
    res = yolo_model.predict(img, conf=conf_floor, imgsz=imgsz, device=device, verbose=False)[0]
    if res is None or res.boxes is None or len(res.boxes) == 0:
        return []
    if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
        return []
    xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.asarray(res.boxes.xyxy)
    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.asarray(res.boxes.conf)
    kpts_xy = res.keypoints.xy.cpu().numpy() if hasattr(res.keypoints.xy, "cpu") else np.asarray(res.keypoints.xy)
    kpts_cf = (
        res.keypoints.conf.cpu().numpy()
        if getattr(res.keypoints, "conf", None) is not None and hasattr(res.keypoints.conf, "cpu")
        else (np.asarray(res.keypoints.conf) if getattr(res.keypoints, "conf", None) is not None else None)
    )
    out: list[PredDet] = []
    for i in range(len(xyxy)):
        if i >= len(kpts_xy):
            break
        box = tuple(float(v) for v in xyxy[i][:4])
        kc = np.asarray(kpts_cf[i], dtype=np.float64) if kpts_cf is not None else np.ones(21, dtype=np.float64)
        out.append(PredDet(
            box_xyxy=box,
            conf=float(confs[i]),
            kps_xy=np.asarray(kpts_xy[i], dtype=np.float64).reshape(21, 2),
            kps_conf=kc.reshape(21),
        ))
    return out


# ------------------------------------------------------------------ stress-scale & negative

def make_scaled_sample(
    gt: SampleGT,
    img_bgr: np.ndarray,
    target_hand_px: float,
    rng: np.random.Generator,
    canvas_wh: tuple[int, int] = (640, 480),
    pad_val: int = 114,
) -> tuple[np.ndarray, SampleGT]:
    """Resize an image so the hand size equals target_hand_px and paste onto a neutral canvas."""
    cw, ch = canvas_wh
    scale = float(target_hand_px) / max(gt.size_px, 1.0)
    nw = max(4, int(round(gt.width * scale)))
    nh = max(4, int(round(gt.height * scale)))
    sx = nw / float(gt.width)
    sy = nh / float(gt.height)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)

    canvas = np.full((ch, cw, 3), pad_val, dtype=np.uint8)
    # Center the hand inside safe bounds of the canvas
    hx1, hy1, hx2, hy2 = gt.box_xyxy[0] * sx, gt.box_xyxy[1] * sy, gt.box_xyxy[2] * sx, gt.box_xyxy[3] * sy
    hcx, hcy = 0.5 * (hx1 + hx2), 0.5 * (hy1 + hy2)
    margin = max(24.0, 0.75 * target_hand_px)
    dst_cx = float(rng.uniform(margin, cw - margin))
    dst_cy = float(rng.uniform(margin, ch - margin))
    ox = int(round(dst_cx - hcx))
    oy = int(round(dst_cy - hcy))

    src_x1 = max(0, -ox)
    src_y1 = max(0, -oy)
    src_x2 = min(nw, cw - ox)
    src_y2 = min(nh, ch - oy)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[oy + src_y1:oy + src_y2, ox + src_x1:ox + src_x2] = resized[src_y1:src_y2, src_x1:src_x2]

    new_box = (
        float(np.clip(hx1 + ox, 0.0, float(cw))),
        float(np.clip(hy1 + oy, 0.0, float(ch))),
        float(np.clip(hx2 + ox, 0.0, float(cw))),
        float(np.clip(hy2 + oy, 0.0, float(ch))),
    )
    new_kps = np.column_stack([gt.kps_xy[:, 0] * sx + ox, gt.kps_xy[:, 1] * sy + oy])
    in_b = (new_kps[:, 0] >= 0.0) & (new_kps[:, 0] <= cw) & (new_kps[:, 1] >= 0.0) & (new_kps[:, 1] <= ch)
    in_f = gt.kps_vis & in_b
    new_size = max(new_box[2] - new_box[0], new_box[3] - new_box[1])
    new_palm = hand_size(new_kps) if all(bool(in_f[i]) for i in PALM_INDICES) else None
    new_gt = SampleGT(
        image_path=f"{gt.image_path}#scale{int(target_hand_px)}",
        sha256=gt.sha256,
        width=cw,
        height=ch,
        box_xyxy=new_box,
        size_px=float(new_size),
        size_bucket=size_bucket(new_size),
        kps_xy=new_kps,
        kps_vis=gt.kps_vis.copy(),
        kps_in_frame=in_f,
        palm_px=new_palm,
        all14_in_frame=all(bool(in_f[i]) for i in CONTROL_KPT_INDICES),
        oob_box=False,
        oob_kp=int((gt.kps_vis & ~in_b).sum()) > 0,
        oob_kp_count=int((gt.kps_vis & ~in_b).sum()),
    )
    return canvas, new_gt


def run_stress_scales(
    samples: list[SampleGT],
    predict_fn: Callable[[np.ndarray], list[PredDet]],
    *,
    scales: list[int],
    count: int = 300,
    seed: int = 42,
    op_conf: float = 0.5,
) -> dict[str, Any]:
    """Controlled downscale diagnostic on complete-hand samples (explicitly labeled non-real-distance)."""
    min_src_px = float(max(scales)) if scales else 0.0
    eligible = [
        s for s in samples
        if s.all14_in_frame and s.palm_px is not None and s.size_px >= min_src_px
    ]
    if not eligible:
        eligible = [s for s in samples if s.all14_in_frame and s.palm_px is not None]
    rng = np.random.default_rng(seed)
    chosen = eligible
    if len(eligible) > count:
        idx = sorted(rng.choice(len(eligible), size=count, replace=False).tolist())
        chosen = [eligible[i] for i in idx]

    out: dict[str, Any] = {
        "diagnostic_tag": "SCALED_DIAGNOSTIC_NOT_REAL_DISTANCE",
        "source_count": len(chosen),
        "min_source_hand_px": min_src_px,
        "scales": {},
    }
    for sc in scales:
        sc_rng = np.random.default_rng(seed + sc)
        outcomes: list[SampleOutcome] = []
        pr_all: list[tuple[float, bool]] = []
        for s in chosen:
            img = cv2.imread(s.image_path)
            if img is None:
                continue
            canvas, s_gt = make_scaled_sample(s, img, float(sc), sc_rng)
            preds = predict_fn(canvas)
            oc, prs = evaluate_sample_metrics(s_gt, preds, op_conf=op_conf)
            outcomes.append(oc)
            pr_all.extend(prs)
        agg = aggregate_outcomes(outcomes, pr_all, op_conf=op_conf)
        out["scales"][f"{sc}px"] = {
            "n": agg["n_images"],
            "recall_05": agg["detection"]["recall_05"],
            "ap50": agg["detection"]["ap50"],
            "pck_box_overall": agg["keypoints"]["pck_box_overall"]["rate"],
            "pck_palm14_overall": agg["keypoints"]["pck_palm14_overall"]["rate"],
            "gate14_gt14_tp": agg["gate14"]["pass_gt14_tp"],
        }
    return out


def run_negative_fpr(
    neg_dir: str | Path,
    predict_fn: Callable[[np.ndarray], list[PredDet]],
    *,
    op_conf: float = 0.5,
) -> dict[str, Any]:
    """Measure background false-positive rate on a folder of hand-free images."""
    paths = _collect_images(Path(neg_dir).resolve())
    imgs_with_fp = 0
    total_fp_boxes = 0
    evaluated = 0
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        evaluated += 1
        preds = [d for d in predict_fn(img) if d.conf >= op_conf]
        if preds:
            imgs_with_fp += 1
            total_fp_boxes += len(preds)
    return {
        "neg_dir": str(Path(neg_dir).resolve()),
        "n_images": evaluated,
        "fpr": _prop_dict(imgs_with_fp, evaluated),
        "total_fp_boxes": total_fp_boxes,
        "fp_boxes_per_100_images": (100.0 * total_fp_boxes / evaluated) if evaluated > 0 else None,
    }


# ------------------------------------------------------------------ latency & ultralytics val

def parse_latency_sizes(spec: str) -> tuple[tuple[int, int], ...]:
    """Parse comma-separated WxH frame sizes (e.g. '640x480,1280x720')."""
    sizes: list[tuple[int, int]] = []
    for token in spec.split(","):
        tok = token.strip().lower()
        if not tok:
            continue
        if "x" not in tok:
            raise ValueError(f"invalid latency size {token!r}, expected WxH (e.g. 640x480)")
        w_s, h_s = tok.split("x", 1)
        w, h = int(w_s), int(h_s)
        if w <= 0 or h <= 0:
            raise ValueError(f"latency size dimensions must be positive: {token!r}")
        sizes.append((w, h))
    return tuple(sizes) if sizes else ((640, 480),)


def benchmark_latency(
    spec: ModelSpec,
    sample_images: list[np.ndarray],
    *,
    warmup: int = 20,
    iters: int = 200,
    devices: tuple[str, ...] = ("0", "cpu"),
    frame_sizes: tuple[tuple[int, int], ...] = ((640, 480), (1280, 720)),
) -> dict[str, Any]:
    """Measure steady-state single-frame inference latency on GPU and CPU at webcam frame sizes."""
    import torch
    from ultralytics import YOLO

    if not sample_images:
        return {}

    by_res: dict[str, dict[str, Any]] = {}
    resized_sets: dict[str, list[np.ndarray]] = {}
    for fw, fh in frame_sizes:
        key_res = f"{fw}x{fh}"
        resized_sets[key_res] = [
            im if (im.shape[1] == fw and im.shape[0] == fh) else cv2.resize(im, (fw, fh), interpolation=cv2.INTER_LINEAR)
            for im in sample_images
        ]
        by_res[key_res] = {}

    for dev in devices:
        if dev != "cpu" and not torch.cuda.is_available():
            continue
        yolo = YOLO(spec.path, task="pose")
        dev_key = "gpu" if dev != "cpu" else "cpu"
        for key_res, imgs in resized_sets.items():
            providers: list[str] | None = None
            for i in range(warmup):
                yolo.predict(imgs[i % len(imgs)], conf=0.5, imgsz=spec.imgsz, device=dev, verbose=False)
            if spec.is_onnx:
                try:
                    providers = list(yolo.predictor.model.session.get_providers())
                except Exception:
                    providers = None
            if dev != "cpu" and not spec.is_onnx:
                torch.cuda.synchronize()

            times_ms: list[float] = []
            for i in range(iters):
                img = imgs[i % len(imgs)]
                if dev != "cpu" and not spec.is_onnx:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                yolo.predict(img, conf=0.5, imgsz=spec.imgsz, device=dev, verbose=False)
                if dev != "cpu" and not spec.is_onnx:
                    torch.cuda.synchronize()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

            arr = np.asarray(times_ms, dtype=np.float64)
            mean_ms = float(np.mean(arr))
            by_res[key_res][dev_key] = {
                "device": dev,
                "resolution": key_res,
                "providers": providers,
                "warmup": warmup,
                "iters": iters,
                "mean_ms": round(mean_ms, 3),
                "p50_ms": round(float(np.percentile(arr, 50)), 3),
                "p95_ms": round(float(np.percentile(arr, 95)), 3),
                "fps": round(1000.0 / mean_ms, 2) if mean_ms > 0 else 0.0,
            }

    primary_res = f"{frame_sizes[0][0]}x{frame_sizes[0][1]}"
    out: dict[str, Any] = {
        "primary_resolution": primary_res,
        "by_resolution": by_res,
    }
    for k in ("gpu", "cpu"):
        if k in by_res.get(primary_res, {}):
            out[k] = by_res[primary_res][k]
    return out


def build_clean_val_yaml(
    data_dir: str | Path,
    clean_list_path: Path,
    out_yaml_path: Path,
    *,
    use_dataset_oks_sigmas: bool = False,
) -> tuple[Path, str]:
    """Create clean_val.yaml for Ultralytics .val() using standard OKS sigmas (1/21) by default."""
    import yaml

    root = Path(data_dir).resolve()
    # Prefer standard data.yaml over custom training variants (e.g. data_fixed.yaml with 0.12 OKS sigmas)
    orig_yaml = root / "data.yaml"
    if not orig_yaml.exists():
        orig_yaml = root / "data_fixed.yaml"
    orig_cfg = yaml.safe_load(orig_yaml.read_text(encoding="utf-8")) if orig_yaml.exists() else {}
    clean_yaml_cfg: dict[str, Any] = {
        "path": str(out_yaml_path.parent),
        "train": str(clean_list_path),
        "val": str(clean_list_path),
        "kpt_shape": orig_cfg.get("kpt_shape", [21, 3]),
        "names": orig_cfg.get("names", {0: "hand"}),
    }
    if "flip_idx" in orig_cfg:
        clean_yaml_cfg["flip_idx"] = orig_cfg["flip_idx"]
    oks_mode = "ultralytics_default_1_over_21"
    if use_dataset_oks_sigmas and "kpt_oks_sigmas" in orig_cfg:
        clean_yaml_cfg["kpt_oks_sigmas"] = orig_cfg["kpt_oks_sigmas"]
        oks_mode = f"custom_yaml_sigmas({orig_yaml.name})"
    out_yaml_path.write_text(yaml.safe_dump(clean_yaml_cfg), encoding="utf-8")
    return out_yaml_path, oks_mode


def run_ultralytics_val(
    spec: ModelSpec,
    clean_val_yaml: Path,
    *,
    device: str = "0",
    workers: int = 0,
    oks_sigma_mode: str = "ultralytics_default_1_over_21",
) -> dict[str, Any]:
    """Run official Ultralytics .val() on clean_val_list.txt with rect=False for strict .pt/.onnx parity."""
    from ultralytics import YOLO

    yolo = YOLO(spec.path, task="pose")
    res = yolo.val(
        data=str(clean_val_yaml),
        imgsz=spec.imgsz,
        batch=1 if spec.is_onnx else 16,
        rect=False,
        device=device,
        plots=False,
        verbose=False,
        workers=workers,
    )
    n_imgs = int(getattr(res.box, "nt_per_image", np.array([])).size) if hasattr(res, "box") else None
    if not n_imgs:
        try:
            n_imgs = int(sum(res.nt_per_class))
        except Exception:
            n_imgs = None
    return {
        "oks_sigma_mode": oks_sigma_mode,
        "n_instances_evaluated": n_imgs,
        "box_map50": float(res.box.map50),
        "box_map50_95": float(res.box.map),
        "pose_map50": float(res.pose.map50),
        "pose_map50_95": float(res.pose.map),
    }


# ------------------------------------------------------------------ visual audit

def _render_audit_image(outcome: SampleOutcome, min_width: int = 540) -> np.ndarray | None:
    src = cv2.imread(outcome.gt.image_path)
    if src is None:
        return None
    H0, W0 = src.shape[:2]
    scale = max(1.0, float(min_width) / max(float(W0), 1.0))
    W = int(round(W0 * scale))
    H = int(round(H0 * scale))
    img = cv2.resize(src, (W, H), interpolation=cv2.INTER_LINEAR) if scale > 1.0 else src.copy()

    def _pt(xy):
        return (int(round(xy[0] * scale)), int(round(xy[1] * scale)))

    # Draw GT box & 21 keypoints in GREEN
    gx1, gy1, gx2, gy2 = (int(round(v * scale)) for v in outcome.gt.box_xyxy)
    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 220, 0), 2)
    for a, b in HAND_CONNECTIONS:
        if outcome.gt.kps_in_frame[a] and outcome.gt.kps_in_frame[b]:
            cv2.line(img, _pt(outcome.gt.kps_xy[a]), _pt(outcome.gt.kps_xy[b]), (0, 180, 0), 2)
    for i in range(21):
        if outcome.gt.kps_in_frame[i]:
            cv2.circle(img, _pt(outcome.gt.kps_xy[i]), 4, (0, 255, 0), -1)

    # Draw matched Pred in RED, or highest-conf unmatched Pred on FN in ORANGE
    draw_pred = outcome.matched_pred or outcome.top_unmatched_pred
    if draw_pred is not None:
        is_matched = outcome.matched_pred is not None
        box_col = (0, 0, 255) if is_matched else (0, 165, 255)
        line_col = (40, 40, 220) if is_matched else (0, 140, 230)
        px1, py1, px2, py2 = (int(round(v * scale)) for v in draw_pred.box_xyxy)
        cv2.rectangle(img, (px1, py1), (px2, py2), box_col, 2)
        if not is_matched:
            cv2.putText(img, f"unmatched conf={draw_pred.conf:.2f}",
                        (max(4, px1), max(16, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_col, 1, cv2.LINE_AA)
        for a, b in HAND_CONNECTIONS:
            p1, p2 = _pt(draw_pred.kps_xy[a]), _pt(draw_pred.kps_xy[b])
            if (p1[0] > 0 or p1[1] > 0) and (p2[0] > 0 or p2[1] > 0):
                cv2.line(img, p1, p2, line_col, 2)
        failed_set = set(outcome.gate14_failed_joints) if is_matched else set()
        for i in range(21):
            p = _pt(draw_pred.kps_xy[i])
            if p[0] > 0 or p[1] > 0:
                cv2.circle(img, p, 4, box_col, -1)
            if i in failed_set:
                cv2.circle(img, p, 9, (0, 255, 255), 2)
                cv2.putText(img, f"{i}:{draw_pred.kps_conf[i]:.2f}", (min(W - 70, p[0] + 8), max(16, p[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)

    # Clean 2-line header banner padded above image (never clips on small 224px images)
    pck_b = (outcome.pck_box_hits / outcome.pck_box_total * 100.0) if outcome.pck_box_total > 0 else 0.0
    name = os.path.basename(outcome.gt.image_path)
    banner = np.full((52, W, 3), 24, dtype=np.uint8)
    if outcome.matched_pred is None and outcome.top_unmatched_pred is not None:
        conf_tag = f"unmC={outcome.top_unmatched_pred.conf:.2f}"
    else:
        conf_tag = f"minC14={outcome.gate14_min_conf:.2f}"
    line1 = f"{name} | IoU={outcome.iou:.2f} | PCK_box={pck_b:.0f}% | {conf_tag} | gt_px={outcome.gt.size_px:.0f}"
    line2 = "GT=green | Matched=red | UnmatchedPred=orange | Gate14Fail=yellow"
    cv2.putText(banner, line1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(banner, line2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (195, 195, 195), 1, cv2.LINE_AA)
    return np.vstack([banner, img])


def save_visual_audit(outcomes: list[SampleOutcome], audit_dir: Path, max_per_cat: int = 20) -> dict[str, int]:
    """Save Top-20 worst_pck, gate14_drops and false_negatives (largest + smallest) overlays."""
    counts = {"worst_pck": 0, "gate14_drops": 0, "false_negatives": 0}
    tp_list = [o for o in outcomes if o.tp and o.pck_box_total > 0]
    worst_pck = sorted(tp_list, key=lambda o: (o.pck_box_hits / o.pck_box_total, o.iou))[:max_per_cat]
    gate_drops = sorted(
        [o for o in outcomes if o.tp and o.gt.all14_in_frame and not o.gate14_pass],
        key=lambda o: (o.gate14_min_conf, -len(o.gate14_failed_joints)),
    )[:max_per_cat]

    all_fns = [o for o in outcomes if not o.tp]
    if len(all_fns) <= max_per_cat:
        fns = sorted(all_fns, key=lambda o: o.gt.size_px, reverse=True)
    else:
        half = max_per_cat // 2
        largest = sorted(all_fns, key=lambda o: o.gt.size_px, reverse=True)[:half]
        picked_ids = {id(o) for o in largest}
        smallest = [o for o in sorted(all_fns, key=lambda o: o.gt.size_px) if id(o) not in picked_ids][:max_per_cat - half]
        fns = largest + smallest

    for cat, items in (("worst_pck", worst_pck), ("gate14_drops", gate_drops), ("false_negatives", fns)):
        sub = audit_dir / cat
        sub.mkdir(parents=True, exist_ok=True)
        for old_png in sub.glob("*.png"):
            old_png.unlink()
        for rank, oc in enumerate(items, start=1):
            vis = _render_audit_image(oc)
            if vis is None:
                continue
            stem = Path(oc.gt.image_path).stem
            cv2.imwrite(str(sub / f"{rank:02d}_{stem}.png"), vis)
            counts[cat] += 1
    return counts


# ------------------------------------------------------------------ report writer

def _pct(val: float | None) -> str:
    return "N/A" if val is None else f"{val * 100.0:.2f}%"


def _pct_ci(prop: dict[str, Any] | None) -> str:
    if not prop or prop.get("rate") is None:
        return "N/A"
    lo, hi = prop["ci95"]
    return f"{prop['rate'] * 100.0:.2f}% ({prop['k']}/{prop['n']}, 95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"


def write_tier1_markdown(payload: dict[str, Any], md_path: Path) -> None:
    """Render a comprehensive, self-documenting Tier-1 benchmark report."""
    da = payload["dataset_audit"]
    git = payload["git"]
    lines: list[str] = [
        "# Báo cáo Benchmark Tầng 1 — Phát hiện Tay, 21 Khớp, Cổng Gate14 & Latency",
        "",
        f"- **Thời điểm chạy:** `{payload['timestamp']}`",
        f"- **Git commit:** `{git['commit'][:12]}` (dirty=`{git['dirty']}`)",
        f"- **Dataset nguồn:** `{da['data_dir']}` (*in-distribution validation split*)",
        f"- **Làm sạch dữ liệu (SHA-256):** `val` gốc `{da['val_total_files']}` ảnh → loại **`{da['val_leaked_from_train_count']}`** ảnh rò rỉ `train ∩ val` và **`{da['val_intra_dups_count']}`** ảnh trùng nội bộ → chấm trên **`{da['val_evaluated']}` ảnh sạch**.",
    ]
    if da.get("dhash_near_duplicates"):
        dh = da["dhash_near_duplicates"]
        lines.append(
            f"- **Cảnh báo gần trùng nội bộ `val` (`dHash ≤ {dh['max_hamming']}`):** phát hiện `{dh['pair_count']}` cặp "
            f"(`{dh['unique_images_involved']}` ảnh liên quan trong `val` sạch)."
        )
    if da.get("dhash_cross_train_val"):
        cd = da["dhash_cross_train_val"]
        lines.append(
            f"- **Cảnh báo gần trùng chéo `train ∩ val` (`dHash ≤ {cd['max_hamming']}`):** "
            f"`{cd['val_images_matched']}/{da['val_evaluated']}` ảnh `val` có ảnh rất giống trong `train` (`{cd['pair_count']}` cặp)."
        )
    small_cnt = int(da["size_buckets"].get("<40", 0))
    small_pct = (100.0 * small_cnt / da["val_evaluated"]) if da["val_evaluated"] > 0 else 0.0
    med_px = da["size_px_percentiles"].get("50", "N/A")
    oks_tag = payload.get("oks_sigma_mode") or "ultralytics_default_1_over_21"
    oks_short = "σ=1/21" if "1_over_21" in oks_tag else oks_tag
    lines.extend([
        f"- **Đặc điểm nhãn trên tập sạch:** `oob_box={da['oob_box_count']}`, `oob_kp={da['oob_kp_count']}`, "
        f"`missing_palm5={da['missing_palm5_count']}`, `all14_in_frame={da['all14_in_frame_count']}/{da['val_evaluated']}`, "
        f"`multi_hand_imgs={da.get('multi_hand_images_count', 0)}`.",
        f"- **Phân bố kích thước tay (`max(bw, bh)` px):** {da['size_buckets']} · percentiles={da['size_px_percentiles']}.",
        f"- **Chuẩn OKS của Ultralytics `.val()`:** `{oks_tag}` (`rect=False` công bằng cho `.pt` & `.onnx`).",
        "",
        f"> **Lưu ý phạm vi kết luận:** `hand-keypoints/val` dùng chung nguồn với `train` (và đã dùng để chọn `best.pt` qua 25 epoch), đồng thời chỉ có **{small_cnt}/{da['val_evaluated']} ({small_pct:.2f}%) ảnh tay < 40 px** (trung vị `{med_px} px`). Đây là điểm chuẩn **cận cảnh / nội bộ** để so trọng số, kiểm tra ONNX (`rect=False` công bằng), soi khớp yếu và `Gate14` — chưa thay thế đánh giá video thực tế ở 1,5–3 m (Tầng 2 & 3).",
        "",
        "## 1. Bảng so sánh tổng hợp các trọng số (Tầng 1 — Ngưỡng vận hành `conf ≥ 0.50`)",
        "",
        f"| Model | `imgsz` | SHA-256 | Recall@0.5 | Precision@0.5 | Extra FP | AP@0.5 | Box mAP50–95 (Ultra) | Pose mAP50–95 ({oks_short}) | PCK@0.2 Box | PCK@0.2 Palm (14 kpt) | Gate14 (GT14 TP) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for m in payload["models"]:
        det = m["metrics"]["detection"]
        kp = m["metrics"]["keypoints"]
        g14 = m["metrics"]["gate14"]
        ul = m.get("ultralytics_val") or {}
        alias_tag = f" *(≡ {', '.join(m['aliases'])})*" if m["aliases"] else ""
        lines.append(
            f"| `{m['display_name']}`{alias_tag} | {m['imgsz']} | `{m['sha256'][:12]}` | "
            f"{_pct(det['recall_05']['rate'])} | {_pct(det['precision_05']['rate'])} | {det['extra_fp_boxes']} | "
            f"{_pct(det['ap50'])} | {_pct(ul.get('box_map50_95'))} | {_pct(ul.get('pose_map50_95'))} | "
            f"{_pct(kp['pck_box_overall']['rate'])} | {_pct(kp['pck_palm14_overall']['rate'])} | "
            f"{_pct(g14['pass_gt14_tp']['rate'])} |"
        )

    lines.extend(["", "## 2. Chi tiết mẫu số `k/n`, 95% CI và Lát cắt loại trừ `val.txt`", ""])
    for m in payload["models"]:
        det = m["metrics"]["detection"]
        kp = m["metrics"]["keypoints"]
        g14 = m["metrics"]["gate14"]
        ul = m.get("ultralytics_val")
        lines.append(f"### `{m['display_name']}` (`imgsz={m['imgsz']}`, SHA-256 `{m['sha256'][:16]}`)")
        top1_str = f" · **Top-1 Conf Recall@0.5:** {_pct(det['top1_conf_recall_05']['rate'])}" if det.get("top1_conf_recall_05") else ""
        lines.append(
            f"- **Recall@0.5:** {_pct_ci(det['recall_05'])}{top1_str} · "
            f"**Precision@0.5:** {_pct_ci(det['precision_05'])} · **Mean IoU (TP):** `{det['mean_iou_tp']:.4f}`"
        )
        lines.append(
            f"- **PCK@0.2 Box:** Overall `{_pct(kp['pck_box_overall']['rate'])}` / on TP `{_pct(kp['pck_box_on_tp']['rate'])}` "
            f"(trên `100%` ảnh `{m['metrics']['n_images']}`; loại `{m['metrics']['oob_gt_kpts_excluded']}` khớp GT ngoài khung)."
        )
        pv = kp["palm_valid_images"]
        lines.append(
            f"- **PCK@0.2 Palm (mẫu số `{pv['k']}/{pv['n']}` ảnh đủ 5 khớp lòng bàn tay):** "
            f"21 khớp Overall `{_pct(kp['pck_palm21_overall']['rate'])}` (on TP `{_pct(kp['pck_palm21_on_tp']['rate'])}`) · "
            f"**14 khớp điều khiển** Overall `{_pct(kp['pck_palm14_overall']['rate'])}` (on TP `{_pct(kp['pck_palm14_on_tp']['rate'])}`)."
        )
        lines.append(
            f"- **Cổng `Gate14` (`all 14 kpts conf ≥ 0.5`):** "
            f"All TP = {_pct_ci(g14['pass_all_tp'])} · "
            f"**GT đủ 14 khớp TP** = {_pct_ci(g14['pass_gt14_tp'])}."
        )
        if ul:
            lines.append(
                f"- **Ultralytics `.val()` (`{ul.get('oks_sigma_mode', oks_tag)}`, mẫu số `{ul.get('n_instances_evaluated')}` instances):** "
                f"Box mAP50 `{_pct(ul['box_map50'])}`, Box mAP50–95 `{_pct(ul['box_map50_95'])}`, "
                f"Pose mAP50 `{_pct(ul['pose_map50'])}`, Pose mAP50–95 `{_pct(ul['pose_map50_95'])}`."
            )
        if m.get("strict_heldout_metrics"):
            sh = m["strict_heldout_metrics"]
            lines.append(
                f"- **Lát cắt loại trừ `{payload['exclude_audit_list']}` (`n={sh['n_images']}`):** "
                f"Recall@0.5 `{_pct(sh['detection']['recall_05']['rate'])}`, AP@0.5 `{_pct(sh['detection']['ap50'])}`, "
                f"PCK@0.2 Box `{_pct(sh['keypoints']['pck_box_overall']['rate'])}`, "
                f"PCK@0.2 Palm 14-kpt `{_pct(sh['keypoints']['pck_palm14_overall']['rate'])}`, "
                f"Gate14 (GT14 TP) `{_pct(sh['gate14']['pass_gt14_tp']['rate'])}`."
            )
        # Top-5 worst joints & top-5 Gate14 drops
        w5 = ", ".join(f"`{j['name']}` ({j['mean_err_palm_pct']:.1f}% palm / {j['mean_err_px']:.1f}px)"
                       for j in kp["top5_worst_joints"])
        d5 = ", ".join(f"`{j['name']}` ({j['drops_gt14_tp']} lượt GT14 / {j['drops_all_tp']} tổng)"
                       for j in g14["top5_drop_joints"])
        lines.append(f"- **Top-5 khớp sai lệch lớn nhất:** {w5 or 'không có'}")
        lines.append(f"- **Top-5 khớp làm rớt Gate14 nhiều nhất:** {d5 or 'không có'}")
        lines.append("")

    # Size bucket comparison for first (reference) model
    ref = payload["models"][0]
    lines.extend([
        f"## 3. Phân tầng theo kích thước bàn tay (`px`) — Model tham chiếu `{ref['display_name']}`",
        "",
        "| Bucket (`max(bw,bh)`) | Số ảnh `n` | Recall@0.5 | PCK@0.2 Box | PCK@0.2 Palm (14 kpt) | Gate14 (GT14 TP) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for bname, b in ref["metrics"]["by_size_bucket"].items():
        lines.append(
            f"| `{bname} px` | {b['n']} | {_pct(b['recall_05']['rate'])} | "
            f"{_pct(b['pck_box_overall'])} | {_pct(b['pck_palm14_overall'])} | {_pct(b['gate14_gt14_tp']['rate'])} |"
        )

    # Stress-scale diagnostic
    if any(m.get("stress_scales") for m in payload["models"]):
        lines.extend([
            "",
            "## 4. Chẩn đoán thu nhỏ bàn tay `[SCALED DIAGNOSTIC — KHÔNG PHẢI KHOẢNG CÁCH THẬT]`",
            "",
            "| Model | Mức thu nhỏ | Số ảnh | Recall@0.5 | AP@0.5 | PCK@0.2 Box | PCK@0.2 Palm (14 kpt) | Gate14 (GT14 TP) |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for m in payload["models"]:
            ss = m.get("stress_scales") or {}
            for sname, vals in ss.get("scales", {}).items():
                lines.append(
                    f"| `{m['display_name']}` | `{sname}` | {vals['n']} | "
                    f"{_pct(vals['recall_05']['rate'])} | {_pct(vals['ap50'])} | "
                    f"{_pct(vals['pck_box_overall'])} | {_pct(vals['pck_palm14_overall'])} | "
                    f"{_pct(vals['gate14_gt14_tp']['rate'])} |"
                )

    # Latency table (webcam resolutions, GPU & CPU, explicit GPU & CPU ONNX providers)
    lat_res_keys: list[str] = []
    for m in payload["models"]:
        br = (m.get("latency") or {}).get("by_resolution") or {}
        for rk in br:
            if rk not in lat_res_keys:
                lat_res_keys.append(rk)
    if not lat_res_keys:
        lat_res_keys = ["default"]

    lines.extend([
        "",
        "## 5. Tốc độ suy luận thực tế trên khung webcam (20 warmup + 200 ảnh, tách GPU & CPU)",
        "",
        "| Model | `imgsz` | Khung đo | GPU Mean (ms) | GPU P50 / P95 (ms) | GPU FPS | CPU Mean (ms) | CPU P50 / P95 (ms) | CPU FPS | ONNX Providers (GPU / CPU) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for rk in lat_res_keys:
        for m in payload["models"]:
            lat = m.get("latency") or {}
            block = (lat.get("by_resolution") or {}).get(rk) if rk != "default" else lat
            if not block:
                continue
            g = block.get("gpu") or {}
            c = block.get("cpu") or {}
            gp = ", ".join(g["providers"]) if isinstance(g.get("providers"), list) else "-"
            cp = ", ".join(c["providers"]) if isinstance(c.get("providers"), list) else "-"
            prov_str = "-" if (gp == "-" and cp == "-") else f"GPU: {gp} \\| CPU: {cp}"
            lines.append(
                f"| `{m['display_name']}` | {m['imgsz']} | `{rk}` | "
                f"{g.get('mean_ms', 'N/A')} | {g.get('p50_ms', 'N/A')} / {g.get('p95_ms', 'N/A')} | {g.get('fps', 'N/A')} | "
                f"{c.get('mean_ms', 'N/A')} | {c.get('p50_ms', 'N/A')} / {c.get('p95_ms', 'N/A')} | {c.get('fps', 'N/A')} | `{prov_str}` |"
            )

    if payload.get("audit_images"):
        ai = payload["audit_images"]
        lines.extend([
            "",
            "## 6. Bằng chứng trực quan kiểm chứng bằng mắt (`audit_top20_errors/`)",
            "",
            f"- Model tham chiếu: `{ref['display_name']}`",
            f"- `audit_top20_errors/worst_pck/`: **{ai['worst_pck']}** ảnh TP lệch khớp nặng nhất.",
            f"- `audit_top20_errors/gate14_drops/`: **{ai['gate14_drops']}** ảnh GT đủ 14 khớp nhưng rớt cổng `Gate14`.",
            f"- `audit_top20_errors/false_negatives/`: **{ai['false_negatives']}** ảnh model bỏ sót (10 bàn tay lớn nhất + 10 bàn tay nhỏ nhất).",
            "- Quy ước màu: **Xanh lá = GT**, **Đỏ = Dự đoán khớp**, **Cam = Dự đoán lệch trên ca FN**, **Vòng vàng = khớp làm rớt `Gate14` kèm `conf`**.",
        ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("../datasets/hand-keypoints"),
                    help="YOLO hand-keypoints dataset root")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model weight paths; append @<imgsz> to override PyTorch imgsz (e.g. hand_pose.pt@640)")
    ap.add_argument("--output", type=Path, default=Path("benchmark/results/tier1"))
    ap.add_argument("--max-images", type=int, default=None, help="Subsample clean val for quick runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--op-conf", type=float, default=0.5, help="Operating confidence threshold")
    ap.add_argument("--device", default="0", help="Primary inference device ('0' for CUDA GPU, 'cpu' for CPU)")
    ap.add_argument("--exclude-audit-list", type=Path, default=None,
                    help="Optional .txt file of image paths (e.g. runs/hand_only_trial/val.txt) to also report a strict held-out slice")
    ap.add_argument("--stress-scales", default="", help="Comma-separated hand px targets for downscale diagnostic, e.g. 36,64")
    ap.add_argument("--stress-count", type=int, default=300)
    ap.add_argument("--neg-dir", type=Path, default=None, help="Optional folder of hand-free images to measure Background FPR")
    ap.add_argument("--latency-warmup", type=int, default=20)
    ap.add_argument("--latency-iters", type=int, default=200)
    ap.add_argument("--latency-sizes", default="640x480,1280x720",
                    help="Comma-separated webcam WxH resolutions for latency benchmark (default: 640x480,1280x720)")
    ap.add_argument("--val-workers", type=int, default=0,
                    help="DataLoader workers for Ultralytics .val() (0 avoids post-CUDA fork deadlocks)")
    ap.add_argument("--use-dataset-oks-sigmas", action="store_true",
                    help="Copy custom kpt_oks_sigmas from dataset YAML instead of standard Ultralytics 1/21")
    ap.add_argument("--skip-ultralytics-val", action="store_true")
    ap.add_argument("--skip-dhash", action="store_true")
    ap.add_argument("--skip-cpu-latency", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch
    from ultralytics import YOLO

    args = parse_args(argv)
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    print(f"[Stage 1] Auditing & deduplicating dataset: {args.data}")
    samples, audit = audit_and_clean_split(
        args.data, max_images=args.max_images, seed=args.seed, check_dhash=not args.skip_dhash
    )
    clean_list_path = out_dir / "clean_val_list.txt"
    clean_list_path.write_text("".join(f"{s.image_path}\n" for s in samples), encoding="utf-8")
    (out_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    # Build clean_val.yaml for Ultralytics .val() using standard OKS sigmas (1/21) by default
    clean_yaml_path, oks_sigma_mode = build_clean_val_yaml(
        args.data,
        clean_list_path,
        out_dir / "clean_val.yaml",
        use_dataset_oks_sigmas=args.use_dataset_oks_sigmas,
    )

    exclude_set: set[str] = set()
    if args.exclude_audit_list and args.exclude_audit_list.exists():
        exclude_set = {
            str(Path(ln.strip()).resolve())
            for ln in args.exclude_audit_list.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        }

    print(f"[Stage 2] Resolving {len(args.models)} model specs...")
    specs = resolve_models(args.models)
    for sp in specs:
        print(f"  - {sp.display_name} (imgsz={sp.imgsz}, sha256={sp.sha256[:12]}, aliases={list(sp.aliases)})")

    scales = [int(s.strip()) for s in args.stress_scales.split(",") if s.strip()]
    lat_sizes = parse_latency_sizes(args.latency_sizes)
    # Pre-load images for latency benchmarking
    lat_imgs: list[np.ndarray] = []
    for s in samples[:max(args.latency_warmup, min(args.latency_iters, len(samples)))]:
        im = cv2.imread(s.image_path)
        if im is not None:
            lat_imgs.append(im)

    model_reports: list[dict[str, Any]] = []
    audit_counts: dict[str, int] | None = None

    for idx, sp in enumerate(specs):
        print(f"\n[Stage 3] Evaluating {sp.display_name} on {len(samples)} clean val images (dev={device})...")
        t_start = time.perf_counter()
        yolo = YOLO(sp.path, task="pose")
        outcomes: list[SampleOutcome] = []
        pr_all: list[tuple[float, bool]] = []
        strict_outcomes: list[SampleOutcome] = []
        strict_pr: list[tuple[float, bool]] = []

        for k, s in enumerate(samples):
            img = cv2.imread(s.image_path)
            if img is None:
                continue
            preds = _predict_single(yolo, img, sp.imgsz, device=device, conf_floor=0.01)
            oc, prs = evaluate_sample_metrics(s, preds, op_conf=args.op_conf)
            outcomes.append(oc)
            pr_all.extend(prs)
            if exclude_set and s.image_path not in exclude_set:
                strict_outcomes.append(oc)
                strict_pr.extend(prs)
            if (k + 1) % 2000 == 0 or (k + 1) == len(samples):
                print(f"    {k + 1}/{len(samples)} ({time.perf_counter() - t_start:.1f}s)")

        agg = aggregate_outcomes(outcomes, pr_all, op_conf=args.op_conf)
        strict_agg = (
            aggregate_outcomes(strict_outcomes, strict_pr, op_conf=args.op_conf)
            if strict_outcomes else None
        )

        ul_val = None
        if not args.skip_ultralytics_val:
            print(f"  [Ultralytics .val()] {sp.display_name} ({oks_sigma_mode}, workers={args.val_workers})...")
            ul_val = run_ultralytics_val(
                sp, clean_yaml_path, device=device, workers=args.val_workers, oks_sigma_mode=oks_sigma_mode
            )

        stress_res = None
        if scales:
            print(f"  [Stage 4] Stress-scale diagnostic {scales} px (n={args.stress_count})...")
            stress_res = run_stress_scales(
                samples,
                lambda im: _predict_single(yolo, im, sp.imgsz, device=device, conf_floor=0.01),
                scales=scales,
                count=args.stress_count,
                seed=args.seed,
                op_conf=args.op_conf,
            )

        neg_res = None
        if args.neg_dir and args.neg_dir.is_dir():
            print(f"  [Stage 4B] Background FPR on {args.neg_dir}...")
            neg_res = run_negative_fpr(
                args.neg_dir,
                lambda im: _predict_single(yolo, im, sp.imgsz, device=device, conf_floor=args.op_conf),
                op_conf=args.op_conf,
            )

        if device == "cpu":
            lat_devs: tuple[str, ...] = ("cpu",)
        elif args.skip_cpu_latency:
            lat_devs = (device,)
        else:
            lat_devs = (device, "cpu")
        print(
            f"  [Stage 5] Benchmarking latency ({args.latency_warmup} warmup + {args.latency_iters} iters, "
            f"devs={lat_devs}, sizes={[f'{w}x{h}' for w, h in lat_sizes]})..."
        )
        lat_res = benchmark_latency(
            sp,
            lat_imgs,
            warmup=args.latency_warmup,
            iters=args.latency_iters,
            devices=lat_devs,
            frame_sizes=lat_sizes,
        )

        if idx == 0:
            print("  [Stage 6] Saving visual error audit overlays (audit_top20_errors/)...")
            audit_counts = save_visual_audit(outcomes, out_dir / "audit_top20_errors", max_per_cat=20)

        model_reports.append({
            "display_name": sp.display_name,
            "path": sp.path,
            "sha256": sp.sha256,
            "imgsz": sp.imgsz,
            "is_onnx": sp.is_onnx,
            "params": sp.params,
            "aliases": list(sp.aliases),
            "eval_seconds": round(time.perf_counter() - t_start, 2),
            "metrics": agg,
            "strict_heldout_metrics": strict_agg,
            "ultralytics_val": ul_val,
            "stress_scales": stress_res,
            "negative_fpr": neg_res,
            "latency": lat_res,
        })

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_commit_info(REPO_ROOT),
        "oks_sigma_mode": oks_sigma_mode,
        "exclude_audit_list": str(args.exclude_audit_list) if args.exclude_audit_list else None,
        "dataset_audit": audit,
        "audit_images": audit_counts,
        "models": model_reports,
    }
    json_path = out_dir / "tier1_results.json"
    md_path = out_dir / "tier1_report.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_tier1_markdown(payload, md_path)
    print(f"\nDone! Report written to:\n  - {md_path}\n  - {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
