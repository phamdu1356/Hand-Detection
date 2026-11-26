"""Multi-hand OOD detection/keypoint/Gate14 metrics & gesture/FPR evaluation.

Pure Python + NumPy (no YOLO/PyTorch dependency), designed for zero-leakage
cross-dataset evaluation across FreiHAND, CMU HandDB, COCO-WholeBody, and HaGRID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmark.schema import (
    CONTROL_KPT_INDICES,
    KPT_NAMES,
    PALM_INDICES,
    SIZE_BUCKETS,
    size_bucket,
    wilson_ci,
)
from dronectl.body_frame import hand_size
from tools.eval_tier1_static import PredDet, box_iou_xyxy, check_gate14, compute_ap50


@dataclass(frozen=True)
class GTInstance:
    box_xyxy: tuple[float, float, float, float]
    size_px: float
    size_bucket: str
    kps_xy: np.ndarray | None           # (21, 2) float64 pixels or None (bbox-only)
    kps_in_frame: np.ndarray | None     # (21,) bool or None
    palm_px: float | None
    all14_in_frame: bool
    oob_kp_count: int = 0
    gesture: str | None = None
    source_tag: str = ""


@dataclass(frozen=True)
class OODImageSample:
    dataset: str
    image_path: str
    sha256: str
    width: int
    height: int
    instances: tuple[GTInstance, ...]
    ignore_boxes: tuple[tuple[float, float, float, float], ...] = ()
    image_gesture: str | None = None


@dataclass
class InstanceOutcome:
    gt: GTInstance
    image_path: str
    matched_pred: PredDet | None
    tp: bool
    iou: float
    pck_box_hits: int
    pck_box_total: int
    pck_palm21_hits: int
    pck_palm21_total: int
    pck_palm14_hits: int
    pck_palm14_total: int
    joint_err_px: np.ndarray            # (21,) float64
    joint_err_palm: np.ndarray          # (21,) float64
    gate14_pass: bool
    gate14_failed_joints: list[int] = field(default_factory=list)
    gate14_min_conf: float = 0.0


@dataclass
class ImageOutcome:
    sample: OODImageSample
    instance_outcomes: list[InstanceOutcome]
    fp_count: int
    ignored_pred_count: int
    unmatched_preds: list[PredDet]


def derive_box_from_kps(
    kps_xy: np.ndarray,
    valid_mask: np.ndarray,
    width: int,
    height: int,
    pad_frac: float = 0.15,
) -> tuple[float, float, float, float] | None:
    """Derive a tight padded [x1, y1, x2, y2] box from valid 2D hand keypoints."""
    pts = np.asarray(kps_xy, dtype=np.float64)[np.asarray(valid_mask, dtype=bool)]
    if pts.shape[0] < 2:
        return None
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    span = max(x2 - x1, y2 - y1, 4.0)
    pad = pad_frac * span
    bx1 = float(np.clip(x1 - pad, 0.0, float(width)))
    by1 = float(np.clip(y1 - pad, 0.0, float(height)))
    bx2 = float(np.clip(x2 + pad, 0.0, float(width)))
    by2 = float(np.clip(y2 + pad, 0.0, float(height)))
    if bx2 - bx1 <= 1e-3 or by2 - by1 <= 1e-3:
        return None
    return (bx1, by1, bx2, by2)


def make_gt_instance(
    box_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    kps_xy: np.ndarray | None = None,
    kps_vis: np.ndarray | None = None,
    *,
    gesture: str | None = None,
    source_tag: str = "",
) -> GTInstance | None:
    """Construct a normalized GTInstance from clipped box + optional 21 keypoints."""
    x1 = float(np.clip(box_xyxy[0], 0.0, float(width)))
    y1 = float(np.clip(box_xyxy[1], 0.0, float(height)))
    x2 = float(np.clip(box_xyxy[2], 0.0, float(width)))
    y2 = float(np.clip(box_xyxy[3], 0.0, float(height)))
    sz = max(x2 - x1, y2 - y1)
    if sz <= 1e-6:
        return None

    kps_arr: np.ndarray | None = None
    in_frame: np.ndarray | None = None
    palm_px: float | None = None
    all14 = False
    oob_cnt = 0
    if kps_xy is not None:
        kps_arr = np.asarray(kps_xy, dtype=np.float64).reshape(21, 2)
        vis = np.ones(21, dtype=bool) if kps_vis is None else np.asarray(kps_vis, dtype=bool).reshape(21)
        finite = np.isfinite(kps_arr).all(axis=1)
        in_b = (
            finite
            & (kps_arr[:, 0] >= 0.0) & (kps_arr[:, 0] <= float(width))
            & (kps_arr[:, 1] >= 0.0) & (kps_arr[:, 1] <= float(height))
        )
        in_frame = vis & in_b
        oob_cnt = int((vis & ~in_b).sum())
        if all(bool(in_frame[i]) for i in PALM_INDICES):
            palm_px = hand_size(kps_arr)
        all14 = all(bool(in_frame[i]) for i in CONTROL_KPT_INDICES)

    return GTInstance(
        box_xyxy=(x1, y1, x2, y2),
        size_px=float(sz),
        size_bucket=size_bucket(sz),
        kps_xy=kps_arr,
        kps_in_frame=in_frame,
        palm_px=palm_px,
        all14_in_frame=all14,
        oob_kp_count=oob_cnt,
        gesture=gesture,
        source_tag=source_tag,
    )


def _overlaps_any_ignore(
    box: tuple[float, float, float, float],
    ignore_boxes: tuple[tuple[float, float, float, float], ...],
    iou_thres: float,
) -> bool:
    for ig in ignore_boxes:
        if box_iou_xyxy(box, ig) >= iou_thres:
            return True
    return False


def evaluate_multihand_image(
    sample: OODImageSample,
    preds: list[PredDet],
    *,
    op_conf: float = 0.5,
    iou_thres: float = 0.5,
    pck_thresh: float = 0.2,
    gate_conf: float = 0.5,
) -> tuple[ImageOutcome, list[tuple[float, bool]]]:
    """Greedy-by-confidence multi-GT evaluation matching COCO/VOC semantics."""
    ordered = sorted(preds, key=lambda d: d.conf, reverse=True)
    gts = sample.instances

    # 1. Full PR-curve matching (all preds conf >= 0.01)
    pr_records: list[tuple[float, bool]] = []
    claimed_pr = [False] * len(gts)
    for det in ordered:
        best_i = -1
        best_iou = 0.0
        for gi, gt in enumerate(gts):
            if claimed_pr[gi]:
                continue
            iou = box_iou_xyxy(det.box_xyxy, gt.box_xyxy)
            if iou >= iou_thres and iou > best_iou:
                best_iou = iou
                best_i = gi
        if best_i >= 0:
            claimed_pr[best_i] = True
            pr_records.append((float(det.conf), True))
        elif not _overlaps_any_ignore(det.box_xyxy, sample.ignore_boxes, iou_thres):
            pr_records.append((float(det.conf), False))

    # 2. Operating-point matching (conf >= op_conf)
    op_preds = [d for d in ordered if d.conf >= op_conf]
    matched_for_gt: list[tuple[PredDet, float] | None] = [None] * len(gts)
    unmatched_preds: list[PredDet] = []
    ignored_preds = 0
    for det in op_preds:
        best_i = -1
        best_iou = 0.0
        for gi, gt in enumerate(gts):
            if matched_for_gt[gi] is not None:
                continue
            iou = box_iou_xyxy(det.box_xyxy, gt.box_xyxy)
            if iou >= iou_thres and iou > best_iou:
                best_iou = iou
                best_i = gi
        if best_i >= 0:
            matched_for_gt[best_i] = (det, best_iou)
        elif _overlaps_any_ignore(det.box_xyxy, sample.ignore_boxes, iou_thres):
            ignored_preds += 1
        else:
            unmatched_preds.append(det)

    # 3. Compute per-GT keypoint & Gate14 outcomes
    palm14_mask = np.zeros(21, dtype=bool)
    palm14_mask[list(CONTROL_KPT_INDICES)] = True
    inst_outcomes: list[InstanceOutcome] = []

    for gi, gt in enumerate(gts):
        pair = matched_for_gt[gi]
        matched = pair[0] if pair is not None else None
        iou = pair[1] if pair is not None else 0.0
        tp = matched is not None

        has_kps = gt.kps_xy is not None and gt.kps_in_frame is not None
        box_total = int(gt.kps_in_frame.sum()) if has_kps else 0
        palm_ok = has_kps and gt.palm_px is not None and gt.palm_px > 1e-6
        palm21_total = box_total if palm_ok else 0
        palm14_eval_mask = (gt.kps_in_frame & palm14_mask) if has_kps else np.zeros(21, dtype=bool)
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
            if has_kps:
                assert gt.kps_xy is not None and gt.kps_in_frame is not None
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

        inst_outcomes.append(InstanceOutcome(
            gt=gt,
            image_path=sample.image_path,
            matched_pred=matched,
            tp=tp,
            iou=float(iou),
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
        ))

    img_out = ImageOutcome(
        sample=sample,
        instance_outcomes=inst_outcomes,
        fp_count=len(unmatched_preds),
        ignored_pred_count=ignored_preds,
        unmatched_preds=unmatched_preds,
    )
    return img_out, pr_records


def _prop(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_ci(k, n)
    return {
        "k": int(k),
        "n": int(n),
        "rate": (k / n) if n > 0 else None,
        "ci95": [round(lo, 6), round(hi, 6)] if n > 0 else None,
    }


def aggregate_ood_outcomes(
    image_outcomes: list[ImageOutcome],
    pr_records: list[tuple[float, bool]],
    *,
    op_conf: float = 0.5,
) -> dict[str, Any]:
    """Aggregate multi-hand image outcomes into instance-level detection, PCK, Gate14 & size buckets."""
    insts = [io for imo in image_outcomes for io in imo.instance_outcomes]
    n_imgs = len(image_outcomes)
    n_gt = len(insts)
    tp = sum(1 for o in insts if o.tp)
    fp = sum(imo.fp_count for imo in image_outcomes)
    ignored = sum(imo.ignored_pred_count for imo in image_outcomes)
    fn = n_gt - tp

    rec = tp / n_gt if n_gt > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    ap50 = compute_ap50(pr_records, n_gt)
    tp_ious = [o.iou for o in insts if o.tp]
    mean_iou_tp = (sum(tp_ious) / len(tp_ious)) if tp_ious else 0.0

    # Negative-image FPR if any 0-GT images were included
    neg_imgs = [imo for imo in image_outcomes if len(imo.sample.instances) == 0]
    neg_fp_imgs = sum(1 for imo in neg_imgs if imo.fp_count > 0)

    box_hits = sum(o.pck_box_hits for o in insts)
    box_tot_all = sum(o.pck_box_total for o in insts)
    box_tot_tp = sum(o.pck_box_total for o in insts if o.tp)

    palm21_hits = sum(o.pck_palm21_hits for o in insts)
    palm21_tot_all = sum(o.pck_palm21_total for o in insts)
    palm21_tot_tp = sum(o.pck_palm21_total for o in insts if o.tp)

    palm14_hits = sum(o.pck_palm14_hits for o in insts)
    palm14_tot_all = sum(o.pck_palm14_total for o in insts)
    palm14_tot_tp = sum(o.pck_palm14_total for o in insts if o.tp)
    n_palm_insts = sum(1 for o in insts if o.gt.palm_px is not None)

    gate_pass_all_tp = sum(1 for o in insts if o.tp and o.gate14_pass)
    tp_gt14 = [o for o in insts if o.tp and o.gt.all14_in_frame]
    gate_pass_gt14_tp = sum(1 for o in tp_gt14 if o.gate14_pass)

    drop_all = {KPT_NAMES[i]: 0 for i in CONTROL_KPT_INDICES}
    drop_gt14 = {KPT_NAMES[i]: 0 for i in CONTROL_KPT_INDICES}
    for o in insts:
        if o.tp and not o.gate14_pass:
            for j in o.gate14_failed_joints:
                drop_all[KPT_NAMES[j]] += 1
                if o.gt.all14_in_frame:
                    drop_gt14[KPT_NAMES[j]] += 1

    per_joint: dict[str, dict[str, Any]] = {}
    if insts:
        px_mat = np.stack([o.joint_err_px for o in insts], axis=0)
        palm_mat = np.stack([o.joint_err_palm for o in insts], axis=0)
    else:
        px_mat = np.zeros((0, 21))
        palm_mat = np.zeros((0, 21))
    for j, name in enumerate(KPT_NAMES):
        vpx = px_mat[:, j][np.isfinite(px_mat[:, j])]
        vpm = palm_mat[:, j][np.isfinite(palm_mat[:, j])]
        per_joint[name] = {
            "index": j,
            "n_tp": int(vpx.size),
            "mean_err_px": float(np.mean(vpx)) if vpx.size > 0 else None,
            "mean_err_palm_pct": float(np.mean(vpm) * 100.0) if vpm.size > 0 else None,
        }
    top5_worst = sorted(
        [v | {"name": k} for k, v in per_joint.items() if v["mean_err_palm_pct"] is not None],
        key=lambda x: x["mean_err_palm_pct"],
        reverse=True,
    )[:5]
    top5_drops = sorted(
        [{"name": k, "drops_all_tp": v, "drops_gt14_tp": drop_gt14[k]} for k, v in drop_all.items() if v > 0],
        key=lambda x: x["drops_gt14_tp"],
        reverse=True,
    )[:5]

    by_bucket: dict[str, dict[str, Any]] = {}
    for bname, _, _ in SIZE_BUCKETS:
        sub = [o for o in insts if o.gt.size_bucket == bname]
        sn = len(sub)
        stp = sum(1 for o in sub if o.tp)
        sb_h = sum(o.pck_box_hits for o in sub)
        sb_t = sum(o.pck_box_total for o in sub)
        sp14_h = sum(o.pck_palm14_hits for o in sub)
        sp14_t = sum(o.pck_palm14_total for o in sub)
        sub_g14 = [o for o in sub if o.tp and o.gt.all14_in_frame]
        sg_pass = sum(1 for o in sub_g14 if o.gate14_pass)
        by_bucket[bname] = {
            "n": sn,
            "recall_05": _prop(stp, sn),
            "pck_box_overall": (sb_h / sb_t) if sb_t > 0 else None,
            "pck_palm14_overall": (sp14_h / sp14_t) if sp14_t > 0 else None,
            "gate14_gt14_tp": _prop(sg_pass, len(sub_g14)),
        }

    return {
        "n_images": n_imgs,
        "n_gt_instances": n_gt,
        "op_conf": op_conf,
        "oob_gt_kpts_excluded": int(sum(o.gt.oob_kp_count for o in insts)),
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ignored_preds": ignored,
            "recall_05": _prop(tp, n_gt),
            "precision_05": _prop(tp, tp + fp),
            "f1_05": f1,
            "ap50": ap50,
            "mean_iou_tp": mean_iou_tp,
            "neg_image_fpr": _prop(neg_fp_imgs, len(neg_imgs)) if neg_imgs else None,
        },
        "keypoints": {
            "pck_box_overall": {"hits": box_hits, "total": box_tot_all,
                                "rate": (box_hits / box_tot_all) if box_tot_all > 0 else None},
            "pck_box_on_tp": {"hits": box_hits, "total": box_tot_tp,
                              "rate": (box_hits / box_tot_tp) if box_tot_tp > 0 else None},
            "palm_valid_instances": {"k": n_palm_insts, "n": n_gt},
            "pck_palm21_overall": {"hits": palm21_hits, "total": palm21_tot_all,
                                   "rate": (palm21_hits / palm21_tot_all) if palm21_tot_all > 0 else None},
            "pck_palm21_on_tp": {"hits": palm21_hits, "total": palm21_tot_tp,
                                 "rate": (palm21_hits / palm21_tot_tp) if palm21_tot_tp > 0 else None},
            "pck_palm14_overall": {"hits": palm14_hits, "total": palm14_tot_all,
                                   "rate": (palm14_hits / palm14_tot_all) if palm14_tot_all > 0 else None},
            "pck_palm14_on_tp": {"hits": palm14_hits, "total": palm14_tot_tp,
                                 "rate": (palm14_hits / palm14_tot_tp) if palm14_tot_tp > 0 else None},
            "top5_worst_joints": top5_worst,
        },
        "gate14": {
            "pass_all_tp": _prop(gate_pass_all_tp, tp),
            "pass_gt14_tp": _prop(gate_pass_gt14_tp, len(tp_gt14)),
            "top5_drop_joints": top5_drops,
        },
        "by_size_bucket": by_bucket,
    }


# ------------------------------------------------------------------ HaGRID gesture & FPR scoring

HAGRID_TO_CANONICAL: dict[str, str] = {
    "like": "like",
    "ok": "ok",
    "palm": "palm",
    "stop": "palm",
    "four": "palm",
    "fist": "fist",
    "one": "one",
    "no_gesture": "none",
}


def canonical_from_hand_command(cmd: str) -> str:
    """Map dronectl.gestures.hand_command() output into HaGRID-comparable canonical label."""
    if cmd.startswith("point_"):
        return "one"
    if cmd in ("like", "ok", "palm", "fist"):
        return cmd
    return "none"


def evaluate_gesture_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-class Recall/Precision/F1, Macro-F1, No-Gesture FPR & 4-bucket error attribution.

    Each record must contain:
      - gt_label: canonical GT class ('like', 'ok', 'palm', 'fist', 'one', 'none')
      - pred_label: canonical predicted class from hand_command()
      - hand_detected: bool (detector returned a hand at conf >= op_conf)
      - gate14_pass: bool (predicted hand passed Gate14)
    """
    pos_classes = sorted({r["gt_label"] for r in records if r["gt_label"] != "none"})
    all_labels = pos_classes + ["none"]
    cm: dict[str, dict[str, int]] = {g: {p: 0 for p in all_labels} for g in all_labels}
    attribution = {"correct": 0, "no_hand_detected": 0, "gate14_rejected": 0, "wrong_gesture": 0}

    for r in records:
        g = r["gt_label"]
        p = r["pred_label"] if r["pred_label"] in all_labels else "none"
        cm.setdefault(g, {k: 0 for k in all_labels})
        cm[g][p] = cm[g].get(p, 0) + 1

        if g != "none":
            if p == g:
                attribution["correct"] += 1
            elif not r["hand_detected"]:
                attribution["no_hand_detected"] += 1
            elif not r["gate14_pass"]:
                attribution["gate14_rejected"] += 1
            else:
                attribution["wrong_gesture"] += 1

    per_class: dict[str, dict[str, Any]] = {}
    f1_vals: list[float] = []
    rec_vals: list[float] = []
    for cls in pos_classes:
        tp = cm[cls][cls]
        gt_tot = sum(cm[cls].values())
        pred_tot = sum(cm[row][cls] for row in cm)
        rec = tp / gt_tot if gt_tot > 0 else 0.0
        prec = tp / pred_tot if pred_tot > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[cls] = {
            "tp": tp,
            "gt_total": gt_tot,
            "pred_total": pred_tot,
            "recall": _prop(tp, gt_tot),
            "precision": _prop(tp, pred_tot),
            "f1": f1,
        }
        f1_vals.append(f1)
        rec_vals.append(rec)

    pos_total = sum(1 for r in records if r["gt_label"] != "none")
    pos_correct = attribution["correct"]
    neg_records = [r for r in records if r["gt_label"] == "none"]
    neg_fp_cmds = sum(1 for r in neg_records if r["pred_label"] != "none")
    neg_det_hands = sum(1 for r in neg_records if r["hand_detected"])

    return {
        "n_total": len(records),
        "n_positive_gestures": pos_total,
        "positive_accuracy": _prop(pos_correct, pos_total),
        "macro_recall": float(np.mean(rec_vals)) if rec_vals else None,
        "macro_f1": float(np.mean(f1_vals)) if f1_vals else None,
        "no_gesture_command_fpr": _prop(neg_fp_cmds, len(neg_records)) if neg_records else None,
        "no_gesture_hand_det_rate": _prop(neg_det_hands, len(neg_records)) if neg_records else None,
        "error_attribution_positive": attribution,
        "per_class": per_class,
        "confusion_matrix": cm,
    }
