"""Zero-leakage Out-of-Distribution (OOD) benchmark suite across 4 independent public datasets:

1. FreiHAND Eval v2       — 3,960 real 224x224 hands with 3D-projected 21-keypoint GT (U. Freiburg).
2. CMU Panoptic HandDB    — 2,758 manually labeled 21-keypoint hands (MPII + NZSL: 846 test + 1,912 train).
3. COCO-WholeBody val2017 — 3,956 full-scene multi-hand 21-keypoint GT hands (extreme scale/occlusion)
                            + 2,307 person-free real-world scenes for Background FPR.
4. HaGRID FullHD Test     — 480 full-res 0.5-4 m gesture & no_gesture images (like, ok, palm, no_gesture)
                            scoring end-to-end dronectl.gestures.hand_command() + 4-bucket error attribution.

Every image is verified by SHA-256 against ../datasets/hand-keypoints/images/train to prove 0% byte leakage.
Visual GT keypoint-order verification PNGs are saved before scoring so 0..20 finger ordering is audited by eye.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from benchmark.ood_metrics import (
    HAGRID_TO_CANONICAL,
    GTInstance,
    ImageOutcome,
    OODImageSample,
    aggregate_ood_outcomes,
    canonical_from_hand_command,
    derive_box_from_kps,
    evaluate_gesture_records,
    evaluate_multihand_image,
    make_gt_instance,
)
from benchmark.schema import CONTROL_KPT_INDICES, SIZE_BUCKETS, file_sha256, git_commit_info
from dronectl.config import ControlConfig
from dronectl.gestures import hand_command
from dronectl.types import HandObs
from tools.eval_tier1_static import (
    HAND_CONNECTIONS,
    ModelSpec,
    PredDet,
    _collect_images,
    _predict_single,
    check_gate14,
    resolve_models,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Distinct BGR colors per finger to visually verify 0..20 keypoint order on GT PNGs
_FINGER_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),  # wrist = white
    1: (0, 0, 255), 2: (0, 0, 255), 3: (0, 0, 255), 4: (0, 0, 255),            # thumb = red
    5: (0, 255, 255), 6: (0, 255, 255), 7: (0, 255, 255), 8: (0, 255, 255),    # index = yellow
    9: (0, 255, 0), 10: (0, 255, 0), 11: (0, 255, 0), 12: (0, 255, 0),         # middle = green
    13: (255, 255, 0), 14: (255, 255, 0), 15: (255, 255, 0), 16: (255, 255, 0),# ring = cyan
    17: (255, 0, 255), 18: (255, 0, 255), 19: (255, 0, 255), 20: (255, 0, 255),# pinky = magenta
}


# ------------------------------------------------------------------ train SHA-256 leak guard

def load_train_sha256_set(train_dir: Path) -> set[str]:
    if not train_dir.is_dir():
        return set()
    return {file_sha256(p) for p in _collect_images(train_dir)}


# ------------------------------------------------------------------ dataset loaders

def _find_file(root: Path, filename: str) -> Path | None:
    hits = sorted(root.rglob(filename))
    return hits[0] if hits else None


def load_freihand_eval(
    root: Path,
    train_hashes: set[str],
    *,
    max_images: int | None = None,
) -> tuple[list[OODImageSample], dict[str, Any]]:
    """Load FreiHAND v2 evaluation split (3,960 images, 3D xyz projected via camera intrinsics K)."""
    xyz_path = _find_file(root, "evaluation_xyz.json")
    k_path = _find_file(root, "evaluation_K.json")
    if xyz_path is None or k_path is None:
        raise FileNotFoundError(f"FreiHAND evaluation_xyz.json / evaluation_K.json not found under {root}")
    rgb_dir = xyz_path.parent / "evaluation" / "rgb"
    if not rgb_dir.is_dir():
        candidates = [p for p in root.rglob("rgb") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"FreiHAND rgb directory not found under {root}")
        rgb_dir = candidates[0]

    xyz_list = json.loads(xyz_path.read_text(encoding="utf-8"))
    k_list = json.loads(k_path.read_text(encoding="utf-8"))
    n_total = min(len(xyz_list), len(k_list))

    samples: list[OODImageSample] = []
    leaks = 0
    seen: set[str] = set()
    dups = 0
    for i in range(n_total):
        img_path = rgb_dir / f"{i:08d}.jpg"
        if not img_path.exists():
            continue
        sha = file_sha256(img_path)
        if sha in train_hashes:
            leaks += 1
            continue
        if sha in seen:
            dups += 1
            continue
        seen.add(sha)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        xyz = np.asarray(xyz_list[i], dtype=np.float64).reshape(21, 3)
        K = np.asarray(k_list[i], dtype=np.float64).reshape(3, 3)
        uv = (K @ xyz.T).T
        z = np.maximum(uv[:, 2:3], 1e-9)
        kps_xy = uv[:, :2] / z
        vis = np.isfinite(kps_xy).all(axis=1)
        box = derive_box_from_kps(kps_xy, vis, W, H, pad_frac=0.15)
        if box is None:
            continue
        gt = make_gt_instance(box, W, H, kps_xy, vis, source_tag="freihand_eval")
        if gt is None:
            continue
        samples.append(OODImageSample(
            dataset="freihand_eval",
            image_path=str(img_path.resolve()),
            sha256=sha,
            width=W,
            height=H,
            instances=(gt,),
        ))
        if max_images and len(samples) >= max_images:
            break

    audit = {
        "dataset": "freihand_eval",
        "source_total": n_total,
        "train_byte_leaks_removed": leaks,
        "intra_dups_removed": dups,
        "n_images": len(samples),
        "n_gt_instances": sum(len(s.instances) for s in samples),
    }
    return samples, audit


def load_cmu_handdb(
    root: Path,
    train_hashes: set[str],
    *,
    splits: tuple[str, ...] = ("manual_test", "manual_train"),
    max_images: int | None = None,
) -> tuple[list[OODImageSample], dict[str, Any]]:
    """Load CMU Panoptic HandDB manually annotated real hands (MPII + NZSL).

    CMU stores one byte-identical JPG copy per labeled hand (e.g. *_01_l.jpg, *_01_r.jpg).
    We group by image SHA-256 so all hands in the same scene are merged into one OODImageSample.
    """
    by_sha: dict[str, dict[str, Any]] = {}
    leaks = 0
    total_jsons = 0

    for sp in splits:
        sp_dirs = [p for p in root.rglob(sp) if p.is_dir()]
        if not sp_dirs:
            continue
        sp_dir = sp_dirs[0]
        for jp in sorted(sp_dir.glob("*.json")):
            img_p = jp.with_suffix(".jpg")
            if not img_p.exists():
                stem = jp.stem
                for suf in ("_l", "_r", "_left", "_right"):
                    if stem.endswith(suf):
                        cand = jp.with_name(stem[: -len(suf)] + ".jpg")
                        if cand.exists():
                            img_p = cand
                            break
            if not img_p.exists():
                continue
            total_jsons += 1
            sha = file_sha256(img_p)
            if sha in train_hashes:
                leaks += 1
                continue
            entry = by_sha.get(sha)
            if entry is None:
                entry = {"img_path": img_p.resolve(), "ann_pairs": []}
                by_sha[sha] = entry
            entry["ann_pairs"].append((jp, sp))

    samples: list[OODImageSample] = []
    split_counts: dict[str, int] = {sp: 0 for sp in splits}
    dup_hands_skipped = 0

    for sha in sorted(by_sha.keys()):
        entry = by_sha[sha]
        img_path: Path = entry["img_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        insts: list[GTInstance] = []
        seen_wrist_keys: set[tuple[int, int, int, int]] = set()
        for jp, sp in entry["ann_pairs"]:
            data = json.loads(jp.read_text(encoding="utf-8"))
            pts = np.asarray(data.get("hand_pts", []), dtype=np.float64)
            if pts.shape != (21, 3):
                continue
            kps_xy = pts[:, :2]
            vis = pts[:, 2] > 0.0
            if int(vis.sum()) < 5:
                continue
            # Deduplicate identical hand annotations if any
            wkey = (
                int(round(kps_xy[0, 0])), int(round(kps_xy[0, 1])),
                int(round(kps_xy[9, 0])), int(round(kps_xy[9, 1])),
            )
            if wkey in seen_wrist_keys:
                dup_hands_skipped += 1
                continue
            seen_wrist_keys.add(wkey)
            box = derive_box_from_kps(kps_xy, vis, W, H, pad_frac=0.15)
            if box is None:
                continue
            gt = make_gt_instance(box, W, H, kps_xy, vis, source_tag=f"cmu_{sp}")
            if gt is not None:
                insts.append(gt)
                split_counts[sp] = split_counts.get(sp, 0) + 1

        if not insts:
            continue
        samples.append(OODImageSample(
            dataset="cmu_handdb",
            image_path=str(img_path),
            sha256=sha,
            width=W,
            height=H,
            instances=tuple(insts),
        ))
        if max_images and len(samples) >= max_images:
            break

    audit = {
        "dataset": "cmu_handdb",
        "splits_loaded": list(splits),
        "total_json_files": total_jsons,
        "instances_by_split": split_counts,
        "train_byte_leaks_removed": leaks,
        "dup_hands_skipped": dup_hands_skipped,
        "n_images": len(samples),
        "n_gt_instances": sum(len(s.instances) for s in samples),
    }
    return samples, audit


def load_coco_wholebody(
    root: Path,
    train_hashes: set[str],
    *,
    max_pos_images: int | None = None,
    max_neg_images: int | None = None,
) -> tuple[list[OODImageSample], list[OODImageSample], dict[str, Any]]:
    """Load COCO-WholeBody val2017 into (positive valid-hand images, person-free negative images)."""
    ann_path = _find_file(root, "coco_wholebody_val_v1.0.json")
    if ann_path is None:
        raise FileNotFoundError(f"coco_wholebody_val_v1.0.json not found under {root}")
    val_dirs = [p for p in root.rglob("val2017") if p.is_dir()]
    if not val_dirs:
        raise FileNotFoundError(f"val2017 directory not found under {root}")
    val_dir = val_dirs[0]

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    img_by_id = {int(im["id"]): im for im in data["images"]}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    pos_samples: list[OODImageSample] = []
    neg_samples: list[OODImageSample] = []
    leaks = 0
    ignored_total = 0

    for img_id in sorted(img_by_id.keys()):
        meta = img_by_id[img_id]
        img_path = val_dir / meta["file_name"]
        if not img_path.exists():
            continue
        W, H = int(meta["width"]), int(meta["height"])
        anns = anns_by_img.get(img_id, [])

        if not anns:
            # Pure person-free / hand-free real-world negative scene
            if max_neg_images is not None and len(neg_samples) >= max_neg_images:
                continue
            sha = file_sha256(img_path)
            if sha in train_hashes:
                leaks += 1
                continue
            neg_samples.append(OODImageSample(
                dataset="coco_val2017_neg",
                image_path=str(img_path.resolve()),
                sha256=sha,
                width=W,
                height=H,
                instances=(),
            ))
            continue

        if max_pos_images is not None and len(pos_samples) >= max_pos_images:
            continue

        insts: list[GTInstance] = []
        ignores: list[tuple[float, float, float, float]] = []
        for ann in anns:
            for side in ("lefthand", "righthand"):
                valid = bool(ann.get(f"{side}_valid", False))
                box_xywh = ann.get(f"{side}_box")
                kpts_flat = ann.get(f"{side}_kpts")
                if not box_xywh or len(box_xywh) != 4:
                    continue
                bx, by, bw, bh = (float(v) for v in box_xywh)
                if bw <= 1.0 or bh <= 1.0:
                    continue
                xyxy = (bx, by, bx + bw, by + bh)
                if valid and kpts_flat and len(kpts_flat) == 63:
                    raw = np.asarray(kpts_flat, dtype=np.float64).reshape(21, 3)
                    kps_xy = raw[:, :2]
                    vis = raw[:, 2] > 0.0
                    gt = make_gt_instance(xyxy, W, H, kps_xy, vis, source_tag="coco_wb_val2017")
                    if gt is not None:
                        insts.append(gt)
                else:
                    ignores.append((
                        float(np.clip(xyxy[0], 0.0, W)),
                        float(np.clip(xyxy[1], 0.0, H)),
                        float(np.clip(xyxy[2], 0.0, W)),
                        float(np.clip(xyxy[3], 0.0, H)),
                    ))

        if not insts:
            continue
        sha = file_sha256(img_path)
        if sha in train_hashes:
            leaks += 1
            continue
        ignored_total += len(ignores)
        pos_samples.append(OODImageSample(
            dataset="coco_wholebody_val2017",
            image_path=str(img_path.resolve()),
            sha256=sha,
            width=W,
            height=H,
            instances=tuple(insts),
            ignore_boxes=tuple(ignores),
        ))

    audit = {
        "dataset": "coco_wholebody_val2017",
        "train_byte_leaks_removed": leaks,
        "pos_images": len(pos_samples),
        "pos_gt_hands": sum(len(s.instances) for s in pos_samples),
        "ignore_hand_boxes": ignored_total,
        "neg_person_free_images": len(neg_samples),
    }
    return pos_samples, neg_samples, audit


def load_hagrid_local(
    root: Path,
    train_hashes: set[str],
) -> tuple[list[tuple[Path, str, str]], dict[str, Any]]:
    """Load local HaGRID test folder (like, ok, palm, no_gesture) deduplicated by SHA-256."""
    items: list[tuple[Path, str, str]] = []
    seen: set[str] = set()
    leaks = 0
    dups = 0
    counts_by_cls: dict[str, int] = {}

    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        raw_cls = cls_dir.name
        if raw_cls not in HAGRID_TO_CANONICAL:
            continue
        canon = HAGRID_TO_CANONICAL[raw_cls]
        for img_p in _collect_images(cls_dir):
            sha = file_sha256(img_p)
            if sha in train_hashes:
                leaks += 1
                continue
            if sha in seen:
                dups += 1
                continue
            seen.add(sha)
            items.append((img_p.resolve(), canon, sha))
            counts_by_cls[canon] = counts_by_cls.get(canon, 0) + 1

    audit = {
        "dataset": "hagrid_local_test",
        "root": str(root.resolve()),
        "train_byte_leaks_removed": leaks,
        "intra_dups_removed": dups,
        "n_images": len(items),
        "counts_by_class": counts_by_cls,
    }
    return items, audit


# ------------------------------------------------------------------ GT keypoint-order visual verification

def save_gt_verification_samples(
    samples_by_dataset: dict[str, list[OODImageSample]],
    out_dir: Path,
    per_dataset: int = 3,
) -> dict[str, list[str]]:
    """Save color-coded GT skeleton overlays (red=thumb, yellow=index, green=mid, cyan=ring, magenta=pinky)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, list[str]] = {}
    for ds_name, samples in samples_by_dataset.items():
        saved[ds_name] = []
        picked = [s for s in samples if s.instances and s.instances[0].all14_in_frame][:per_dataset]
        if not picked:
            picked = [s for s in samples if s.instances][:per_dataset]
        for idx, s in enumerate(picked, start=1):
            src = cv2.imread(s.image_path)
            if src is None:
                continue
            H0, W0 = src.shape[:2]
            scale = max(1.0, 540.0 / max(float(W0), 1.0)) if W0 < 540 else 1.0
            W, H = int(round(W0 * scale)), int(round(H0 * scale))
            img = cv2.resize(src, (W, H), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else src.copy()

            for gt in s.instances:
                gx1, gy1, gx2, gy2 = (int(round(v * scale)) for v in gt.box_xyxy)
                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (200, 200, 200), 2)
                if gt.kps_xy is not None and gt.kps_in_frame is not None:
                    for a, b in HAND_CONNECTIONS:
                        if gt.kps_in_frame[a] and gt.kps_in_frame[b]:
                            p1 = (int(round(gt.kps_xy[a, 0] * scale)), int(round(gt.kps_xy[a, 1] * scale)))
                            p2 = (int(round(gt.kps_xy[b, 0] * scale)), int(round(gt.kps_xy[b, 1] * scale)))
                            cv2.line(img, p1, p2, _FINGER_COLORS.get(b, (255, 255, 255)), 2)
                    for k in range(21):
                        if gt.kps_in_frame[k]:
                            pt = (int(round(gt.kps_xy[k, 0] * scale)), int(round(gt.kps_xy[k, 1] * scale)))
                            cv2.circle(img, pt, 4, _FINGER_COLORS[k], -1)
                            if k in (0, 4, 8, 12, 16, 20):
                                cv2.putText(img, str(k), (pt[0] + 4, pt[1] - 4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, _FINGER_COLORS[k], 1, cv2.LINE_AA)

            banner = np.full((34, W, 3), 24, dtype=np.uint8)
            cv2.putText(banner, f"{ds_name} | 0=W 4=Thumb(R) 8=Idx(Y) 12=Mid(G) 16=Ring(C) 20=Pinky(M)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
            vis = np.vstack([banner, img])
            out_p = out_dir / f"{ds_name}_{idx:02d}.png"
            cv2.imwrite(str(out_p), vis)
            saved[ds_name].append(str(out_p))
    return saved


# ------------------------------------------------------------------ evaluators

def eval_keypoint_dataset(
    yolo_model,
    spec: ModelSpec,
    samples: list[OODImageSample],
    *,
    device: str = "0",
    op_conf: float = 0.5,
) -> tuple[dict[str, Any], list[ImageOutcome], list[list[tuple[float, bool]]]]:
    outcomes: list[ImageOutcome] = []
    pr_per_img: list[list[tuple[float, bool]]] = []
    t0 = time.perf_counter()
    for k, s in enumerate(samples):
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        preds = _predict_single(yolo_model, img, spec.imgsz, device=device, conf_floor=0.01)
        im_out, prs = evaluate_multihand_image(s, preds, op_conf=op_conf)
        outcomes.append(im_out)
        pr_per_img.append(prs)
        if (k + 1) % 1000 == 0 or (k + 1) == len(samples):
            print(f"      [{s.dataset}] {k + 1}/{len(samples)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    pr_flat = [r for sub in pr_per_img for r in sub]
    agg = aggregate_ood_outcomes(outcomes, pr_flat, op_conf=op_conf)
    return agg, outcomes, pr_per_img


def eval_neg_background_fpr(
    yolo_model,
    spec: ModelSpec,
    neg_samples: list[OODImageSample],
    *,
    device: str = "0",
    op_conf: float = 0.5,
) -> dict[str, Any]:
    """Measure pure Background FPR + Gate14-passing false-command FPR on hand-free scenes."""
    cfg = ControlConfig()
    imgs_with_box_fp = 0
    imgs_with_cmd_fp = 0
    total_fp_boxes = 0
    n_eval = 0
    t0 = time.perf_counter()
    for k, s in enumerate(neg_samples):
        img = cv2.imread(s.image_path)
        if img is None:
            continue
        n_eval += 1
        preds = _predict_single(yolo_model, img, spec.imgsz, device=device, conf_floor=op_conf)
        op = [d for d in preds if d.conf >= op_conf]
        if op:
            imgs_with_box_fp += 1
            total_fp_boxes += len(op)
            best = max(op, key=lambda d: d.conf)
            x1, y1, x2, y2 = best.box_xyxy
            hand = HandObs(
                kps=best.kps_xy,
                kps_conf=best.kps_conf,
                box=(x1, y1, x2 - x1, y2 - y1),
                conf=best.conf,
                gesture_raw="none",
            )
            if hand_command(hand, cfg) != "none":
                imgs_with_cmd_fp += 1
        if (k + 1) % 1000 == 0 or (k + 1) == len(neg_samples):
            print(f"      [coco_neg] {k + 1}/{len(neg_samples)} ({time.perf_counter() - t0:.1f}s)", flush=True)

    from benchmark.ood_metrics import _prop
    return {
        "n_images": n_eval,
        "box_fpr": _prop(imgs_with_box_fp, n_eval),
        "command_fpr": _prop(imgs_with_cmd_fp, n_eval),
        "total_fp_boxes": total_fp_boxes,
        "fp_boxes_per_100_images": round(100.0 * total_fp_boxes / n_eval, 3) if n_eval else None,
    }


def eval_hagrid_gestures(
    yolo_model,
    spec: ModelSpec,
    items: list[tuple[Path, str, str]],
    *,
    device: str = "0",
    op_conf: float = 0.5,
    use_perception_roi: bool = False,
) -> dict[str, Any]:
    """Evaluate HaGRID FullHD gestures through hand_command() (raw YOLO or full Perception ROI pipeline)."""
    cfg = ControlConfig()
    perc = None
    if use_perception_roi:
        sys.path.insert(0, str(REPO_ROOT / "demo"))
        from perception import Perception
        perc = Perception(pose_model_path=spec.path, far_field="auto", conf=op_conf, hand_only=True)

    records: list[dict[str, Any]] = []
    for img_p, gt_cls, _sha in items:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        hand: HandObs | None = None
        gate_ok = False
        if perc is not None:
            perc.ctl.reset_state()
            obs = perc.perceive(img, t=0.0)
            hand = obs.hand
            if hand is not None:
                pred_det = PredDet(
                    box_xyxy=(hand.box[0], hand.box[1], hand.box[0] + hand.box[2], hand.box[1] + hand.box[3]),
                    conf=hand.conf,
                    kps_xy=hand.kps,
                    kps_conf=hand.kps_conf,
                )
                gate_ok, _, _ = check_gate14(pred_det, conf_min=cfg.hand_conf_min)
        else:
            preds = _predict_single(yolo_model, img, spec.imgsz, device=device, conf_floor=op_conf)
            op = [d for d in preds if d.conf >= op_conf]
            if op:
                best = max(op, key=lambda d: d.conf)
                gate_ok, _, _ = check_gate14(best, conf_min=cfg.hand_conf_min)
                x1, y1, x2, y2 = best.box_xyxy
                hand = HandObs(
                    kps=best.kps_xy,
                    kps_conf=best.kps_conf,
                    box=(x1, y1, x2 - x1, y2 - y1),
                    conf=best.conf,
                    gesture_raw="none",
                )
        raw_cmd = hand_command(hand, cfg) if hand is not None else "none"
        records.append({
            "gt_label": gt_cls,
            "pred_label": canonical_from_hand_command(raw_cmd),
            "hand_detected": hand is not None,
            "gate14_pass": gate_ok,
        })
    return evaluate_gesture_records(records)


# ------------------------------------------------------------------ markdown report

def _pct(val: float | None) -> str:
    return "N/A" if val is None else f"{val * 100.0:.2f}%"


def _pct_ci(prop: dict[str, Any] | None) -> str:
    if not prop or prop.get("rate") is None:
        return "N/A"
    lo, hi = prop["ci95"]
    return f"{prop['rate'] * 100.0:.2f}% ({prop['k']}/{prop['n']}, 95% CI {lo * 100:.1f}–{hi * 100:.1f}%)"


def write_ood_markdown(payload: dict[str, Any], md_path: Path) -> None:
    git = payload["git"]
    aud = payload["dataset_audits"]
    lines: list[str] = [
        "# Báo cáo Benchmark Độc lập Ngoài Phân phối (Zero-Leakage OOD Suite)",
        "",
        f"- **Thời điểm chạy:** `{payload['timestamp']}`",
        f"- **Git commit:** `{git['commit'][:12]}` (dirty=`{git['dirty']}`)",
        f"- **Kiểm chứng chống rò rỉ (SHA-256 vs `hand-keypoints/train`, `{payload['train_sha256_count']}` ảnh):**",
    ]
    for name, a in aud.items():
        if not a:
            continue
        lines.append(f"  - `{name}`: `{a.get('train_byte_leaks_removed', 0)}` tệp trùng byte với train · chi tiết: `{ {k: v for k, v in a.items() if k not in ('dataset', 'root', 'train_byte_leaks_removed')} }`")

    lines.extend([
        "",
        "## 1. Bảng tổng hợp OOD trên toàn bộ 3 tập 21 khớp gộp lại (Zero-Leakage Pooled + Macro)",
        "",
        "| Model | `imgsz` | Pooled Recall@0.5 | Pooled Prec@0.5 | Pooled AP@0.5 | Pooled PCK@0.2 Box | Pooled PCK@0.2 Palm (14 kpt) | Pooled Gate14 (GT14 TP) | Macro Recall (3 bộ) | Macro PCK14 (3 bộ) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for m in payload["models"]:
        po = m.get("pooled_keypoint_ood") or {}
        ma = m.get("macro_keypoint_ood") or {}
        det = po.get("detection") or {}
        kp = po.get("keypoints") or {}
        g14 = po.get("gate14") or {}
        lines.append(
            f"| `{m['display_name']}` | {m['imgsz']} | {_pct((det.get('recall_05') or {}).get('rate'))} | "
            f"{_pct((det.get('precision_05') or {}).get('rate'))} | {_pct(det.get('ap50'))} | "
            f"{_pct((kp.get('pck_box_overall') or {}).get('rate'))} | {_pct((kp.get('pck_palm14_overall') or {}).get('rate'))} | "
            f"{_pct((g14.get('pass_gt14_tp') or {}).get('rate'))} | {_pct(ma.get('recall_05'))} | {_pct(ma.get('pck_palm14_overall'))} |"
        )

    # Section 2: Per-dataset breakdown
    ds_keys = [k for k in ("freihand_eval", "cmu_handdb_test", "cmu_handdb_all", "coco_wholebody_val2017")
               if any(k in (m.get("datasets") or {}) for m in payload["models"])]
    lines.extend([
        "",
        "> **Lưu ý phương pháp:** Trên `CMU HandDB` và `COCO-WholeBody val2017` (ảnh toàn cảnh nhiều người), một số bàn tay phụ trong nền không được gán nhãn (`valid=False` có box rỗng `[0,0,0,0]`), nên cột **Precision@0.5** của hai bộ này là **cận dưới bảo thủ**; tỷ lệ báo giả thực (FPR) được đo chính xác ở Mục 4 (`2.307` ảnh nền COCO không có người) và Mục 5 (`119` ảnh HaGRID `no_gesture`). Các chỉ số `Recall@0.5`, `PCK@0.2 Box/Palm (14 khớp)` và `Gate14 (GT14 TP)` là chính xác tuyệt đối.",
        "",
        "## 2. Chi tiết theo từng bộ dataset 21 khớp OOD (tách riêng domain cận cảnh vs toàn cảnh)",
        "",
        "| Dataset | Số tay GT | Model | Recall@0.5 | Precision@0.5 | AP@0.5 | PCK@0.2 Box | PCK@0.2 Palm (14 kpt) | PCK14 (on TP) | Gate14 (GT14 TP) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dk in ds_keys:
        for m in payload["models"]:
            dm = (m.get("datasets") or {}).get(dk)
            if not dm:
                continue
            det = dm["detection"]
            kp = dm["keypoints"]
            g14 = dm["gate14"]
            lines.append(
                f"| `{dk}` | {dm['n_gt_instances']} | `{m['display_name']}` | "
                f"{_pct(det['recall_05']['rate'])} | {_pct(det['precision_05']['rate'])} | {_pct(det['ap50'])} | "
                f"{_pct(kp['pck_box_overall']['rate'])} | {_pct(kp['pck_palm14_overall']['rate'])} | "
                f"{_pct(kp['pck_palm14_on_tp']['rate'])} | {_pct(g14['pass_gt14_tp']['rate'])} |"
            )

    # Section 3: Real-world size buckets across pooled OOD
    lines.extend([
        "",
        "## 3. Phân tầng kích thước bàn tay THỰC TẾ trên toàn bộ OOD (gồm tay nhỏ `<40 px` & `40–80 px` ngoài đời)",
        "",
        "| Model | `<40 px` (Rec / PCK14) | `40-80 px` (Rec / PCK14) | `80-160 px` (Rec / PCK14) | `>=160 px` (Rec / PCK14) |",
        "|---|---:|---:|---:|---:|",
    ])
    for m in payload["models"]:
        po = m.get("pooled_keypoint_ood") or {}
        bb = po.get("by_size_bucket") or {}
        cells = []
        for bname, _, _ in SIZE_BUCKETS:
            b = bb.get(bname) or {}
            cells.append(f"{_pct((b.get('recall_05') or {}).get('rate'))} / {_pct(b.get('pck_palm14_overall'))} (n={b.get('n', 0)})")
        lines.append(f"| `{m['display_name']}` | " + " | ".join(cells) + " |")

    # Section 4: COCO person-free negative Background FPR
    if any(m.get("coco_neg_fpr") for m in payload["models"]):
        lines.extend([
            "",
            "## 4. Tỷ lệ báo giả trên ảnh nền đời thực không có người/tay (`COCO val2017` person-free)",
            "",
            "| Model | Số ảnh nền | Box FPR (`conf ≥ 0.5`) | Command FPR (`hand_command != none`) | FP Boxes / 100 ảnh |",
            "|---|---:|---:|---:|---:|",
        ])
        for m in payload["models"]:
            nf = m.get("coco_neg_fpr")
            if not nf:
                continue
            lines.append(
                f"| `{m['display_name']}` | {nf['n_images']} | {_pct_ci(nf['box_fpr'])} | "
                f"{_pct_ci(nf['command_fpr'])} | {nf['fp_boxes_per_100_images']} |"
            )

    # Section 5: HaGRID FullHD End-to-End Gesture + no_gesture FPR
    if any(m.get("hagrid_raw") for m in payload["models"]):
        lines.extend([
            "",
            "## 5. Cử chỉ đời thực FullHD 0,5–4 m (`HaGRID Test`: `like`, `ok`, `palm`, `no_gesture`) qua `hand_command()`",
            "",
            "| Model & Chế độ | Macro-F1 (3 cử chỉ) | Accuracy (360 cử chỉ) | `like` Rec | `ok` Rec | `palm` Rec | `no_gesture` Command FPR | Nguyên nhân sai (Mất tay / Rớt Gate14 / Sai cử chỉ) |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for m in payload["models"]:
            for key, tag in (("hagrid_raw", "Raw YOLO"), ("hagrid_perception", "Perception ROI-Zoom")):
                hg = m.get(key)
                if not hg:
                    continue
                pc = hg["per_class"]
                att = hg["error_attribution_positive"]
                lines.append(
                    f"| `{m['display_name']}` ({tag}) | {_pct(hg['macro_f1'])} | {_pct(hg['positive_accuracy']['rate'])} | "
                    f"{_pct((pc.get('like') or {}).get('recall', {}).get('rate'))} | "
                    f"{_pct((pc.get('ok') or {}).get('recall', {}).get('rate'))} | "
                    f"{_pct((pc.get('palm') or {}).get('recall', {}).get('rate'))} | "
                    f"{_pct_ci(hg['no_gesture_command_fpr'])} | "
                    f"{att['no_hand_detected']} / {att['gate14_rejected']} / {att['wrong_gesture']} |"
                )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ood-root", type=Path, default=Path("../datasets/ood_bench"))
    ap.add_argument("--train-ref", type=Path, default=Path("../datasets/hand-keypoints/images/train"))
    ap.add_argument("--hagrid-dir", type=Path, default=Path("data/rgb-hagrid/a95c8e47b17dc2fc28f85294461114d8719bf0bb/data/test"))
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", type=Path, default=Path("benchmark/results/ood"))
    ap.add_argument("--device", default="0")
    ap.add_argument("--op-conf", type=float, default=0.5)
    ap.add_argument("--verify-gt-only", action="store_true", help="Only load datasets and export GT keypoint-order PNGs")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import torch
    from ultralytics import YOLO

    args = parse_args(argv)
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cpu" if (args.device == "cpu" or not torch.cuda.is_available()) else args.device

    print("[Step 1] Hashing training reference set for zero-leakage proof...", flush=True)
    train_hashes = load_train_sha256_set(args.train_ref.resolve())
    print(f"  Loaded {len(train_hashes)} unique train SHA-256 hashes.", flush=True)

    audits: dict[str, Any] = {}
    kp_datasets: dict[str, list[OODImageSample]] = {}

    fh_dir = args.ood_root / "freihand_eval"
    if fh_dir.is_dir() and _find_file(fh_dir, "evaluation_xyz.json"):
        print("[Load] FreiHAND Eval v2...", flush=True)
        kp_datasets["freihand_eval"], audits["freihand_eval"] = load_freihand_eval(fh_dir, train_hashes)
        print(f"  -> {audits['freihand_eval']}", flush=True)

    cmu_dir = args.ood_root / "cmu_handdb"
    if cmu_dir.is_dir() and any(cmu_dir.rglob("manual_test")):
        print("[Load] CMU Panoptic HandDB (manual_test + manual_train)...", flush=True)
        kp_datasets["cmu_handdb_test"], audits["cmu_handdb_test"] = load_cmu_handdb(
            cmu_dir, train_hashes, splits=("manual_test",)
        )
        kp_datasets["cmu_handdb_all"], audits["cmu_handdb_all"] = load_cmu_handdb(
            cmu_dir, train_hashes, splits=("manual_test", "manual_train")
        )
        print(f"  -> test: {audits['cmu_handdb_test']} | all: {audits['cmu_handdb_all']}", flush=True)

    coco_dir = args.ood_root / "coco_wholebody"
    coco_neg: list[OODImageSample] = []
    if coco_dir.is_dir() and any(coco_dir.rglob("val2017")):
        print("[Load] COCO-WholeBody val2017...", flush=True)
        kp_datasets["coco_wholebody_val2017"], coco_neg, audits["coco_wholebody_val2017"] = load_coco_wholebody(
            coco_dir, train_hashes
        )
        print(f"  -> {audits['coco_wholebody_val2017']}", flush=True)

    hagrid_items: list[tuple[Path, str, str]] = []
    if args.hagrid_dir.is_dir():
        print("[Load] HaGRID local test...", flush=True)
        hagrid_items, audits["hagrid_local_test"] = load_hagrid_local(args.hagrid_dir, train_hashes)
        print(f"  -> {audits['hagrid_local_test']}", flush=True)

    # Export color-coded GT verification PNGs so keypoint order 0..20 is visually verified
    gt_verify_map = {k: v for k, v in kp_datasets.items() if k != "cmu_handdb_test"}
    gt_saved = save_gt_verification_samples(gt_verify_map, out_dir / "gt_verify", per_dataset=3)
    print(f"[GT Verify] Saved verification PNGs: {gt_saved}", flush=True)
    if args.verify_gt_only:
        return 0

    specs = resolve_models(args.models)
    model_reports: list[dict[str, Any]] = []
    # Disjoint pool keys: freihand_eval + cmu_handdb_all + coco_wholebody_val2017 (never double-count cmu_test!)
    pool_keys = [k for k in ("freihand_eval", "cmu_handdb_all", "coco_wholebody_val2017") if k in kp_datasets]

    for sp in specs:
        print(f"\n[Model] {sp.display_name} (imgsz={sp.imgsz}, sha256={sp.sha256[:12]}, dev={device})", flush=True)
        t_model = time.perf_counter()
        yolo = YOLO(sp.path, task="pose")

        ds_metrics: dict[str, Any] = {}
        pooled_outcomes: list[ImageOutcome] = []
        pooled_pr: list[tuple[float, bool]] = []
        cmu_all_outcomes: list[ImageOutcome] | None = None
        cmu_all_pr_per_img: list[list[tuple[float, bool]]] | None = None

        for dk in pool_keys:
            agg, ocs, prs_img = eval_keypoint_dataset(yolo, sp, kp_datasets[dk], device=device, op_conf=args.op_conf)
            ds_metrics[dk] = agg
            pooled_outcomes.extend(ocs)
            for sub in prs_img:
                pooled_pr.extend(sub)
            if dk == "cmu_handdb_all":
                cmu_all_outcomes = ocs
                cmu_all_pr_per_img = prs_img

        # Derive official cmu_handdb_test slice without re-running inference
        if "cmu_handdb_test" in kp_datasets and cmu_all_outcomes is not None and cmu_all_pr_per_img is not None:
            test_shas = {s.sha256 for s in kp_datasets["cmu_handdb_test"]}
            t_ocs: list[ImageOutcome] = []
            t_prs: list[tuple[float, bool]] = []
            for oc, sub_pr in zip(cmu_all_outcomes, cmu_all_pr_per_img):
                if oc.sample.sha256 in test_shas:
                    t_ocs.append(oc)
                    t_prs.extend(sub_pr)
            ds_metrics["cmu_handdb_test"] = aggregate_ood_outcomes(t_ocs, t_prs, op_conf=args.op_conf)

        pooled_agg = aggregate_ood_outcomes(pooled_outcomes, pooled_pr, op_conf=args.op_conf) if pooled_outcomes else None
        macro_rec = [ds_metrics[k]["detection"]["recall_05"]["rate"] for k in pool_keys
                     if ds_metrics[k]["detection"]["recall_05"]["rate"] is not None]
        macro_pck14 = [ds_metrics[k]["keypoints"]["pck_palm14_overall"]["rate"] for k in pool_keys
                       if ds_metrics[k]["keypoints"]["pck_palm14_overall"]["rate"] is not None]
        macro_summary = {
            "datasets_included": pool_keys,
            "recall_05": float(np.mean(macro_rec)) if macro_rec else None,
            "pck_palm14_overall": float(np.mean(macro_pck14)) if macro_pck14 else None,
        }

        neg_fpr = None
        if coco_neg:
            print(f"    Evaluating Background FPR on {len(coco_neg)} person-free COCO val2017 images...", flush=True)
            neg_fpr = eval_neg_background_fpr(yolo, sp, coco_neg, device=device, op_conf=args.op_conf)

        hagrid_raw = None
        hagrid_perc = None
        if hagrid_items:
            print(f"    Evaluating HaGRID gestures ({len(hagrid_items)} FullHD images, raw + Perception ROI)...", flush=True)
            hagrid_raw = eval_hagrid_gestures(yolo, sp, hagrid_items, device=device, op_conf=args.op_conf, use_perception_roi=False)
            hagrid_perc = eval_hagrid_gestures(yolo, sp, hagrid_items, device=device, op_conf=args.op_conf, use_perception_roi=True)

        model_reports.append({
            "display_name": sp.display_name,
            "path": sp.path,
            "sha256": sp.sha256,
            "imgsz": sp.imgsz,
            "is_onnx": sp.is_onnx,
            "aliases": list(sp.aliases),
            "eval_seconds": round(time.perf_counter() - t_model, 2),
            "pooled_keypoint_ood": pooled_agg,
            "macro_keypoint_ood": macro_summary,
            "datasets": ds_metrics,
            "coco_neg_fpr": neg_fpr,
            "hagrid_raw": hagrid_raw,
            "hagrid_perception": hagrid_perc,
        })

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git": git_commit_info(REPO_ROOT),
        "train_sha256_count": len(train_hashes),
        "dataset_audits": audits,
        "gt_verify_pngs": gt_saved,
        "models": model_reports,
    }
    json_path = out_dir / "ood_results.json"
    md_path = out_dir / "ood_report.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_ood_markdown(payload, md_path)
    print(f"\nDone! OOD report written to:\n  - {md_path}\n  - {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
