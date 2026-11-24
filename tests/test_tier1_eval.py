"""Deterministic unit, golden-math and sanity-invariant tests for Tier-1 evaluation & shared schema."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmark.schema import (
    CONTROL_KPT_INDICES,
    KPT_NAMES,
    LABEL_COLUMNS,
    MANIFEST_COLUMNS,
    dhash64,
    file_sha256,
    hamming_distance,
    size_bucket,
    validate_label_rows,
    validate_manifest_rows,
    wilson_ci,
)
from tools.eval_tier1_static import (
    PredDet,
    SampleGT,
    aggregate_outcomes,
    audit_and_clean_split,
    box_iou_xyxy,
    check_gate14,
    compute_ap50,
    evaluate_sample_metrics,
    save_visual_audit,
)


def _make_hand_kps(origin=(200.0, 260.0), scale=100.0) -> np.ndarray:
    """Create a well-formed 21-keypoint hand with palm length == scale."""
    ox, oy = origin
    kps = np.zeros((21, 2), dtype=np.float64)
    kps[0] = (ox, oy)                           # wrist
    # 5 MCPs & fingers upward
    xs = (-0.45, -0.30, 0.0, 0.25, 0.45)
    bases = (1, 5, 9, 13, 17)
    for xoff, b in zip(xs, bases):
        for step in range(4):
            y_frac = 0.4 + 0.3 * step if b == 1 else 1.0 + 0.35 * step
            kps[b + step] = (ox + xoff * scale, oy - y_frac * scale)
    # Place middle MCP (9) at exact distance `scale` above wrist (0)
    kps[9] = (ox, oy - scale)
    return kps


def _make_sample(
    path="img.jpg",
    size_px=150.0,
    kps=None,
    vis_mask=None,
    in_frame_mask=None,
    width=640,
    height=480,
) -> SampleGT:
    kps = _make_hand_kps() if kps is None else np.asarray(kps, dtype=np.float64)
    vis = np.ones(21, dtype=bool) if vis_mask is None else np.asarray(vis_mask, dtype=bool)
    in_f = vis.copy() if in_frame_mask is None else np.asarray(in_frame_mask, dtype=bool)
    oob_cnt = int((vis & ~in_f).sum())
    x1, y1 = 120.0, 50.0
    box = (x1, y1, x1 + size_px, y1 + size_px)
    from dronectl.body_frame import hand_size
    palm = hand_size(kps) if all(bool(in_f[i]) for i in (0, 5, 9, 13, 17)) else None
    all14 = all(bool(in_f[i]) for i in CONTROL_KPT_INDICES)
    return SampleGT(
        image_path=path,
        sha256="deadbeef",
        width=width,
        height=height,
        box_xyxy=box,
        size_px=size_px,
        size_bucket=size_bucket(size_px),
        kps_xy=kps,
        kps_vis=vis,
        kps_in_frame=in_f,
        palm_px=palm,
        all14_in_frame=all14,
        oob_box=False,
        oob_kp=oob_cnt > 0,
        oob_kp_count=oob_cnt,
    )


def _write_yolo_pair(root: Path, split: str, stem: str, img: np.ndarray, kps_norm=None):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    ip = img_dir / f"{stem}.jpg"
    cv2.imwrite(str(ip), img)
    if kps_norm is None:
        kps_px = _make_hand_kps(origin=(160.0, 200.0), scale=60.0)
        H, W = img.shape[:2]
        kps_norm = np.column_stack([kps_px[:, 0] / W, kps_px[:, 1] / H, np.full(21, 2.0)])
    parts = ["0", "0.5", "0.5", "0.35", "0.45"] + [f"{v:.6f}" for v in kps_norm.reshape(-1)]
    (lbl_dir / f"{stem}.txt").write_text(" ".join(parts) + "\n", encoding="utf-8")
    return ip


# ------------------------------------------------------------------ 1. Schema & Dedup

def test_file_sha256_and_clean_split_removes_leaks_and_dups(tmp_path):
    rng = np.random.default_rng(7)
    im_train1 = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    im_train2 = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    im_clean1 = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    im_clean2 = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    im_clean3 = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)

    p_tr1 = _write_yolo_pair(tmp_path, "train", "tr1", im_train1)
    _write_yolo_pair(tmp_path, "train", "tr2", im_train2)
    # Byte-identical copies in val: 1 leaked from train, 1 intra-val dup, 3 unique clean
    p_v1 = _write_yolo_pair(tmp_path, "val", "v1_clean", im_clean1)
    _write_yolo_pair(tmp_path, "val", "v2_clean", im_clean2)
    _write_yolo_pair(tmp_path, "val", "v3_clean", im_clean3)
    p_v_leak = _write_yolo_pair(tmp_path, "val", "v4_leak", im_train1)
    p_v_dup = _write_yolo_pair(tmp_path, "val", "v5_dup", im_clean1)
    # Ensure exact byte equality for leak and dup
    p_v_leak.write_bytes(p_tr1.read_bytes())
    p_v_dup.write_bytes(p_v1.read_bytes())

    assert file_sha256(p_v_leak) == file_sha256(p_tr1)
    samples, audit = audit_and_clean_split(tmp_path, check_dhash=True)
    assert audit["val_leaked_from_train_count"] == 1
    assert audit["val_intra_dups_count"] == 1
    assert [Path(s.image_path).stem for s in samples] == ["v1_clean", "v2_clean", "v3_clean"]


def test_dhash64_catches_recompressed_near_duplicate(tmp_path):
    # Smooth gradient + geometric shapes so JPEG recompression changes bytes but preserves structure
    y, x = np.mgrid[0:240, 0:320]
    base = np.stack([(x * 255 // 320), (y * 255 // 240), ((x + y) * 127 // 560)], axis=-1).astype(np.uint8)
    cv2.circle(base, (120, 100), 45, (20, 220, 240), -1)
    p95 = tmp_path / "q95.jpg"
    p45 = tmp_path / "q45.jpg"
    cv2.imwrite(str(p95), base, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(p45), base, [cv2.IMWRITE_JPEG_QUALITY, 45])
    assert file_sha256(p95) != file_sha256(p45)
    assert hamming_distance(dhash64(p95), dhash64(p45)) <= 4

    other = np.flip(base, axis=(0, 1)).copy()
    cv2.rectangle(other, (10, 10), (150, 200), (255, 0, 0), -1)
    assert hamming_distance(dhash64(p95), dhash64(other)) > 15


def test_validate_manifest_rejects_empty_and_cross_split_leak():
    assert validate_manifest_rows([], list(MANIFEST_COLUMNS))
    assert validate_manifest_rows([{"trial_id": "t1"}], ["trial_id"])
    base = {c: "x" for c in MANIFEST_COLUMNS}
    r1 = {**base, "trial_id": "t1", "split": "dev", "participant_id": "p1", "video_sha256": "aaa"}
    r2 = {**base, "trial_id": "t2", "split": "test", "participant_id": "p1", "video_sha256": "aaa"}
    errs = validate_manifest_rows([r1, r2], list(MANIFEST_COLUMNS))
    assert any("video_sha256 leaked" in e for e in errs)
    assert any("participant_id leaked" in e for e in errs)


def test_validate_labels_rejects_bad_times_unknown_cmd_and_overlap():
    base = {c: "" for c in LABEL_COLUMNS}
    bad = [
        {**base, "trial_id": "t1", "event_id": "e1", "kind": "intent", "command": "takeoff", "onset_s": "5.0", "offset_s": "4.0"},
        {**base, "trial_id": "t1", "event_id": "e2", "kind": "intent", "command": "fly_moon", "onset_s": "1.0", "offset_s": "3.0"},
        {**base, "trial_id": "t1", "event_id": "e3", "kind": "intent", "command": "stop", "onset_s": "2.5", "offset_s": "4.0"},
    ]
    errs = validate_label_rows(bad, list(LABEL_COLUMNS))
    assert any("onset_s < offset_s" in e for e in errs)
    assert any("unknown command" in e for e in errs)
    assert any("overlapping intents" in e for e in errs)


# ------------------------------------------------------------------ 2. Golden Hand-Calculated Math

def test_box_iou_hand_calculated():
    assert box_iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50.0 / 150.0)
    assert box_iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)


def test_compute_ap50_hand_calculated():
    # N=3 GT; ranked preds: [0.9:TP, 0.8:FP, 0.7:TP, 0.6:TP]
    # Recalls:    [1/3, 1/3, 2/3, 3/3]
    # Precisions: [1/1, 1/2, 2/3, 3/4] -> envelope: [1.0, 0.75, 0.75, 0.75]
    # AP = (1/3)*1.0 + (2/3)*0.75 = 1/3 + 1/2 = 5/6
    ap = compute_ap50([(0.9, True), (0.8, False), (0.7, True), (0.6, True)], total_gt=3)
    assert ap == pytest.approx(5.0 / 6.0, abs=1e-6)


def test_pck_box_and_palm_21_and_14_hand_calculated():
    gt = _make_sample(size_px=150.0)
    assert gt.palm_px is not None
    pred_kps = gt.kps_xy.copy()
    # Shift exactly 4 of the 14 control keypoints past 0.2 * palm_px (and past 0.2 * 150 = 30 px)
    for idx in (4, 8, 12, 20):
        pred_kps[idx] += np.array([0.25 * gt.palm_px + 10.0, 0.0])
    pred = PredDet(box_xyxy=gt.box_xyxy, conf=0.9, kps_xy=pred_kps, kps_conf=np.full(21, 0.9))
    oc, _ = evaluate_sample_metrics(gt, [pred])
    assert oc.tp is True
    assert oc.pck_palm14_hits == 10 and oc.pck_palm14_total == 14
    assert oc.pck_palm14_hits / oc.pck_palm14_total == pytest.approx(10.0 / 14.0, abs=1e-6)
    assert oc.pck_palm21_hits == 17 and oc.pck_palm21_total == 21


def test_missing_palm_gt_excluded_from_palm_pck_but_kept_in_box_pck():
    vis = np.ones(21, dtype=bool)
    vis[0] = False  # wrist missing -> palm_px must be None
    gt = _make_sample(vis_mask=vis)
    assert gt.palm_px is None
    pred = PredDet(box_xyxy=gt.box_xyxy, conf=0.9, kps_xy=gt.kps_xy.copy(), kps_conf=np.full(21, 0.9))
    oc, prs = evaluate_sample_metrics(gt, [pred])
    agg = aggregate_outcomes([oc], prs)
    assert agg["keypoints"]["pck_box_overall"]["rate"] == pytest.approx(1.0)
    assert agg["keypoints"]["pck_palm14_overall"]["rate"] is None
    assert agg["keypoints"]["palm_valid_images"]["k"] == 0


def test_out_of_frame_gt_keypoints_excluded_from_pck_and_counted():
    kps = _make_hand_kps()
    kps[20] = (-15.0, 100.0)  # pinky tip physically outside left edge
    in_f = np.ones(21, dtype=bool)
    in_f[20] = False
    gt = _make_sample(kps=kps, in_frame_mask=in_f)
    pred_kps = kps.copy()
    pred_kps[20] = (0.0, 100.0)  # clipped inside frame -> far from -15
    pred = PredDet(box_xyxy=gt.box_xyxy, conf=0.9, kps_xy=pred_kps, kps_conf=np.full(21, 0.9))
    oc, prs = evaluate_sample_metrics(gt, [pred])
    agg = aggregate_outcomes([oc], prs)
    assert oc.pck_box_total == 20 and oc.pck_box_hits == 20
    assert agg["oob_gt_kpts_excluded"] == 1


def test_wilson_ci_hand_calculated():
    lo30, hi30 = wilson_ci(27, 30)
    assert lo30 == pytest.approx(0.7438, abs=1e-3)
    assert hi30 == pytest.approx(0.9654, abs=1e-3)
    lo5, hi5 = wilson_ci(4, 5)
    assert lo5 == pytest.approx(0.3755, abs=1e-3)
    assert hi5 == pytest.approx(0.9638, abs=1e-3)
    lo0, hi0 = wilson_ci(0, 0)
    assert math.isnan(lo0) and math.isnan(hi0)


# ------------------------------------------------------------------ 3. Gate14 & Size Buckets

def test_gate14_single_low_conf_joint_fails_and_attributes_joint_name():
    gt = _make_sample()
    kconf = np.full(21, 0.95)
    kconf[20] = 0.49  # pinky_tip drops below 0.50
    pred = PredDet(box_xyxy=gt.box_xyxy, conf=0.9, kps_xy=gt.kps_xy.copy(), kps_conf=kconf)
    passed, failed, min_c = check_gate14(pred, conf_min=0.5)
    assert passed is False and failed == [20] and min_c == pytest.approx(0.49)
    oc, prs = evaluate_sample_metrics(gt, [pred])
    agg = aggregate_outcomes([oc], prs)
    assert agg["gate14"]["drop_counts_gt14_tp"][KPT_NAMES[20]] == 1


def test_gate14_dual_denominators_and_size_buckets():
    assert size_bucket(39.9) == "<40"
    assert size_bucket(40.0) == "40-80"
    assert size_bucket(80.0) == "80-160"
    assert size_bucket(160.0) == ">=160"

    gt_full = _make_sample(size_px=50.0)
    vis_partial = np.ones(21, dtype=bool)
    vis_partial[20] = False
    gt_part = _make_sample(size_px=120.0, vis_mask=vis_partial)
    p_ok = PredDet(box_xyxy=gt_full.box_xyxy, conf=0.9, kps_xy=gt_full.kps_xy.copy(), kps_conf=np.full(21, 0.9))
    kconf_drop = np.full(21, 0.9)
    kconf_drop[20] = 0.2
    p_drop = PredDet(box_xyxy=gt_part.box_xyxy, conf=0.9, kps_xy=gt_part.kps_xy.copy(), kps_conf=kconf_drop)

    oc1, pr1 = evaluate_sample_metrics(gt_full, [p_ok])
    oc2, pr2 = evaluate_sample_metrics(gt_part, [p_drop])
    agg = aggregate_outcomes([oc1, oc2], pr1 + pr2)
    assert agg["gate14"]["pass_all_tp"]["k"] == 1 and agg["gate14"]["pass_all_tp"]["n"] == 2
    assert agg["gate14"]["pass_gt14_tp"]["k"] == 1 and agg["gate14"]["pass_gt14_tp"]["n"] == 1
    assert agg["by_size_bucket"]["40-80"]["n"] == 1 and agg["by_size_bucket"]["80-160"]["n"] == 1


# ------------------------------------------------------------------ 4. Sanity Invariants

def test_sanity_oracle_predictor_scores_100_percent():
    gts = [_make_sample(path=f"s{i}.jpg", size_px=80.0 + 20.0 * i) for i in range(5)]
    outcomes, prs = [], []
    for gt in gts:
        oracle = PredDet(box_xyxy=gt.box_xyxy, conf=1.0, kps_xy=gt.kps_xy.copy(), kps_conf=np.ones(21))
        oc, p = evaluate_sample_metrics(gt, [oracle])
        outcomes.append(oc)
        prs.extend(p)
    agg = aggregate_outcomes(outcomes, prs)
    assert agg["detection"]["recall_05"]["rate"] == pytest.approx(1.0)
    assert agg["detection"]["precision_05"]["rate"] == pytest.approx(1.0)
    assert agg["detection"]["ap50"] == pytest.approx(1.0)
    assert agg["keypoints"]["pck_box_overall"]["rate"] == pytest.approx(1.0)
    assert agg["keypoints"]["pck_palm14_overall"]["rate"] == pytest.approx(1.0)
    assert agg["gate14"]["pass_gt14_tp"]["rate"] == pytest.approx(1.0)


def test_sanity_null_predictor_scores_0_percent():
    gts = [_make_sample(path=f"s{i}.jpg") for i in range(5)]
    outcomes, prs = [], []
    for gt in gts:
        oc, p = evaluate_sample_metrics(gt, [])
        outcomes.append(oc)
        prs.extend(p)
    agg = aggregate_outcomes(outcomes, prs)
    assert agg["detection"]["recall_05"]["rate"] == pytest.approx(0.0)
    assert agg["detection"]["ap50"] == pytest.approx(0.0)
    assert agg["keypoints"]["pck_box_overall"]["rate"] == pytest.approx(0.0)
    assert agg["keypoints"]["pck_palm14_overall"]["rate"] == pytest.approx(0.0)
    assert agg["gate14"]["pass_all_tp"]["n"] == 0 and agg["gate14"]["pass_all_tp"]["rate"] is None


def test_sanity_shifted_predictor_and_extra_fp_penalized():
    gt = _make_sample()
    assert gt.palm_px is not None
    shifted = gt.kps_xy + np.array([0.30 * gt.palm_px, 0.0])
    good_box = PredDet(box_xyxy=gt.box_xyxy, conf=0.95, kps_xy=shifted, kps_conf=np.ones(21))
    extra_fp = PredDet(box_xyxy=(0.0, 0.0, 40.0, 40.0), conf=0.80, kps_xy=shifted, kps_conf=np.ones(21))
    oc, prs = evaluate_sample_metrics(gt, [good_box, extra_fp])
    agg = aggregate_outcomes([oc], prs)
    assert agg["detection"]["recall_05"]["rate"] == pytest.approx(1.0)
    assert agg["detection"]["precision_05"]["rate"] == pytest.approx(0.5)
    assert agg["detection"]["extra_fp_boxes"] == 1
    assert agg["keypoints"]["pck_palm14_overall"]["rate"] == pytest.approx(0.0)


# ------------------------------------------------------------------ 5. Visual Audit & Reproducibility

def test_visual_audit_outputs_and_deterministic_json(tmp_path):
    img = np.full((320, 420, 3), 90, dtype=np.uint8)
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img)
    before_bytes = p1.read_bytes()

    gt1 = _make_sample(path=str(p1), width=420, height=320)
    gt2 = _make_sample(path=str(p2), width=420, height=320)
    kconf = np.full(21, 0.9)
    kconf[8] = 0.25
    pred1 = PredDet(box_xyxy=gt1.box_xyxy, conf=0.9, kps_xy=gt1.kps_xy.copy(), kps_conf=kconf)
    oc1, pr1 = evaluate_sample_metrics(gt1, [pred1])
    oc2, pr2 = evaluate_sample_metrics(gt2, [])

    counts = save_visual_audit([oc1, oc2], tmp_path / "audit", max_per_cat=5)
    assert counts == {"worst_pck": 1, "gate14_drops": 1, "false_negatives": 1}
    assert p1.read_bytes() == before_bytes  # source image untouched
    gate_png = cv2.imread(str(tmp_path / "audit" / "gate14_drops" / "01_a.png"))
    fn_png = cv2.imread(str(tmp_path / "audit" / "false_negatives" / "01_b.png"))
    assert gate_png is not None and gate_png.shape[1] >= 540
    assert fn_png is not None and fn_png.shape[1] >= 540

    j1 = json.dumps(aggregate_outcomes([oc1, oc2], pr1 + pr2), sort_keys=True)
    j2 = json.dumps(aggregate_outcomes([oc1, oc2], pr1 + pr2), sort_keys=True)
    assert j1 == j2


def test_unmatched_pred_captured_on_fn_and_cross_split_dhash(tmp_path):
    from tools.eval_tier1_static import find_cross_split_dhash

    gt = _make_sample()
    far_pred = PredDet(box_xyxy=(0.0, 0.0, 40.0, 40.0), conf=0.92, kps_xy=gt.kps_xy.copy(), kps_conf=np.ones(21))
    oc, _ = evaluate_sample_metrics(gt, [far_pred])
    assert oc.tp is False
    assert oc.top_unmatched_pred is not None and oc.top_unmatched_pred.conf == pytest.approx(0.92)

    cross = find_cross_split_dhash([0b1010, 0b111111111111], ["v1.jpg", "v2.jpg"], [0b1011], ["t1.jpg"], max_hamming=2)
    assert cross["val_images_matched"] == 1 and cross["pair_count"] == 1


# ------------------------------------------------------------------ 6. Step-1 Audit Hardening Tests

def test_schema_strict_manifest_and_label_guards():
    base_m = {c: "x" for c in MANIFEST_COLUMNS}
    # Empty participant_id or video_sha256 must fail (cannot bypass dev/test leak checks)
    r_empty = {**base_m, "trial_id": "t1", "split": "dev", "participant_id": "", "video_sha256": ""}
    m_errs = validate_manifest_rows([r_empty], list(MANIFEST_COLUMNS))
    assert any("empty participant_id" in e for e in m_errs)
    assert any("empty video_sha256" in e for e in m_errs)

    base_l = {c: "" for c in LABEL_COLUMNS}
    rows_l = [
        {**base_l, "trial_id": "t1", "event_id": "e1", "kind": "intent", "command": "neutral", "onset_s": "1.0", "offset_s": "2.0"},
        {**base_l, "trial_id": "t1", "event_id": "e2", "kind": "intent", "command": "takeoff", "onset_s": "3.0", "offset_s": "5.0", "cue_s": "6.0", "timeout_s": "-1"},
        {**base_l, "trial_id": "t1", "event_id": "e2", "kind": "neutral", "command": "neutral", "onset_s": "4.5", "offset_s": "8.0"},
    ]
    l_errs = validate_label_rows(rows_l, list(LABEL_COLUMNS))
    assert any("intent row cannot use command='neutral'" in e for e in l_errs)
    assert any("duplicate event_id" in e for e in l_errs)
    assert any("overlaps intent" in e for e in l_errs)
    assert any("cue_s" in e for e in l_errs)
    assert any("timeout_s" in e for e in l_errs)


def test_greedy_by_conf_matching_and_top1_conf_recall():
    gt = _make_sample(size_px=100.0)
    # Case A: two TP boxes -> higher-conf box must win (matching PR-curve order) even if lower-conf has higher IoU
    x1, y1, x2, y2 = gt.box_xyxy
    high_conf_box = PredDet(box_xyxy=(x1 + 10, y1, x2 + 10, y2), conf=0.95, kps_xy=gt.kps_xy.copy(), kps_conf=np.ones(21))
    high_iou_box = PredDet(box_xyxy=gt.box_xyxy, conf=0.60, kps_xy=gt.kps_xy + 50.0, kps_conf=np.ones(21))
    oc_a, _ = evaluate_sample_metrics(gt, [high_iou_box, high_conf_box])
    assert oc_a.tp is True and oc_a.top1_conf_tp is True
    assert oc_a.matched_pred is high_conf_box

    # Case B: top-1 conf box misses GT, 2nd box hits GT -> recall=1, top1_conf_recall=0
    wrong_top = PredDet(box_xyxy=(0.0, 0.0, 20.0, 20.0), conf=0.95, kps_xy=gt.kps_xy.copy(), kps_conf=np.ones(21))
    right_second = PredDet(box_xyxy=gt.box_xyxy, conf=0.60, kps_xy=gt.kps_xy.copy(), kps_conf=np.ones(21))
    oc_b, prs_b = evaluate_sample_metrics(gt, [wrong_top, right_second])
    agg = aggregate_outcomes([oc_b], prs_b)
    assert agg["detection"]["recall_05"]["rate"] == pytest.approx(1.0)
    assert agg["detection"]["top1_conf_recall_05"]["rate"] == pytest.approx(0.0)


def test_label_path_in_images_parent_and_clean_yaml_default_oks(tmp_path):
    import yaml
    from tools.eval_tier1_static import build_clean_val_yaml, parse_latency_sizes

    # Dataset nested inside a parent folder literally named 'images'
    ds = tmp_path / "images" / "ds"
    img = np.full((240, 320, 3), 80, dtype=np.uint8)
    _write_yolo_pair(ds, "val", "sample1", img)
    samples, audit = audit_and_clean_split(ds, check_dhash=False)
    assert len(samples) == 1 and audit["multi_hand_images_count"] == 0

    # Even when data_fixed.yaml has custom kpt_oks_sigmas=0.12, default clean_val.yaml omits it (uses 1/21)
    (ds / "data.yaml").write_text(yaml.safe_dump({"kpt_shape": [21, 3], "names": {0: "hand"}}))
    (ds / "data_fixed.yaml").write_text(yaml.safe_dump({"kpt_shape": [21, 3], "names": {0: "hand"}, "kpt_oks_sigmas": [0.12] * 21}))
    out_yaml, mode = build_clean_val_yaml(ds, tmp_path / "list.txt", tmp_path / "clean_val.yaml")
    cfg = yaml.safe_load(out_yaml.read_text())
    assert mode == "ultralytics_default_1_over_21"
    assert "kpt_oks_sigmas" not in cfg

    assert parse_latency_sizes("640x480,1280x720") == ((640, 480), (1280, 720))
    with pytest.raises(ValueError):
        parse_latency_sizes("badsize")


def test_stress_scales_never_upscales_and_audit_cleans_stale_and_picks_large_fns(tmp_path):
    from tools.eval_tier1_static import run_stress_scales

    img = np.full((240, 320, 3), 100, dtype=np.uint8)
    p_small = tmp_path / "small.jpg"
    p_big = tmp_path / "big.jpg"
    cv2.imwrite(str(p_small), img)
    cv2.imwrite(str(p_big), img)
    gt_small = _make_sample(path=str(p_small), size_px=50.0, width=320, height=240)
    gt_big = _make_sample(path=str(p_big), size_px=150.0, width=320, height=240)
    res = run_stress_scales([gt_small, gt_big], lambda _im: [], scales=[36, 64], count=10)
    # Only gt_big (150 >= 64) should be eligible; gt_small (50 < 64) must not be upscaled
    assert res["source_count"] == 1 and res["min_source_hand_px"] == 64.0

    # Stale PNG removal + largest & smallest FN coverage
    audit_dir = tmp_path / "audit"
    (audit_dir / "false_negatives").mkdir(parents=True)
    stale = audit_dir / "false_negatives" / "99_stale.png"
    stale.write_bytes(b"stale")
    gts = []
    for sz in (30.0, 50.0, 120.0, 180.0):
        fp = tmp_path / f"fn_{int(sz)}.jpg"
        cv2.imwrite(str(fp), img)
        gts.append(_make_sample(path=str(fp), size_px=sz, width=320, height=240))
    ocs = [evaluate_sample_metrics(g, [])[0] for g in gts]
    save_visual_audit(ocs, audit_dir, max_per_cat=2)
    assert not stale.exists()
    fn_files = sorted(p.name for p in (audit_dir / "false_negatives").glob("*.png"))
    # With max_per_cat=2 -> 1 largest (180) + 1 smallest (30)
    assert fn_files == ["01_fn_180.png", "02_fn_30.png"]
