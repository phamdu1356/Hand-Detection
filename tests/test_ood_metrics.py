"""Deterministic unit tests for multi-hand OOD metrics & HaGRID gesture/FPR scoring."""
from __future__ import annotations

import numpy as np
import pytest

from benchmark.ood_metrics import (
    OODImageSample,
    aggregate_ood_outcomes,
    canonical_from_hand_command,
    derive_box_from_kps,
    evaluate_gesture_records,
    evaluate_multihand_image,
    make_gt_instance,
)
from tests.test_tier1_eval import _make_hand_kps
from tools.eval_tier1_static import PredDet


def test_multihand_matching_ignore_boxes_and_neg_fpr():
    k1 = _make_hand_kps(origin=(120.0, 220.0), scale=60.0)
    k2 = _make_hand_kps(origin=(420.0, 220.0), scale=60.0)
    b1 = derive_box_from_kps(k1, np.ones(21, dtype=bool), 640, 480, pad_frac=0.15)
    b2 = derive_box_from_kps(k2, np.ones(21, dtype=bool), 640, 480, pad_frac=0.15)
    assert b1 is not None and b2 is not None

    gt1 = make_gt_instance(b1, 640, 480, k1)
    gt2 = make_gt_instance(b2, 640, 480, k2)
    assert gt1 is not None and gt2 is not None
    ignore_b = (500.0, 350.0, 560.0, 410.0)

    sample_pos = OODImageSample(
        dataset="coco_wb",
        image_path="pos.jpg",
        sha256="aaa",
        width=640,
        height=480,
        instances=(gt1, gt2),
        ignore_boxes=(ignore_b,),
    )
    p1 = PredDet(box_xyxy=gt1.box_xyxy, conf=0.95, kps_xy=k1.copy(), kps_conf=np.ones(21))
    p2 = PredDet(box_xyxy=gt2.box_xyxy, conf=0.90, kps_xy=k2.copy(), kps_conf=np.ones(21))
    # Prediction overlapping ignore box should NOT count as FP
    p_ign = PredDet(box_xyxy=ignore_b, conf=0.80, kps_xy=k1.copy(), kps_conf=np.ones(21))
    # Genuine FP on empty background
    p_fp = PredDet(box_xyxy=(0.0, 0.0, 40.0, 40.0), conf=0.70, kps_xy=k1.copy(), kps_conf=np.ones(21))

    im_pos, pr_pos = evaluate_multihand_image(sample_pos, [p1, p2, p_ign, p_fp])
    assert im_pos.fp_count == 1
    assert im_pos.ignored_pred_count == 1
    assert all(io.tp for io in im_pos.instance_outcomes)

    # Zero-GT negative image: 1 clean image + 1 with FP
    neg1 = OODImageSample(dataset="neg", image_path="n1.jpg", sha256="b", width=640, height=480, instances=())
    neg2 = OODImageSample(dataset="neg", image_path="n2.jpg", sha256="c", width=640, height=480, instances=())
    im_n1, pr_n1 = evaluate_multihand_image(neg1, [])
    im_n2, pr_n2 = evaluate_multihand_image(neg2, [p_fp])

    agg = aggregate_ood_outcomes([im_pos, im_n1, im_n2], pr_pos + pr_n1 + pr_n2)
    assert agg["n_gt_instances"] == 2
    assert agg["detection"]["tp"] == 2 and agg["detection"]["fp"] == 2
    assert agg["detection"]["recall_05"]["rate"] == pytest.approx(1.0)
    assert agg["detection"]["precision_05"]["rate"] == pytest.approx(0.5)
    assert agg["detection"]["neg_image_fpr"]["k"] == 1 and agg["detection"]["neg_image_fpr"]["n"] == 2
    assert agg["keypoints"]["pck_palm14_overall"]["rate"] == pytest.approx(1.0)
    assert agg["gate14"]["pass_gt14_tp"]["rate"] == pytest.approx(1.0)


def test_gesture_records_macro_f1_fpr_and_4bucket_attribution():
    assert canonical_from_hand_command("point_left") == "one"
    assert canonical_from_hand_command("like") == "like"
    assert canonical_from_hand_command("none") == "none"

    records = [
        # 2 'like': 1 correct, 1 missed (no hand)
        {"gt_label": "like", "pred_label": "like", "hand_detected": True, "gate14_pass": True},
        {"gt_label": "like", "pred_label": "none", "hand_detected": False, "gate14_pass": False},
        # 2 'ok': 1 gate14 rejected, 1 wrong gesture (predicted 'like')
        {"gt_label": "ok", "pred_label": "none", "hand_detected": True, "gate14_pass": False},
        {"gt_label": "ok", "pred_label": "like", "hand_detected": True, "gate14_pass": True},
        # 2 'none' (no_gesture): 1 clean, 1 false command 'ok'
        {"gt_label": "none", "pred_label": "none", "hand_detected": False, "gate14_pass": False},
        {"gt_label": "none", "pred_label": "ok", "hand_detected": True, "gate14_pass": True},
    ]
    res = evaluate_gesture_records(records)
    assert res["error_attribution_positive"] == {
        "correct": 1,
        "no_hand_detected": 1,
        "gate14_rejected": 1,
        "wrong_gesture": 1,
    }
    # like: TP=1, GT=2 (Rec=0.5), Pred=2 (Prec=0.5) -> F1=0.5
    # ok:   TP=0, GT=2 (Rec=0.0), Pred=1 (Prec=0.0) -> F1=0.0
    # Macro-F1 = 0.25
    assert res["per_class"]["like"]["f1"] == pytest.approx(0.5)
    assert res["per_class"]["ok"]["f1"] == pytest.approx(0.0)
    assert res["macro_f1"] == pytest.approx(0.25)
    assert res["no_gesture_command_fpr"]["k"] == 1 and res["no_gesture_command_fpr"]["n"] == 2
