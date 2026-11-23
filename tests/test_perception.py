"""Perception-layer checks on demo/test_model.py: body-pose sharing, main-person choice, fields.

Tests that need the real models skip when the weights are not present. The shared-run and
selection tests use a fake person model and need no weights.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "demo"))

import test_model as tm  # noqa: E402  (demo/test_model.py)

HAND_POSE = "/home/pham-du/code/học và hành/Hand-Detection/hand_pose.pt"
BODY_POSE = "/home/pham-du/code/học và hành/Hand-Detection/yolo11n-pose.pt"


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.conf = np.asarray(conf, dtype=float)

    def __len__(self):
        return len(self.xyxy)


class FakePersonModel:
    """Stands in for YOLO pose: returns fixed people and counts predict calls."""

    def __init__(self, boxes, kpts):
        self.boxes = np.asarray(boxes, dtype=float)
        self.kpts = np.asarray(kpts, dtype=float)
        self.calls = 0

    def predict(self, frame, conf=None, imgsz=None, verbose=None):
        self.calls += 1
        n = len(self.boxes)
        res = SimpleNamespace(
            boxes=_Boxes(self.boxes, np.full(n, 0.9)),
            keypoints=SimpleNamespace(xy=self.kpts, conf=np.full((n, 17), 0.9)),
        )
        return [res]


def _person_kpts(x0, y0, scale=1.0):
    kp = np.zeros((17, 2))
    for i in range(17):
        kp[i] = (x0 + scale * 10 * (i % 4), y0 + scale * 10 * (i // 4))
    return kp


def _controller_without_weights():
    ctl = tm.OptimalFlycamController(os.path.dirname(ROOT), None, far_field_mode="off",
                                     heatmap_model_path="")
    return ctl


def test_select_main_person_largest_box_without_hand():
    boxes = [[0, 0, 100, 100], [200, 0, 500, 400]]
    kpts = np.stack([_person_kpts(10, 10), _person_kpts(250, 50)])
    assert tm.select_main_person(boxes, kpts, None) == 1


def test_select_main_person_prefers_person_whose_wrist_is_near_hand():
    boxes = [[0, 0, 100, 100], [200, 0, 500, 400]]
    kpts = np.stack([_person_kpts(10, 10), _person_kpts(250, 50)])
    kpts[0][tm.COCO_RIGHT_WRIST] = (60.0, 60.0)     # small person, wrist near the hand
    hand_center = (62.0, 58.0)
    assert tm.select_main_person(boxes, kpts, hand_center) == 0


def test_select_main_person_keeps_the_previous_person_while_visible():
    boxes = [[0, 0, 100, 200], [300, 0, 400, 200]]
    kpts = np.stack([_person_kpts(10, 10), _person_kpts(310, 10)])
    kpts[0][tm.COCO_RIGHT_WRIST] = (90.0, 100.0)      # bystander's wrist is now nearer the hand
    hand_center = (95.0, 100.0)
    prev = np.array([305.0, 5.0, 405.0, 205.0])      # followed person, moved a little
    assert tm.select_main_person(boxes, kpts, hand_center, prev_box=prev) == 1


def test_select_main_person_reselects_when_the_previous_person_is_gone():
    boxes = [[0, 0, 100, 200]]
    kpts = np.stack([_person_kpts(10, 10)])
    prev = np.array([300.0, 0.0, 400.0, 200.0])
    assert tm.select_main_person(boxes, kpts, None, prev_box=prev) == 0


def test_select_main_person_none_when_empty():
    assert tm.select_main_person(np.zeros((0, 4)), np.zeros((0, 17, 2)), None) is None


def test_body_pose_runs_once_per_frame_and_fields_are_filled():
    ctl = _controller_without_weights()
    fake = FakePersonModel([[100, 50, 300, 500]], [_person_kpts(120, 60)])
    ctl.body_pose_model = fake
    frame = np.full((720, 1280, 3), 128, np.uint8)

    first = ctl._run_body_pose(frame)
    second = ctl._run_body_pose(frame)
    assert first is second
    assert fake.calls == 1

    fields = ctl._body_fields(frame, None)
    assert fields["person_box"] == (100.0, 50.0, 200.0, 450.0)
    assert fields["body_keypoints"].shape == (17, 2)
    assert fields["body_kpt_conf"].shape == (17,)

    other = np.zeros_like(frame)                     # a new frame object is not served from cache
    ctl._run_body_pose(other)
    assert fake.calls == 2
    assert ctl._body_fields(other, None)["person_box"] is not None


def test_body_fields_empty_without_body_run_on_this_frame():
    ctl = _controller_without_weights()
    frame = np.zeros((48, 64, 3), np.uint8)
    fields = ctl._body_fields(frame, None)
    assert fields == {"body_keypoints": None, "body_kpt_conf": None, "person_box": None}


def test_detect_frame_does_not_mutate_frame_or_run_legacy_state(monkeypatch):
    import cv2

    ctl = _controller_without_weights()
    fake_hand_kps = np.tile(np.array([[100.0, 120.0]], dtype=np.float32), (21, 1))
    fake_hand_kps[9] = (100.0, 80.0)
    ctl.pose_model = object()
    monkeypatch.setattr(ctl, "_detect_hand_pose", lambda *a, **kw: {
        "box": (80.0, 60.0, 50.0, 70.0),
        "keypoints": fake_hand_kps,
        "kpt_conf": np.full(21, 0.9, dtype=np.float32),
        "conf": 0.92,
        "stage": "full_frame",
        "roi": None,
        "rois_checked": 0,
    })

    def _forbid(*_args, **_kwargs):
        raise AssertionError("detect_frame must not draw with OpenCV")

    monkeypatch.setattr(cv2, "line", _forbid)
    monkeypatch.setattr(cv2, "circle", _forbid)
    monkeypatch.setattr(cv2, "rectangle", _forbid)
    monkeypatch.setattr(cv2, "putText", _forbid)

    frame = np.full((240, 320, 3), 77, np.uint8)
    before = frame.copy()
    res = ctl.detect_frame(frame, conf_thres=0.5)
    assert res["detected"] is True
    assert np.array_equal(frame, before)
    assert len(ctl.wrist_history) == 0 and len(ctl.gesture_window) == 0


@pytest.mark.skipif(not (os.path.exists(HAND_POSE) and os.path.exists(BODY_POSE)),
                    reason="hand_pose.pt or yolo11n-pose.pt not present")
def test_real_models_on_generated_image_share_one_body_run():
    ctl = tm.OptimalFlycamController(ROOT, None, pose_model_path=HAND_POSE,
                                     body_pose_model_path=BODY_POSE, far_field_mode="always",
                                     heatmap_model_path="", always_body_pose=True)
    frame = np.full((720, 1280, 3), 200, np.uint8)
    frame[200:520, 500:780] = (60, 90, 140)          # a plain generated block, no real person
    body_model = ctl.body_pose_model
    calls = {"n": 0}
    original = body_model.predict

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    body_model.predict = counting
    ctl.process_frame(frame, conf_thres=0.5, far_field="always")
    assert calls["n"] == 1
    res = ctl.last_result
    assert res is not None
    for key in ("body_keypoints", "body_kpt_conf", "person_box"):
        assert key in res
