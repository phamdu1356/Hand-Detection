"""Body frame invariants: scale, roll, hand size under rotation, fallback and validity."""
import math

import numpy as np
import pytest

from dronectl.body_frame import (
    fallback_frame,
    hand_angle,
    hand_shoulder_ratio,
    hand_size,
    image_frame,
    palm_center,
    shoulder_frame,
    to_body,
)
from dronectl.config import ControlConfig
from dronectl.hand_rules import _is_valid_kp
from dronectl.types import BodyObs
from tests.synth import BodyPose, HandPose, Scene, hand_points, make_observation

PALM_BODY = (0.1, -0.6)


def _palm_and_hand(scene_kwargs, hand_kwargs=None):
    scene = Scene(**scene_kwargs)
    hand_kwargs = hand_kwargs or {}
    obs = make_observation(0.0, scene, body=BodyPose(),
                           hand=HandPose(gesture="palm", palm=PALM_BODY, **hand_kwargs))
    return obs


def _measure(obs):
    frame = shoulder_frame(obs.body)
    kps = obs.hand.kps
    return (to_body(frame, palm_center(kps)), hand_shoulder_ratio(kps, frame),
            hand_angle(kps, frame))


# ---------------------------------------------------------------- M7 building block

def test_same_pose_gives_same_body_coordinates_at_scale_1_and_quarter():
    near = _measure(_palm_and_hand({"shoulder_px": 160.0}))
    far = _measure(_palm_and_hand({"shoulder_px": 40.0}))
    assert np.allclose(near[0], far[0], atol=1e-9)
    assert near[0] == pytest.approx(np.array(PALM_BODY), abs=1e-9)
    assert near[1] == pytest.approx(far[1], rel=1e-9)
    assert near[2] == pytest.approx(far[2], abs=1e-9)


def test_tilting_the_person_does_not_change_body_coordinates():
    upright = _measure(_palm_and_hand({"tilt": 0.0}))
    tilted = _measure(_palm_and_hand({"tilt": math.radians(15)}))
    assert np.allclose(upright[0], tilted[0], atol=1e-9)
    assert upright[1] == pytest.approx(tilted[1], rel=1e-9)
    assert upright[2] == pytest.approx(tilted[2], abs=1e-9)


def test_hand_size_stable_under_40_degree_roll_about_palm_axis():
    # Orthographic roll about the wrist-to-middle axis: x coordinates shrink by cos(40 deg).
    local = hand_points("palm")
    rolled = local.copy()
    rolled[:, 0] *= math.cos(math.radians(40))

    def size(kps):
        # scale to pixels and move off the origin, because (0, 0) counts as a missing point
        return hand_size(kps * 100.0 + 300.0)

    ratio = size(rolled) / size(local)
    assert abs(ratio - 1.0) < 0.10


# ---------------------------------------------------------------- shoulder frame

def test_shoulder_frame_axes_are_orthonormal_and_point_to_users_right_and_up():
    obs = make_observation(0.0, Scene(), body=BodyPose())
    frame = shoulder_frame(obs.body)
    assert frame.kind == "shoulder"
    assert frame.scale == pytest.approx(160.0)
    assert np.dot(frame.ex, frame.ey) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(frame.ex) == pytest.approx(1.0)
    assert frame.ex[0] < -0.99         # the user's right is the image left on a raw camera
    assert frame.ey[1] < -0.99         # up is smaller v on screen


def test_shoulder_frame_matches_a_real_coco_person_facing_the_camera():
    # yolo11n-pose on ultralytics' zidane.jpg: left shoulder (COCO 5) at u=1048, right (6) at u=854
    kps = np.zeros((17, 2))
    kps[5], kps[6] = (1048.0, 300.0), (854.0, 300.0)
    body = BodyObs(kps=kps, kps_conf=np.full(17, 0.9), box=(800.0, 200.0, 300.0, 500.0))
    frame = shoulder_frame(body)
    above_right_hand = to_body(frame, (854.0, 150.0))   # above the user's right shoulder
    assert above_right_hand[0] > 0.4 and above_right_hand[1] > 0.7


def test_fallback_frame_uses_the_same_orientation_as_the_shoulder_frame():
    frame = fallback_frame(10.0)
    assert np.allclose(frame.ex, (-1.0, 0.0)) and np.allclose(frame.ey, (0.0, -1.0))


def test_shoulder_frame_is_none_when_shoulder_conf_is_low():
    obs = make_observation(0.0, Scene(), body=BodyPose(drop=(6,)))
    assert shoulder_frame(obs.body) is None


def test_shoulder_frame_is_none_when_shoulders_too_close():
    obs = make_observation(0.0, Scene(shoulder_px=10.0), body=BodyPose())
    assert shoulder_frame(obs.body) is None


def test_shoulder_frame_is_none_without_body():
    assert shoulder_frame(None) is None


def test_shoulder_frame_respects_custom_min_width():
    obs = make_observation(0.0, Scene(shoulder_px=30.0), body=BodyPose())
    assert shoulder_frame(obs.body) is not None
    strict = ControlConfig(shoulder_min_px=50.0)
    assert shoulder_frame(obs.body, strict) is None


def test_shoulder_frame_rejects_invalid_coordinates():
    obs = make_observation(0.0, Scene(), body=BodyPose())
    obs.body.kps[5] = (0.0, 0.0)       # invalid by _is_valid_kp
    assert shoulder_frame(obs.body) is None


# ---------------------------------------------------------------- hand measurements

def test_palm_centre_skips_invalid_points():
    kps = hand_points("palm") * 50.0 + 300.0
    full = palm_center(kps)
    kps2 = kps.copy()
    kps2[5] = (0.0, 0.0)
    partial = palm_center(kps2)
    assert partial is not None and not np.allclose(partial, full)
    # the remaining valid points are kp0, kp9, kp13, kp17
    expected = np.mean([kps2[i] for i in (0, 9, 13, 17)], axis=0)
    assert np.allclose(partial, expected)


def test_palm_centre_is_none_when_nothing_is_valid():
    assert palm_center(np.zeros((21, 2))) is None


def test_hand_size_is_none_when_wrist_missing():
    kps = hand_points("palm") * 50.0 + 300.0
    kps[0] = (0.0, 0.0)
    assert hand_size(kps) is None


def test_hand_size_is_none_for_degenerate_hand():
    kps = np.full((21, 2), 5.0)
    assert hand_size(kps) is None


def test_hand_size_equals_length_for_upright_palm():
    obs = make_observation(0.0, Scene(shoulder_px=200.0), hand=HandPose(gesture="palm", ratio=0.45))
    assert hand_size(obs.hand.kps) == pytest.approx(0.45 * 200.0, rel=0.03)


def test_hand_angle_increases_for_clockwise_turn():
    base = make_observation(0.0, Scene(), hand=HandPose(gesture="fist"))
    turned = make_observation(0.0, Scene(), hand=HandPose(gesture="fist", rot=math.radians(20)))
    a0 = hand_angle(base.hand.kps)
    a1 = hand_angle(turned.hand.kps)
    assert a1 - a0 == pytest.approx(math.radians(20), abs=1e-9)


def test_hand_angle_is_zero_for_upright_hand_in_image_frame():
    obs = make_observation(0.0, Scene(), hand=HandPose(gesture="fist"))
    assert hand_angle(obs.hand.kps) == pytest.approx(0.0, abs=1e-9)


def test_hand_angle_none_without_wrist():
    kps = hand_points("palm") * 50.0 + 300.0
    kps[9] = (0.0, 0.0)
    assert hand_angle(kps) is None


def test_hand_angle_uses_body_frame_when_given():
    obs = make_observation(0.0, Scene(tilt=math.radians(30)), body=BodyPose(),
                           hand=HandPose(gesture="fist"))
    frame = shoulder_frame(obs.body)
    assert hand_angle(obs.hand.kps, frame) == pytest.approx(0.0, abs=1e-9)
    assert hand_angle(obs.hand.kps) != pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------- fallback

def test_fallback_frame_uses_unit_and_is_marked():
    frame = fallback_frame(12.0)
    assert frame.kind == "hand_fallback"
    assert frame.scale == 12.0
    with pytest.raises(ValueError):
        fallback_frame(0.0)


def test_fallback_ratio_is_one_over_k_at_anchor_time():
    cfg = ControlConfig()
    obs = make_observation(0.0, Scene(), hand=HandPose(gesture="palm", ratio=0.45))
    size = hand_size(obs.hand.kps)
    frame = fallback_frame(cfg.fallback_k * size)
    assert hand_shoulder_ratio(obs.hand.kps, frame) == pytest.approx(1.0 / cfg.fallback_k, rel=1e-9)


def test_image_frame_is_identity_scale():
    frame = image_frame()
    assert np.allclose(to_body(frame, (3.0, 4.0)), (-3.0, -4.0))   # user's right = image left


def test_is_valid_kp_rules():
    assert not _is_valid_kp(None)
    assert not _is_valid_kp((0.0, 0.0))
    assert not _is_valid_kp((float("nan"), 3.0))
    assert not _is_valid_kp((1.0,))
    assert _is_valid_kp((1.0, 0.0))
