"""Checks that the synthetic keypoints have the geometry the other tests rely on."""
import math

import numpy as np
import pytest

from tests.synth import (
    GESTURES,
    BodyPose,
    HandPose,
    Scene,
    arm_points,
    body_points,
    hand_points,
    make_observation,
)


def _angle_deg(v):
    return math.degrees(math.atan2(v[1], v[0]))


def test_shoulder_width_matches_scene_scale():
    kp = body_points(BodyPose())
    scene = Scene(shoulder_px=160.0)
    px = np.array([scene.body_to_px(p) for p in kp])
    assert np.linalg.norm(px[5] - px[6]) == pytest.approx(160.0)
    assert px.shape == (17, 2)


def test_tilt_rolls_the_person_but_keeps_shoulder_width():
    kp = body_points(BodyPose())
    scene = Scene(shoulder_px=160.0, tilt=math.radians(15))
    px = np.array([scene.body_to_px(p) for p in kp])
    d = px[6] - px[5]
    assert np.linalg.norm(d) == pytest.approx(160.0)
    assert _angle_deg(d) == pytest.approx(15.0)


def test_upright_person_has_head_above_shoulders_in_pixels():
    scene = Scene()
    px = np.array([scene.body_to_px(p) for p in body_points(BodyPose())])
    assert px[0][1] < px[5][1]          # nose has smaller v than shoulder (image y points down)


@pytest.mark.parametrize("kind", ["down", "up", "t", "point_right", "point_left", "point_up", "point_down"])
def test_arm_kinds_build_expected_wrists(kind):
    el, wr = arm_points(-1, kind)
    assert el.shape == (2,) and wr.shape == (2,)
    if kind == "up":
        assert wr[1] > 1.0
    if kind == "t":
        assert wr[0] == pytest.approx(-2.0) and abs(wr[1]) < 1e-9
    if kind == "point_right":
        assert _angle_deg(wr - np.array([-0.5, 0.0])) == pytest.approx(0.0)
    if kind == "point_up":
        assert _angle_deg(wr - np.array([-0.5, 0.0])) == pytest.approx(90.0)


def test_numeric_point_angle_is_honoured():
    el, wr = arm_points(1, "point_45")
    assert _angle_deg(wr - np.array([0.5, 0.0])) == pytest.approx(45.0)


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        arm_points(1, "wave")
    with pytest.raises(ValueError):
        hand_points("thumbs_sideways")


def test_hand_palm_centre_lands_on_requested_point():
    scene = Scene(tilt=math.radians(-20))
    obs = make_observation(0.0, scene, hand=HandPose(gesture="ok", palm=(0.4, -0.6)))
    palm = obs.hand.kps[[0, 5, 9, 13, 17]].mean(axis=0)
    assert np.allclose(palm, scene.body_to_px((0.4, -0.6)))


def test_hand_length_is_ratio_times_shoulder_width():
    scene = Scene(shoulder_px=200.0)
    obs = make_observation(0.0, scene, hand=HandPose(gesture="fist", ratio=0.45))
    wrist_to_middle = np.linalg.norm(obs.hand.kps[9] - obs.hand.kps[0])
    assert wrist_to_middle == pytest.approx(0.45 * 200.0, rel=1e-6)


@pytest.mark.parametrize("gesture", GESTURES)
def test_every_gesture_has_21_finite_points(gesture):
    pts = hand_points(gesture)
    assert pts.shape == (21, 2) and np.all(np.isfinite(pts))
    assert np.allclose(pts[0], 0.0)


def test_open_finger_is_longer_than_curled_finger():
    open_pts = hand_points("palm")
    closed_pts = hand_points("fist")
    assert np.linalg.norm(open_pts[8]) > np.linalg.norm(closed_pts[8])


def test_missing_body_and_hand_are_optional():
    scene = Scene()
    obs = make_observation(1.5, scene)
    assert obs.body is None and obs.hand is None and obs.t == 1.5


def test_drop_sets_confidence_to_zero_only():
    scene = Scene()
    obs = make_observation(0.0, scene, body=BodyPose(drop=(9, 10)))
    assert obs.body.kps_conf[9] == 0.0 and obs.body.kps_conf[10] == 0.0
    assert obs.body.kps_conf[5] == pytest.approx(0.9)
    assert np.all(np.isfinite(obs.body.kps))


def test_noise_is_reproducible_and_zero_noise_is_exact():
    scene = Scene()
    clean = make_observation(0.0, scene, hand=HandPose(gesture="palm"))
    again = make_observation(0.0, scene, hand=HandPose(gesture="palm"))
    assert np.array_equal(clean.hand.kps, again.hand.kps)
    a = make_observation(0.0, scene, hand=HandPose(gesture="palm"), noise_px=2.0,
                         rng=np.random.default_rng(7))
    b = make_observation(0.0, scene, hand=HandPose(gesture="palm"), noise_px=2.0,
                         rng=np.random.default_rng(7))
    assert np.array_equal(a.hand.kps, b.hand.kps)
    assert not np.array_equal(a.hand.kps, clean.hand.kps)
