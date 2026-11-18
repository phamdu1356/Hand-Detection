"""Gesture labels (near rules, far arm poses) and the GestureFilter (gap tolerance, latch)."""
import numpy as np
import pytest

from dronectl.body_frame import shoulder_frame
from dronectl.config import ControlConfig
from dronectl.gestures import GestureFilter, far_gesture, near_gesture
from tests.synth import BodyPose, HandPose, Scene, make_observation

NEAR_EXPECTED = {
    "palm": "palm",
    "four": "palm",
    "fist": "fist",
    "like": "like",
    "ok": "ok",
    "one": "none",
    "three": "none",
    "peace_inverted": "none",
}


def _hand(gesture, conf=0.9, noise=0.0, seed=0):
    obs = make_observation(0.0, Scene(), hand=HandPose(gesture=gesture, conf=conf),
                           noise_px=noise, rng=np.random.default_rng(seed))
    return obs.hand


@pytest.mark.parametrize("gesture, label", sorted(NEAR_EXPECTED.items()))
def test_near_rule_mapping(gesture, label):
    assert near_gesture(_hand(gesture)) == label


def test_near_is_none_without_hand():
    assert near_gesture(None) == "none"


def test_near_is_none_when_mean_confidence_is_low():
    assert near_gesture(_hand("palm", conf=0.3)) == "none"


def test_near_palm_survives_one_pixel_noise():
    for seed in range(20):
        assert near_gesture(_hand("palm", noise=1.0, seed=seed)) == "palm"
        assert near_gesture(_hand("fist", noise=1.0, seed=seed)) == "fist"


def test_near_respects_custom_confidence_threshold():
    cfg = ControlConfig(hand_conf_min=0.95)
    assert near_gesture(_hand("palm", conf=0.9), cfg) == "none"


def _far(left="down", right="down", drop=(), conf=0.9, tilt=0.0, shoulder_px=160.0):
    scene = Scene(tilt=tilt, shoulder_px=shoulder_px)
    obs = make_observation(0.0, scene, body=BodyPose(left=left, right=right, drop=drop, conf=conf))
    frame = shoulder_frame(obs.body)
    return far_gesture(obs.body, frame)


def test_arms_up_is_recognised():
    assert _far("up", "up") == "arms_up"


def test_t_pose_is_recognised():
    assert _far("t", "t") == "t_pose"


@pytest.mark.parametrize("right, label", [
    ("point_right", "point_right"),
    ("point_left", "point_left"),
    ("point_up", "point_up"),
    ("point_down", "point_down"),
])
def test_pointing_with_one_arm(right, label):
    assert _far("down", right) == label


def test_pointing_on_left_arm_is_recognised():
    assert _far("point_left", "down") == "point_left"


def test_pointing_tolerance_rejects_diagonal():
    assert _far("down", "point_45") == "none"


def test_pointing_tolerance_accepts_within_30_degrees():
    assert _far("down", "point_20") == "point_right"


def test_arms_down_is_none():
    assert _far("down", "down") == "none"


def test_single_straight_arm_raised_is_point_up_by_spec():
    # Spec 4.2: one straight arm pointing up with the other arm hanging is point_up, not arms_up.
    assert _far("up", "down") == "point_up"


def test_both_arms_extended_is_none():
    assert _far("point_up", "point_right") == "none"


def test_opposite_horizontal_arms_are_t_pose_not_pointing():
    assert _far("point_left", "point_right") == "t_pose"


def test_pointing_down_needs_reach_margin_over_hanging_arm():
    assert _far("down", "point_down") == "point_down"


def test_pointing_down_with_no_reach_margin_is_none():
    # both arms hang at the same length: nothing reaches further than the other
    assert _far("down", "down") == "none"


def test_pointing_arm_that_is_not_straight_is_none():
    obs = make_observation(0.0, Scene(), body=BodyPose(left="down", right="point_right"))
    obs.body.kps[8] = obs.body.kps[6] + np.array([0.0, 60.0])   # bent elbow, pixel offsets
    frame = shoulder_frame(obs.body)
    assert far_gesture(obs.body, frame) == "none"


def test_missing_wrist_is_none():
    assert _far("up", "up", drop=(9,)) == "none"


def test_low_confidence_pose_is_none():
    assert _far("up", "up", conf=0.2) == "none"


def test_far_needs_body_and_shoulder_frame():
    assert far_gesture(None, None) == "none"
    obs = make_observation(0.0, Scene(), body=BodyPose(drop=(5,)))
    assert far_gesture(obs.body, shoulder_frame(obs.body)) == "none"


def test_far_is_none_in_fallback_frame():
    from dronectl.body_frame import fallback_frame

    obs = make_observation(0.0, Scene(), body=BodyPose(left="up", right="up"))
    assert far_gesture(obs.body, fallback_frame(20.0)) == "none"


def test_far_is_invariant_to_person_tilt():
    assert _far("up", "up", tilt=0.3) == "arms_up"
    assert _far("t", "t", tilt=-0.2) == "t_pose"
    assert _far("down", "point_right", tilt=0.25) == "point_right"


# ---------------------------------------------------------------- GestureFilter

def _feed(flt, samples):
    for t, raw in samples:
        flt.update(t, raw)


def test_label_becomes_stable_after_min_stable_time():
    flt = GestureFilter()
    _feed(flt, [(0.0, "palm"), (0.03, "palm")])
    assert flt.stable() == "none"
    _feed(flt, [(0.07, "palm")])
    assert flt.stable() == "palm"


def test_held_time_counts_from_first_appearance():
    flt = GestureFilter()
    _feed(flt, [(0.0, "fist"), (0.5, "fist"), (1.0, "fist")])
    assert flt.held("fist") == pytest.approx(1.0)
    assert flt.held("palm") == 0.0


def test_short_none_gap_does_not_reset():
    flt = GestureFilter()
    _feed(flt, [(0.0, "fist"), (0.5, "fist"), (0.55, "none"), (0.58, "fist"), (0.6, "fist")])
    assert flt.label == "fist"
    assert flt.held("fist") == pytest.approx(0.6)


def test_gap_of_exactly_tolerance_does_not_reset():
    flt = GestureFilter()
    _feed(flt, [(0.0, "fist"), (0.5, "fist"), (0.6, "none")])
    assert flt.label == "fist"


def test_gap_longer_than_tolerance_resets():
    flt = GestureFilter()
    _feed(flt, [(0.0, "fist"), (0.5, "fist"), (0.55, "none"), (0.7, "none")])
    assert flt.label == "none"
    _feed(flt, [(0.75, "fist")])
    assert flt.held("fist") == pytest.approx(0.0)


def test_different_label_switches_immediately():
    flt = GestureFilter()
    _feed(flt, [(0.0, "fist"), (0.3, "palm")])
    assert flt.label == "palm" and flt.held("palm") == pytest.approx(0.0)


def test_latch_blocks_held_until_released():
    flt = GestureFilter()
    _feed(flt, [(0.0, "like"), (1.0, "like")])
    assert flt.held("like") == pytest.approx(1.0)
    flt.latch("like")
    assert flt.held("like") == 0.0
    _feed(flt, [(1.2, "like"), (3.0, "like")])
    assert flt.held("like") == 0.0            # still held, still latched


def test_latch_releases_after_other_label_for_release_time():
    flt = GestureFilter()
    _feed(flt, [(0.0, "like"), (1.0, "like")])
    flt.latch("like")
    _feed(flt, [(1.1, "none"), (1.2, "none")])
    assert flt.latched == "like"              # 0.1 s of "away" is not enough
    _feed(flt, [(1.45, "none")])
    assert flt.latched is None
    _feed(flt, [(1.5, "like"), (2.0, "like")])
    assert flt.held("like") == pytest.approx(0.5)


def test_return_to_latched_label_cancels_release_timer():
    flt = GestureFilter()
    _feed(flt, [(0.0, "ok"), (1.0, "ok")])
    flt.latch("ok")
    _feed(flt, [(1.1, "none"), (1.2, "ok"), (1.6, "none"), (1.7, "ok")])
    assert flt.latched == "ok"                # the timer restarted at 1.6


def test_latch_on_other_label_does_not_block_current_label():
    flt = GestureFilter()
    _feed(flt, [(0.0, "palm"), (0.5, "palm")])
    flt.latch("like")
    assert flt.held("palm") == pytest.approx(0.5)


def test_reset_clears_state():
    flt = GestureFilter()
    _feed(flt, [(0.0, "fist"), (0.5, "fist")])
    flt.latch("fist")
    flt.reset()
    assert flt.label == "none" and flt.latched is None and flt.held_s == 0.0
