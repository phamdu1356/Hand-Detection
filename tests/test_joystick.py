"""Joystick: dead zone, full scale, expo curve, rate limits, horizontal cap and M7 scale invariance."""
import math

import numpy as np
import pytest

from dronectl.body_frame import shoulder_frame
from dronectl.config import ControlConfig
from dronectl.joystick import Joystick, rate_limit, shape, wrap_angle
from tests.synth import BodyPose, HandPose, Scene, make_observation

CFG = ControlConfig()
DT = 1.0 / 30.0


def _pose(palm=(0.0, -0.6), ratio=0.45, rot=0.0, scale_px=160.0, tilt=0.0):
    scene = Scene(shoulder_px=scale_px, tilt=tilt)
    obs = make_observation(0.0, scene, body=BodyPose(),
                           hand=HandPose(gesture="fist", palm=palm, ratio=ratio, rot=rot))
    return shoulder_frame(obs.body), obs.hand.kps


def _err(anchor, current, joy):
    """Anchor on pose `anchor`, return the normalised error for pose `current`."""
    frame, kps = anchor
    joy.anchor(frame, kps)
    frame2, kps2 = current
    return joy.raw_error(frame2, kps2)


# ---------------------------------------------------------------- shape()

@pytest.mark.parametrize("e", [0.0, 0.05, 0.1875 - 1e-6, -0.18])
def test_inside_dead_zone_is_zero(e):
    assert shape(e, dead=0.1875, alpha=0.5) == 0.0


def test_full_deflection_gives_full_output():
    assert shape(1.0, 0.1875, 0.5) == pytest.approx(1.0)
    assert shape(-1.0, 0.1875, 0.5) == pytest.approx(-1.0)


def test_beyond_full_is_clipped():
    assert shape(3.0, 0.2, 0.5) == pytest.approx(1.0)


def test_expo_formula_at_half_way():
    # s = 0.5 with no dead zone: alpha*s^3 + (1-alpha)*s = 0.5*0.125 + 0.5*0.5
    assert shape(0.5, 0.0, 0.5) == pytest.approx(0.3125)


def test_shape_is_odd_monotone_and_bounded():
    grid = np.linspace(-1.0, 1.0, 401)
    vals = np.array([shape(e, 0.1875, 0.5) for e in grid])
    assert np.all(np.diff(vals) >= -1e-12)
    assert np.allclose(vals, -vals[::-1], atol=1e-12)
    assert np.all(np.abs(vals) <= 1.0 + 1e-12)


def test_alpha_zero_is_linear_and_alpha_one_is_cubic():
    assert shape(0.6, 0.0, 0.0) == pytest.approx(0.6)
    assert shape(0.6, 0.0, 1.0) == pytest.approx(0.216)


# ---------------------------------------------------------------- rate_limit()

def test_accel_limits_growing_magnitude():
    assert rate_limit(0.0, 1.0, DT, accel=2.0, decel=4.0) == pytest.approx(2.0 * DT)


def test_decel_is_faster_than_accel():
    dec_step = 1.0 - rate_limit(1.0, 0.0, DT, accel=2.0, decel=4.0)
    acc_step = rate_limit(0.0, 1.0, DT, accel=2.0, decel=4.0)
    assert dec_step == pytest.approx(4.0 * DT)
    assert dec_step > acc_step


def test_sign_flip_uses_decel():
    v = rate_limit(0.5, -0.5, DT, accel=2.0, decel=4.0)
    assert v == pytest.approx(0.5 - 4.0 * DT)


def test_small_step_reaches_target_exactly():
    assert rate_limit(0.0, 0.01, DT, accel=2.0, decel=4.0) == pytest.approx(0.01)


def test_rate_limit_never_overshoots():
    assert rate_limit(0.99, 1.0, DT, accel=2.0, decel=4.0) == pytest.approx(1.0)


# ---------------------------------------------------------------- targets from the hand

def test_palm_movement_inside_dead_zone_gives_no_command():
    joy = Joystick()
    err = _err(_pose(palm=(0.0, -0.6)), _pose(palm=(0.1, -0.6)), joy)
    assert np.allclose(joy.targets(err), 0.0)


def test_full_right_gives_v_h_to_the_right():
    joy = Joystick()
    err = _err(_pose(palm=(0.0, -0.6)), _pose(palm=(0.8, -0.6)), joy)
    assert err[0] == pytest.approx(1.0)
    t = joy.targets(err)
    assert t[1] == pytest.approx(CFG.v_h)
    assert t[0] == pytest.approx(0.0, abs=1e-12)


def test_full_left_gives_minus_v_h():
    joy = Joystick()
    err = _err(_pose(palm=(0.0, -0.6)), _pose(palm=(-0.8, -0.6)), joy)
    assert joy.targets(err)[1] == pytest.approx(-CFG.v_h)


def test_full_up_gives_v_v():
    joy = Joystick()
    err = _err(_pose(palm=(0.0, -0.6)), _pose(palm=(0.0, 0.2)), joy)
    assert err[1] == pytest.approx(1.0)
    assert joy.targets(err)[2] == pytest.approx(CFG.v_v)


def test_hand_nearer_by_35_percent_gives_full_forward():
    joy = Joystick()
    err = _err(_pose(ratio=0.45), _pose(ratio=0.45 * 1.35), joy)
    assert err[2] == pytest.approx(1.0)
    assert joy.targets(err)[0] == pytest.approx(CFG.v_h)


def test_smaller_hand_gives_backward():
    joy = Joystick()
    err = _err(_pose(ratio=0.45), _pose(ratio=0.45 / 1.35), joy)
    assert joy.targets(err)[0] == pytest.approx(-CFG.v_h)


def test_clockwise_turn_yaws_right():
    joy = Joystick()
    err = _err(_pose(rot=0.0), _pose(rot=math.radians(35)), joy)
    assert err[3] == pytest.approx(1.0)
    yaw = joy.targets(err)[3]
    assert yaw == pytest.approx(-math.radians(CFG.w_yaw_deg))   # negative = turn right


def test_counter_clockwise_turn_yaws_left():
    joy = Joystick()
    err = _err(_pose(rot=0.0), _pose(rot=-math.radians(35)), joy)
    assert joy.targets(err)[3] == pytest.approx(math.radians(CFG.w_yaw_deg))


def test_yaw_dead_zone():
    joy = Joystick()
    err = _err(_pose(rot=0.0), _pose(rot=math.radians(8)), joy)
    assert joy.targets(err)[3] == 0.0


def test_wrap_angle_range():
    assert wrap_angle(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    assert wrap_angle(0.1) == pytest.approx(0.1)


def test_error_without_origin_is_none():
    joy = Joystick()
    assert joy.raw_error(*_pose()) is None


def test_anchor_fails_without_measurable_hand():
    joy = Joystick()
    frame, kps = _pose()
    kps = kps.copy()
    kps[0] = 0.0
    assert joy.anchor(frame, kps) is False
    assert joy.origin is None


def test_error_is_none_when_frame_kind_changed():
    joy = Joystick()
    frame, kps = _pose()
    joy.anchor(frame, kps)
    other = shoulder_frame(make_observation(0.0, Scene(), body=BodyPose()).body)
    other.kind = "hand_fallback"
    assert joy.raw_error(other, kps) is None


def test_error_is_none_when_hand_unmeasurable():
    joy = Joystick()
    frame, kps = _pose()
    joy.anchor(frame, kps)
    bad = kps.copy()
    bad[0] = 0.0
    bad[9] = 0.0
    assert joy.raw_error(frame, bad) is None


def test_point_targets_follow_v_far():
    joy = Joystick()
    assert joy.point_target("right")[1] == pytest.approx(CFG.v_far)
    assert joy.point_target("left")[1] == pytest.approx(-CFG.v_far)
    assert joy.point_target("up")[2] == pytest.approx(CFG.v_far)
    assert joy.point_target("down")[2] == pytest.approx(-CFG.v_far)
    assert np.allclose(joy.point_target("up")[[0, 1, 3]], 0.0)
    with pytest.raises(ValueError):
        joy.point_target("forward")


# ---------------------------------------------------------------- stepping

def test_first_output_respects_accel_limit():
    joy = Joystick()
    cmd = joy.step(np.array([0.0, CFG.v_h, 0.0, 0.0]), DT)
    assert cmd.right <= CFG.acc_h * DT + 1e-12
    assert cmd.right > 0.0


def test_output_converges_to_target():
    joy = Joystick()
    tgt = np.array([0.0, 0.7, 0.0, 0.0])
    for _ in range(300):
        cmd = joy.step(tgt, DT)
    assert cmd.right == pytest.approx(0.7, abs=1e-6)


def test_braking_is_faster_than_acceleration():
    joy = Joystick()
    for _ in range(300):
        joy.step(np.array([0.0, CFG.v_h, 0.0, 0.0]), DT)
    prev = joy.step(np.zeros(4), DT).right
    for _ in range(60):
        cur = joy.step(np.zeros(4), DT).right
        assert prev - cur <= CFG.dec_lin * DT + 1e-9   # never faster than the decel limit
        prev = cur
    assert joy.step(np.zeros(4), DT).right == pytest.approx(0.0, abs=1e-6)


def test_horizontal_total_never_exceeds_v_h():
    rng = np.random.default_rng(3)
    joy = Joystick()
    for _ in range(500):
        tgt = np.array([rng.uniform(-CFG.v_h, CFG.v_h), rng.uniform(-CFG.v_h, CFG.v_h),
                        rng.uniform(-1, 1), rng.uniform(-1, 1)])
        cmd = joy.step(tgt, DT)
        assert math.hypot(cmd.forward, cmd.right) <= CFG.v_h + 1e-9


def test_diagonal_full_deflection_is_capped():
    joy = Joystick()
    for _ in range(300):
        cmd = joy.step(np.array([CFG.v_h, CFG.v_h, 0.0, 0.0]), DT)
    assert math.hypot(cmd.forward, cmd.right) == pytest.approx(CFG.v_h, rel=1e-6)


def test_dt_is_clamped():
    joy = Joystick()
    cmd = joy.step(np.array([0.0, 1.0, 0.0, 0.0]), 0.0)     # treated as 1 ms
    assert cmd.right <= CFG.acc_h * 1e-3 + 1e-12
    joy2 = Joystick()
    cmd2 = joy2.step(np.array([0.0, 1.0, 0.0, 0.0]), 10.0)  # treated as 0.2 s
    assert cmd2.right <= CFG.acc_h * 0.2 + 1e-9


def test_reset_clears_origin_and_output():
    joy = Joystick()
    joy.anchor(*_pose())
    joy.step(np.array([0.0, 1.0, 0.0, 0.0]), DT)
    joy.reset()
    assert joy.origin is None
    assert joy.step(np.zeros(4), DT).right == 0.0


def test_zero_target_from_rest_gives_zero_command():
    joy = Joystick()
    assert joy.step(np.zeros(4), DT).is_zero()


def test_positive_yaw_target_gives_positive_yaw_rate():
    joy = Joystick()
    cmd = joy.step(np.array([0.0, 0.0, 0.0, 0.5]), DT)
    assert cmd.yaw_rate > 0.0 and cmd.right == 0.0


# ---------------------------------------------------------------- M7: scale invariance

def _run_sequence(scale_px):
    """Same body-unit gesture sequence at a given person scale; returns the commands."""
    path = [
        dict(palm=(0.0, -0.6), ratio=0.45, rot=0.0),
        dict(palm=(0.25, -0.5), ratio=0.45, rot=0.1),
        dict(palm=(0.6, -0.2), ratio=0.5, rot=0.2),
        dict(palm=(0.8, 0.1), ratio=0.6, rot=0.35),
        dict(palm=(0.3, 0.25), ratio=0.55, rot=-0.25),
        dict(palm=(-0.4, -0.8), ratio=0.4, rot=-0.5),
        dict(palm=(-0.8, -0.3), ratio=0.45, rot=0.0),
    ]
    joy = Joystick()
    anchor = _pose(scale_px=scale_px, **path[0])
    joy.anchor(*anchor)
    out = []
    for step in path * 4:
        err = joy.raw_error(*_pose(scale_px=scale_px, **step))
        cmd = joy.step(joy.targets(err), DT)
        out.append((err, (cmd.forward, cmd.right, cmd.up, cmd.yaw_rate)))
    return out


def test_m7_commands_match_between_full_scale_and_quarter_scale():
    near = _run_sequence(160.0)      # person at scale 1.0
    far = _run_sequence(40.0)        # person at scale 0.25
    worst = 0.0
    for (e1, c1), (e2, c2) in zip(near, far):
        assert np.allclose(e1, e2, atol=1e-9)
        for a, b in zip(c1, c2):
            denom = max(abs(a), abs(b), 1e-9)
            worst = max(worst, abs(a - b) / denom)
    assert worst <= 0.01                       # M7: <= 1 % difference
    assert worst < 1e-6                        # in practice the two agree exactly


def test_m7_full_deflection_is_same_at_both_scales():
    for scale in (160.0, 40.0):
        joy = Joystick()
        joy.anchor(*_pose(scale_px=scale, palm=(0.0, -0.6)))
        err = joy.raw_error(*_pose(scale_px=scale, palm=(0.8, -0.6)))
        assert joy.targets(err)[1] == pytest.approx(CFG.v_h)
