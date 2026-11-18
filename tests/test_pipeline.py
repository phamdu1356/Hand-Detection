"""End-to-end scripts through Controller + MockBackend, and range selection."""
import math

import numpy as np
import pytest

from dronectl.config import ControlConfig
from dronectl.pipeline import Controller, choose_range
from dronectl.types import FLYING_MODES, MOVING_MODES, Mode
from dronectl.backends.mock import MockBackend
from tests.synth import BodyPose, HandPose, Scene, make_observation

FPS = 30.0
DT = 1.0 / FPS
DEFAULT_BODY = BodyPose()


class Runner:
    """Feeds synthetic frames to the controller and applies its outputs to the mock drone."""

    def __init__(self, cfg=None, scene=None):
        self.cfg = cfg or ControlConfig()
        self.ctl = Controller(self.cfg)
        self.mock = MockBackend()
        self.scene = scene or Scene()
        self.t = 0.0
        self.outs = []

    def frame(self, body, hand):
        obs = make_observation(self.t, self.scene, body=body, hand=hand)
        out = self.ctl.update(obs, self.mock.state())
        if out.action == "takeoff":
            self.mock.takeoff(self.cfg.takeoff_altitude)
        elif out.action == "land":
            self.mock.land()
        self.mock.send_velocity(out.cmd)
        self.mock.step(DT)
        self.outs.append(out)
        self.t += DT
        return out

    def run(self, seconds, body, hand_at=lambda k, n: None):
        """hand_at(k, n) -> HandPose or None for frame k of n."""
        n = int(round(seconds * FPS))
        for k in range(n):
            self.frame(body, hand_at(k, n) if callable(hand_at) else hand_at)

    def wait_until(self, pred, body, hand, max_s=15.0):
        start = self.t
        while not pred(self) and self.t - start < max_s:
            self.frame(body, hand)
        assert pred(self), f"condition not reached in {max_s}s, mode={self.ctl.mode}"

    def actions(self):
        return [o.action for o in self.outs if o.action]


def _const(h):
    return lambda k, n: h


# ---------------------------------------------------------------- range selection

def _frame_for(hand_px_ratio=0.45, body=True):
    scene = Scene()
    obs = make_observation(0.0, scene, body=DEFAULT_BODY if body else None,
                           hand=HandPose(gesture="palm", ratio=hand_px_ratio) if hand_px_ratio else None)
    return obs


@pytest.mark.parametrize("hand_ok, has_frame, hs, prev, expected", [
    (True, False, 50.0, "none", "near"),          # hand but no shoulders: near (fallback)
    (True, True, 0.40 * 160.0, "none", "near"),   # q = 0.40 >= 0.35
    (True, True, 0.20 * 160.0, "near", "far"),    # q = 0.20 < 0.25
    (True, True, 0.30 * 160.0, "near", "near"),   # band: keep previous
    (True, True, 0.30 * 160.0, "far", "far"),     # band: keep previous
    (True, True, 0.30 * 160.0, "none", "far"),    # band with no previous: far
    (False, True, None, "near", "far"),           # person only
    (False, False, None, "near", "none"),         # nothing
])
def test_choose_range_table(hand_ok, has_frame, hs, prev, expected):
    frame = None
    if has_frame:
        frame = _frame_for(body=True)
        from dronectl.body_frame import shoulder_frame
        frame = shoulder_frame(frame.body)
    assert choose_range(ControlConfig(), hand_ok, frame, hs, prev) == expected


def test_range_does_not_flicker_inside_hysteresis_band():
    ctl = Controller()
    seen = set()
    scene = Scene()
    for i in range(300):
        ratio = 0.30 + 0.03 * math.sin(i / 5.0)     # oscillates within [0.27, 0.33]
        obs = make_observation(i / FPS, scene, body=DEFAULT_BODY,
                               hand=HandPose(gesture="palm", ratio=ratio))
        seen.add(ctl.update(obs).range_mode)
    assert seen == {"far"}                          # started far, never switched


def test_range_stays_near_inside_band_once_near():
    ctl = Controller()
    scene = Scene()
    modes = []
    for i in range(300):
        ratio = 0.45 if i < 5 else 0.30 + 0.03 * math.sin(i / 5.0)
        obs = make_observation(i / FPS, scene, body=DEFAULT_BODY,
                               hand=HandPose(gesture="palm", ratio=ratio))
        modes.append(ctl.update(obs).range_mode)
    assert set(modes) == {"near"}


def test_range_switches_to_far_when_hand_shrinks():
    ctl = Controller()
    scene = Scene()
    obs = make_observation(0.0, scene, body=DEFAULT_BODY, hand=HandPose(gesture="palm", ratio=0.45))
    assert ctl.update(obs).range_mode == "near"
    obs = make_observation(1 / FPS, scene, body=DEFAULT_BODY, hand=HandPose(gesture="palm", ratio=0.2))
    assert ctl.update(obs).range_mode == "far"


# ---------------------------------------------------------------- scripted scenarios

def test_full_script_like_fist_move_palm_ok_lands():
    r = Runner()
    body = DEFAULT_BODY
    r.run(1.2, body, _const(HandPose("like")))                             # takeoff
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, body, HandPose("fist"))
    positions = []

    def move(k, n):
        return HandPose("fist", palm=(0.4 * k / (n - 1), -0.6))           # 0.4 shoulder widths in 2 s

    for k in range(int(2 * FPS)):
        r.frame(body, move(k, int(2 * FPS)))
        positions.append(r.mock.right_pos)
    assert r.ctl.mode is Mode.FOLLOW
    r.run(0.4, body, _const(HandPose("palm", palm=(0.4, -0.6))))          # stop
    assert r.ctl.mode is Mode.HOVER
    r.run(1.2, body, _const(HandPose("ok", palm=(0.4, -0.6))))            # land
    assert r.ctl.mode is Mode.LANDING
    r.wait_until(lambda r: r.ctl.mode is Mode.LANDED, body,
                 HandPose("three", palm=(0.4, -0.6)))                      # neutral hand while landing

    assert r.actions() == ["takeoff", "land"]
    assert r.mock.events == ["takeoff", "land", "landed"]
    assert r.mock.right_pos > 0.15                                         # drone moved to the right
    assert all(b >= a - 1e-9 for a, b in zip(positions, positions[1:]))   # moved right, never back
    assert r.ctl.mode is Mode.LANDED


def test_commands_only_in_moving_modes_and_none_on_the_ground():
    r = Runner()
    body = DEFAULT_BODY
    r.run(1.2, body, _const(HandPose("like")))
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, body, HandPose("fist"))
    r.run(1.0, body, lambda k, n: HandPose("fist", palm=(0.3 * k / n, -0.6)))
    r.run(0.4, body, _const(HandPose("palm")))
    r.run(1.2, body, _const(HandPose("ok")))
    r.wait_until(lambda r: r.ctl.mode is Mode.LANDED, body, HandPose("three"))
    for o in r.outs:
        if not o.cmd.is_zero():
            assert o.mode in MOVING_MODES
        if o.mode is Mode.LANDED:
            assert o.cmd.is_zero() and o.action != "land"
    assert r.mock.ground_commands == 0


def test_arms_up_for_5_seconds_takes_off_exactly_once():
    r = Runner()
    r.run(5.0, BodyPose(left="up", right="up"))
    assert r.actions() == ["takeoff"]
    assert r.ctl.mode is Mode.HOVER
    assert r.mock.altitude == pytest.approx(1.5, abs=0.05)


def test_like_held_0_9_seconds_does_not_take_off():
    r = Runner()
    r.run(0.9, DEFAULT_BODY, _const(HandPose("like")))
    r.run(0.5, DEFAULT_BODY)
    assert r.actions() == []
    assert r.mock.events == [] and r.ctl.mode is Mode.LANDED


def test_hand_and_person_lost_in_follow_hovers_within_0_6_seconds():
    r = Runner()
    r.run(1.2, DEFAULT_BODY, _const(HandPose("like")))
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, DEFAULT_BODY, HandPose("fist"))
    r.run(0.5, DEFAULT_BODY, _const(HandPose("fist")))
    assert r.ctl.mode is Mode.FOLLOW
    start = r.t
    while r.ctl.mode is Mode.FOLLOW and r.t - start < 2.0:
        r.frame(None, None)
    assert r.ctl.mode is Mode.HOVER
    assert r.t - start <= 0.6                                             # M3
    r.run(0.5, DEFAULT_BODY)
    assert np.allclose(r.mock.vel, 0.0, atol=1e-9)                       # drone does not drift on its own


def test_hand_lost_while_person_visible_hovers_at_once():
    r = Runner()
    r.run(1.2, DEFAULT_BODY, _const(HandPose("like")))
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, DEFAULT_BODY, HandPose("fist"))
    r.run(0.5, DEFAULT_BODY, _const(HandPose("fist")))
    r.frame(DEFAULT_BODY, None)                                           # far range now
    assert r.ctl.mode is Mode.HOVER


def test_pointing_right_moves_right_in_far_range():
    r = Runner()
    r.run(1.6, BodyPose(left="up", right="up"))                          # takeoff with arms up
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, DEFAULT_BODY, None)
    # The latch on arms_up blocks landing; releasing the arms first is what re-arms it.
    r.run(0.5, DEFAULT_BODY)
    r.run(1.0, BodyPose(right="point_right"))
    assert r.ctl.mode is Mode.POINT_MOVE
    assert r.ctl.fsm.point_dir == "right"
    assert r.mock.right_pos > 0.1          # 0.5 s of POINT_MOVE, accel and EMA limited
    r.run(0.5, DEFAULT_BODY)
    assert r.ctl.mode is Mode.HOVER


def test_fallback_frame_moves_drone_right_when_palm_moves_right_without_shoulders():
    r = Runner()
    r.run(1.2, None, _const(HandPose("like")))
    assert r.ctl.mode is Mode.TAKING_OFF
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, None, HandPose("fist"))
    r.run(0.4, None, _const(HandPose("fist", palm=(0.0, -0.6))))
    assert r.ctl.mode is Mode.FOLLOW
    assert r.outs[-1].debug["frame"] == "hand_fallback"
    before = r.mock.right_pos
    r.run(1.0, None, lambda k, n: HandPose("fist", palm=(0.1 * k / n, -0.6)))
    assert r.mock.right_pos > before + 0.05
    assert r.ctl.mode is Mode.FOLLOW


def test_reanchor_when_shoulders_reappear_keeps_command_zero_that_frame():
    r = Runner()
    r.run(1.2, None, _const(HandPose("like")))
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, None, HandPose("fist"))
    r.run(0.4, None, _const(HandPose("fist")))
    assert r.ctl.mode is Mode.FOLLOW
    out = r.frame(DEFAULT_BODY, HandPose("fist"))                        # frame kind changes
    assert out.debug.get("reanchored") is True
    assert out.cmd.is_zero()


def test_scale_invariance_end_to_end_m7():
    """The same script at person scale 1.0 and 0.25 gives the same commands (within 1 %)."""
    def script(shoulder_px):
        r = Runner(scene=Scene(shoulder_px=shoulder_px))
        r.run(1.2, DEFAULT_BODY, _const(HandPose("like")))
        r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, DEFAULT_BODY, HandPose("fist"))
        r.run(0.4, DEFAULT_BODY, _const(HandPose("fist")))
        r.run(1.5, DEFAULT_BODY, lambda k, n: HandPose("fist", palm=(0.3 * k / n, -0.6 + 0.1 * k / n),
                                                    ratio=0.45 * (1 + 0.2 * k / n)))
        return [(o.cmd.forward, o.cmd.right, o.cmd.up, o.cmd.yaw_rate)
                for o in r.outs if o.mode is Mode.FOLLOW]

    full = script(160.0)
    quarter = script(40.0)
    assert len(full) == len(quarter) > 30
    worst = 0.0
    for a_vec, b_vec in zip(full, quarter):
        for a, b in zip(a_vec, b_vec):
            denom = max(abs(a), abs(b), 1e-9)
            worst = max(worst, abs(a - b) / denom)
    assert worst <= 0.01                                                  # M7 (1 %)
    assert max(abs(v) for vec in full for v in vec) > 0.05                # commands are real


def test_reset_returns_controller_to_landed():
    r = Runner()
    r.run(1.2, DEFAULT_BODY, _const(HandPose("like")))
    assert r.ctl.mode is Mode.TAKING_OFF
    r.ctl.reset()
    assert r.ctl.mode is Mode.LANDED


def test_debug_fields_present_in_follow():
    r = Runner()
    r.run(1.2, DEFAULT_BODY, _const(HandPose("like")))
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, DEFAULT_BODY, HandPose("fist"))
    r.run(0.4, DEFAULT_BODY, _const(HandPose("fist")))
    dbg = r.outs[-1].debug
    for key in ("frame", "label", "stable", "held_s", "reason", "hx", "hy", "d", "err"):
        assert key in dbg
    assert dbg["frame"] == "shoulder"


def test_point_debug_contains_direction():
    r = Runner()
    r.run(1.6, BodyPose(left="up", right="up"))
    r.wait_until(lambda r: r.ctl.mode is Mode.HOVER, DEFAULT_BODY, None)
    r.run(0.5, DEFAULT_BODY)
    r.run(0.8, BodyPose(right="point_up"))
    assert r.ctl.mode is Mode.POINT_MOVE and r.outs[-1].debug["point"] == "up"


def test_controller_accepts_missing_drone_state():
    ctl = Controller()
    out = ctl.update(make_observation(0.0, Scene(), body=DEFAULT_BODY))
    assert out.mode is Mode.LANDED and out.cmd.is_zero()


def test_moving_modes_are_subset_of_flying_modes():
    assert MOVING_MODES <= FLYING_MODES
