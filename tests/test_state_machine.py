"""Flight state machine: one test per row of table 6.2, plus the latch and the invariant property tests."""
import random

import pytest

from dronectl.config import ControlConfig
from dronectl.gestures import GestureFilter
from dronectl.state_machine import StateMachine
from dronectl.types import FLYING_MODES, MOVING_MODES, Mode

FPS = 30.0
DT = 1.0 / FPS


class Sim:
    """Drives a StateMachine with a GestureFilter at 30 Hz and a simple altitude model."""

    def __init__(self, cfg=None, altitude=0.0):
        self.cfg = cfg or ControlConfig()
        self.fsm = StateMachine(self.cfg)
        self.flt = GestureFilter(self.cfg)
        self.t = 0.0
        self.altitude = altitude
        self.none_s = 0.0
        self.none_since = None
        self.actions = []
        self.last = None

    @property
    def landed(self):
        return self.altitude <= 0.0 and self.fsm.mode is Mode.LANDING

    def frame(self, label, range_mode):
        self.flt.update(self.t, label)
        if range_mode == "none":
            if self.none_since is None:
                self.none_since = self.t
            self.none_s = self.t - self.none_since
        else:
            self.none_since = None
            self.none_s = 0.0
        d = self.fsm.step(self.t, range_mode, self.flt, self.none_s, self.altitude, self.landed)
        if d.action:
            self.actions.append((self.t, d.action))
        self.last = d
        if self.fsm.mode is Mode.TAKING_OFF:
            self.altitude += 0.5 * DT
        elif self.fsm.mode is Mode.LANDING:
            self.altitude = max(0.0, self.altitude - 0.4 * DT)
        self.t += DT
        return d

    def hold(self, label, range_mode, seconds):
        n = int(round(seconds * FPS))
        for _ in range(n):
            self.frame(label, range_mode)
        return self.last

    def settle(self, seconds=0.5):
        """Let the drone finish TAKING_OFF or LANDING while showing no gesture."""
        return self.hold("none", "none", seconds)


def _flying():
    """Return a Sim in HOVER, by taking off with like and waiting for the altitude."""
    sim = Sim()
    sim.hold("like", "near", 1.2)
    sim.settle(10.0)
    assert sim.fsm.mode is Mode.HOVER
    return sim


def _taking_off():
    """Return a Sim in TAKING_OFF (the drone is still on the ground, so it stays there)."""
    sim = Sim()
    sim.hold("like", "near", 1.2)
    assert sim.fsm.mode is Mode.TAKING_OFF
    return sim


# ---------------------------------------------------------------- row 1: stop on palm / t-pose

@pytest.mark.parametrize("state", [Mode.TAKING_OFF, Mode.HOVER, Mode.FOLLOW, Mode.POINT_MOVE, Mode.LANDING])
def test_row1_palm_stops_every_flying_mode(state):
    sim = Sim(altitude=1.0)          # airborne, so LANDING does not complete at once
    sim.fsm.mode = state
    sim.fsm.point_dir = "right" if state is Mode.POINT_MOVE else None
    sim.hold("palm", "near", 0.2)
    assert sim.fsm.mode is Mode.HOVER


def test_row1_palm_cancels_landing():
    sim = _flying()
    sim.hold("ok", "near", 1.2)
    assert sim.fsm.mode is Mode.LANDING
    sim.hold("palm", "near", 0.2)
    assert sim.fsm.mode is Mode.HOVER
    assert [a for _, a in sim.actions] == ["takeoff", "land"]


def test_row1_t_pose_held_0_2s_stops_in_far_range():
    sim = _flying()
    sim.hold("fist", "near", 0.5)
    assert sim.fsm.mode is Mode.FOLLOW
    sim.hold("t_pose", "far", 0.5)
    assert sim.fsm.mode is Mode.HOVER


def test_row1_short_t_pose_does_not_stop():
    sim = _taking_off()
    sim.hold("t_pose", "far", 0.1)
    assert sim.fsm.mode is Mode.TAKING_OFF


def test_row1_palm_does_not_act_while_landed():
    sim = Sim()
    sim.hold("palm", "near", 1.0)
    assert sim.fsm.mode is Mode.LANDED and sim.actions == []


# ---------------------------------------------------------------- row 2: no person or hand for 0.5 s

def test_row2_range_none_for_half_second_stops_flight():
    sim = _taking_off()
    sim.hold("none", "none", 0.4)
    assert sim.fsm.mode is Mode.TAKING_OFF   # below 0.5 s
    sim.hold("none", "none", 0.3)
    assert sim.fsm.mode is Mode.HOVER


def test_row2_range_none_stops_follow_at_once_through_row8():
    sim = _flying()
    sim.hold("fist", "near", 0.5)
    sim.hold("none", "none", 0.05)
    assert sim.fsm.mode is Mode.HOVER        # range not near: row 8, before row 2 time runs out


def test_row2_short_gap_in_range_none_keeps_flying():
    sim = _taking_off()
    sim.hold("none", "none", 0.2)
    sim.hold("like", "near", 0.1)
    assert sim.fsm.mode is Mode.TAKING_OFF


# ---------------------------------------------------------------- row 3: land on ok (near) / arms_up (far)

def test_row3_ok_held_1s_lands_in_near():
    sim = _flying()
    sim.hold("ok", "near", 0.9)
    assert sim.fsm.mode is Mode.HOVER
    sim.hold("ok", "near", 0.2)
    assert sim.fsm.mode is Mode.LANDING
    assert sim.actions[-1][1] == "land"


def test_row3_arms_up_held_1_5s_lands_in_far():
    sim = _flying()
    sim.hold("arms_up", "far", 1.4)
    assert sim.fsm.mode is Mode.HOVER
    sim.hold("arms_up", "far", 0.2)
    assert sim.fsm.mode is Mode.LANDING


def test_row3_ok_does_not_land_in_far_range():
    sim = _flying()
    sim.hold("arms_up", "far", 0.5)
    assert sim.fsm.mode is Mode.HOVER


# ---------------------------------------------------------------- row 4: take off on like (near) / arms_up (far)

def test_row4_like_held_1s_takes_off():
    sim = Sim()
    sim.hold("like", "near", 0.9)
    assert sim.fsm.mode is Mode.LANDED
    sim.hold("like", "near", 0.3)
    assert sim.fsm.mode is Mode.TAKING_OFF
    assert [a for _, a in sim.actions] == ["takeoff"]


def test_row4_like_held_0_9s_does_not_take_off():
    sim = Sim()
    sim.hold("like", "near", 0.9)
    sim.hold("none", "none", 0.2)
    assert sim.fsm.mode is Mode.LANDED and sim.actions == []


def test_row4_arms_up_held_1_5s_takes_off_in_far():
    sim = Sim()
    sim.hold("arms_up", "far", 1.6)
    assert sim.fsm.mode is Mode.TAKING_OFF
    assert sim.actions[0][1] == "takeoff"


def test_row4_ok_does_nothing_while_landed():
    sim = Sim()
    sim.hold("ok", "near", 3.0)
    assert sim.fsm.mode is Mode.LANDED and sim.actions == []


# ---------------------------------------------------------------- row 5: takeoff finished

def test_row5_takeoff_ends_when_altitude_reached():
    sim = Sim()
    sim.hold("like", "near", 1.2)
    assert sim.fsm.mode is Mode.TAKING_OFF
    sim.settle(0.0)
    sim.altitude = 1.5 * 0.95
    sim.frame("none", "none")
    assert sim.fsm.mode is Mode.HOVER


def test_row5_takeoff_ends_by_timeout():
    sim = Sim()
    sim.hold("like", "near", 1.2)
    sim.altitude = 0.0       # the drone never climbs
    for _ in range(int(9.0 * FPS)):
        sim.frame("none", "none")
        if sim.fsm.mode is not Mode.TAKING_OFF:
            break
    assert sim.fsm.mode is Mode.HOVER


def test_row5_takeoff_holds_while_climbing():
    sim = Sim()
    sim.hold("like", "near", 1.2)
    sim.altitude = 0.5
    sim.hold("none", "far", 1.0)             # altitude 0.5 -> 1.0, target is 1.425
    assert sim.fsm.mode is Mode.TAKING_OFF


# ---------------------------------------------------------------- row 6: landing finished

def test_row6_landing_ends_when_backend_reports_landed():
    sim = _flying()
    sim.hold("ok", "near", 1.2)
    assert sim.fsm.mode is Mode.LANDING
    sim.altitude = 0.0
    sim.frame("none", "none")
    assert sim.fsm.mode is Mode.LANDED


def test_row6_landing_continues_while_airborne():
    sim = _flying()
    sim.hold("ok", "near", 1.2)
    sim.altitude = 0.5
    sim.hold("none", "none", 0.3)
    assert sim.fsm.mode is Mode.LANDING


# ---------------------------------------------------------------- row 7: fist -> FOLLOW

def test_row7_fist_held_0_2s_enters_follow_in_near():
    sim = _flying()
    sim.hold("fist", "near", 0.1)
    assert sim.fsm.mode is Mode.HOVER
    sim.hold("fist", "near", 0.3)
    assert sim.fsm.mode is Mode.FOLLOW


def test_row7_fist_in_far_range_does_not_enter_follow():
    sim = _flying()
    sim.hold("fist", "far", 0.5)
    assert sim.fsm.mode is Mode.HOVER


# ---------------------------------------------------------------- row 8: leave FOLLOW

def test_row8_other_label_for_0_15s_leaves_follow():
    sim = _flying()
    sim.hold("fist", "near", 0.3)
    sim.hold("like", "near", 0.1)
    assert sim.fsm.mode is Mode.FOLLOW
    sim.hold("like", "near", 0.1)
    assert sim.fsm.mode is Mode.HOVER


def test_row8_range_not_near_leaves_follow_immediately():
    sim = _flying()
    sim.hold("fist", "near", 0.3)
    sim.frame("fist", "far")
    assert sim.fsm.mode is Mode.HOVER


def test_row8_fist_keeps_follow():
    sim = _flying()
    sim.hold("fist", "near", 3.0)
    assert sim.fsm.mode is Mode.FOLLOW


def test_row8_hand_lost_in_near_leaves_follow_within_0_6s():
    sim = _flying()
    sim.hold("fist", "near", 0.3)
    start = sim.t
    while sim.fsm.mode is Mode.FOLLOW and sim.t - start < 2.0:
        sim.frame("none", "near")
    assert sim.fsm.mode is Mode.HOVER
    assert sim.t - start <= 0.6                                  # M3


# ---------------------------------------------------------------- row 9: point -> POINT_MOVE

@pytest.mark.parametrize("direction", ["right", "left", "up", "down"])
def test_row9_point_held_0_5s_enters_point_move(direction):
    sim = _flying()
    sim.hold(f"point_{direction}", "far", 0.4)
    assert sim.fsm.mode is Mode.HOVER
    sim.hold(f"point_{direction}", "far", 0.4)
    assert sim.fsm.mode is Mode.POINT_MOVE
    assert sim.fsm.point_dir == direction


def test_row9_point_in_near_range_does_nothing():
    sim = _flying()
    sim.hold("point_right", "near", 1.0)
    assert sim.fsm.mode is Mode.HOVER


# ---------------------------------------------------------------- row 10: leave POINT_MOVE

def test_row10_other_direction_for_0_3s_leaves_point_move():
    sim = _flying()
    sim.hold("point_right", "far", 0.6)
    assert sim.fsm.mode is Mode.POINT_MOVE
    sim.hold("point_up", "far", 0.1)
    assert sim.fsm.mode is Mode.POINT_MOVE
    sim.hold("point_up", "far", 0.4)
    assert sim.fsm.mode is Mode.HOVER
    assert sim.fsm.point_dir is None


def test_row10_same_direction_keeps_point_move():
    sim = _flying()
    sim.hold("point_left", "far", 3.0)
    assert sim.fsm.mode is Mode.POINT_MOVE and sim.fsm.point_dir == "left"


def test_row10_short_flicker_keeps_point_move():
    sim = _flying()
    sim.hold("point_down", "far", 0.6)
    sim.hold("none", "far", 0.05)
    sim.hold("point_down", "far", 0.3)
    assert sim.fsm.mode is Mode.POINT_MOVE


# ---------------------------------------------------------------- latch

def test_arms_up_for_5s_takes_off_exactly_once():
    sim = Sim()
    sim.hold("arms_up", "far", 5.0)
    assert [a for _, a in sim.actions] == ["takeoff"]
    assert sim.fsm.mode is Mode.HOVER


def test_arms_up_held_after_takeoff_does_not_land():
    sim = Sim()
    sim.hold("arms_up", "far", 5.0)
    assert all(a != "land" for _, a in sim.actions)


def test_releasing_then_raising_again_lands():
    sim = Sim()
    sim.hold("arms_up", "far", 5.0)
    sim.hold("none", "none", 0.5)
    sim.hold("arms_up", "far", 1.6)
    assert [a for _, a in sim.actions] == ["takeoff", "land"]


def test_like_latch_blocks_repeated_takeoff_after_landing_with_like_still_held():
    sim = Sim()
    sim.hold("like", "near", 1.2)
    sim.altitude = 0.0
    sim.settle(0.0)
    sim.hold("like", "near", 2.0)
    assert [a for _, a in sim.actions] == ["takeoff"]


# ---------------------------------------------------------------- property tests

_LABELS_NEAR = ["palm", "fist", "like", "ok", "none"]
_LABELS_FAR = ["arms_up", "t_pose", "point_right", "point_left", "point_up", "point_down", "none"]


def _random_sequence(rng, frames):
    """Yield (label, range_mode) with dwell times, mixing near, far and none."""
    out = []
    while len(out) < frames:
        dwell = rng.randint(1, 60)
        kind = rng.random()
        if kind < 0.4:
            label, rng_mode = rng.choice(_LABELS_NEAR), "near"
        elif kind < 0.8:
            label, rng_mode = rng.choice(_LABELS_FAR), "far"
        else:
            label, rng_mode = "none", "none"
        out.extend([(label, rng_mode)] * dwell)
    return out[:frames]


def test_property_fsm_transition_rules_hold_on_1000_random_sequences():
    rng = random.Random(20240601)
    visited = set()
    for _ in range(1000):
        sim = Sim()
        seq = _random_sequence(rng, 300)
        prev_mode = sim.fsm.mode
        for label, range_mode in seq:
            d = sim.frame(label, range_mode)
            mode = d.mode
            visited.add(mode)
            if d.action == "takeoff":
                assert prev_mode is Mode.LANDED and mode is Mode.TAKING_OFF
            if d.action == "land":
                assert prev_mode is Mode.HOVER and mode is Mode.LANDING
            if mode is Mode.LANDED:
                assert d.action is None
            if mode in MOVING_MODES and prev_mode not in MOVING_MODES:
                assert prev_mode is Mode.HOVER          # moving modes are entered from HOVER only
            if prev_mode is Mode.LANDED and mode is not Mode.LANDED:
                assert mode is Mode.TAKING_OFF          # LANDED can only go to TAKING_OFF
            prev_mode = mode
            assert mode in FLYING_MODES or mode is Mode.LANDED


def test_property_actions_are_emitted_exactly_once_per_transition():
    rng = random.Random(7)
    for _ in range(200):
        sim = Sim()
        seq = _random_sequence(rng, 200)
        for label, range_mode in seq:
            before = sim.fsm.mode
            d = sim.frame(label, range_mode)
            if d.action is not None:
                assert d.changed and before is not d.mode
            else:
                assert not (d.changed and d.mode in (Mode.TAKING_OFF, Mode.LANDING))


def test_property_sequences_reach_every_mode():
    """Guards the property tests above against passing vacuously."""
    rng = random.Random(20240601)
    visited = set()
    for _ in range(200):
        sim = Sim()
        for label, range_mode in _random_sequence(rng, 300):
            visited.add(sim.frame(label, range_mode).mode)
    assert visited == set(Mode)
