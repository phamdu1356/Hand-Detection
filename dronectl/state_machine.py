"""Flight state machine (plan section 6.2).

Rows are checked in table order and the first matching row fires, so one call makes at
most one transition. Row numbers in the comments refer to table 6.2.

Invariants (tested):
- cmd != 0 only in FOLLOW and POINT_MOVE. The Controller enforces this from the mode.
- LANDED never emits action "land" or a velocity command.
- action "takeoff" only leaves LANDED, and action "land" only leaves HOVER.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ControlConfig
from .types import FLYING_MODES, Mode


@dataclass
class Decision:
    mode: Mode
    action: str | None   # "takeoff" | "land" | None
    changed: bool        # True when the mode changed on this call
    reason: str


class StateMachine:
    def __init__(self, cfg: ControlConfig | None = None):
        self.cfg = cfg or ControlConfig()
        self.reset()

    def reset(self):
        self.mode = Mode.LANDED
        self.point_dir: str | None = None   # "right" | "left" | "up" | "down" while POINT_MOVE
        self.entered_t = 0.0
        self._follow_off_since: float | None = None   # FOLLOW without a usable near hand

    def step(self, t: float, range_mode: str, flt, none_s: float,
             altitude: float, landed: bool) -> Decision:
        """Advance one frame.

        flt is a gestures.GestureFilter. none_s is how long range_mode has been "none".
        altitude and landed come from the backend DroneState.
        """
        c = self.cfg
        near = range_mode == "near"
        far = range_mode == "far"
        mode = self.mode
        prev = mode

        def go(new: Mode, action: str | None = None, reason: str = "") -> Decision:
            self.mode = new
            self.entered_t = t
            self._follow_off_since = None
            if new is not Mode.POINT_MOVE:
                self.point_dir = None
            return Decision(new, action, new is not prev, reason)

        # 1, 2: stop from any flying mode
        if mode in FLYING_MODES:
            if (near and flt.stable() == "palm") or (far and flt.held("t_pose") >= c.t_pose_stop_s):
                return go(Mode.HOVER, reason="stop: palm or t-pose")
            if none_s >= c.range_none_stop_s:
                return go(Mode.HOVER, reason="stop: no person or hand")

        # 3: land on ok (near) or both arms up (far)
        if mode is Mode.HOVER:
            if near and flt.held("ok") >= c.land_hold_near_s:
                flt.latch("ok")
                return go(Mode.LANDING, "land", "ok held")
            if far and flt.held("arms_up") >= c.land_hold_far_s:
                flt.latch("arms_up")
                return go(Mode.LANDING, "land", "arms_up held")

        # 4: take off on like (near) or both arms up (far)
        if mode is Mode.LANDED:
            if near and flt.held("like") >= c.takeoff_hold_near_s:
                flt.latch("like")
                return go(Mode.TAKING_OFF, "takeoff", "like held")
            if far and flt.held("arms_up") >= c.takeoff_hold_far_s:
                flt.latch("arms_up")
                return go(Mode.TAKING_OFF, "takeoff", "arms_up held")

        # 5: takeoff finished (altitude reached, or timed out)
        if mode is Mode.TAKING_OFF:
            if altitude >= c.takeoff_ratio * c.takeoff_altitude:
                return go(Mode.HOVER, reason="takeoff: altitude reached")
            if t - self.entered_t >= c.takeoff_timeout_s:
                return go(Mode.HOVER, reason="takeoff: timeout")

        # 6: landing finished
        if mode is Mode.LANDING and landed:
            return go(Mode.LANDED, reason="landed")

        if mode is Mode.HOVER:
            # 7: fist held -> FOLLOW (near only)
            if near and flt.held("fist") >= c.follow_hold_s:
                return go(Mode.FOLLOW, reason="fist held: follow")
            # 9: point held -> POINT_MOVE (far only)
            if far and flt.label.startswith("point_") and flt.held(flt.label) >= c.point_hold_s:
                direction = flt.label[len("point_"):]
                go(Mode.POINT_MOVE, reason=f"point {direction} held")
                self.point_dir = direction
                return Decision(Mode.POINT_MOVE, None, True, f"point {direction} held")

        if mode is Mode.FOLLOW:
            # 8: leave FOLLOW when the hand is gone (or range not near) for follow_lost_s, or
            # on another label held follow_exit_s. "like" counts as fist here: with the thumb
            # half out the rules read a fist as like, and like means nothing while flying.
            if not near or flt.label == "none":
                if self._follow_off_since is None:
                    self._follow_off_since = t
                if t - self._follow_off_since >= c.follow_lost_s:
                    return go(Mode.HOVER, reason="hand lost")
            else:
                self._follow_off_since = None
                if flt.label not in ("fist", "like") and flt.held_s >= c.follow_exit_s:
                    return go(Mode.HOVER, reason="fist released")

        if mode is Mode.POINT_MOVE:
            # 10: leave POINT_MOVE when the label is not the same direction for a while
            if flt.label != f"point_{self.point_dir}" and flt.held_s >= c.point_exit_s:
                return go(Mode.HOVER, reason="point released")

        return Decision(self.mode, None, False, "")
