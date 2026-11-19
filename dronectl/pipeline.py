"""Controller: one call per frame turns an Observation and the drone state into a ControlOutput.

Order per frame (plan 6.3): body frame, range choice, gesture labels and filter, state
machine, joystick for FOLLOW or POINT_MOVE, then the ControlOutput.
"""
from __future__ import annotations

import numpy as np

from .backends.base import DroneState
from .body_frame import Frame, fallback_frame, hand_size, palm_center, shoulder_frame, to_body
from .config import ControlConfig
from .gestures import GestureFilter, far_gesture, near_gesture
from .joystick import Joystick
from .state_machine import StateMachine
from .types import ControlOutput, HandObs, Mode, Observation, VelocityCommand

FIRST_DT = 1.0 / 30.0
_MIN_DT, _MAX_DT = 1e-3, 0.2


def choose_range(cfg: ControlConfig, hand_ok: bool, body_frame: Frame | None,
                 hand_px: float | None, prev: str, lock_near: bool = False) -> str:
    """near / far / none (plan 6.1). Between far_enter_q and near_enter_q the previous range is kept.

    lock_near: FOLLOW is active. A usable hand then stays near whatever its size, because a
    hand pulled back to fly backward shrinks toward far_enter_q.
    """
    if hand_ok:
        if body_frame is None or lock_near:
            return "near"                  # no shoulders: too close, use fallback
        q = hand_px / body_frame.scale
        if q >= cfg.near_enter_q:
            return "near"
        if q < cfg.far_enter_q:
            return "far"
        return prev if prev in ("near", "far") else "far"
    if body_frame is not None:
        return "far"
    return "none"


def _hand_usable(hand: HandObs | None, cfg: ControlConfig) -> bool:
    return (hand is not None and float(np.mean(hand.kps_conf)) >= cfg.hand_conf_min
            and hand_size(hand.kps) is not None)


class Controller:
    def __init__(self, cfg: ControlConfig | None = None):
        self.cfg = cfg or ControlConfig()
        self.reset()

    def reset(self):
        self.fsm = StateMachine(self.cfg)
        self.flt = GestureFilter(self.cfg)
        self.joy = Joystick(self.cfg)
        self._range = "none"
        self._none_since: float | None = None
        self._last_t: float | None = None

    @property
    def mode(self) -> Mode:
        return self.fsm.mode

    def update(self, obs: Observation, drone: DroneState | None = None) -> ControlOutput:
        cfg = self.cfg
        drone = drone or DroneState()
        t = float(obs.t)
        dt = FIRST_DT if self._last_t is None else min(max(t - self._last_t, _MIN_DT), _MAX_DT)
        self._last_t = t

        hand = obs.hand if _hand_usable(obs.hand, cfg) else None
        hand_px = hand_size(hand.kps) if hand is not None else None
        body_frame = shoulder_frame(obs.body, cfg)
        frame = body_frame
        if frame is None and hand is not None:
            anchored = self.joy.origin is not None and self.joy.origin.kind == "hand_fallback"
            unit_px = self.joy.origin.hs0 if anchored else hand_px
            frame = fallback_frame(cfg.fallback_k * unit_px)

        range_mode = choose_range(cfg, hand is not None, body_frame, hand_px, self._range,
                                  lock_near=self.fsm.mode is Mode.FOLLOW)
        self._range = range_mode

        if range_mode == "near":
            label = near_gesture(hand, cfg)
        elif range_mode == "far":
            label = far_gesture(obs.body, body_frame, cfg)
        else:
            label = "none"
        self.flt.update(t, label)

        if range_mode == "none":
            if self._none_since is None:
                self._none_since = t
            none_s = t - self._none_since
        else:
            self._none_since = None
            none_s = 0.0

        prev_mode = self.fsm.mode
        dec = self.fsm.step(t, range_mode, self.flt, none_s, drone.altitude, drone.landed)
        mode = dec.mode

        cmd, dbg = VelocityCommand(), {}
        if mode is Mode.FOLLOW:
            cmd, dbg = self._follow(prev_mode, frame, hand, dt)
        elif mode is Mode.POINT_MOVE:
            if prev_mode is not Mode.POINT_MOVE:
                self.joy.reset()
            cmd = self.joy.step(self.joy.point_target(self.fsm.point_dir), dt)
            dbg["point"] = self.fsm.point_dir
        else:
            if prev_mode is Mode.FOLLOW or prev_mode is Mode.POINT_MOVE or self.joy.origin is not None:
                self.joy.reset()

        debug = {
            "frame": frame.kind if frame is not None else None,
            "label": label,
            "stable": self.flt.stable(),
            "held_s": self.flt.held_s,
            "reason": dec.reason,
        }
        debug.update(dbg)
        if hand is not None and frame is not None:
            palm = palm_center(hand.kps)
            if palm is not None:
                bxy = to_body(frame, palm)
                debug["hx"], debug["hy"] = float(bxy[0]), float(bxy[1])
            debug["d"] = hand_px / frame.scale
        return ControlOutput(mode=mode, cmd=cmd, action=dec.action,
                             range_mode=range_mode, debug=debug)

    def _follow(self, prev_mode: Mode, frame: Frame | None, hand: HandObs | None, dt: float):
        """Joystick step in FOLLOW. Re-anchors on entry, and when the frame kind changes."""
        dbg: dict = {}
        if prev_mode is not Mode.FOLLOW:
            self.joy.reset()
            if frame is not None and hand is not None:
                self.joy.anchor(frame, hand.kps)
            dbg["anchored"] = self.joy.origin is not None
            return self.joy.step(np.zeros(4), dt), dbg
        if (frame is not None and self.joy.origin is not None
                and frame.kind != self.joy.origin.kind):
            self.joy.reset()
            if hand is not None:
                self.joy.anchor(frame, hand.kps)
            dbg["reanchored"] = True
            return self.joy.step(np.zeros(4), dt), dbg
        err = self.joy.raw_error(frame, hand.kps) if hand is not None else None
        if err is None:
            # hand lost for a moment: keep the last target, the state machine ends FOLLOW
            # after follow_lost_s
            dbg["held"] = True
            return self.joy.step(self.joy.last_target, dt), dbg
        err = self.joy.smooth_error(err)
        dbg["err"] = tuple(float(v) for v in err)
        return self.joy.step(self.joy.targets(err), dt), dbg
