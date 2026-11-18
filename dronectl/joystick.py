"""Virtual joystick for the near range (plan section 5).

The hand is measured relative to an origin stored when FOLLOW starts: position in the
shoulder frame, hand-size ratio (forward/back) and hand bearing (yaw). The error on each
axis is shaped by a dead zone and an expo curve, then smoothed (EMA) and rate limited.
Units: m/s for translation, rad/s for yaw. Command order: forward, right, up, yaw_rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .body_frame import Frame, hand_angle, hand_size, palm_center, to_body
from .config import ControlConfig
from .types import VelocityCommand

FORWARD, RIGHT, UP, YAW = 0, 1, 2, 3


@dataclass
class Origin:
    kind: str       # frame kind the origin was measured in
    x0: float       # body x of the palm centre
    y0: float       # body y of the palm centre
    r0: float       # hand size / frame scale
    th0: float      # hand bearing, radians
    hs0: float      # hand size in pixels


def shape(e: float, dead: float, alpha: float) -> float:
    """Dead zone followed by an expo curve. Odd, monotone, maps |e| = 1 to 1."""
    a = abs(e)
    if a < dead:
        return 0.0
    s = min(1.0, (a - dead) / (1.0 - dead))
    return math.copysign(alpha * s ** 3 + (1.0 - alpha) * s, e)


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rate_limit(prev: float, target: float, dt: float, accel: float, decel: float) -> float:
    """Move prev toward target. Growing magnitude is limited by accel, shrinking by decel."""
    away = abs(target) > abs(prev) and (prev == 0.0 or target * prev > 0.0)
    rate = accel if away else decel
    step = max(-rate * dt, min(rate * dt, target - prev))
    return prev + step


class Joystick:
    def __init__(self, cfg: ControlConfig | None = None):
        self.cfg = cfg or ControlConfig()
        self.origin: Origin | None = None
        self._ema = np.zeros(4)
        self._out = np.zeros(4)

    def reset(self):
        self.origin = None
        self._ema = np.zeros(4)
        self._out = np.zeros(4)

    def anchor(self, frame: Frame, hand_kps) -> bool:
        """Store the origin from the current hand. Returns False when the hand cannot be measured."""
        palm = palm_center(hand_kps)
        size = hand_size(hand_kps)
        ang = hand_angle(hand_kps, frame)
        if palm is None or size is None or ang is None:
            return False
        b = to_body(frame, palm)
        self.origin = Origin(kind=frame.kind, x0=float(b[0]), y0=float(b[1]),
                             r0=size / frame.scale, th0=float(ang), hs0=size)
        return True

    def raw_error(self, frame: Frame | None, hand_kps):
        """Normalised error (ex, ey, ez, eyaw) against the origin, or None if not measurable."""
        o = self.origin
        if o is None or frame is None or frame.kind != o.kind:
            return None
        palm = palm_center(hand_kps)
        size = hand_size(hand_kps)
        ang = hand_angle(hand_kps, frame)
        if palm is None or size is None or ang is None:
            return None
        b = to_body(frame, palm)
        c = self.cfg
        ex = (b[0] - o.x0) / c.full_xy
        ey = (b[1] - o.y0) / c.full_xy
        ez = math.log((size / frame.scale) / o.r0) / math.log(c.full_z_ratio)
        eyaw = wrap_angle(ang - o.th0) / math.radians(c.full_yaw_deg)
        return ex, ey, ez, eyaw

    def targets(self, err) -> np.ndarray:
        """Shaped velocity targets (forward, right, up, yaw_rate) for a normalised error."""
        ex, ey, ez, eyaw = err
        c = self.cfg
        dz_xy = c.dead_xy / c.full_xy
        dz_yaw = c.dead_yaw_deg / c.full_yaw_deg
        fwd = shape(ez, c.dead_z, c.expo_alpha) * c.v_h
        right = shape(ex, dz_xy, c.expo_alpha) * c.v_h
        up = shape(ey, dz_xy, c.expo_alpha) * c.v_v
        yaw = -shape(eyaw, dz_yaw, c.expo_alpha) * math.radians(c.w_yaw_deg)
        return np.array([fwd, right, up, yaw])

    def point_target(self, direction: str) -> np.ndarray:
        """Constant far-range target for POINT_MOVE, in the same units as targets()."""
        v = self.cfg.v_far
        t = np.zeros(4)
        if direction == "right":
            t[RIGHT] = v
        elif direction == "left":
            t[RIGHT] = -v
        elif direction == "up":
            t[UP] = v
        elif direction == "down":
            t[UP] = -v
        else:
            raise ValueError(f"unknown direction: {direction!r}")
        return t

    def step(self, target, dt: float) -> VelocityCommand:
        """EMA, acceleration limits, then the horizontal speed cap. Call once per frame."""
        c = self.cfg
        dt = min(max(float(dt), 1e-3), 0.2)
        target = np.asarray(target, dtype=float)
        self._ema = self._ema + c.ema_alpha * (target - self._ema)
        accel = (c.acc_h, c.acc_h, c.acc_v, math.radians(c.acc_yaw_deg))
        decel = (c.dec_lin, c.dec_lin, c.dec_lin, math.radians(c.dec_yaw_deg))
        out = self._out.copy()
        for i in range(4):
            out[i] = rate_limit(out[i], self._ema[i], dt, accel[i], decel[i])
        h = math.hypot(out[FORWARD], out[RIGHT])
        if h > c.v_h:
            out[FORWARD] *= c.v_h / h
            out[RIGHT] *= c.v_h / h
        self._out = out
        return VelocityCommand(forward=float(out[FORWARD]), right=float(out[RIGHT]),
                               up=float(out[UP]), yaw_rate=float(out[YAW]))
