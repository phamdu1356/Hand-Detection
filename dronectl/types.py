"""Data types shared by the control layer (plan section 2.3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class HandObs:
    kps: np.ndarray        # (21, 2) pixel coordinates
    kps_conf: np.ndarray   # (21,)
    box: tuple             # x, y, w, h in pixels
    conf: float
    gesture_raw: str = "none"


@dataclass
class BodyObs:
    kps: np.ndarray        # (17, 2) COCO keypoints in pixels
    kps_conf: np.ndarray   # (17,)
    box: tuple


@dataclass
class Observation:
    t: float               # seconds, time.monotonic()
    frame_wh: tuple
    hand: HandObs | None
    body: BodyObs | None
    latency_ms: float = 0.0


class Mode(Enum):
    LANDED = "LANDED"
    TAKING_OFF = "TAKING_OFF"
    HOVER = "HOVER"
    FOLLOW = "FOLLOW"
    POINT_MOVE = "POINT_MOVE"
    LANDING = "LANDING"


# Modes in which the drone is airborne ("đang bay" in table 6.2).
FLYING_MODES = frozenset({Mode.TAKING_OFF, Mode.HOVER, Mode.FOLLOW, Mode.POINT_MOVE, Mode.LANDING})
# The only modes in which a non-zero velocity command may be sent (invariant M1).
MOVING_MODES = frozenset({Mode.FOLLOW, Mode.POINT_MOVE})


@dataclass
class VelocityCommand:
    """Headless user frame: forward away from the user, right of the user, up, yaw left positive.

    Units: m/s for translation, rad/s for yaw_rate.
    """
    forward: float = 0.0
    right: float = 0.0
    up: float = 0.0
    yaw_rate: float = 0.0

    def is_zero(self) -> bool:
        return self.forward == 0.0 and self.right == 0.0 and self.up == 0.0 and self.yaw_rate == 0.0


@dataclass
class ControlOutput:
    mode: Mode
    cmd: VelocityCommand
    action: str | None     # "takeoff" | "land" | None; emitted once per state change
    range_mode: str        # "near" | "far" | "none"
    debug: dict = field(default_factory=dict)
