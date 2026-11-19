"""Synthetic keypoints for the tests.

Body keypoints are built in body units: x points to the user's right, y points up, and
1 unit is the shoulder width. Hand keypoints are built in hand units: the wrist is at the
origin and the middle-finger MCP is at (0, 1). ``Scene`` maps both into pixels of a raw
(not mirrored) camera facing the user: u points right, v points down, and the user's right
is on the image left. Tilt is a clockwise roll on screen, in radians.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dronectl.types import BodyObs, HandObs, Observation


def rot(angle: float) -> np.ndarray:
    """2x2 rotation matrix. Positive angles turn clockwise on screen (v points down)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


@dataclass
class Scene:
    shoulder_px: float = 160.0
    center: tuple = (640.0, 360.0)   # shoulder midpoint in pixels
    tilt: float = 0.0                # body roll, radians, clockwise on screen
    frame_wh: tuple = (1280, 720)

    def body_to_px(self, p) -> np.ndarray:
        v = np.array([-p[0], -p[1]], dtype=float) * self.shoulder_px   # user's right -> image left
        return rot(self.tilt) @ v + np.asarray(self.center, dtype=float)


# ---------------------------------------------------------------- body

_POINT_DEG = {"right": 0.0, "up": 90.0, "left": 180.0, "down": -90.0}


def arm_points(side: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Elbow and wrist in body units. side = -1 for the user's left arm, +1 for the user's right arm.

    kind: "down", "up" (arm raised above the head), "t" (arm straight out sideways),
    or "point_<dir>" / "point_<degrees>" (arm straight in that direction, body frame).
    """
    shoulder = np.array([0.5 * side, 0.0])
    if kind == "down":
        return np.array([0.6 * side, -0.6]), np.array([0.7 * side, -1.3])
    if kind == "up":
        elbow = shoulder + np.array([0.25 * side, 0.9])
        return elbow, elbow + np.array([0.2 * side, 0.95])
    if kind == "t":
        return np.array([1.25 * side, 0.0]), np.array([2.0 * side, 0.0])
    if kind.startswith("point_"):
        name = kind[len("point_"):]
        deg = _POINT_DEG[name] if name in _POINT_DEG else float(name)
        d = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        return shoulder + 0.8 * d, shoulder + 1.6 * d
    raise ValueError(f"unknown arm kind: {kind!r}")


@dataclass
class BodyPose:
    left: str = "down"      # the user's left arm (COCO 5, 7, 9), seen on the image right
    right: str = "down"     # the user's right arm (COCO 6, 8, 10), seen on the image left
    conf: float = 0.9
    drop: tuple = ()        # COCO indices whose confidence is set to 0


def body_points(pose: BodyPose) -> np.ndarray:
    """COCO-17 keypoints in body units."""
    kp = np.zeros((17, 2))
    kp[0] = (0.0, 0.8)                       # nose
    kp[1] = (-0.1, 0.85)
    kp[2] = (0.1, 0.85)
    kp[3] = (-0.3, 0.75)
    kp[4] = (0.3, 0.75)
    kp[5] = (-0.5, 0.0)                      # shoulders
    kp[6] = (0.5, 0.0)
    kp[7], kp[9] = arm_points(-1, pose.left)
    kp[8], kp[10] = arm_points(1, pose.right)
    kp[11] = (-0.35, -1.2)                   # hips
    kp[12] = (0.35, -1.2)
    kp[13] = (-0.35, -2.2)                   # knees
    kp[14] = (0.35, -2.2)
    kp[15] = (-0.35, -3.2)                   # ankles
    kp[16] = (0.35, -3.2)
    return kp


# ---------------------------------------------------------------- hand

# gesture -> ((index, middle, ring, pinky) open?, thumb open? or "touch")
GESTURE_SHAPES = {
    "palm": ((True, True, True, True), True),
    "four": ((True, True, True, True), False),
    "fist": ((False, False, False, False), False),
    "like": ((False, False, False, False), True),
    "ok": ((True, True, True, True), "touch"),
    "one": ((True, False, False, False), False),
    "peace_inverted": ((True, True, False, False), False),
    "three": ((True, True, True, False), False),
}
GESTURES = tuple(GESTURE_SHAPES)
_PALM_IDX = [0, 5, 9, 13, 17]


def hand_points(gesture: str) -> np.ndarray:
    """21 hand keypoints in hand units, y up, wrist at the origin."""
    if gesture not in GESTURE_SHAPES:
        raise ValueError(f"unknown gesture: {gesture!r}")
    fingers, thumb = GESTURE_SHAPES[gesture]
    kp = np.zeros((21, 2))
    mcp = {5: (-0.42, 0.92), 9: (0.0, 1.0), 13: (0.30, 0.94), 17: (0.62, 0.80)}
    for base, is_open in zip((5, 9, 13, 17), fingers):
        mx, my = mcp[base]
        kp[base] = (mx, my)
        if is_open:
            kp[base + 1] = (mx, my + 0.5)
            kp[base + 2] = (mx, my + 0.85)
            kp[base + 3] = (mx, my + 1.15)
        else:  # curled: tip folds back toward the palm
            kp[base + 1] = (mx, my + 0.45)
            kp[base + 2] = (mx, my + 0.3)
            kp[base + 3] = (mx, my + 0.1)
    kp[1] = (-0.35, 0.05)
    kp[2] = (-0.7, 0.45)
    if thumb is True:
        kp[3], kp[4] = (-0.9, 0.75), (-1.12, 1.02)
    elif thumb is False:
        kp[3], kp[4] = (-0.45, 0.35), (-0.2, 0.3)
    else:  # "touch": thumb tip meets the index fingertip
        kp[3] = (-0.6, 1.3)
        kp[4] = kp[8] + np.array([0.06, 0.04])
    return kp


def hand_px(gesture: str, target_px, length_px: float, angle: float) -> np.ndarray:
    """Hand keypoints in pixels. The palm centre lands on target_px; angle is clockwise."""
    local = hand_points(gesture) * np.array([1.0, -1.0])     # up -> image v axis
    centre = local[_PALM_IDX].mean(axis=0)
    return (local - centre) @ rot(angle).T * length_px + np.asarray(target_px, dtype=float)


def hand_body(hand: "HandPose") -> np.ndarray:
    """Hand keypoints in body units. hand.rot turns the hand clockwise as the user sees it."""
    local = hand_points(hand.gesture)
    centre = local[_PALM_IDX].mean(axis=0)
    c, s = math.cos(hand.rot), math.sin(hand.rot)
    cw = np.array([[c, s], [-s, c]])            # clockwise in y-up coordinates
    return (local - centre) @ cw.T * hand.ratio + np.asarray(hand.palm, dtype=float)


@dataclass
class HandPose:
    gesture: str = "palm"
    palm: tuple = (0.0, -0.6)   # palm centre in body units (shoulder midpoint = origin)
    ratio: float = 0.45         # hand length / shoulder width
    rot: float = 0.0            # clockwise rotation as the user sees it, radians
    conf: float = 0.9


# ---------------------------------------------------------------- observation

def _box(pts: np.ndarray) -> tuple:
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def make_observation(t: float, scene: Scene, body: BodyPose | None = None,
                     hand: HandPose | None = None, noise_px: float = 0.0,
                     rng: np.random.Generator | None = None,
                     latency_ms: float = 0.0) -> Observation:
    """Build an Observation. Pass body=None for no person and hand=None for no hand."""
    rng = rng if rng is not None else np.random.default_rng(0)
    body_obs = None
    if body is not None:
        pts = np.array([scene.body_to_px(p) for p in body_points(body)])
        if noise_px > 0:
            pts = pts + rng.normal(0.0, noise_px, pts.shape)
        conf = np.full(17, body.conf)
        conf[list(body.drop)] = 0.0
        body_obs = BodyObs(kps=pts, kps_conf=conf, box=_box(pts))
    hand_obs = None
    if hand is not None:
        pts = np.array([scene.body_to_px(p) for p in hand_body(hand)])
        if noise_px > 0:
            pts = pts + rng.normal(0.0, noise_px, pts.shape)
        hand_obs = HandObs(kps=pts, kps_conf=np.full(21, hand.conf), box=_box(pts), conf=hand.conf)
    return Observation(t=t, frame_wh=scene.frame_wh, hand=hand_obs, body=body_obs,
                       latency_ms=latency_ms)
