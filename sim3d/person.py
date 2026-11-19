"""A jointed 3D person: COCO-17 body keypoints and a 21-keypoint right hand, in world metres.

World frame is the simulator's FLU frame (x forward, y left, z up). The person stands on the
ground at ``position`` facing ``facing`` (0 means facing +x, toward the drone's start point).
Body coordinates are (right, up, forward) in metres from the shoulder midpoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .hand_template import PALM_IDX, hand_points

SHOULDER_W = 0.34     # distance between the COCO shoulder keypoints, m
SHOULDER_H = 1.42     # shoulder height above the ground, m
HAND_LEN = 0.10       # wrist to middle-finger MCP, m
UPPER_ARM = 0.29
FOREARM = 0.26
REACH = 0.98 * (UPPER_ARM + FOREARM)

# head and legs, (right, up, forward) from the shoulder midpoint
_FIXED = {
    0: (0.0, 0.26, 0.06),      # nose
    1: (-0.035, 0.30, 0.04),   # left eye
    2: (0.035, 0.30, 0.04),
    3: (-0.08, 0.27, -0.02),   # left ear
    4: (0.08, 0.27, -0.02),
    11: (-0.12, -0.50, 0.0),   # left hip
    12: (0.12, -0.50, 0.0),
    13: (-0.12, -0.92, 0.02),  # knees
    14: (0.12, -0.92, 0.02),
    15: (-0.12, -1.36, 0.0),   # ankles
    16: (0.12, -1.36, 0.0),
}
_POINT_DEG = {"right": 0.0, "up": 90.0, "left": 180.0, "down": -90.0}


@dataclass
class PersonPose:
    """What the person is doing at one instant.

    left_arm / right_arm: "down", "up" (raised above the head), "t" (straight out sideways),
    "point_<right|left|up|down>", or "reach" (wrist placed so that the palm centre is at
    ``palm``). The hand is only drawn on the right arm, and only when show_hand is True.
    roll_deg turns the hand clockwise as the user sees it.
    """
    left_arm: str = "down"
    right_arm: str = "reach"
    palm: tuple = (0.10, -0.10, 0.34)
    gesture: str = "palm"
    roll_deg: float = 0.0
    show_hand: bool = True


def two_bone_ik(shoulder, target, side: float):
    """Elbow and wrist for an arm reaching toward target (body coordinates).

    The elbow bends down and outward. A target out of reach is pulled back onto the sphere
    the arm can reach, so the returned wrist may differ from target.
    """
    s = np.asarray(shoulder, dtype=float)
    d = np.asarray(target, dtype=float) - s
    c = float(np.linalg.norm(d))
    if c < 1e-6:
        d, c = np.array([0.0, -1.0, 0.0]), 1e-6
    e = d / c
    c = min(max(c, abs(UPPER_ARM - FOREARM) + 1e-3), REACH)
    wrist = s + e * c
    cos_a = (UPPER_ARM ** 2 + c ** 2 - FOREARM ** 2) / (2.0 * UPPER_ARM * c)
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    pole = np.array([0.5 * side, -1.0, -0.2])
    n = pole - np.dot(pole, e) * e
    if np.linalg.norm(n) < 1e-6:
        n = np.array([0.0, 0.0, -1.0]) - np.dot([0.0, 0.0, -1.0], e) * e
    n = n / np.linalg.norm(n)
    elbow = s + UPPER_ARM * (math.cos(a) * e + math.sin(a) * n)
    return elbow, wrist


def _arm_template(side: float, kind: str):
    sh = np.array([0.5 * SHOULDER_W * side, 0.0, 0.0])
    if kind == "down":
        return sh + [0.04 * side, -0.28, 0.03], sh + [0.06 * side, -0.53, 0.06]
    if kind == "up":
        return sh + [0.08 * side, 0.27, 0.0], sh + [0.13 * side, 0.52, 0.0]
    if kind == "t":
        return sh + [UPPER_ARM * side, 0.0, 0.0], sh + [(UPPER_ARM + FOREARM) * side, 0.0, 0.0]
    if kind.startswith("point_"):
        deg = _POINT_DEG[kind[len("point_"):]]
        d = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0])
        return sh + UPPER_ARM * d, sh + (UPPER_ARM + FOREARM) * d
    raise ValueError(f"unknown arm pose: {kind!r}")


def hand_body(gesture: str, roll_deg: float, palm_target):
    """Right-hand keypoints (21, 3) in body coordinates with the palm centre at palm_target.

    The hand is flat, palm toward the camera. Returns (keypoints, wrist).
    """
    local = hand_points(gesture)
    centre = local[PALM_IDX].mean(axis=0)
    r = math.radians(roll_deg)
    c, s = math.cos(r), math.sin(r)
    hr = np.array([c, -s, 0.0])   # template x axis: clockwise roll moves it down
    hu = np.array([s, c, 0.0])    # template y axis: clockwise roll moves it right
    pts = HAND_LEN * ((local[:, :1] - centre[0]) * hr + (local[:, 1:] - centre[1]) * hu)
    pts = pts + np.asarray(palm_target, dtype=float)
    return pts, pts[0].copy()


class Person:
    def __init__(self, position=(-1.6, 0.0, 0.0), facing: float = 0.0):
        self.position = np.asarray(position, dtype=float)
        self.facing = float(facing)

    # axes of the body frame in the world
    @property
    def axes(self):
        f = np.array([math.cos(self.facing), math.sin(self.facing), 0.0])
        r = np.array([math.sin(self.facing), -math.cos(self.facing), 0.0])
        return r, np.array([0.0, 0.0, 1.0]), f

    @property
    def shoulder_mid(self) -> np.ndarray:
        return self.position + np.array([0.0, 0.0, SHOULDER_H])

    def to_world(self, body_pts) -> np.ndarray:
        r, u, f = self.axes
        b = np.asarray(body_pts, dtype=float).reshape(-1, 3)
        return self.shoulder_mid + b[:, :1] * r + b[:, 1:2] * u + b[:, 2:3] * f

    def body_keypoints(self, pose: PersonPose):
        """(body17, hand21 or None) in body coordinates."""
        kp = np.zeros((17, 3))
        for i, v in _FIXED.items():
            kp[i] = v
        kp[5] = (-0.5 * SHOULDER_W, 0.0, 0.0)
        kp[6] = (0.5 * SHOULDER_W, 0.0, 0.0)
        hand = None
        for side, kind, (ie, iw) in ((-1.0, pose.left_arm, (7, 9)), (1.0, pose.right_arm, (8, 10))):
            if kind == "reach":
                if side < 0:
                    raise ValueError("only the right arm can reach")
                pts, wrist = hand_body(pose.gesture, pose.roll_deg, pose.palm)
                elbow, wrist_ik = two_bone_ik(kp[6], wrist, side)
                pts = pts + (wrist_ik - wrist)       # out of reach: the hand follows the wrist
                kp[ie], kp[iw] = elbow, wrist_ik
                hand = pts
            else:
                kp[ie], kp[iw] = _arm_template(side, kind)
        if not pose.show_hand:
            hand = None
        return kp, hand

    def keypoints(self, pose: PersonPose):
        """(body17, hand21 or None) in world metres."""
        body, hand = self.body_keypoints(pose)
        return self.to_world(body), (self.to_world(hand) if hand is not None else None)
