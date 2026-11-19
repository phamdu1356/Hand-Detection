"""Flat 21-keypoint hand templates in hand units: wrist at the origin, middle MCP at (0, 1).

x points to the user's right, y up, for a right hand with the palm facing the camera.
"""
from __future__ import annotations

import numpy as np

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
PALM_IDX = [0, 5, 9, 13, 17]


def hand_points(gesture: str) -> np.ndarray:
    """21 hand keypoints in hand units, y up, wrist at the origin."""
    if gesture not in GESTURE_SHAPES:
        raise ValueError(f"unknown gesture: {gesture!r}")
    fingers, thumb = GESTURE_SHAPES[gesture]
    kp = np.zeros((21, 2))
    # MCP span 0.81 of the wrist-to-middle-MCP length, close to real hands (median 0.73 in
    # the 2D val labels, which includes foreshortened ones)
    mcp = {5: (-0.32, 0.92), 9: (0.0, 1.0), 13: (0.24, 0.94), 17: (0.48, 0.80)}
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
