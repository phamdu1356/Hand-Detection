"""Numpy-only copies of the hand rules in demo/test_model.py.

They are copied rather than imported so that dronectl does not pull in OpenCV or the demo
package. demo/test_model.py is not changed in this phase.
"""
from __future__ import annotations

import math

import numpy as np


def _is_valid_kp(pt):
    return (
        pt is not None
        and len(pt) >= 2
        and math.isfinite(float(pt[0]))
        and math.isfinite(float(pt[1]))
        and (float(pt[0]) > 0.0 or float(pt[1]) > 0.0)
    )


def analyze_hand_keypoints(kps):
    """From 21 (x, y) keypoints, extract the static gesture, the index pointing angle and the palm bone length."""
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape[0] < 21 or not _is_valid_kp(kps[0]) or not _is_valid_kp(kps[9]):
        return "UNKNOWN", 0.0, 0.0

    wrist = kps[0]
    middle_mcp = kps[9]
    palm_bone_len = float(np.linalg.norm(middle_mcp - wrist))
    if palm_bone_len < 1e-3:
        return "UNKNOWN", 0.0, 0.0

    # Open check for the four long fingers: index (8,6), middle (12,10), ring (16,14), pinky (20,18)
    fingers = []
    for tip_idx, pip_idx in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if _is_valid_kp(kps[tip_idx]) and _is_valid_kp(kps[pip_idx]):
            is_open = np.linalg.norm(kps[tip_idx] - wrist) > np.linalg.norm(kps[pip_idx] - wrist) * 1.15
            fingers.append(bool(is_open))
        else:
            fingers.append(False)

    if _is_valid_kp(kps[4]) and _is_valid_kp(kps[2]) and _is_valid_kp(kps[17]):
        thumb_open = bool(np.linalg.norm(kps[4] - kps[17]) > np.linalg.norm(kps[2] - kps[17]) * 1.18)
    else:
        thumb_open = False

    if _is_valid_kp(kps[4]) and _is_valid_kp(kps[8]):
        thumb_index_dist = float(np.linalg.norm(kps[4] - kps[8])) / palm_bone_len
    else:
        thumb_index_dist = 999.0

    # Pointing angle of the index finger (kp5 -> kp8) in the camera plane (-180..180 degrees)
    if _is_valid_kp(kps[5]) and _is_valid_kp(kps[8]):
        dx, dy = float(kps[8][0] - kps[5][0]), float(kps[5][1] - kps[8][1])
        point_angle = math.degrees(math.atan2(dy, dx))
    else:
        point_angle = 0.0

    count_4 = sum(fingers)
    if thumb_index_dist < 0.42 and sum(fingers[1:]) >= 2:
        gesture = "ok"
    elif count_4 == 0 and not thumb_open:
        gesture = "fist"
    elif count_4 == 0 and thumb_open:
        gesture = "like"
    elif fingers[0] and count_4 == 1:
        gesture = "one"
    elif fingers[0] and fingers[1] and count_4 == 2:
        gesture = "peace_inverted"
    elif count_4 == 3:
        gesture = "three"
    elif count_4 == 4 and not thumb_open:
        gesture = "four"
    elif count_4 == 4 and thumb_open:
        gesture = "palm"
    else:
        gesture = "palm" if count_4 >= 3 else "fist"

    return gesture, point_angle, palm_bone_len
