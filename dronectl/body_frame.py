"""Shoulder-referenced body frame and hand measurements (plan section 3).

Conventions: image pixels have u to the right and v down, as a webcam delivers them (not
mirrored). The user faces the camera, so their right hand side is on the image left and the
COCO left shoulder (index 5) is on the image right. A Frame maps a pixel p to body
coordinates (x, y), where x points to the user's right, y points up, and one unit is the
scale S of the frame (shoulder width in pixels, or a fallback unit).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import ControlConfig
from .hand_rules import _is_valid_kp

PALM_IDX = (0, 5, 9, 13, 17)
WRIST, MIDDLE_MCP = 0, 9
SHOULDER_L, SHOULDER_R = 5, 6  # COCO indices


@dataclass(eq=False)
class Frame:
    kind: str           # "shoulder" or "hand_fallback"
    origin: np.ndarray  # pixel position of the origin, shape (2,)
    ex: np.ndarray      # unit vector along body x (user's right = image left), shape (2,)
    ey: np.ndarray      # unit vector along body y (up), shape (2,)
    scale: float        # pixels per body unit


# Body axes on a raw camera image when no shoulders are seen: user's right = image left, up = -v.
_FALLBACK_EX = np.array([-1.0, 0.0])
_FALLBACK_EY = np.array([0.0, -1.0])


def image_frame() -> Frame:
    """Image axes in the user's orientation, one pixel per unit. Used when no body frame exists."""
    return Frame("hand_fallback", np.zeros(2), _FALLBACK_EX.copy(), _FALLBACK_EY.copy(), 1.0)


def shoulder_frame(body, cfg: ControlConfig | None = None) -> Frame | None:
    """Frame from the shoulder line. Returns None when the shoulders are missing or too close."""
    if body is None:
        return None
    cfg = cfg or ControlConfig()
    conf = np.asarray(body.kps_conf, dtype=float)
    if conf[SHOULDER_L] < cfg.body_conf_min or conf[SHOULDER_R] < cfg.body_conf_min:
        return None
    left = np.asarray(body.kps[SHOULDER_L], dtype=float)
    right = np.asarray(body.kps[SHOULDER_R], dtype=float)
    if not (_is_valid_kp(left) and _is_valid_kp(right)):
        return None
    d = right - left
    scale = float(np.linalg.norm(d))
    if scale < cfg.shoulder_min_px:
        return None
    ex = d / scale                     # left shoulder -> right shoulder: the user's right
    ey = np.array([-ex[1], ex[0]])     # perpendicular, pointing up on screen
    return Frame("shoulder", (left + right) / 2.0, ex, ey, scale)


def fallback_frame(unit_px: float) -> Frame:
    """Image-axes frame whose unit is unit_px. Used when no shoulders are visible."""
    if unit_px <= 0:
        raise ValueError("fallback unit must be positive")
    return Frame("hand_fallback", np.zeros(2), _FALLBACK_EX.copy(), _FALLBACK_EY.copy(), float(unit_px))


def to_body(frame: Frame, p) -> np.ndarray:
    """Pixel p -> body coordinates (x, y) in units of frame.scale."""
    d = np.asarray(p, dtype=float) - frame.origin
    return np.array([d @ frame.ex, d @ frame.ey]) / frame.scale


def _valid_points(kps, idx):
    return [np.asarray(kps[i], dtype=float) for i in idx if _is_valid_kp(kps[i])]


def palm_center(hand_kps):
    """Mean of the valid palm keypoints (0, 5, 9, 13, 17), or None when none are valid."""
    pts = _valid_points(hand_kps, PALM_IDX)
    if not pts:
        return None
    return np.mean(pts, axis=0)


def hand_size(hand_kps):
    """Palm size in pixels: the larger of the median wrist-to-MCP length and 1.25 x the MCP span.

    Foreshortening only shortens a length. A hand seen from below (a drone on the ground)
    shortens the wrist-to-MCP lengths, a hand turned sideways shortens the span, so the larger
    of the two is the better estimate of the true size.
    """
    kps = np.asarray(hand_kps, dtype=float)
    needed = (0, 5, 9, 13, 17)
    if not all(_is_valid_kp(kps[i]) for i in needed):
        return None
    w = kps[0]
    lengths = [
        np.linalg.norm(kps[5] - w),
        np.linalg.norm(kps[9] - w),
        np.linalg.norm(kps[13] - w),
        np.linalg.norm(kps[17] - w),
    ]
    size = max(float(np.median(lengths)), 1.25 * float(np.linalg.norm(kps[5] - kps[17])))
    return size if size > 1e-6 else None


def hand_angle(hand_kps, frame: Frame | None = None):
    """Bearing of the wrist-to-middle-MCP vector in the body frame, in radians.

    0 means the hand points up. Positive values are clockwise, so turning the wrist
    clockwise (as the user sees it) increases the angle.
    """
    frame = frame or image_frame()
    kps = np.asarray(hand_kps, dtype=float)
    if not (_is_valid_kp(kps[WRIST]) and _is_valid_kp(kps[MIDDLE_MCP])):
        return None
    b = to_body(frame, kps[MIDDLE_MCP]) - to_body(frame, kps[WRIST])
    return math.atan2(b[0], b[1])


def hand_shoulder_ratio(hand_kps, frame: Frame):
    """Hand size divided by the frame scale. Scale-free, used for near/far and the forward axis."""
    size = hand_size(hand_kps)
    if size is None:
        return None
    return size / frame.scale
