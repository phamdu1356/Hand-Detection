"""Gesture labels and the gesture filter (plan section 4).

near_gesture: static hand shape from the hand keypoints (rule based until phase 4).
far_gesture:  arm poses from the body keypoints in the shoulder frame.
GestureFilter: time-based filter with a gap tolerance (gap_tol_s) and a latch.
"""
from __future__ import annotations

import math

import numpy as np

from .body_frame import Frame, to_body
from .config import ControlConfig
from .hand_rules import analyze_hand_keypoints

NEAR_LABELS = ("palm", "fist", "like", "ok", "none")
FAR_LABELS = ("arms_up", "t_pose", "point_right", "point_left", "point_up", "point_down", "none")

_RULE_TO_NEAR = {"palm": "palm", "four": "palm", "fist": "fist", "like": "like", "ok": "ok"}
_POINT_DIRS = {"right": 0.0, "up": 90.0, "left": 180.0, "down": -90.0}
_FAR_NEEDED = (0, 5, 6, 7, 8, 9, 10)  # nose, shoulders, elbows, wrists (COCO)


def near_gesture(hand, cfg: ControlConfig | None = None) -> str:
    """Label from the hand shape. Returns "none" when the hand is missing or low confidence."""
    cfg = cfg or ControlConfig()
    if hand is None or float(np.mean(hand.kps_conf)) < cfg.hand_conf_min:
        return "none"
    rule, _, _ = analyze_hand_keypoints(hand.kps)
    return _RULE_TO_NEAR.get(rule, "none")


def _angle_at(a, b, c) -> float:
    """Angle at vertex b, in degrees. 0 when a segment has zero length."""
    u, v = a - b, c - b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def _quantise_direction(vec, tol_deg: float) -> str:
    ang = math.degrees(math.atan2(vec[1], vec[0]))
    best, best_gap = "none", float("inf")
    for name, target in _POINT_DIRS.items():
        gap = abs((ang - target + 180.0) % 360.0 - 180.0)
        if gap < best_gap:
            best, best_gap = name, gap
    return f"point_{best}" if best_gap <= tol_deg else "none"


def far_gesture(body, frame: Frame | None, cfg: ControlConfig | None = None) -> str:
    """Label from the arm pose, measured in the shoulder frame. Needs a shoulder frame."""
    cfg = cfg or ControlConfig()
    if body is None or frame is None or frame.kind != "shoulder":
        return "none"
    conf = np.asarray(body.kps_conf, dtype=float)
    if min(conf[i] for i in _FAR_NEEDED) < cfg.body_conf_min:
        return "none"
    p = {i: to_body(frame, body.kps[i]) for i in _FAR_NEEDED}
    nose, sl, sr = p[0], p[5], p[6]
    el, er, wl, wr = p[7], p[8], p[9], p[10]

    if (wl[1] > cfg.arms_up_y_min and wr[1] > cfg.arms_up_y_min
            and wl[1] > nose[1] and wr[1] > nose[1]):
        return "arms_up"

    straight = cfg.elbow_straight_deg
    th_l = _angle_at(sl, el, wl)
    th_r = _angle_at(sr, er, wr)
    if (abs(wl[1]) < cfg.t_pose_y_max and abs(wr[1]) < cfg.t_pose_y_max
            and wl[0] < -cfg.t_pose_x_min and wr[0] > cfg.t_pose_x_min
            and th_l > straight and th_r > straight):
        return "t_pose"

    arms = {
        "l": (sl, wl, th_l),
        "r": (sr, wr, th_r),
    }
    length = {k: float(np.linalg.norm(w - s)) for k, (s, w, _) in arms.items()}
    ext = {k: th > straight and length[k] > cfg.point_extend_min for k, (_, _, th) in arms.items()}
    rest = {"l": wl[1] < cfg.point_rest_y_max, "r": wr[1] < cfg.point_rest_y_max}

    # The pointing arm is extended and the other arm hangs below the shoulder. A straight
    # hanging arm is also "extended" by length, so when both arms hang, the pointing arm
    # must reach clearly further (point_len_margin). This is what makes point_down work.
    pointers = []
    for p, q in (("l", "r"), ("r", "l")):
        if not (ext[p] and rest[q]):
            continue
        if not rest[p] or length[p] - length[q] > cfg.point_len_margin:
            pointers.append(p)
    if len(pointers) != 1:
        return "none"
    s, w, _ = arms[pointers[0]]
    return _quantise_direction(w - s, cfg.point_tol_deg)


class GestureFilter:
    """Turns per-frame raw labels into a held-for-duration label.

    - A run of "none" up to gap_tol_s does not reset the current label.
    - held_s is how long the current label has been held; held(label) is 0 while
      that label is latched.
    - latch(label) blocks held(label) until a different label has been seen for
      latch_release_s. The FSM uses this so that one held gesture cannot trigger
      takeoff and landing back to back.
    """

    def __init__(self, cfg: ControlConfig | None = None):
        self.cfg = cfg or ControlConfig()
        self.reset()

    def reset(self):
        self.label = "none"
        self._t = 0.0
        self._start = 0.0
        self._last_seen = 0.0
        self._started = False
        self._latched = None
        self._away_since = None

    def update(self, t: float, raw: str):
        if not self._started:
            self._started = True
            self.label, self._start, self._last_seen = raw, t, t
        self._t = t
        self._update_latch(t, raw)
        if raw == self.label:
            self._last_seen = t
            return
        if raw == "none" and t - self._last_seen <= self.cfg.gap_tol_s:
            return  # short dropout: keep the current label
        self.label, self._start, self._last_seen = raw, t, t

    def _update_latch(self, t: float, raw: str):
        if self._latched is None:
            return
        if raw == self._latched:
            self._away_since = None
            return
        if self._away_since is None:
            self._away_since = t
        if t - self._away_since >= self.cfg.latch_release_s:
            self._latched = None
            self._away_since = None

    @property
    def held_s(self) -> float:
        return self._t - self._start if self._started else 0.0

    @property
    def latched(self):
        return self._latched

    def held(self, label: str) -> float:
        if label != self.label or label == self._latched:
            return 0.0
        return self.held_s

    def stable(self) -> str:
        return self.label if self.held_s >= self.cfg.min_stable_s else "none"

    def latch(self, label: str):
        self._latched = label
        self._away_since = None
