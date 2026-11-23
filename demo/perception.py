"""Perception for the drone demo: a camera frame in, an Observation out (plan 2.2).

In hand-only mode, the detector uses only the hand model. The optional mixed mode also
estimates a body pose for the shoulder frame and arm gestures.
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dronectl.types import BodyObs, HandObs, Mode, Observation  # noqa: E402
from test_model import HAND_CONNECTIONS, OptimalFlycamController  # noqa: E402

_NO_GESTURE = {"", "none", "unknown", "NONE", "UNKNOWN"}
_MODE_COLOR = {
    Mode.LANDED: (160, 160, 160),
    Mode.TAKING_OFF: (0, 200, 255),
    Mode.HOVER: (255, 200, 0),
    Mode.FOLLOW: (0, 220, 0),
    Mode.POINT_MOVE: (255, 0, 255),
    Mode.LANDING: (0, 128, 255),
}


def _conf_or_ones(conf, n):
    if conf is None:
        return np.ones(n, dtype=float)
    return np.asarray(conf, dtype=float).reshape(n)


class Perception:
    """Camera frame -> Observation. Needs the hand pose model; the person model is optional."""

    def __init__(self, pose_model_path=None, body_model_path=None, heatmap_model_path=None,
                 far_field="auto", conf=0.5, base_dir=None, hand_only=False):
        base_dir = base_dir or os.path.dirname(_HERE)
        self.conf = float(conf)
        self.ctl = OptimalFlycamController(
            base_dir,
            None,
            pose_model_path=pose_model_path,
            body_pose_model_path=body_model_path,
            heatmap_model_path=heatmap_model_path,
            far_field_mode=far_field,
            enable_tile_fallback=(not hand_only or far_field == "always"),
            always_body_pose=not hand_only,
            enable_body_pose=not hand_only,
        )
        if self.ctl.pose_model is None:
            raise RuntimeError("no hand pose model found: pass pose_model_path (hand_pose.pt or .onnx)")

    def perceive(self, frame, t: float | None = None) -> Observation:
        """Run pure inference on ``frame`` (without mutating or drawing on it) and return an Observation."""
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        res = self.ctl.detect_frame(frame, conf_thres=self.conf) or {}
        latency_ms = (time.perf_counter() - t0) * 1000.0

        hand = None
        kps = res.get("keypoints")
        if res.get("detected") and kps is not None:
            kps = np.asarray(kps, dtype=float).reshape(21, 2)
            kconf = _conf_or_ones(res.get("kpt_conf"), 21)
            x, y, bw, bh = res["box"] if res.get("box") is not None else (0.0, 0.0, 0.0, 0.0)
            raw = str(res.get("raw_gesture", "none"))
            hand = HandObs(kps=kps, kps_conf=kconf, box=(float(x), float(y), float(bw), float(bh)),
                           conf=float(res.get("conf", 0.0)),
                           gesture_raw="none" if raw in _NO_GESTURE else raw.lower())

        body = None
        bkps = res.get("body_keypoints")
        if bkps is not None:
            bkps = np.asarray(bkps, dtype=float).reshape(17, 2)
            body = BodyObs(kps=bkps, kps_conf=_conf_or_ones(res.get("body_kpt_conf"), 17),
                           box=tuple(float(v) for v in res["person_box"]))

        return Observation(t=float(time.monotonic() if t is None else t), frame_wh=(int(w), int(h)),
                           hand=hand, body=body, latency_ms=float(latency_ms))


def draw_overlay(frame, obs: Observation, ctl_out, backend_state=None, fps: float | None = None):
    """Draw the HUD of plan 7.4 on a copy of ``frame`` and return it."""
    img = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    if obs.body is not None:
        for x, y in obs.body.kps:
            if x > 0 or y > 0:
                cv2.circle(img, (int(x), int(y)), 3, (200, 200, 0), -1)
    if obs.hand is not None:
        k = obs.hand.kps
        for i, j in HAND_CONNECTIONS:
            if (k[i] > 0).any() and (k[j] > 0).any():
                cv2.line(img, tuple(int(v) for v in k[i]), tuple(int(v) for v in k[j]), (255, 180, 0), 2)
        for x, y in k:
            if x > 0 or y > 0:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)
        palm = k[[0, 5, 9, 13, 17]].mean(axis=0)
        cv2.circle(img, (int(palm[0]), int(palm[1])), 5, (0, 0, 255), -1)

    mode = ctl_out.mode
    color = _MODE_COLOR.get(mode, (255, 255, 255))
    cv2.putText(img, f"MODE {mode.value}  range={ctl_out.range_mode}", (10, 28), font, 0.8, color, 2)
    label = ctl_out.debug.get("label", "none")
    held = ctl_out.debug.get("held_s", 0.0)
    cv2.putText(img, f"gesture={label} held={held:.2f}s", (10, 56), font, 0.6, (255, 255, 255), 2)

    cmd = ctl_out.cmd
    bars = [("fwd", cmd.forward, 1.5), ("right", cmd.right, 1.5), ("up", cmd.up, 1.0),
            ("yaw", cmd.yaw_rate, 0.8)]
    h = img.shape[0]
    for i, (name, val, full) in enumerate(bars):
        y0 = h - 30 - 22 * i
        mid = 160
        length = int(max(-1.0, min(1.0, val / full)) * 100)
        cv2.rectangle(img, (mid - 100, y0 - 10), (mid + 100, y0 + 4), (60, 60, 60), 1)
        cv2.line(img, (mid, y0 - 10), (mid, y0 + 4), (120, 120, 120), 1)
        cv2.rectangle(img, (mid, y0 - 8), (mid + length, y0 + 2), color, -1)
        cv2.putText(img, f"{name} {val:+.2f}", (mid + 110, y0 + 2), font, 0.45, (255, 255, 255), 1)

    if backend_state is not None:
        cv2.putText(img, f"alt={backend_state.altitude:.2f}m landed={backend_state.landed}",
                    (10, 84), font, 0.6, (255, 255, 255), 2)
    lat = obs.latency_ms
    tail = f"lat={lat:.0f}ms" + (f" fps={fps:.0f}" if fps is not None else "")
    cv2.putText(img, tail, (10, h - 10), font, 0.5, (255, 255, 255), 1)
    return img
