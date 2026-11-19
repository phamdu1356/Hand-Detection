"""Webcam-like degradation of perfect projected keypoints.

What a real detector does, as measured on a laptop webcam run (logs/run1.csv, 2026-11-19):
keypoint jitter, short bursts with no hand, a fist read with the thumb out (like), and a
couple of frames of latency. Bursts are two-state Markov chains: a burst starts with
probability *_enter per frame and ends with *_exit per frame (mean length 1 / *_exit frames).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from dronectl.types import BodyObs, HandObs, Observation


@dataclass(frozen=True)
class NoiseConfig:
    hand_sigma: float = 0.03      # keypoint jitter, fraction of the projected hand size
    body_sigma: float = 0.012     # keypoint jitter, fraction of the projected shoulder width
    drop_enter: float = 0.04      # hand dropout bursts
    drop_exit: float = 0.3
    thumb_enter: float = 0.04     # fist rendered with the thumb out
    thumb_exit: float = 0.3
    delay_frames: int = 2         # observation latency
    hand_conf: float = 0.85
    body_conf: float = 0.9


PRESETS = {
    "none": NoiseConfig(hand_sigma=0.0, body_sigma=0.0, drop_enter=0.0, thumb_enter=0.0, delay_frames=0),
    "webcam": NoiseConfig(),
    "harsh": NoiseConfig(hand_sigma=0.05, body_sigma=0.02, drop_enter=0.08, drop_exit=0.2,
                         thumb_enter=0.1, thumb_exit=0.2, delay_frames=3),
}


def preset(name: str) -> NoiseConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown noise preset {name!r}, choose from {sorted(PRESETS)}")
    return PRESETS[name]


def _box(pts):
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


class Burst:
    def __init__(self, enter: float, exit_: float, rng):
        self.enter, self.exit, self.rng, self.on = enter, exit_, rng, False

    def step(self) -> bool:
        if self.on:
            self.on = self.rng.random() >= self.exit
        else:
            self.on = self.enter > 0 and self.rng.random() < self.enter
        return self.on


class WebcamNoise:
    def __init__(self, cfg: NoiseConfig | str = "webcam", seed: int = 0):
        self.cfg = preset(cfg) if isinstance(cfg, str) else cfg
        self.rng = np.random.default_rng(seed)
        self._drop = Burst(self.cfg.drop_enter, self.cfg.drop_exit, self.rng)
        self._thumb = Burst(self.cfg.thumb_enter, self.cfg.thumb_exit, self.rng)
        self._queue: deque = deque()

    def render_gesture(self, gesture: str) -> str:
        """The hand shape the detector will report this frame."""
        misread = self._thumb.step()
        return "like" if gesture == "fist" and misread else gesture

    def observe(self, t: float, frame_wh, body_uv, body_vis, hand_uv, hand_vis) -> Observation:
        """Build this frame's Observation from projected keypoints, then apply the latency."""
        c = self.cfg
        body = None
        if body_uv is not None and body_vis[5] and body_vis[6]:
            sw = float(np.linalg.norm(body_uv[5] - body_uv[6]))
            pts = body_uv + self.rng.normal(0.0, c.body_sigma * sw, body_uv.shape)
            conf = np.where(body_vis, c.body_conf, 0.0)
            body = BodyObs(kps=pts, kps_conf=conf, box=_box(pts[body_vis]))
        hand = None
        dropped = self._drop.step()
        if hand_uv is not None and bool(np.all(hand_vis)) and not dropped:
            size = float(np.linalg.norm(hand_uv[9] - hand_uv[0]))
            pts = hand_uv + self.rng.normal(0.0, c.hand_sigma * size, hand_uv.shape)
            hand = HandObs(kps=pts, kps_conf=np.full(21, c.hand_conf), box=_box(pts), conf=c.hand_conf)
        obs = Observation(t=t, frame_wh=tuple(frame_wh), hand=hand, body=body)
        self._queue.append(obs)
        if len(self._queue) <= c.delay_frames:
            return replace(self._queue[0], t=t, hand=None, body=None)
        old = self._queue.popleft()
        return replace(old, t=t)         # stale content, stamped with the current time
