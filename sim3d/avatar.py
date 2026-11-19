"""Pose the 3D person from a real webcam Observation (simulator input mode "webcam").

The webcam sees the user from ``webcam_distance`` metres. Body keypoints are taken into the
shoulder frame and scaled to metres with the shoulder width. The hand's distance from the
webcam comes from its size relative to the shoulders, which gives how far it is pushed
forward. The result is re-projected through the drone camera by the simulator, so the
controller sees the user's real motion from the drone's point of view.
"""
from __future__ import annotations

import numpy as np

from dronectl.body_frame import palm_center, shoulder_frame, to_body, hand_size
from dronectl.config import ControlConfig

from .person import HAND_LEN, SHOULDER_W, Person, PersonPose

MAX_PUSH = 0.5      # the hand cannot be further in front of the shoulders than this, m


def lift(obs, person: Person, webcam_distance: float = 0.7, cfg: ControlConfig | None = None):
    """(body_world (17, 3), body_conf (17,), hand_world (21, 3) or None).

    Without a shoulder frame the person keeps a default standing pose and no hand.
    """
    cfg = cfg or ControlConfig()
    rest_body, _ = person.body_keypoints(PersonPose(right_arm="down", show_hand=False))
    frame = shoulder_frame(obs.body, cfg) if obs.body is not None else None
    if frame is None:
        return person.to_world(rest_body), np.ones(17), None
    body = rest_body.copy()
    conf = np.asarray(obs.body.kps_conf, dtype=float)
    for i in range(17):
        if conf[i] >= cfg.body_conf_min:
            bx, by = to_body(frame, obs.body.kps[i])
            body[i] = (bx * SHOULDER_W, by * SHOULDER_W, rest_body[i][2])
    hand_w = None
    hand = obs.hand
    if hand is not None and float(np.mean(hand.kps_conf)) >= cfg.hand_conf_min:
        size = hand_size(hand.kps)
        palm = palm_center(hand.kps)
        if size is not None and palm is not None:
            q = size / frame.scale
            depth = (HAND_LEN / SHOULDER_W) * webcam_distance / q     # webcam to hand, m
            push = float(np.clip(webcam_distance - depth, 0.0, MAX_PUSH))
            near = depth / webcam_distance                            # hand looks bigger by 1/near
            pb = np.asarray(to_body(frame, palm)) * SHOULDER_W * near
            m_per_px = HAND_LEN / size
            pts = np.zeros((21, 3))
            for i in range(21):
                off = (np.asarray(to_body(frame, hand.kps[i])) - to_body(frame, palm)) * frame.scale * m_per_px
                pts[i] = (pb[0] + off[0], pb[1] + off[1], push)
            hand_w = person.to_world(pts)
            body[10] = pts[0]                                         # wrist joins the hand
            body[8] = 0.5 * (body[6] + pts[0]) + np.array([0.04, -0.08, 0.0])
    return person.to_world(body), conf if obs.body is not None else np.ones(17), hand_w
