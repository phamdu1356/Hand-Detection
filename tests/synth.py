"""Backwards-compatible re-export; the canonical module now lives in sim3d.synth."""
from sim3d.synth import (  # noqa: F401
    GESTURES,
    GESTURE_SHAPES,
    BodyPose,
    HandPose,
    Scene,
    arm_points,
    body_points,
    hand_body,
    hand_points,
    hand_px,
    make_observation,
    rot,
)
