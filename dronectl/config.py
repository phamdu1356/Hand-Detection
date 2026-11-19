"""Control constants from the design spec (sections 3 to 6) and an optional YAML override.

Use ``load_config()`` for defaults, or ``load_config(path)`` to override values from a YAML
file. Unknown keys are rejected so that a typo cannot silently keep the default.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class ControlConfig:
    # body frame (section 3)
    body_conf_min: float = 0.4        # shoulder and far-pose keypoint confidence
    shoulder_min_px: float = 20.0     # shorter shoulder line is treated as invalid
    fallback_k: float = 0.3           # fallback unit = fallback_k * hand size (no shoulders)

    # hand (section 4.1)
    hand_conf_min: float = 0.5        # mean hand keypoint confidence
    classifier: str = "rules"         # "rules" only until phase 4 adds "mlp"

    # range selection (section 6.1)
    near_enter_q: float = 0.30        # hand size / shoulder width to enter near
    far_enter_q: float = 0.20         # below this, switch to far (not while FOLLOW holds a hand)

    # far pose rules (section 4.2)
    arms_up_y_min: float = 1.0
    t_pose_y_max: float = 0.35
    t_pose_x_min: float = 1.3
    elbow_straight_deg: float = 150.0
    point_extend_min: float = 1.2
    point_rest_y_max: float = -0.8    # the other wrist must be below this
    point_tol_deg: float = 30.0
    point_len_margin: float = 0.2     # pointing arm must reach this much further than a hanging arm

    # gesture filter (section 4.3)
    min_stable_s: float = 0.066
    gap_tol_s: float = 0.25        # detector dropouts come in bursts of up to ~7 frames
    latch_release_s: float = 0.3

    # state machine (section 6.2)
    t_pose_stop_s: float = 0.2
    range_none_stop_s: float = 0.5
    land_hold_near_s: float = 1.0     # ok held
    land_hold_far_s: float = 1.5      # arms_up held
    takeoff_hold_near_s: float = 1.0  # like held
    takeoff_hold_far_s: float = 1.5   # arms_up held
    takeoff_altitude: float = 1.5
    takeoff_ratio: float = 0.95
    takeoff_timeout_s: float = 8.0
    follow_hold_s: float = 0.2        # fist held to enter FOLLOW
    follow_exit_s: float = 0.15       # other label held to leave FOLLOW
    follow_lost_s: float = 0.3        # hand lost or range not near this long leaves FOLLOW
    point_hold_s: float = 0.5         # point held to enter POINT_MOVE
    point_exit_s: float = 0.2         # other label held to leave POINT_MOVE (after gap_tol_s)

    # joystick (section 5)
    full_xy: float = 0.8              # body units giving full lateral and vertical input
    dead_xy: float = 0.15             # dead zone, body units
    full_z_ratio: float = 1.35        # hand size ratio giving full forward input
    dead_z: float = 0.2               # dead zone on the normalised log ratio
    full_yaw_deg: float = 35.0
    dead_yaw_deg: float = 10.0
    expo_alpha: float = 0.5
    v_h: float = 1.5                  # m/s, horizontal maximum
    v_v: float = 1.0                  # m/s, vertical maximum
    w_yaw_deg: float = 45.0           # deg/s, yaw maximum
    v_far: float = 0.5                # m/s, POINT_MOVE speed
    ema_alpha: float = 0.35
    ez_alpha: float = 0.3             # extra EMA on the raw forward error (hand size is noisy)
    acc_h: float = 2.0                # m/s^2, horizontal increase limit
    acc_v: float = 1.5                # m/s^2, vertical increase limit
    acc_yaw_deg: float = 90.0         # deg/s^2, yaw increase limit
    dec_lin: float = 4.0              # m/s^2, decrease limit for translation
    dec_yaw_deg: float = 180.0        # deg/s^2, decrease limit for yaw


_ALLOWED_CLASSIFIERS = ("rules",)


def validate(cfg: ControlConfig) -> ControlConfig:
    """Raise ValueError when the values cannot describe a working controller."""
    if cfg.classifier not in _ALLOWED_CLASSIFIERS:
        raise ValueError(f"classifier must be one of {_ALLOWED_CLASSIFIERS}, got {cfg.classifier!r}")
    if not 0.0 < cfg.far_enter_q < cfg.near_enter_q:
        raise ValueError("require 0 < far_enter_q < near_enter_q")
    if not 0.0 < cfg.dead_xy < cfg.full_xy:
        raise ValueError("require 0 < dead_xy < full_xy")
    if not 0.0 <= cfg.dead_z < 1.0 or cfg.full_z_ratio <= 1.0:
        raise ValueError("require 0 <= dead_z < 1 and full_z_ratio > 1")
    if not 0.0 < cfg.dead_yaw_deg < cfg.full_yaw_deg:
        raise ValueError("require 0 < dead_yaw_deg < full_yaw_deg")
    if not 0.0 < cfg.ema_alpha <= 1.0 or not 0.0 <= cfg.expo_alpha <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1] and expo_alpha in [0, 1]")
    if not 0.0 < cfg.ez_alpha <= 1.0:
        raise ValueError("ez_alpha must be in (0, 1]")
    if cfg.follow_lost_s < 0.0:
        raise ValueError("follow_lost_s must not be negative")
    if min(cfg.v_h, cfg.v_v, cfg.w_yaw_deg, cfg.v_far, cfg.acc_h, cfg.acc_v, cfg.acc_yaw_deg,
           cfg.dec_lin, cfg.dec_yaw_deg, cfg.fallback_k) <= 0.0:
        raise ValueError("speed, acceleration and fallback_k must be positive")
    return cfg


def load_config(path: str | None = None, overrides: dict | None = None) -> ControlConfig:
    """Build a ControlConfig from defaults, then a YAML file, then an explicit dict."""
    values: dict = {}
    if path is not None:
        import yaml  # imported lazily: only needed when a file is given

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config file {path!r} must contain a mapping")
        values.update(data)
    if overrides:
        values.update(overrides)
    known = {f.name for f in fields(ControlConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise KeyError(f"unknown config keys: {unknown}")
    return validate(ControlConfig(**values))
