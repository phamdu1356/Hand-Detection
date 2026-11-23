"""Closed loop: person keypoints -> drone camera -> noise -> Controller -> PyBullet drone."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from dronectl.backends.base import apply_control
from dronectl.config import ControlConfig
from dronectl.pipeline import Controller
from dronectl.types import Mode

from .camera import DroneCamera
from .noise import WebcamNoise
from .person import Person, PersonPose

FPS = 30.0
AIM_BELOW = 0.15     # the gimbal aims this far below the shoulders, where the hand usually is


class Sim3D:
    def __init__(self, cfg: ControlConfig | None = None, noise="webcam", seed: int = 0,
                 gui: bool = False, person: Person | None = None, camera: DroneCamera | None = None,
                 wind: float = 0.0, fps: float = FPS, hand_only: bool = False,
                 input_mode: str | None = None):
        from dronectl.backends.pybullet_sim import PyBulletSimBackend

        self.cfg = cfg or ControlConfig()
        self.ctl = Controller(self.cfg, hand_only=hand_only, input_mode=input_mode)
        self.noise = noise if isinstance(noise, WebcamNoise) else WebcamNoise(noise, seed)
        self.person = person or Person()
        self.camera = camera or (DroneCamera(gimbal=False, pitch_up_deg=6.0) if self.ctl.hand_only else DroneCamera())
        self.backend = PyBulletSimBackend(gui=gui, wind=wind, seed=seed)
        self.dt = 1.0 / fps
        self.t = 0.0
        self.records: list[dict] = []
        self.last_obs = self.last_out = None
        self.view = None
        if gui:
            from .render import SceneView
            self.view = SceneView(self.backend.client, self.camera)

    def step_pose(self, pose: PersonPose, tag: str = "") -> dict:
        shown = replace(pose, gesture=self.noise.render_gesture(pose.gesture))
        body, hand = self.person.keypoints(shown)
        return self.step(body, hand, tag=tag)

    def step(self, body_w, hand_w, body_conf=None, tag: str = "", estop: bool = False) -> dict:
        """One frame from world keypoints. body_conf masks body keypoints the source lacks.

        estop: emergency hover, zero command this frame and the state machine goes to HOVER.
        """
        st = self.backend.state()
        cam = self.camera
        cam.aim(st.pos, st.yaw, self.person.shoulder_mid - np.array([0.0, 0.0, AIM_BELOW]), self.dt)
        b_uv, _, b_vis = cam.project(body_w, st.pos, st.yaw)
        if body_conf is not None:
            b_vis = b_vis & (np.asarray(body_conf) > 0.3)
        h_uv = h_vis = None
        if hand_w is not None:
            h_uv, _, h_vis = cam.project(hand_w, st.pos, st.yaw)
        obs = self.noise.observe(self.t, (cam.width, cam.height), b_uv, b_vis, h_uv, h_vis)
        out = self.ctl.update(obs, st, estop=estop)
        self.last_obs, self.last_out = obs, out
        apply_control(self.backend, out, self.cfg.takeoff_altitude)
        self.backend.step(self.dt)
        after = self.backend.state()
        torso = self.person.shoulder_mid
        rec = dict(
            t=self.t, tag=tag, mode=out.mode.value, range=out.range_mode,
            label=out.debug.get("label"), cmd=(out.cmd.forward, out.cmd.right, out.cmd.up, out.cmd.yaw_rate),
            pos=tuple(float(v) for v in after.pos), yaw=float(after.yaw), altitude=float(after.altitude),
            hand_seen=obs.hand is not None, body_seen=obs.body is not None,
            hand_in_view=bool(h_vis is not None and np.all(h_vis)),
            dist=float(np.linalg.norm(np.asarray(after.pos)[:2] - torso[:2])),
        )
        self.records.append(rec)
        if self.view is not None:
            self.view.draw(body_w, hand_w, after, obs, out)
        self.t += self.dt
        return rec

    def run_script(self, script, realtime: bool = False, on_frame=None):
        import time
        wall0 = time.perf_counter()
        n = int(math.ceil(script.duration / self.dt))
        for _ in range(n):
            pose, tag = script.pose_at(self.t)
            rec = self.step_pose(pose, tag)
            if on_frame is not None and on_frame(self, rec) is False:
                break
            if realtime:
                lag = self.t - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
        return self.records

    def summary(self) -> dict:
        r = self.records
        modes = [x["mode"] for x in r]
        follow_exits = sum(1 for a, b in zip(modes, modes[1:]) if a == Mode.FOLLOW.value and b != a)
        airborne = [x for x in r if x["altitude"] > 0.3]
        return dict(
            frames=len(r),
            follow_exits=follow_exits,
            hand_missing=float(np.mean([not x["hand_seen"] for x in airborne])) if airborne else 0.0,
            person_lost=float(np.mean([not x["body_seen"] for x in airborne])) if airborne else 0.0,
            min_dist=min((x["dist"] for x in airborne), default=float("nan")),
            landed=bool(r and r[-1]["altitude"] < 0.05),
        )

    def close(self):
        if self.view is not None:
            self.view.close()
        self.backend.close()
