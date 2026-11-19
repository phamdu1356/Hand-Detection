"""Pinhole camera carried by the drone.

The simulated drone's heading 0 points away from the user (+x). A follow drone films the
user, so the camera looks backward: optical axis heading = drone yaw + pi. The camera sits on
a pitch gimbal: with gimbal=True, aim() tilts it toward a target (the person's chest) at a
limited rate, so the person stays in view from the ground and as the drone climbs. The
camera does not yaw on its own: a yaw command turns the view. Airframe roll and pitch are
ignored (stabilised).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DroneCamera:
    width: int = 960
    height: int = 720
    hfov_deg: float = 82.6          # Tello-like
    pitch_up_deg: float = 20.0
    mount_back: float = 0.12        # lens offset from the drone centre, toward the user
    gimbal: bool = True
    gimbal_rate_deg: float = 60.0   # deg/s
    gimbal_limit_deg: float = 35.0

    def aim(self, drone_pos, drone_yaw: float, target, dt: float):
        """Tilt toward target at gimbal_rate_deg. No-op without a gimbal."""
        if not self.gimbal:
            return
        centre, _, _, _ = self.pose(drone_pos, drone_yaw)
        d = np.asarray(target, dtype=float) - centre
        want = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
        want = max(-self.gimbal_limit_deg, min(self.gimbal_limit_deg, want))
        step = self.gimbal_rate_deg * dt
        self.pitch_up_deg += max(-step, min(step, want - self.pitch_up_deg))

    @property
    def f(self) -> float:
        return 0.5 * self.width / math.tan(math.radians(self.hfov_deg) / 2.0)

    def pose(self, drone_pos, drone_yaw: float):
        """Camera centre and (forward, right, down) unit axes in the world."""
        psi = float(drone_yaw) + math.pi
        th = math.radians(self.pitch_up_deg)
        fwd = np.array([math.cos(psi) * math.cos(th), math.sin(psi) * math.cos(th), math.sin(th)])
        right = np.array([math.sin(psi), -math.cos(psi), 0.0])
        down = np.cross(fwd, right)
        centre = np.asarray(drone_pos, dtype=float) + self.mount_back * np.array([math.cos(psi), math.sin(psi), 0.0])
        return centre, fwd, right, down

    def project(self, points, drone_pos, drone_yaw: float):
        """Pixels (N, 2) with u right and v down, depth (N,), and an in-view mask (N,)."""
        centre, fwd, right, down = self.pose(drone_pos, drone_yaw)
        p = np.asarray(points, dtype=float).reshape(-1, 3) - centre
        z = p @ fwd
        safe = np.where(z > 1e-3, z, 1e-3)
        u = self.f * (p @ right) / safe + 0.5 * self.width
        v = self.f * (p @ down) / safe + 0.5 * self.height
        uv = np.column_stack([u, v])
        vis = (z > 0.15) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return uv, z, vis
