"""PyBullet backend: adapts sim.quad.QuadSim to the DroneBackend interface.

The controller and the simulator use the same headless user frame (forward, right, up,
yaw left positive), so the conversion is only a rename. The simulator is imported here,
not from dronectl's other modules, so the rest of dronectl stays free of PyBullet.
"""
from __future__ import annotations

import numpy as np

from ..types import VelocityCommand
from .base import DroneState


class PyBulletSimBackend:
    def __init__(self, gui: bool = False, wind: float = 0.0, seed: int = 0):
        from sim.quad import QuadSim  # lazy: needs pybullet and the repository's sim/ package

        self._sim = QuadSim(gui=gui, wind=wind, seed=seed)

    def takeoff(self, altitude: float = 1.5) -> None:
        self._sim.takeoff(altitude)

    def land(self) -> None:
        self._sim.land()

    def hover(self) -> None:
        self._sim.hover()

    def send_velocity(self, cmd: VelocityCommand) -> None:
        self._sim.set_velocity(cmd.forward, cmd.right, cmd.up, cmd.yaw_rate)

    def step(self, dt: float) -> None:
        self._sim.step(dt)

    def state(self) -> DroneState:
        st = self._sim.state()
        return DroneState(pos=np.asarray(st.pos, dtype=float), vel=np.asarray(st.vel, dtype=float),
                          yaw=float(st.yaw), altitude=float(st.altitude), landed=bool(st.landed))

    @property
    def client(self):
        """The PyBullet client of the simulated world (for extra debug drawing)."""
        return self._sim.client

    def close(self) -> None:
        self._sim.close()
