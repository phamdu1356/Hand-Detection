"""Backend interface between the controller and a drone (plan section 7.1).

Commands use the headless user frame of VelocityCommand. A simulator or a real drone adapter
converts them to its own axes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..types import ControlOutput, VelocityCommand


@dataclass
class DroneState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))   # FLU: x forward, y left, z up
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    altitude: float = 0.0
    landed: bool = True


class DroneBackend(Protocol):
    def takeoff(self, altitude: float = 1.5) -> None: ...

    def land(self) -> None: ...

    def hover(self) -> None: ...

    def send_velocity(self, cmd: VelocityCommand) -> None: ...

    def step(self, dt: float) -> None: ...

    def state(self) -> DroneState: ...

    def close(self) -> None: ...


def apply_control(backend: DroneBackend, out: ControlOutput, takeoff_altitude: float = 1.5) -> None:
    """Dispatch one ControlOutput (takeoff / land / hover action and velocity) to a backend."""
    if out.action == "takeoff":
        backend.takeoff(takeoff_altitude)
    elif out.action == "land":
        backend.land()
    elif out.action == "hover":
        backend.hover()
    backend.send_velocity(out.cmd)
