"""Mock backend: ideal velocity tracking, no physics. Used by tests and runs without PyBullet."""
from __future__ import annotations

import numpy as np

from ..types import VelocityCommand
from .base import DroneState

CLIMB_RATE = 0.5    # m/s, matches the simulator's takeoff rate
DESCENT_RATE = 0.4  # m/s, matches the simulator's landing rate


class MockBackend:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.yaw = 0.0
        self.altitude = 0.0
        self.landed = True
        self._mode = "idle"        # "idle" | "climb" | "hover" | "descend"
        self._target_alt = 0.0
        self._cmd = VelocityCommand()
        self.events: list[str] = []
        self.ground_commands = 0   # non-zero commands received while on the ground

    @property
    def right_pos(self) -> float:
        """Position to the user's right, in metres (the FLU y axis points left)."""
        return float(-self.pos[1])

    def takeoff(self, altitude: float = 1.5) -> None:
        self.events.append("takeoff")
        self.landed = False
        self._mode = "climb"
        self._target_alt = float(altitude)

    def land(self) -> None:
        self.events.append("land")
        if not self.landed:
            self._mode = "descend"

    def send_velocity(self, cmd: VelocityCommand) -> None:
        if self.landed and not cmd.is_zero():
            self.ground_commands += 1
        self._cmd = VelocityCommand(cmd.forward, cmd.right, cmd.up, cmd.yaw_rate)
        if self.verbose and not cmd.is_zero():
            print(f"cmd f={cmd.forward:+.2f} r={cmd.right:+.2f} u={cmd.up:+.2f} yaw={cmd.yaw_rate:+.2f}")

    def step(self, dt: float) -> None:
        vertical = self._cmd.up
        if self._mode == "climb":
            self.altitude = min(self._target_alt, self.altitude + CLIMB_RATE * dt)
            vertical = CLIMB_RATE if self.altitude < self._target_alt else 0.0
            if self.altitude >= self._target_alt:
                self._mode = "hover"
        elif self._mode == "descend":
            self.altitude = max(0.0, self.altitude - DESCENT_RATE * dt)
            vertical = -DESCENT_RATE
            if self.altitude <= 0.0:
                self._mode = "idle"
                self.landed = True
                self._cmd = VelocityCommand()
                self.events.append("landed")
        elif not self.landed:
            next_altitude = max(0.0, self.altitude + vertical * dt)
            if dt > 0:
                vertical = (next_altitude - self.altitude) / dt
            self.altitude = next_altitude
        if self.landed:
            self.vel[:] = 0.0
            return
        self.vel = np.array([self._cmd.forward, -self._cmd.right, vertical])
        self.pos += self.vel * dt
        self.pos[2] = self.altitude
        self.yaw += self._cmd.yaw_rate * dt

    def state(self) -> DroneState:
        return DroneState(pos=self.pos.copy(), vel=self.vel.copy(), yaw=self.yaw,
                          altitude=self.altitude, landed=self.landed)

    def close(self) -> None:
        self._cmd = VelocityCommand()
