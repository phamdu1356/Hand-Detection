"""Quadrotor simulator in PyBullet with a velocity-tracking cascade controller.

Frames
------
World is FLU (ROS style): X forward, Y left, Z up. The user stands at the
origin facing +X. Velocity commands are given in the user frame and mapped to
the world as ``vx = forward``, ``vy = -right``, ``vz = up``. ``yaw_rate`` is
positive counter-clockwise seen from above. Commands do not depend on the
drone's own heading.

Control cascade (every 240 Hz physics step)
-------------------------------------------
1. Velocity PI loop -> desired acceleration -> thrust vector ``m (a + g)``.
2. Thrust magnitude is clamped to ``[0, 2.5 m g]``; the tilt of the thrust
   direction is limited to 25 degrees from vertical.
3. Attitude PD loop on the rotation-matrix error gives body torques; yaw rate
   is tracked by a P loop on the body yaw rate.

This module does not depend on any other project package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

MASS = 0.8  # kg, total
G = 9.81  # m/s^2
SIM_HZ = 240
DT_PHYS = 1.0 / SIM_HZ

BASE_HALF = (0.125, 0.125, 0.03)  # central box, 0.25 x 0.25 x 0.06 m
BASE_MASS = 0.72
ARM_RADIUS = 0.05  # rotor pillars (cylinders along Z)
ARM_HALF_LEN = 0.01
ARM_MASS = 0.02
ARM_OFFSET = 0.14  # pillars sit at (+-ARM_OFFSET, +-ARM_OFFSET)
REST_Z = BASE_HALF[2]  # COM height when resting on the ground

TILT_MAX = math.radians(25.0)
THRUST_MAX = 2.5 * MASS * G
TORQUE_MAX = 0.5  # N*m per axis

KP_V = 2.5  # velocity loop P gain (1/s)
KI_V = 0.5  # velocity loop I gain (1/s^2)
INT_MAX = 2.0  # anti-windup clamp on the integral (m/s)

OMEGA_ATT = 10.0  # attitude loop bandwidth (rad/s), roll/pitch
ZETA_ATT = 0.8
OMEGA_YAW = 15.0  # yaw-rate loop bandwidth (rad/s)

TAKEOFF_SPEED = 0.5  # m/s, max climb speed during takeoff
TAKEOFF_GAIN = 2.0  # 1/s, approach gain near the target altitude
LAND_SPEED = 0.4  # m/s, descent speed during landing
TOUCHDOWN_ALT = 0.02  # m, altitude below which touchdown is checked

FENCE_GAIN = 2.0  # 1/s, speed limit approaching a fence wall
BOUNDS_X = (-2.0, 15.0)
BOUNDS_Y = (-8.0, 8.0)
BOUNDS_Z = (0.0, 6.0)  # altitude, not world Z

WIND_TAU = 2.0  # s, correlation time of the gust process

_WORLD = p.WORLD_FRAME


@dataclass
class DroneState:
    pos: np.ndarray  # world position of the body COM (m)
    vel: np.ndarray  # world linear velocity (m/s)
    yaw: float  # unwrapped yaw (rad), counter-clockwise positive
    altitude: float  # COM height above rest height (m)
    landed: bool


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _vee(s: np.ndarray) -> np.ndarray:
    return np.array([s[2, 1], s[0, 2], s[1, 0]])


def _rot_from_zb_yaw(zb: np.ndarray, yaw: float) -> np.ndarray:
    xc = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    yb = np.cross(zb, xc)
    yb /= np.linalg.norm(yb)
    xb = np.cross(yb, zb)
    return np.column_stack([xb, yb, zb])


def _fence(v: np.ndarray, pos: np.ndarray, altitude: float) -> np.ndarray:
    """Limit the commanded velocity so the drone decelerates before each wall.

    Upper bound: ``v <= K (hi - x)``; lower bound: ``v >= K (lo - x)``. Inside
    the box these leave the command unchanged when it points inward; outside
    the box they push back toward the inside.
    """
    out = v.copy()
    out[0] = min(max(out[0], FENCE_GAIN * (BOUNDS_X[0] - pos[0])), FENCE_GAIN * (BOUNDS_X[1] - pos[0]))
    out[1] = min(max(out[1], FENCE_GAIN * (BOUNDS_Y[0] - pos[1])), FENCE_GAIN * (BOUNDS_Y[1] - pos[1]))
    out[2] = min(max(out[2], FENCE_GAIN * (BOUNDS_Z[0] - altitude)), FENCE_GAIN * (BOUNDS_Z[1] - altitude))
    return out


class QuadSim:
    """Velocity-commanded quadrotor in PyBullet.

    Modes: ``landed`` (motors off, resting on the ground), ``takeoff``,
    ``flying`` (tracks the user command), ``landing``.
    """

    def __init__(self, gui: bool = False, wind: float = 0.0, seed: int = 0):
        self.wind = float(wind)
        self._rng = np.random.default_rng(seed)
        self._wind_acc = np.zeros(2)
        self._mode = "landed"
        self._target_alt = 0.0
        self._cmd = np.zeros(4)  # forward, right, up, yaw_rate
        self._int_err = np.zeros(3)
        self._yaw = 0.0
        self._yaw_prev = 0.0
        self._acc = 0.0

        self._pb = bullet_client.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
        pb = self._pb
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(DT_PHYS)
        pb.setGravity(0.0, 0.0, -G)
        pb.loadURDF("plane.urdf")

        base_col = pb.createCollisionShape(p.GEOM_BOX, halfExtents=BASE_HALF)
        base_vis = pb.createVisualShape(p.GEOM_BOX, halfExtents=BASE_HALF, rgbaColor=[0.15, 0.35, 0.85, 1])
        arm_col = pb.createCollisionShape(p.GEOM_CYLINDER, radius=ARM_RADIUS, height=2 * ARM_HALF_LEN)
        arm_vis = pb.createVisualShape(p.GEOM_CYLINDER, radius=ARM_RADIUS, length=2 * ARM_HALF_LEN,
                                       rgbaColor=[0.1, 0.1, 0.1, 1])
        arm_pos = [[sx * ARM_OFFSET, sy * ARM_OFFSET, 0.0] for sx in (1, -1) for sy in (1, -1)]
        self._body = pb.createMultiBody(
            baseMass=BASE_MASS,
            baseCollisionShapeIndex=base_col,
            baseVisualShapeIndex=base_vis,
            basePosition=[0.0, 0.0, REST_Z],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=[ARM_MASS] * 4,
            linkCollisionShapeIndices=[arm_col] * 4,
            linkVisualShapeIndices=[arm_vis] * 4,
            linkPositions=arm_pos,
            linkOrientations=[[0, 0, 0, 1]] * 4,
            linkInertialFramePositions=[[0, 0, 0]] * 4,
            linkInertialFrameOrientations=[[0, 0, 0, 1]] * 4,
            linkParentIndices=[0] * 4,
            linkJointTypes=[p.JOINT_FIXED] * 4,
            linkJointAxis=[[0, 0, 1]] * 4,
        )
        # No damping: the controller is the only thing that dissipates energy.
        for link in range(-1, 4):
            pb.changeDynamics(self._body, link, linearDamping=0.0, angularDamping=0.0)

        # Gains are scheduled on the measured inertia of the assembled body.
        dyn = pb.getDynamicsInfo(self._body, -1)
        inertia = np.asarray(dyn[2], dtype=float)
        self._kp_att = np.array([inertia[0], inertia[1], 0.0]) * OMEGA_ATT**2
        self._kd_att = np.array([inertia[0], inertia[1], 0.0]) * 2.0 * ZETA_ATT * OMEGA_ATT
        self._kp_yaw = inertia[2] * OMEGA_YAW

    # ------------------------------------------------------------------ API
    def takeoff(self, altitude: float = 1.5) -> None:
        """Climb to ``altitude`` (m above rest height), then hold and accept commands."""
        self._target_alt = float(altitude)
        self._int_err[:] = 0.0
        self._cmd[:] = 0.0
        self._mode = "takeoff"

    def land(self) -> None:
        """Descend at ``LAND_SPEED`` until touchdown, then turn the motors off."""
        if self._mode in ("takeoff", "flying"):
            self._mode = "landing"

    def set_velocity(self, forward: float, right: float, up: float, yaw_rate: float) -> None:
        """Set the user-frame velocity command: m/s forward/right/up and rad/s yaw (CCW)."""
        self._cmd[:] = (float(forward), float(right), float(up), float(yaw_rate))

    def step(self, dt: float) -> None:
        """Advance simulated time by ``dt`` seconds (physics runs at 240 Hz)."""
        self._acc += float(dt)
        while self._acc >= DT_PHYS:
            self._acc -= DT_PHYS
            self._physics_step()

    def state(self) -> DroneState:
        pos, _ = self._pb.getBasePositionAndOrientation(self._body)
        vel, _ = self._pb.getBaseVelocity(self._body)
        pos = np.asarray(pos, dtype=float)
        return DroneState(
            pos=pos,
            vel=np.asarray(vel, dtype=float),
            yaw=self._yaw,
            altitude=float(pos[2] - REST_Z),
            landed=self._mode == "landed",
        )

    def close(self) -> None:
        self._pb.disconnect()

    # ------------------------------------------------------------ internals
    def _physics_step(self) -> None:
        pb = self._pb
        body = self._body
        pos, orn = pb.getBasePositionAndOrientation(body)
        vel, ang_vel = pb.getBaseVelocity(body)
        pos = np.asarray(pos, dtype=float)
        v = np.asarray(vel, dtype=float)
        w_world = np.asarray(ang_vel, dtype=float)
        rot = np.asarray(pb.getMatrixFromQuaternion(orn), dtype=float).reshape(3, 3)

        yaw = math.atan2(rot[1, 0], rot[0, 0])
        self._yaw += _wrap(yaw - self._yaw_prev)
        self._yaw_prev = yaw
        altitude = pos[2] - REST_Z

        if self._mode == "landed":
            pb.stepSimulation()
            return

        # Mode logic: reference velocity and yaw rate.
        if self._mode == "takeoff":
            vz = min(TAKEOFF_SPEED, TAKEOFF_GAIN * (self._target_alt - altitude))
            v_des = np.array([0.0, 0.0, vz])
            yaw_des = 0.0
            if abs(altitude - self._target_alt) < 0.05 and abs(v[2]) < 0.1:
                self._mode = "flying"
                self._cmd[:] = 0.0
        elif self._mode == "flying":
            fwd, right, up, yaw_rate = self._cmd
            v_des = _fence(np.array([fwd, -right, up]), pos, altitude)
            yaw_des = float(yaw_rate)
        else:  # landing
            v_des = np.array([0.0, 0.0, -LAND_SPEED])
            yaw_des = 0.0
            if altitude < TOUCHDOWN_ALT and abs(v[2]) < 0.1:
                self._mode = "landed"
                self._int_err[:] = 0.0
                self._cmd[:] = 0.0
                pb.stepSimulation()
                return

        # Velocity loop -> thrust vector.
        err = v_des - v
        self._int_err = np.clip(self._int_err + err * DT_PHYS, -INT_MAX, INT_MAX)
        a_cmd = KP_V * err + KI_V * self._int_err
        f_des = MASS * (a_cmd + np.array([0.0, 0.0, G]))
        thrust = min(max(float(np.linalg.norm(f_des)), 0.0), THRUST_MAX)
        norm = float(np.linalg.norm(f_des))
        zb = f_des / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        if math.acos(min(1.0, max(-1.0, zb[2]))) > TILT_MAX:
            h = zb[:2]
            hn = float(np.linalg.norm(h))
            d = h / hn if hn > 1e-9 else np.array([1.0, 0.0])
            zb = np.array([math.sin(TILT_MAX) * d[0], math.sin(TILT_MAX) * d[1], math.cos(TILT_MAX)])

        # Attitude loop -> body torques.
        r_des = _rot_from_zb_yaw(zb, self._yaw)
        e_r = 0.5 * _vee(r_des.T @ rot - rot.T @ r_des)
        w_body = rot.T @ w_world
        tau = -self._kp_att * e_r - self._kd_att * w_body
        tau[2] = self._kp_yaw * (yaw_des - w_body[2])
        tau = np.clip(tau, -TORQUE_MAX, TORQUE_MAX)

        f_world = thrust * rot[:, 2]
        if self.wind > 0.0:
            self._wind_acc += (-self._wind_acc / WIND_TAU) * DT_PHYS + self.wind * math.sqrt(
                2.0 * DT_PHYS / WIND_TAU) * self._rng.standard_normal(2)
            f_world = f_world + MASS * np.array([self._wind_acc[0], self._wind_acc[1], 0.0])

        pb.applyExternalForce(body, -1, f_world.tolist(), pos.tolist(), _WORLD)
        pb.applyExternalTorque(body, -1, (rot @ tau).tolist(), _WORLD)
        pb.stepSimulation()
