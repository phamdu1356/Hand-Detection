"""Tests for the PyBullet quadrotor simulator (DIRECT mode, no GUI).

Covers the acceptance criteria of plan item M12: takeoff height and time,
hover drift, velocity step response, stopping, yaw rate, geofence, landing,
and the sign of the lateral axis.
"""

import math

import numpy as np
import pytest

from sim.quad import QuadSim

DT = 1.0 / 240.0


def _run(sim: QuadSim, seconds: float) -> None:
    for _ in range(int(round(seconds / DT))):
        sim.step(DT)


@pytest.fixture
def sim():
    s = QuadSim(gui=False, seed=0)
    yield s
    s.close()


@pytest.fixture
def hovering(sim):
    """Drone already in the air, hovering at 1.5 m, zero command."""
    sim.takeoff(1.5)
    _run(sim, 6.0)
    return sim


def test_takeoff_reaches_1p5m_within_5s(sim):
    sim.takeoff(1.5)
    first_in_band = None
    t = 0.0
    while t < 5.0:
        sim.step(DT)
        t += DT
        if first_in_band is None and abs(sim.state().altitude - 1.5) <= 0.1:
            first_in_band = t
            break
    assert first_in_band is not None, f"altitude {sim.state().altitude:.3f} m never reached 1.5 +- 0.1 m"
    assert first_in_band < 5.0


def test_hover_drift_under_0p1m_over_5s(hovering):
    p0 = hovering.state().pos.copy()
    _run(hovering, 5.0)
    drift = float(np.linalg.norm(hovering.state().pos - p0))
    assert drift < 0.1, f"hover drift {drift:.3f} m"


def test_forward_step_reaches_90pct_fast_without_overshoot(hovering):
    hovering.set_velocity(1.0, 0.0, 0.0, 0.0)
    t90 = None
    peak = 0.0
    t = 0.0
    while t < 3.0:
        hovering.step(DT)
        t += DT
        vx = float(hovering.state().vel[0])
        peak = max(peak, vx)
        if t90 is None and vx >= 0.9:
            t90 = t
    assert t90 is not None and t90 < 1.5, f"vx reached 0.9 m/s at {t90}"
    assert peak < 1.2, f"peak vx {peak:.3f} m/s overshoots 1.0 m/s by more than 20%"


def test_zero_command_stops_within_1s(hovering):
    hovering.set_velocity(1.0, 0.0, 0.0, 0.0)
    _run(hovering, 3.0)
    hovering.set_velocity(0.0, 0.0, 0.0, 0.0)
    t = 0.0
    speed = np.inf
    while t < 1.0:
        hovering.step(DT)
        t += DT
        speed = float(np.linalg.norm(hovering.state().vel))
        if speed < 0.1:
            break
    assert speed < 0.1, f"speed {speed:.3f} m/s after 1 s of zero command"


def test_yaw_45_deg_per_s_for_2s_gives_90_deg(hovering):
    yaw0 = hovering.state().yaw
    hovering.set_velocity(0.0, 0.0, 0.0, math.radians(45.0))
    _run(hovering, 2.0)
    turned = math.degrees(hovering.state().yaw - yaw0)
    assert 80.0 <= turned <= 100.0, f"turned {turned:.2f} deg"


def test_right_positive_moves_toward_negative_y(hovering):
    y0 = hovering.state().pos[1]
    x0 = hovering.state().pos[0]
    hovering.set_velocity(0.0, 1.0, 0.0, 0.0)
    _run(hovering, 2.0)
    pos = hovering.state().pos
    assert pos[1] - y0 < -1.0, f"y moved by {pos[1] - y0:.3f} m, expected toward -Y"
    assert abs(pos[0] - x0) < 0.2


def test_geofence_stops_forward_flight(hovering):
    hovering.set_velocity(1.0, 0.0, 0.0, 0.0)
    _run(hovering, 30.0)
    st = hovering.state()
    assert st.pos[0] <= 15.2, f"x={st.pos[0]:.3f} went past the X=15 wall"
    assert st.pos[0] >= 14.0, f"x={st.pos[0]:.3f}: drone never reached the wall"
    assert abs(st.vel[0]) < 0.1


def test_geofence_stops_lateral_flight(hovering):
    hovering.set_velocity(0.0, 1.0, 0.0, 0.0)
    _run(hovering, 15.0)
    y = hovering.state().pos[1]
    assert -8.2 <= y <= -7.0, f"y={y:.3f} not held at the Y=-8 wall"


def test_land_sets_landed_flag(hovering):
    hovering.land()
    t = 0.0
    while t < 10.0 and not hovering.state().landed:
        hovering.step(DT)
        t += DT
    st = hovering.state()
    assert st.landed, "landed flag not set within 10 s of land()"
    assert st.altitude < 0.05


def test_key_mapping_matches_user_frame():
    from sim.teleop_keyboard import A, D, DOWN, LEFT, RIGHT, SPACE, UP, W, key_command

    assert key_command({ord("w")}) == (1.0, 0.0, 0.0, 0.0)
    assert key_command({ord("S")}) == (-1.0, 0.0, 0.0, 0.0)
    assert key_command(set(D)) == (0.0, 1.0, 0.0, 0.0)
    assert key_command(set(A)) == (0.0, -1.0, 0.0, 0.0)
    assert key_command(set(UP)) == (0.0, 0.0, 1.0, 0.0)
    assert key_command(set(DOWN)) == (0.0, 0.0, -1.0, 0.0)
    assert key_command(set(LEFT))[3] > 0.0  # left turn is counter-clockwise
    assert key_command(set(RIGHT))[3] < 0.0
    assert key_command(set(W) | set(SPACE)) == (0.0, 0.0, 0.0, 0.0)
