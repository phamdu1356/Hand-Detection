"""Keyboard teleop for the PyBullet quadrotor simulator.

Keys (hold to move):
    W / S        forward / backward
    A / D        left / right
    Up / Down    up / down
    Left / Right yaw left / yaw right
    Space        hold to stop (zero command)
    T            takeoff to 1.5 m
    L            land
    Q            quit (GUI only)

Usage:
    python sim/teleop_keyboard.py            # GUI window
    python sim/teleop_keyboard.py --smoke    # ~3 s scripted run, DIRECT mode, no window
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pybullet as p

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.quad import QuadSim  # noqa: E402

VEL = 1.0  # m/s per key
YAW_RATE = math.radians(60.0)  # rad/s per key
FRAME_DT = 1.0 / 60.0
TAKEOFF_ALT = 1.5
SMOKE_SECONDS = 3.0


def _codes(ch: str) -> set[int]:
    return {ord(ch.lower()), ord(ch.upper())}


W, S, A, D = _codes("w"), _codes("s"), _codes("a"), _codes("d")
UP, DOWN = {p.B3G_UP_ARROW}, {p.B3G_DOWN_ARROW}
LEFT, RIGHT = {p.B3G_LEFT_ARROW}, {p.B3G_RIGHT_ARROW}
SPACE = {32}
TAKEOFF, LAND, QUIT = _codes("t"), _codes("l"), _codes("q")


def key_command(down: set[int]) -> tuple[float, float, float, float]:
    """Map held keys to a user-frame command (forward, right, up, yaw_rate)."""
    def held(keys: set[int]) -> bool:
        return not keys.isdisjoint(down)

    if held(SPACE):
        return (0.0, 0.0, 0.0, 0.0)
    forward = VEL * (float(held(W)) - float(held(S)))
    right = VEL * (float(held(D)) - float(held(A)))
    up = VEL * (float(held(UP)) - float(held(DOWN)))
    yaw = YAW_RATE * (float(held(LEFT)) - float(held(RIGHT)))
    return (forward, right, up, yaw)


def _apply_keys(sim: QuadSim, down: set[int], triggered: set[int]) -> None:
    if not triggered.isdisjoint(TAKEOFF):
        sim.takeoff(TAKEOFF_ALT)
    if not triggered.isdisjoint(LAND):
        sim.land()
    sim.set_velocity(*key_command(down))


def _scripted_keys(t: float) -> tuple[set[int], set[int]]:
    """Keys (held, triggered) for the smoke run at simulated time ``t``."""
    if t < FRAME_DT:
        return set(), TAKEOFF
    if 2.5 <= t < 2.5 + FRAME_DT:
        return set(), LAND
    return set(), set()


def run_smoke() -> int:
    sim = QuadSim(gui=False)
    try:
        max_alt = 0.0
        frames = int(round(SMOKE_SECONDS / FRAME_DT))
        for i in range(frames):
            down, triggered = _scripted_keys(i * FRAME_DT)
            _apply_keys(sim, down, triggered)
            sim.step(FRAME_DT)
            max_alt = max(max_alt, sim.state().altitude)
        st = sim.state()
        print(f"smoke: t={SMOKE_SECONDS:.1f}s altitude={st.altitude:.3f}m max_altitude={max_alt:.3f}m "
              f"landed={st.landed} pos={st.pos.round(3).tolist()}")
        ok = max_alt > 0.5 and math.isfinite(float(st.altitude))
        print("smoke: OK" if ok else "smoke: FAILED (drone did not climb)")
        return 0 if ok else 1
    finally:
        sim.close()


def run_gui(wind: float) -> int:
    sim = QuadSim(gui=True, wind=wind)
    print("Keys: W/S forward/back, A/D left/right, Up/Down up/down, Left/Right yaw, "
          "Space stop, T takeoff, L land, Q quit")
    try:
        while True:
            events = sim.keyboard_events()
            down = {k for k, s in events.items() if s & p.KEY_IS_DOWN}
            triggered = {k for k, s in events.items() if s & p.KEY_WAS_TRIGGERED}
            if not triggered.isdisjoint(QUIT):
                break
            _apply_keys(sim, down, triggered)
            t0 = time.perf_counter()
            sim.step(FRAME_DT)
            time.sleep(max(0.0, FRAME_DT - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="scripted ~3 s headless run; exit 0 on success")
    parser.add_argument("--wind", type=float, default=0.0, help="gust strength (m/s^2), GUI mode only")
    args = parser.parse_args(argv)
    if args.smoke:
        return run_smoke()
    return run_gui(args.wind)


if __name__ == "__main__":
    raise SystemExit(main())
