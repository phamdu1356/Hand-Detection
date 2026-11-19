"""demo/fly_sim.py and demo/perception.py without a camera or models (synthetic scenario, HUD)."""
import csv
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "demo"))

import fly_sim  # noqa: E402
from dronectl.config import ControlConfig  # noqa: E402
from dronectl.pipeline import Controller  # noqa: E402
from dronectl.types import Mode  # noqa: E402
from tests.synth import BodyPose, HandPose, Scene, make_observation  # noqa: E402


def test_synthetic_scenario_flies_and_lands_with_mock(tmp_path):
    log = tmp_path / "syn.csv"
    assert fly_sim.main(["--synthetic", "--backend", "mock", "--no-gui", "--log", str(log)]) == 0
    rows = list(csv.DictReader(open(log, newline="", encoding="utf-8")))
    modes = [r["mode"] for r in rows]
    assert "POINT_MOVE" in modes and modes[-1] == "LANDED"
    assert all(r["input_mode"] == "hand" and r["body_present"] == "0" for r in rows)
    assert any(float(r["cmd_right"]) > 0 for r in rows)
    actions = [r["action"] for r in rows if r["action"]]
    assert actions == ["takeoff", "land"]


def test_synthetic_scenario_sends_zero_command_outside_follow(tmp_path):
    log = tmp_path / "syn.csv"
    fly_sim.main(["--synthetic", "--backend", "mock", "--no-gui", "--log", str(log)])
    for r in csv.DictReader(open(log, newline="", encoding="utf-8")):
        if r["mode"] not in ("FOLLOW", "POINT_MOVE"):
            assert float(r["cmd_forward"]) == float(r["cmd_right"]) == float(r["cmd_up"]) == 0.0
            assert float(r["cmd_yaw"]) == 0.0


def test_draw_overlay_draws_on_copy_and_keeps_shape():
    from perception import draw_overlay

    scene = Scene()
    obs = make_observation(0.0, scene, body=BodyPose(), hand=HandPose(gesture="palm", ratio=0.45))
    ctl = Controller(ControlConfig())
    out = ctl.update(obs, fly_sim.MockBackend().state())
    frame = np.zeros((720, 1280, 3), np.uint8)
    img = draw_overlay(frame, obs, out, None, fps=30.0)
    assert img.shape == frame.shape
    assert img.any()
    assert not frame.any()          # the input frame is not modified
    assert out.mode in (Mode.LANDED, Mode.HOVER, Mode.TAKING_OFF, Mode.FOLLOW)


@pytest.mark.parametrize("backend", ["mock", "sim"])
def test_replay_reproduces_recorded_commands(tmp_path, backend):
    log = tmp_path / "rec.csv"
    assert fly_sim.main(["--synthetic", "--backend", backend, "--no-gui", "--log", str(log)]) == 0
    assert fly_sim.main(["--replay", str(log), "--backend", backend, "--no-gui"]) == 0


def test_mock_tracks_vertical_hand_commands():
    from dronectl.types import VelocityCommand

    backend = fly_sim.MockBackend()
    backend.takeoff(1.5)
    backend.step(3.0)
    backend.send_velocity(VelocityCommand(up=0.5))
    backend.step(1.0)
    assert backend.state().altitude == pytest.approx(2.0)
    backend.send_velocity(VelocityCommand(up=-0.25))
    backend.step(1.0)
    assert backend.state().pos[2] == pytest.approx(1.75)
