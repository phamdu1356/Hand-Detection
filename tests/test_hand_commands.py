"""Hand-only flight input from a moving, unmirrored camera."""
import numpy as np
import pytest

from dronectl.pipeline import Controller
from dronectl.types import Mode
from sim3d.camera import DroneCamera
from sim3d.noise import WebcamNoise
from sim3d.person import Person, PersonPose
from tests.synth import BodyPose, HandPose, Scene, make_observation


@pytest.mark.parametrize("shape", ["palm", "fist"])
def test_stationary_neutral_hand_with_moving_camera(shape):
    ctl = Controller(hand_only=True)
    ctl.fsm.mode = Mode.HOVER
    cam = DroneCamera(pitch_up_deg=0.0, gimbal=False)
    _, hand = Person(position=(-2, 0, 0)).keypoints(PersonPose(gesture=shape))
    detector = WebcamNoise("none")
    centres = []
    for i in range(91):
        t = i / 30
        pos = (0.4 * t, 0.25 * np.sin(t), 1.3 + 0.10 * t)
        uv, _, visible = cam.project(hand, pos, 0.10 * t)
        assert visible.all()
        centres.append(uv.mean(axis=0))
        obs = detector.observe(t, (cam.width, cam.height), None, None, uv, visible)
        out = ctl.update(obs)
        assert out.debug["label"] == shape
        assert out.mode is Mode.HOVER and out.cmd.is_zero() and out.action is None
    assert np.linalg.norm(centres[-1] - centres[0]) > 50


def test_arms_alone_cannot_take_off_or_move():
    ctl = Controller(hand_only=True)
    for i in range(90):
        obs = make_observation(i / 30, Scene(), body=BodyPose(left="up", right="up"))
        out = ctl.update(obs)
        assert out.mode is Mode.LANDED and out.cmd.is_zero() and out.action is None


def test_unreliable_fingertip_stops_pointing_on_first_frame():
    ctl = Controller(hand_only=True)
    ctl.fsm.mode = Mode.HOVER
    for i in range(30):
        obs = make_observation(i / 30, Scene(), hand=HandPose(gesture="one"))
        out = ctl.update(obs)
    assert out.mode is Mode.POINT_MOVE and out.cmd.up > 0
    obs.t += 1 / 30
    obs.hand.kps_conf[8] = 0
    out = ctl.update(obs)
    assert out.mode is Mode.HOVER and out.cmd.is_zero()


def test_camera_dropout_cannot_complete_a_takeoff_hold():
    ctl = Controller(hand_only=True)
    for t in (0.0, 0.03, 0.06, 1.5):
        out = ctl.update(make_observation(t, Scene(), hand=HandPose(gesture="like")))
        assert out.action is None and out.mode is Mode.LANDED


def test_hand_only_3d_script_returns_to_ground_after_all_directions():
    from sim3d.loop import Sim3D
    from sim3d.scenarios import DISTANCE, evaluate, expectations, hand_script

    sim = Sim3D(noise="none", hand_only=True, person=Person(position=(-DISTANCE["hand"], 0.0, 0.0)))
    try:
        records = sim.run_script(hand_script())
        assert not sim.camera.gimbal
        assert all(r["ok"] for r in evaluate(records, expectations("hand")))
        assert sim.summary()["landed"] and records[-1]["mode"] == "LANDED"
    finally:
        sim.close()
