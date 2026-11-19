"""3D simulator: geometry, noise, webcam lifting, and the closed loop with the controller."""
import math

import numpy as np
import pytest

from dronectl.body_frame import shoulder_frame, to_body
from sim3d.camera import DroneCamera
from sim3d.noise import WebcamNoise
from sim3d.person import FOREARM, UPPER_ARM, Person, PersonPose, two_bone_ik
from tests.synth import BodyPose, HandPose, Scene, make_observation

pytest.importorskip("pybullet")


def test_camera_centre_ray_projects_to_the_image_centre():
    cam = DroneCamera(pitch_up_deg=0.0, gimbal=False)
    c, fwd, _, _ = cam.pose((0.0, 0.0, 1.5), 0.0)
    uv, z, vis = cam.project([c + 2.0 * fwd], (0.0, 0.0, 1.5), 0.0)
    assert np.allclose(uv[0], (cam.width / 2, cam.height / 2)) and z[0] == pytest.approx(2.0) and vis[0]


def test_drone_camera_sees_the_person_like_a_raw_webcam():
    """COCO left shoulder (5) on the image right, and the shoulder frame x is the user's right."""
    cam = DroneCamera(pitch_up_deg=0.0, gimbal=False)
    body, _ = Person().keypoints(PersonPose(show_hand=False, right_arm="down"))
    uv, _, vis = cam.project(body, (0.0, 0.0, 1.45), 0.0)
    assert vis[5] and vis[6] and uv[5][0] > uv[6][0]
    from dronectl.types import BodyObs
    frame = shoulder_frame(BodyObs(kps=uv, kps_conf=np.full(17, 0.9), box=(0, 0, 1, 1)))
    assert to_body(frame, uv[6])[0] > 0           # right shoulder at +x
    assert to_body(frame, uv[0])[1] > 0           # nose above the shoulders


def test_gimbal_tilts_toward_the_target_at_a_limited_rate():
    cam = DroneCamera(pitch_up_deg=0.0)
    cam.aim((0.0, 0.0, 0.0), 0.0, (-1.0, 0.0, 1.0), 0.1)
    assert cam.pitch_up_deg == pytest.approx(6.0)  # 60 deg/s for 0.1 s


def test_ik_keeps_bone_lengths_and_clamps_out_of_reach_targets():
    sh = np.array([0.17, 0.0, 0.0])
    elbow, wrist = two_bone_ik(sh, sh + [0.0, -0.1, 0.4], 1.0)
    assert np.linalg.norm(elbow - sh) == pytest.approx(UPPER_ARM)
    assert np.linalg.norm(wrist - elbow) == pytest.approx(FOREARM)
    _, far = two_bone_ik(sh, sh + [0.0, 0.0, 2.0], 1.0)
    assert np.linalg.norm(far - sh) < UPPER_ARM + FOREARM


def test_clockwise_wrist_roll_tilts_the_hand_toward_the_users_right():
    p = Person()
    _, h0 = p.body_keypoints(PersonPose(gesture="fist", roll_deg=0.0))
    _, h1 = p.body_keypoints(PersonPose(gesture="fist", roll_deg=25.0))
    d0, d1 = h0[9] - h0[0], h1[9] - h1[0]
    assert abs(d0[0]) < 1e-9 and d1[0] > 0


def test_noise_free_observation_has_the_projected_keypoints_without_delay():
    cam = DroneCamera(pitch_up_deg=0.0, gimbal=False)
    body, hand = Person().keypoints(PersonPose(gesture="palm"))
    b_uv, _, b_vis = cam.project(body, (0.0, 0.0, 1.45), 0.0)
    h_uv, _, h_vis = cam.project(hand, (0.0, 0.0, 1.45), 0.0)
    obs = WebcamNoise("none").observe(0.0, (cam.width, cam.height), b_uv, b_vis, h_uv, h_vis)
    assert np.allclose(obs.hand.kps, h_uv) and np.allclose(obs.body.kps, b_uv)


def test_webcam_noise_delays_observations_and_drops_hands_in_bursts():
    noise = WebcamNoise("webcam", seed=3)
    cam = DroneCamera(pitch_up_deg=0.0, gimbal=False)
    body, hand = Person().keypoints(PersonPose(gesture="palm"))
    b_uv, _, b_vis = cam.project(body, (0.0, 0.0, 1.45), 0.0)
    h_uv, _, h_vis = cam.project(hand, (0.0, 0.0, 1.45), 0.0)
    seen = [noise.observe(k / 30, (960, 720), b_uv, b_vis, h_uv, h_vis).hand is not None for k in range(3000)]
    assert not seen[0] and not seen[1]               # 2 frames of latency
    missing = 1.0 - np.mean(seen[2:])
    assert 0.06 < missing < 0.2                      # 0.04 / (0.04 + 0.3) = 12% expected


def test_lift_puts_a_hand_on_the_users_right_and_pushes_a_bigger_hand_forward():
    from sim3d.avatar import lift
    person = Person()
    scene = Scene()
    small = make_observation(0.0, scene, body=BodyPose(), hand=HandPose("fist", palm=(0.6, -0.4), ratio=0.30))
    big = make_observation(0.0, scene, body=BodyPose(), hand=HandPose("fist", palm=(0.6, -0.4), ratio=0.45))
    _, _, h_small = lift(small, person, 0.7)
    _, _, h_big = lift(big, person, 0.7)
    r, _, f = person.axes
    centre = person.shoulder_mid
    assert (h_small[9] - centre) @ r > 0            # user's right
    assert (h_big[9] - centre) @ f > (h_small[9] - centre) @ f + 0.05


def test_drone_view_image_has_the_scaled_camera_size():
    from dronectl.types import ControlOutput, Mode, VelocityCommand
    from sim3d.render import drone_view
    cam = DroneCamera()
    obs = make_observation(0.0, Scene(), body=BodyPose(), hand=HandPose("fist"))
    img = drone_view(obs, ControlOutput(Mode.HOVER, VelocityCommand(), None, "near"), cam, scale=0.5)
    assert img.shape == (cam.height // 2, cam.width // 2, 3)


# ---------------------------------------------------------------- closed loop

def _run(name, noise, seed=0):
    from sim3d.loop import Sim3D
    from sim3d.scenarios import DISTANCE, SCRIPTS, evaluate, expectations
    sim = Sim3D(noise=noise, seed=seed, person=Person(position=(-DISTANCE[name], 0.0, 0.0)))
    try:
        sim.run_script(SCRIPTS[name]())
        return evaluate(sim.records, expectations(name)), sim.summary()
    finally:
        sim.close()


def test_closed_loop_axes_without_noise_moves_every_axis_the_right_way():
    rows, summary = _run("axes", "none")
    assert [r["tag"] for r in rows if not r["ok"]] == []
    assert summary["min_dist"] > 1.0


def test_closed_loop_far_arm_poses_take_off_point_and_land():
    rows, summary = _run("far", "webcam")
    assert all(r["ok"] for r in rows)
    assert summary["landed"]


def test_closed_loop_with_webcam_noise_takes_off_and_moves_sideways():
    rows, _ = _run("axes", "webcam", seed=3)
    ok = {r["tag"]: r["ok"] for r in rows}
    assert ok["right"] and ok["left"] and ok["up"]
