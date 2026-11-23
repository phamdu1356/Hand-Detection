"""PyBulletSimBackend end to end: synthetic observations -> Controller -> QuadSim (DIRECT mode).

Covers the plan section 10.3 scenario (M2 braking, M6 hover stillness and the right-hand move).
"""
import numpy as np
import pytest

from dronectl.backends.base import apply_control
from dronectl.backends.pybullet_sim import PyBulletSimBackend
from dronectl.config import ControlConfig
from dronectl.pipeline import Controller
from dronectl.types import Mode
from tests.synth import BodyPose, HandPose, Scene, make_observation

FPS = 30.0
DT = 1.0 / FPS
BODY = BodyPose()
NEUTRAL = HandPose(gesture="one", palm=(0.0, -0.6), ratio=0.45)   # label "none", near


class Runner:
    def __init__(self, backend=None, cfg=None, seed=0):
        self.cfg = cfg or ControlConfig()
        self.ctl = Controller(self.cfg)
        self.drone = backend or PyBulletSimBackend(seed=seed)
        self.scene = Scene()
        self.t = 0.0
        self.outs = []
        self.right_max = 0.0

    def frame(self, hand, body=BODY, noise_px=0.0, rng=None, estop=False):
        obs = make_observation(self.t, self.scene, body=body, hand=hand, noise_px=noise_px, rng=rng)
        out = self.ctl.update(obs, self.drone.state(), estop=estop)
        apply_control(self.drone, out, self.cfg.takeoff_altitude)
        self.drone.step(DT)
        self.right_max = max(self.right_max, -float(self.drone.state().pos[1]))
        self.outs.append(out)
        self.t += DT
        return out

    def run(self, seconds, hand):
        for _ in range(int(round(seconds * FPS))):
            self.frame(hand)

    def until(self, pred, hand, max_s=15.0):
        start = self.t
        while not pred(self) and self.t - start < max_s:
            self.frame(hand)
        assert pred(self), f"condition not reached in {max_s}s, mode={self.ctl.mode}"

    def close(self):
        self.drone.close()


@pytest.fixture
def runner():
    r = Runner()
    yield r
    r.close()


def _hand(gesture, palm=(0.0, -0.6)):
    return HandPose(gesture=gesture, palm=palm, ratio=0.45)


def _takeoff_and_hover(r):
    r.run(1.2, _hand("like"))               # like held 1 s -> takeoff
    r.until(lambda s: s.ctl.mode is Mode.HOVER, NEUTRAL)
    assert r.drone.state().altitude >= 1.4
    r.run(0.5, NEUTRAL)


def _enter_follow(r):
    r.run(0.5, _hand("fist"))               # fist held 0.2 s -> FOLLOW, origin set
    assert r.ctl.mode is Mode.FOLLOW


def test_scenario_takeoff_follow_right_palm_land(runner):
    r = runner
    _takeoff_and_hover(r)
    _enter_follow(r)
    r.run(2.0, _hand("fist", palm=(0.4, -0.6)))  # hand 0.4 shoulder widths to the user's right
    assert r.ctl.mode is Mode.FOLLOW
    r.run(0.1, _hand("palm", palm=(0.4, -0.6)))  # stop
    r.run(0.5, NEUTRAL)
    r.run(0.1, _hand("ok", palm=(0.4, -0.6)))
    r.until(lambda s: s.ctl.mode is Mode.LANDING, _hand("ok"))
    r.until(lambda s: s.ctl.mode is Mode.LANDED, NEUTRAL, max_s=20.0)
    assert r.right_max > 0.5, f"drone reached only {r.right_max:.3f} m to the right"
    assert r.drone.state().altitude < 0.05
    actions = [o.action for o in r.outs if o.action]
    assert actions == ["takeoff", "land"]


def test_palm_brakes_below_0p1_mps_within_half_second(runner):
    # About 0.8 m/s at the moment of the palm. At full speed (1.5 m/s) the stop takes ~0.57 s,
    # limited by the 25 degree tilt; see the report for that measurement.
    r = runner
    _takeoff_and_hover(r)
    _enter_follow(r)
    r.run(1.5, _hand("fist", palm=(0.6, -0.6)))  # build up speed
    assert np.linalg.norm(r.drone.state().vel[:2]) > 0.5
    r.frame(_hand("palm", palm=(0.6, -0.6)))     # first frame with palm
    t_palm = r.t
    t_stop = None
    for _ in range(int(round(2.0 * FPS))):
        r.frame(_hand("palm", palm=(0.6, -0.6)))
        if np.linalg.norm(r.drone.state().vel) < 0.1:
            t_stop = r.t - t_palm
            break
    assert t_stop is not None and t_stop <= 0.5, f"braked in {t_stop} s"


def test_still_hand_in_follow_drifts_under_0p2m_in_10s(runner):
    r = runner
    _takeoff_and_hover(r)
    _enter_follow(r)
    p0 = r.drone.state().pos.copy()
    rng = np.random.default_rng(1)
    for _ in range(int(round(10.0 * FPS))):
        r.frame(_hand("fist"), noise_px=2.0, rng=rng)
    assert r.ctl.mode is Mode.FOLLOW
    drift = float(np.linalg.norm(r.drone.state().pos[:2] - p0[:2]))
    assert drift <= 0.2, f"drift {drift:.3f} m in 10 s"


def test_palm_cancels_landing_mid_air_on_pybullet(runner):
    r = runner
    _takeoff_and_hover(r)
    r.until(lambda s: s.ctl.mode is Mode.LANDING, _hand("ok"))
    r.run(1.0, NEUTRAL)
    assert r.ctl.mode is Mode.LANDING
    r.run(0.2, _hand("palm"))
    assert r.ctl.mode is Mode.HOVER
    r.run(2.5, NEUTRAL)
    st = r.drone.state()
    assert not st.landed and st.altitude > 0.5
    assert r.ctl.mode is Mode.HOVER
