"""Fly the simulated drone with hand gestures (plan 7.4).

Each frame: observation (webcam, scripted synthetic scenario, or a replayed log) -> Controller
-> backend (PyBullet sim or mock). Optionally logs one CSV row per frame (plan 7.5).

Examples:
    python demo/fly_sim.py --camera 0 --backend sim --log logs/run1.csv
    python demo/fly_sim.py --camera 0 --backend mock
    python demo/fly_sim.py --synthetic --backend mock --no-gui --log logs/synthetic.csv
    python demo/fly_sim.py --replay logs/synthetic.csv --backend mock --no-gui

Keys in the window: q quit, r reset the controller, space emergency hover (zero command and
the state machine goes to HOVER while airborne), h show or hide the HUD.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dronectl.backends.mock import MockBackend  # noqa: E402
from dronectl.body_frame import hand_angle, hand_shoulder_ratio, shoulder_frame  # noqa: E402
from dronectl.config import load_config  # noqa: E402
from dronectl.pipeline import Controller  # noqa: E402
from dronectl.types import BodyObs, FLYING_MODES, HandObs, Mode, Observation  # noqa: E402

LOG_COLUMNS = [
    "t", "latency_ms", "frame_w", "frame_h",
    "hand_present", "hand_conf", "gesture_raw", "gesture_stable", "held_s",
    "body_present", "S_px", "q_ratio", "range_mode", "frame_type",
    "x", "y", "r", "theta", "x0", "y0", "r0", "theta0",
    "ex", "ey", "ez", "eyaw",
    "mode", "action", "cmd_forward", "cmd_right", "cmd_up", "cmd_yaw",
    "drone_x", "drone_y", "drone_z", "drone_yaw",
    "hand_kps_json", "body_kps_json", "input_mode",
]
FRAME_DT = 1.0 / 30.0


# ---------------------------------------------------------------- backends

def make_backend(name: str, gui: bool, wind: float = 0.0, seed: int = 0):
    if name == "mock":
        return MockBackend()
    if name == "sim":
        from dronectl.backends.pybullet_sim import PyBulletSimBackend
        return PyBulletSimBackend(gui=gui, wind=wind, seed=seed)
    raise ValueError(f"unknown backend {name!r}")


# ---------------------------------------------------------------- observation sources

class SyntheticScenario:
    """Scripted run: like (takeoff), neutral until hover, fist (follow), hand 0.6 shoulder widths
    to the user's right for 2 s, palm (stop), ok (land). Driven by the controller's last mode, so
    it waits for each transition rather than using fixed times."""

    def __init__(self, max_seconds: float = 60.0, hand_only=False):
        from tests.synth import Scene
        self.scene = Scene()
        self.max_seconds = max_seconds
        self.t = 0.0
        self.phase = "like"
        self.phase_t = 0.0
        self.hand_only = hand_only

    def next_obs(self, last_out):
        from tests.synth import BodyPose, HandPose, make_observation
        if self.t >= self.max_seconds:
            return None
        mode = last_out.mode if last_out is not None else Mode.LANDED
        if self.phase == "like" and self.phase_t >= 1.2:
            self._enter("neutral_takeoff")
        elif self.phase == "neutral_takeoff" and mode is Mode.HOVER:
            self._enter("settle")
        elif self.phase == "settle" and self.phase_t >= 0.5:
            self._enter("fist")
        elif self.phase == "fist" and self.phase_t >= 0.5:
            self._enter("move")
        elif self.phase == "move" and self.phase_t >= 2.0:
            self._enter("stop")
        elif self.phase == "stop" and self.phase_t >= 0.5:
            self._enter("neutral_land")
        elif self.phase == "neutral_land" and self.phase_t >= 0.5:
            self._enter("land")
        elif self.phase == "land" and mode is Mode.LANDED:
            return None

        gesture, palm_x = {
            "like": ("like", 0.0), "neutral_takeoff": ("one", 0.0), "settle": ("one", 0.0),
            "fist": ("fist", 0.0), "move": ("fist", 0.6), "stop": ("palm", 0.6),
            "neutral_land": ("one", 0.6), "land": ("ok", 0.6),
        }[self.phase]
        angle = 0.0
        if self.hand_only:
            if self.phase == "move":
                gesture, angle = "one", math.pi / 2
            elif gesture == "one":
                gesture = "palm"
        hand = HandPose(gesture=gesture, palm=(palm_x, -0.6), ratio=0.45, rot=angle)
        obs = make_observation(self.t, self.scene, body=None if self.hand_only else BodyPose(), hand=hand)
        self.t += FRAME_DT
        self.phase_t += FRAME_DT
        return obs

    def _enter(self, name):
        self.phase = name
        self.phase_t = 0.0


class ReplaySource:
    """Rebuilds observations from a CSV log written by this script."""

    def __init__(self, path: str):
        with open(path, newline="", encoding="utf-8") as fh:
            self.rows = list(csv.DictReader(fh))
        self.i = 0

    def next_obs(self, _last_out):
        if self.i >= len(self.rows):
            return None
        row = self.rows[self.i]
        self.i += 1
        return obs_from_row(row)


def obs_from_row(row: dict) -> Observation:
    hand = None
    if row.get("hand_present") == "1":
        arr = np.asarray(json.loads(row["hand_kps_json"]), dtype=float).reshape(21, 3)
        kps = arr[:, :2]
        hand = HandObs(kps=kps, kps_conf=arr[:, 2], box=_xywh(kps), conf=float(row["hand_conf"]),
                       gesture_raw=row.get("gesture_raw") or "none")
    body = None
    if row.get("body_present") == "1":
        arr = np.asarray(json.loads(row["body_kps_json"]), dtype=float).reshape(17, 3)
        kps = arr[:, :2]
        body = BodyObs(kps=kps, kps_conf=arr[:, 2], box=_xywh(kps))
    return Observation(t=float(row["t"]), frame_wh=(int(row["frame_w"]), int(row["frame_h"])),
                       hand=hand, body=body, latency_ms=float(row["latency_ms"]))


def _xywh(kps):
    x0, y0 = kps.min(axis=0)
    x1, y1 = kps.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


class CameraSource:
    def __init__(self, source, perception):
        import cv2
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open the selected camera or video")
        self.recorded = isinstance(source, str) and os.path.isfile(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_number = 0
        self.previous_pts = -1.0
        self.perception = perception
        self.frame = None

    def next_obs(self, _last_out):
        ok, frame = self.cap.read()
        if not ok:
            if self.frame_number == 0 or not self.recorded:
                raise RuntimeError("camera or video stopped delivering frames")
            return None
        self.frame = frame
        t = time.monotonic()
        if self.recorded:
            import cv2
            t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not math.isfinite(t) or t <= self.previous_pts:
                if not math.isfinite(self.fps) or self.fps <= 0:
                    raise ValueError("recorded video has no usable timestamps or frame rate")
                t = max(0.0, self.previous_pts + 1 / self.fps)
            self.previous_pts = t
        self.frame_number += 1
        return self.perception.perceive(frame, t)

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------- log

def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(v)
    return v


def kps_json(kps, conf):
    if kps is None:
        return ""
    arr = np.column_stack([np.asarray(kps, dtype=float), np.asarray(conf, dtype=float)])
    return json.dumps([[round(float(x), 3) for x in row] for row in arr])


def build_row(obs: Observation, out, state, ctl: Controller, cfg, latency_ms: float) -> dict:
    hand, body = obs.hand, obs.body
    frame_s = shoulder_frame(body, cfg) if body is not None else None
    dbg = out.debug
    row = {c: None for c in LOG_COLUMNS}
    row.update(
        t=obs.t, latency_ms=latency_ms, frame_w=obs.frame_wh[0], frame_h=obs.frame_wh[1],
        hand_present=hand is not None, body_present=body is not None,
        hand_conf=hand.conf if hand is not None else None,
        gesture_raw=dbg.get("label"), gesture_stable=dbg.get("stable"), held_s=dbg.get("held_s"),
        S_px=frame_s.scale if frame_s is not None else None,
        q_ratio=dbg.get("d"), range_mode=out.range_mode, frame_type=dbg.get("frame"),
        x=dbg.get("hx"), y=dbg.get("hy"),
        mode=out.mode.value, action=out.action,
        cmd_forward=out.cmd.forward, cmd_right=out.cmd.right, cmd_up=out.cmd.up, cmd_yaw=out.cmd.yaw_rate,
        drone_x=float(state.pos[0]), drone_y=float(state.pos[1]), drone_z=float(state.pos[2]),
        drone_yaw=float(state.yaw),
        hand_kps_json=kps_json(hand.kps, hand.kps_conf) if hand is not None else "",
        body_kps_json=kps_json(body.kps, body.kps_conf) if body is not None else "",
        input_mode="hand" if ctl.hand_only else "mixed",
    )
    if hand is not None and out.range_mode != "none":
        frame = frame_s if frame_s is not None else None
        if frame is not None:
            row["theta"] = hand_angle(hand.kps, frame)
            row["r"] = hand_shoulder_ratio(hand.kps, frame)
    origin = ctl.joy.origin
    if origin is not None:
        row.update(x0=origin.x0, y0=origin.y0, r0=origin.r0, theta0=origin.th0)
    if "err" in dbg:
        row.update(ex=dbg["err"][0], ey=dbg["err"][1], ez=dbg["err"][2], eyaw=dbg["err"][3])
    return {k: _fmt(v) for k, v in row.items()}


class CsvLog:
    def __init__(self, path: str):
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self.fh = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.fh, fieldnames=LOG_COLUMNS)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------- main loop

def run(source, backend, ctl: Controller, cfg, log=None, hud=None, emergency=None):
    """Drive the controller and backend from a source until it is exhausted.

    Returns the list of (t, mode, cmd) tuples, which the replay check uses.
    hud(frame_or_none, obs, out, state) is called per frame when given.
    emergency() -> bool is polled each frame when given.
    """
    seq = []
    last_out = None
    prev_t = None
    while True:
        obs = source.next_obs(last_out)
        if obs is None:
            break
        t0 = time.perf_counter()
        state = backend.state()
        stop_now = bool(emergency and emergency())
        if stop_now and state.altitude > 0.05 and not state.landed:
            ctl.fsm.mode = Mode.HOVER
        out = ctl.update(obs, state)
        if out.action == "takeoff":
            backend.takeoff(cfg.takeoff_altitude)
        elif out.action == "land":
            backend.land()
        if stop_now:
            out.cmd = type(out.cmd)()
        backend.send_velocity(out.cmd)
        dt = FRAME_DT if prev_t is None else min(max(obs.t - prev_t, 1e-3), 0.2)
        prev_t = obs.t
        backend.step(dt)
        latency_ms = obs.latency_ms + (time.perf_counter() - t0) * 1000.0
        state_after = backend.state()
        if log is not None:
            log.write(build_row(obs, out, state_after, ctl, cfg, latency_ms))
        seq.append((obs.t, out.mode.value, out.cmd, out.action))
        if hud is not None:
            hud(source, obs, out, state_after)
        last_out = out
    return seq


def compare_sequences(replayed, logged_rows) -> int:
    """Count frames where the replayed mode, action or command differs from the log."""
    mismatches = 0
    for (t, mode, cmd, action), row in zip(replayed, logged_rows):
        same = (
            mode == row["mode"]
            and (action or "") == (row["action"] or "")
            and all(abs(float(a) - float(b)) < 1e-9 for a, b in (
                (cmd.forward, row["cmd_forward"]), (cmd.right, row["cmd_right"]),
                (cmd.up, row["cmd_up"]), (cmd.yaw_rate, row["cmd_yaw"])))
        )
        mismatches += 0 if same else 1
    mismatches += abs(len(replayed) - len(logged_rows))
    return mismatches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None, help="webcam index")
    src.add_argument("--video", help="onboard camera recording or OpenCV-compatible video stream URL")
    src.add_argument("--synthetic", action="store_true", help="scripted scenario, no camera")
    src.add_argument("--replay", default=None, help="CSV log to replay")
    ap.add_argument("--backend", choices=("sim", "mock"), default="sim")
    ap.add_argument("--config", default=None, help="YAML file overriding the control constants")
    ap.add_argument("--input-mode", choices=("hand", "mixed"), help="default: hand, or the recorded mode on replay")
    ap.add_argument("--log", default=None, help="write one CSV row per frame here")
    ap.add_argument("--no-gui", action="store_true", help="no windows (sim and camera HUD)")
    ap.add_argument("--wind", type=float, default=0.0, help="sim gust strength in m/s^2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-model", default=None, help="hand_pose.pt or .onnx")
    ap.add_argument("--body-model", default=None, help="yolo11n-pose.pt or .onnx")
    ap.add_argument("--heatmap-model", default=None)
    ap.add_argument("--far-field", choices=("auto", "always", "off"), default="auto")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    input_mode = args.input_mode or "hand"
    if args.replay is not None:
        source = ReplaySource(args.replay)
        if args.input_mode is None and source.rows:
            input_mode = source.rows[0].get("input_mode") or "mixed"
        gui = False
    elif args.synthetic:
        source = SyntheticScenario(hand_only=input_mode == "hand")
        gui = False
    else:
        from perception import Perception
        perception = Perception(pose_model_path=args.pose_model, body_model_path=args.body_model,
                                heatmap_model_path=args.heatmap_model, far_field=args.far_field,
                                conf=args.conf, hand_only=input_mode == "hand")
        camera = args.camera if args.camera is not None else 0
        source = CameraSource(args.video or camera, perception)
        gui = not args.no_gui

    backend = make_backend(args.backend, gui=(args.backend == "sim" and gui), wind=args.wind, seed=args.seed)
    ctl = Controller(cfg, hand_only=input_mode == "hand")
    log = CsvLog(args.log) if args.log else None
    keys = {"show_hud": True, "estop": False}
    try:
        if gui:
            import cv2
            from perception import draw_overlay

            def hud(src, obs, out, state):
                frame = getattr(src, "frame", None)
                if frame is None:
                    return
                if keys["show_hud"]:
                    cv2.imshow("fly_sim", draw_overlay(frame, obs, out, state))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt
                if key == ord("r"):
                    ctl.reset()
                if key == ord("h"):
                    keys["show_hud"] = not keys["show_hud"]
                keys["estop"] = key == ord(" ")

            seq = run(source, backend, ctl, cfg, log=log, hud=hud, emergency=lambda: keys["estop"])
        else:
            seq = run(source, backend, ctl, cfg, log=log)
        final = backend.state()
    except KeyboardInterrupt:
        seq = []
        final = backend.state()
    finally:
        if log is not None:
            log.close()
        if hasattr(source, "close"):
            source.close()
        backend.close()

    if args.replay is not None:
        logged = source.rows
        mismatches = compare_sequences(seq, logged)
        print(f"replay: {len(seq)} frames, {mismatches} differ from the log")
        return 0 if mismatches == 0 else 1
    print(f"done: frames={len(seq)} final altitude={final.altitude:.2f}m landed={final.landed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
