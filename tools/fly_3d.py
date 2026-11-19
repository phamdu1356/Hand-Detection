"""3D simulator: a jointed person filmed by a camera on the drone, closed loop with the controller.

Scripted (repeatable, used as a test):
  .venv/bin/python tools/fly_3d.py --scenario hand --no-gui --noise none
  .venv/bin/python tools/fly_3d.py --scenario axes                 # window + drone camera view
  .venv/bin/python tools/fly_3d.py --scenario axes --no-gui --repeat 10 --noise webcam
  .venv/bin/python tools/fly_3d.py --scenario far --no-gui

Webcam drives the 3D person (your real motion, seen from the drone):
  .venv/bin/python tools/fly_3d.py --webcam 0 --pose-model hand_pose.pt --body-model yolo11n-pose.pt

Windows: the PyBullet scene (person, drone, blue trail, yellow camera ray) and "drone camera"
(what the controller receives). Keys in the drone camera window: q quit, space emergency hover.
Exit code 0 when every judged axis moved the right way in every run (scripted mode).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO, os.path.join(_REPO, "demo")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dronectl.config import load_config  # noqa: E402
from sim3d.loop import Sim3D  # noqa: E402
from sim3d.person import Person  # noqa: E402
from sim3d.scenarios import DISTANCE, SCRIPTS, evaluate, expectations  # noqa: E402


def print_table(rows, summary):
    print(f"{'hand / arm motion':28s} {'axis':9s} {'want':4s} {'mean cmd':>10s} {'moved':>9s}  {'modes':16s} result")
    for r in rows:
        unit = "rad" if r["axis"] == "yaw_right" else "m"
        print(f"{r['desc']:28s} {r['axis']:9s} {'+' if r['sign'] > 0 else '-':4s} {r['cmd']:+10.2f} "
              f"{r['moved']:+8.2f}{unit:1s}  {r['modes']:16s} {'OK' if r['ok'] else 'WRONG'}")
    print("follow exits {follow_exits}, airborne frames without hand {hand_missing:.0%}, without person "
          "{person_lost:.0%}, closest to person {min_dist:.2f} m, landed at end {landed}".format(**summary))


def hud_window(sim, keys):
    import cv2
    from sim3d.render import drone_view

    def show(obs, out):
        cv2.imshow("drone camera", drone_view(obs, out, sim.camera))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            keys["quit"] = True
        keys["estop"] = k == ord(" ")
    return show


def run_scripted(args, cfg) -> int:
    all_ok = True
    for i in range(args.repeat):
        seed = args.seed + i
        gui = not args.no_gui and i == 0
        sim = Sim3D(cfg=cfg, noise=args.noise, seed=seed, gui=gui, hand_only=args.scenario == "hand",
                    person=Person(position=(-args.person_distance, 0.0, 0.0)), wind=args.wind)
        keys = {"quit": False, "estop": False}
        show = hud_window(sim, keys) if gui else None
        if gui:
            _run_gui_script(sim, SCRIPTS[args.scenario](), show, keys)
        else:
            sim.run_script(SCRIPTS[args.scenario]())
        rows = evaluate(sim.records, expectations(args.scenario))
        summary = sim.summary()
        sim.close()
        ok = all(r["ok"] for r in rows)
        if args.scenario == "hand":
            stop = [r for r in sim.records if r["tag"] == "stop" and r["label"] == "palm"]
            ok &= (bool(stop) and all(not any(r["cmd"]) for r in stop) and summary["landed"]
                   and sim.records[-1]["mode"] == "LANDED")
        all_ok &= ok
        print(f"== run {i + 1}/{args.repeat} seed {seed} noise {args.noise}: {'all correct' if ok else 'SOME WRONG'}")
        print_table(rows, summary)
    return 0 if all_ok else 1


def _run_gui_script(sim, script, show, keys):
    """Like Sim3D.run_script in real time, with the drone camera window."""
    import math
    wall0 = time.perf_counter()
    for _ in range(int(math.ceil(script.duration / sim.dt))):
        pose, tag = script.pose_at(sim.t)
        sim.step_pose(pose, tag)
        show(sim.last_obs, sim.last_out)
        if keys["quit"]:
            break
        lag = sim.t - (time.perf_counter() - wall0)
        if lag > 0:
            time.sleep(lag)


def run_webcam(args, cfg) -> int:
    import cv2
    from perception import Perception
    from sim3d.avatar import lift

    perception = Perception(pose_model_path=args.pose_model, body_model_path=args.body_model,
                            far_field=args.far_field, conf=args.conf)
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"cannot open camera {args.webcam}")
        return 2
    person = Person(position=(-args.person_distance, 0.0, 0.0))
    sim = Sim3D(cfg=cfg, noise="none", seed=args.seed, gui=not args.no_gui, person=person, wind=args.wind)
    keys = {"quit": False, "estop": False}
    show = hud_window(sim, keys)
    wall0 = time.perf_counter()
    try:
        while not keys["quit"]:
            ok, frame = cap.read()
            if not ok:
                break
            obs = perception.perceive(frame, time.perf_counter() - wall0)
            body_w, body_conf, hand_w = lift(obs, person, args.webcam_distance, cfg)
            sim.step(body_w, hand_w, body_conf=body_conf, estop=keys["estop"])
            show(sim.last_obs, sim.last_out)
            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            cv2.imshow("webcam", small)
    finally:
        cap.release()
        summary = sim.summary()
        sim.close()
    print("frames {frames}, follow exits {follow_exits}, closest to person {min_dist:.2f} m".format(**summary))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCRIPTS),
                    help="default: hand (scripted), axes (experimental webcam avatar)")
    ap.add_argument("--webcam", type=int, default=None, help="drive the 3D person from this webcam")
    ap.add_argument("--noise", choices=("none", "webcam", "harsh"), default="webcam")
    ap.add_argument("--repeat", type=int, default=1, help="scripted runs with consecutive seeds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--wind", type=float, default=0.0)
    ap.add_argument("--person-distance", type=float, default=None,
                    help="distance from person to take-off point in metres")
    ap.add_argument("--webcam-distance", type=float, default=0.7, help="you to your webcam, m")
    ap.add_argument("--config", default=None)
    ap.add_argument("--pose-model", default=None)
    ap.add_argument("--body-model", default=None)
    ap.add_argument("--far-field", choices=("auto", "always", "off"), default="auto")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args(argv)
    if args.scenario is None:
        args.scenario = "hand" if args.webcam is None else "axes"
    if args.webcam is not None and args.scenario == "hand":
        ap.error("use demo/fly_sim.py --camera INDEX for hand-only RGB input")
    cfg = load_config(args.config)
    if args.person_distance is None:
        args.person_distance = DISTANCE[args.scenario] if args.webcam is None else 1.6
    if args.webcam is not None:
        return run_webcam(args, cfg)
    return run_scripted(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
