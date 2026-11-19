"""3D direction check with the real models, no webcam needed.

Real hands are cut out of the val images (gesture read by the hand pose model itself), then
moved, scaled and turned on a camera frame the way a user would move their hand in front of
a raw (not mirrored) webcam. The frames go through Perception -> Controller -> PyBullet quad,
and the drone's displacement is checked on every axis.

Two scenes:
  plain    grey background, no person: the controller uses the hand fallback frame
  person   ultralytics' zidane.jpg, hand next to the right-hand person: shoulder frame.
           Hand and body keypoints still come from the models, but the gesture label is
           taken from the script: on this busy background the rule-based thumb test cannot
           tell fist from like reliably, which is a classifier problem, not an axis problem.

Usage:
  .venv/bin/python tools/test_3d_real.py --pose-model hand_pose.pt --body-model yolo11n-pose.pt \
      --val ../datasets/hand-keypoints/images/val [--repo .] [--log-dir logs/3d]
Exit code 0 when every axis moves the right way in every scene.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np

FPS = 30.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--pose-model", required=True)
    ap.add_argument("--body-model", required=True)
    ap.add_argument("--val", required=True, help="folder of val images with real hands")
    ap.add_argument("--scenes", default="plain,person")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--save-frames", default=None, help="write one frame per segment here")
    return ap.parse_args(argv)


# ---------------------------------------------------------------- sprites

def pick_sprites(pose_model_path, val_dir, analyze, wanted=("palm", "fist", "like"), max_images=2500,
                 validate=None, per_gesture=40):
    """One clear, upright, single-hand crop per gesture: (crop_bgr, palm_xy_in_crop, hand_len_px, path).

    validate(sprite, gesture) -> bool, when given, rejects crops whose gesture is not read the
    same way after the moves the test makes (the thumb rule is close to its threshold on some).
    """
    from ultralytics import YOLO
    model = YOLO(pose_model_path)
    cands = {g: [] for g in wanted}
    for path in sorted(glob.glob(os.path.join(val_dir, "*.jpg")))[:max_images]:
        img = cv2.imread(path)
        r = model.predict(img, conf=0.6, verbose=False)[0]
        if r.boxes is None or len(r.boxes) != 1:
            continue
        kps = r.keypoints.xy.cpu().numpy()[0]
        kc = r.keypoints.conf.cpu().numpy()[0]
        if kc.min() < 0.6:
            continue
        g = analyze(kps)[0].lower()
        if g not in wanted:
            continue
        up = kps[0] - kps[9]                       # wrist below the middle MCP: hand upright
        if up[1] < 0.8 * np.linalg.norm(up):
            continue
        x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[0]
        size = max(x2 - x1, y2 - y1)
        m = 0.25 * size
        h, w = img.shape[:2]
        cx1, cy1 = int(max(0, x1 - m)), int(max(0, y1 - m))
        cx2, cy2 = int(min(w, x2 + m)), int(min(h, y2 + m))
        crop = img[cy1:cy2, cx1:cx2].copy()
        palm = kps[[0, 5, 9, 13, 17]].mean(axis=0) - (cx1, cy1)
        score = float(r.boxes.conf[0]) * min(size, 300.0)
        cands[g].append((score, (crop, palm, float(np.linalg.norm(kps[9] - kps[0])), path)))
    best = {}
    for g in wanted:
        for _score, sprite in sorted(cands[g], key=lambda c: -c[0])[:per_gesture]:
            if validate is None or validate(sprite, g):
                best[g] = sprite
                break
    missing = [g for g in wanted if g not in best]
    if missing:
        raise RuntimeError(f"no clean sprite for {missing} in {val_dir}")
    return best


def paste(canvas, sprite, centre, scale, rot_user_cw_deg):
    """Paste a hand so its palm centre lands on ``centre``.

    rot_user_cw_deg turns the hand clockwise as the USER sees it. On a raw camera that is
    counter-clockwise on screen, which is what cv2's positive angle means.
    """
    crop, palm, _len, _path = sprite
    M = cv2.getRotationMatrix2D((float(palm[0]), float(palm[1])), rot_user_cw_deg, scale)
    M[:, 2] += np.asarray(centre, dtype=float) - palm
    h, w = canvas.shape[:2]
    warped = cv2.warpAffine(crop, M, (w, h))
    mask = cv2.warpAffine(np.full(crop.shape[:2], 255, np.uint8), M, (w, h))
    out = canvas.copy()
    out[mask > 0] = warped[mask > 0]
    return out


# ---------------------------------------------------------------- scenario

# (name, offset in hand lengths along the USER's right / up, scale factor, user-clockwise deg, check)
# On a raw camera the user's right is the image LEFT, so right=+0.6 moves the hand to smaller u.
AXES = [
    ("hand to user's right", (+0.6, 0.0), 1.0, 0.0, ("right", +1)),
    ("hand to user's left", (-0.6, 0.0), 1.0, 0.0, ("right", -1)),
    ("hand up", (0.0, +0.6), 1.0, 0.0, ("up", +1)),
    ("hand down", (0.0, -0.6), 1.0, 0.0, ("up", -1)),
    ("hand closer (bigger)", (0.0, 0.0), 1.45, 0.0, ("forward", +1)),
    ("hand farther (smaller)", (0.0, 0.0), 0.7, 0.0, ("forward", -1)),
    ("wrist turned clockwise", (0.0, 0.0), 1.0, 25.0, ("yaw_right", +1)),
]


def segments():
    """Yield (gesture, seconds, start_pose, end_pose, tag). pose = (right, up, scale, rot)."""
    rest = (0.0, 0.0, 1.0, 0.0)
    yield "palm", 1.0, rest, rest, "idle"
    yield "like", 1.6, rest, rest, "takeoff"
    yield "palm", 4.0, rest, rest, "climb"
    for name, (dr, du), sc, rot, _check in AXES:
        tgt = (dr, du, sc, rot)
        yield "fist", 0.7, rest, rest, f"grab:{name}"
        yield "fist", 0.4, rest, tgt, f"move:{name}"
        yield "fist", 1.5, tgt, tgt, f"hold:{name}"
        yield "palm", 2.0, tgt, tgt, f"stop:{name}"
        yield "palm", 0.5, rest, rest, f"reset:{name}"


class SpriteSource:
    """Frames from the scripted hand, fed through the real Perception."""

    def __init__(self, perception, sprites, background, base_px, hand_len_px, save_dir=None,
                 scripted_gestures=False, offset_gain=1.0):
        self.offset_gain = offset_gain
        self.scripted_gestures = scripted_gestures
        self.perception = perception
        self.sprites = sprites
        self.bg = background
        self.base = np.asarray(base_px, dtype=float)
        self.hand_len = float(hand_len_px)
        self.frames = [(g, int(round(sec * FPS)), a, b, tag) for g, sec, a, b, tag in segments()]
        self.i_seg, self.i_frame, self.n = 0, 0, 0
        self.tags = []          # tag per produced frame
        self.save_dir = save_dir
        self.frame = None

    def next_obs(self, _last_out):
        while self.i_seg < len(self.frames) and self.i_frame >= self.frames[self.i_seg][1]:
            self.i_seg += 1
            self.i_frame = 0
        if self.i_seg >= len(self.frames):
            return None
        g, n, a, b, tag = self.frames[self.i_seg]
        k = self.i_frame / max(1, n - 1)
        pose = [a[j] + (b[j] - a[j]) * k for j in range(4)]
        sprite = self.sprites[g]
        scale0 = self.hand_len / sprite[2]
        centre = self.base + np.array([-pose[0], -pose[1]]) * self.hand_len * self.offset_gain  # user's right = image left
        frame = paste(self.bg, sprite, centre, scale0 * pose[2], pose[3])
        if self.save_dir and self.i_frame == n - 1:
            cv2.imwrite(os.path.join(self.save_dir, f"{self.n:05d}_{tag.replace(' ', '_').replace(':', '-')}.jpg"), frame)
        self.frame = frame
        self.i_frame += 1
        t = self.n / FPS
        self.n += 1
        self.tags.append(tag)
        obs = self.perception.perceive(frame, t)
        if self.scripted_gestures and obs.hand is not None:
            obs.hand.gesture_raw = g
        return obs


def evaluate(tags, states, outs):
    """Per axis: the mean command and the drone displacement over the move and hold frames.

    An axis passes when the mean command has the expected sign and is at least CMD_MIN, and
    the displacement (user frame at the start of the move) has the expected sign too. Size is
    not judged: the expo curve keeps small hand moves gentle by design.
    """
    rows = []
    for name, *_rest, (axis, sign) in AXES:
        idx = [i for i, tg in enumerate(tags) if tg in (f"move:{name}", f"hold:{name}")]
        modes = {outs[i].mode.value for i in idx}
        if not idx:
            rows.append((name, axis, sign, float("nan"), float("nan"), modes, False))
            continue
        field = {"forward": "forward", "right": "right", "up": "up", "yaw_right": "yaw_rate"}[axis]
        cmd = float(np.mean([getattr(outs[i].cmd, field) for i in idx]))
        if axis == "yaw_right":
            cmd = -cmd                     # FLU yaw rate is positive to the left
        s0, s1 = states[idx[0]], states[idx[-1]]
        d = s1.pos - s0.pos
        yaw0 = s0.yaw
        fwd = d[0] * math.cos(yaw0) + d[1] * math.sin(yaw0)       # FLU world -> heading at start
        left = -d[0] * math.sin(yaw0) + d[1] * math.cos(yaw0)
        dyaw = math.atan2(math.sin(s1.yaw - s0.yaw), math.cos(s1.yaw - s0.yaw))
        val = {"forward": fwd, "right": -left, "up": d[2], "yaw_right": -math.degrees(dyaw)}[axis]
        cmd_min = math.radians(2.0) if axis == "yaw_right" else CMD_MIN
        ok = sign * cmd >= cmd_min and sign * val > 0.0
        rows.append((name, axis, sign, cmd, val, modes, ok))
    return rows


CMD_MIN = 0.05   # m/s


def run_scene(args, scene, sprites):
    sys.path[:0] = [args.repo, os.path.join(args.repo, "demo")]
    from dronectl.backends.pybullet_sim import PyBulletSimBackend
    from dronectl.config import ControlConfig
    from dronectl.pipeline import Controller
    from fly_sim import CsvLog, build_row
    from perception import Perception

    import dronectl.pipeline as pipeline_mod
    from dronectl.gestures import near_gesture
    if SCRIPTED_GESTURES[scene]:
        # The controller reads the label from the keypoints; use the scripted one instead,
        # keeping the confidence check.
        def scripted(hand, cfg=None):
            return hand.gesture_raw if near_gesture(hand, cfg) != "none" else "none"
        pipeline_mod.near_gesture = scripted
    else:
        pipeline_mod.near_gesture = near_gesture
    perc = Perception(pose_model_path=args.pose_model, body_model_path=args.body_model)
    bg, base, hand_len = scene_setup(scene)
    save = None
    if args.save_frames:
        save = os.path.join(args.save_frames, scene)
        os.makedirs(save, exist_ok=True)
    src = SpriteSource(perc, sprites, bg, base, hand_len, save, SCRIPTED_GESTURES[scene], OFFSET_GAIN[scene])
    cfg = ControlConfig()
    ctl = Controller(cfg)
    backend = PyBulletSimBackend(gui=False)
    log = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log = CsvLog(os.path.join(args.log_dir, f"3d_{scene}.csv"))
    states, outs, ranges, frames_kind = [], [], [], []
    prev_t = None
    while True:
        obs = src.next_obs(None)
        if obs is None:
            break
        st = backend.state()
        out = ctl.update(obs, st)
        if out.action == "takeoff":
            backend.takeoff(cfg.takeoff_altitude)
        elif out.action == "land":
            backend.land()
        backend.send_velocity(out.cmd)
        dt = 1.0 / FPS if prev_t is None else obs.t - prev_t
        prev_t = obs.t
        backend.step(dt)
        st = backend.state()
        states.append(st)
        outs.append(out)
        ranges.append(out.range_mode)
        frames_kind.append(ctl.joy.origin.kind if ctl.joy.origin is not None else "-")
        if log:
            log.write(build_row(obs, out, st, ctl, cfg, obs.latency_ms))
    if log:
        log.close()
    backend.close()
    rows = evaluate(src.tags, states, outs)
    took_off = any(o.mode.value != "LANDED" for o in outs)
    kinds = sorted({k for k in frames_kind if k != "-"})
    return rows, took_off, kinds


def scene_setup(scene):
    """(background, palm base pixel, hand length px) for a scene."""
    if scene == "plain":
        return np.full((720, 1280, 3), 110, np.uint8), (640.0, 380.0), 110.0
    import ultralytics
    bg = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))
    # The standing person on the right has shoulders near u=854..1048, v~300 (S ~ 195 px).
    # The hand is held up at his right side (image left of his face), in the free space above
    # the other man's head, so he is the closest person and the range is near (q >= 0.35).
    return bg, (790.0, 150.0), 90.0


# Offsets are in hand lengths. In the person scene they are scaled so a move is about
# 0.45 shoulder widths, a normal reach for a real user.
OFFSET_GAIN = {"plain": 1.0, "person": 1.6}


SCRIPTED_GESTURES = {"plain": False, "person": True}


def make_validator(args, scene):
    """Check a sprite is found (and, unless gestures are scripted, read as its gesture) at
    every pose the scenario uses, in that scene."""
    check_gesture = not SCRIPTED_GESTURES[scene]
    sys.path[:0] = [args.repo, os.path.join(args.repo, "demo")]
    from test_model import OptimalFlycamController
    ctl = OptimalFlycamController(args.repo, None, pose_model_path=args.pose_model,
                                  far_field_mode="off", use_temporal_roi=False)
    bg, base, hand_len = scene_setup(scene)
    # end poses and the in-between poses of each move (the label must hold all the way)
    poses = [(0.0, 0.0, 1.0, 0.0)] + [
        (k * dr, k * du, 1.0 + k * (sc - 1.0), k * rot)
        for _n, (dr, du), sc, rot, _c in AXES for k in (0.33, 0.67, 1.0)]

    def ok(sprite, gesture):
        for _ in (0,):
            for dr, du, sc, rot in poses:
                centre = np.asarray(base) + np.array([-dr, -du]) * hand_len * OFFSET_GAIN[scene]
                frame = paste(bg, sprite, centre, hand_len / sprite[2] * sc, rot)
                ctl.ema_box = None
                ctl.wrist_history.clear()
                ctl.gesture_window.clear()
                ctl.process_frame(frame, conf_thres=0.5)
                r = ctl.last_result or {}
                if not r.get("detected"):
                    return False
                if check_gesture and str(r.get("raw_gesture", "")).lower() != gesture:
                    return False
        return True
    return ok


def main(argv=None):
    args = parse_args(argv)
    sys.path[:0] = [args.repo]
    from dronectl.hand_rules import analyze_hand_keypoints
    all_ok = True
    for scene in args.scenes.split(","):
        sprites = pick_sprites(args.pose_model, args.val, analyze_hand_keypoints,
                               validate=make_validator(args, scene))
        for g, sp in sprites.items():
            print(f"[{scene}] sprite {g}: {os.path.basename(sp[3])}")
        rows, took_off, kinds = run_scene(args, scene, sprites)
        how = "scripted labels" if SCRIPTED_GESTURES[scene] else "model + rules"
        print(f"\n== scene {scene}: gestures from {how}, took off={took_off}, joystick frame={kinds}")
        print(f"{'hand motion':26s} {'drone axis':10s} {'expect':6s} {'mean cmd':>10s} {'moved':>10s}  modes          result")
        for name, axis, sign, cmd, val, modes, ok in rows:
            cu, mu = ("deg/s", "deg") if axis == "yaw_right" else ("m/s", "m")
            cshow = math.degrees(cmd) if axis == "yaw_right" else cmd
            print(f"{name:26s} {axis:10s} {'+' if sign > 0 else '-':6s} {cshow:+6.2f}{cu:5s} {val:+7.2f}{mu:3s} "
                  f" {','.join(sorted(modes)):14s} {'OK' if ok else 'WRONG'}")
            all_ok &= ok
        all_ok &= took_off
    print("\nALL AXES CORRECT" if all_ok else "\nSOME AXES WRONG")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
