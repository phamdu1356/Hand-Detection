"""Scripted person timelines and the per-axis check of how the drone moved.

A Script is a list of Steps. Each step holds a hand shape and arm pose for ``dur`` seconds;
the palm position and wrist roll move linearly from the previous step's values during the
first ``move_s`` seconds. Steps with a tag mark the window an axis is judged on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .person import PersonPose

REST_PALM = (0.10, -0.10, 0.34)    # in front of the right chest, arm half extended
TAKEOFF_PALM = (0.10, -0.30, 0.45)  # reaching down toward the drone on the ground


@dataclass
class Step:
    dur: float
    gesture: str = "one"
    palm: tuple | None = None        # None keeps the previous palm position
    roll_deg: float | None = None
    move_s: float = 0.0
    left_arm: str = "down"
    right_arm: str = "reach"
    show_hand: bool = True
    tag: str = ""


class Script:
    def __init__(self, steps):
        self.steps = list(steps)
        self._starts = []
        t, palm, roll = 0.0, np.array(REST_PALM), 0.0
        self._from = []
        for s in self.steps:
            self._starts.append(t)
            end_palm = np.array(s.palm) if s.palm is not None else palm
            end_roll = s.roll_deg if s.roll_deg is not None else roll
            self._from.append((palm, roll, end_palm, end_roll))
            palm, roll, t = end_palm, end_roll, t + s.dur
        self.duration = t

    def pose_at(self, t: float):
        """(PersonPose, tag) at time t. Past the end the last step holds."""
        i = len(self.steps) - 1
        for k, st in enumerate(self._starts):
            if t < st + self.steps[k].dur:
                i = k
                break
        s = self.steps[i]
        p0, r0, p1, r1 = self._from[i]
        a = 1.0 if s.move_s <= 0 else min(1.0, max(0.0, (t - self._starts[i]) / s.move_s))
        palm = p0 + a * (p1 - p0)
        roll = r0 + a * (r1 - r0)
        pose = PersonPose(left_arm=s.left_arm, right_arm=s.right_arm, palm=tuple(palm),
                          gesture=s.gesture, roll_deg=float(roll), show_hand=s.show_hand)
        return pose, s.tag


# axis tag -> (description, user-frame axis, expected sign, hand offset or roll)
AXES = {
    "right": ("hand to user's right", "right", +1, (0.20, 0.0, 0.0), 0.0),
    "left": ("hand to user's left", "right", -1, (-0.20, 0.0, 0.0), 0.0),
    "up": ("hand up", "up", +1, (0.0, 0.20, 0.0), 0.0),
    "down": ("hand down", "up", -1, (0.0, -0.18, 0.0), 0.0),
    "closer": ("hand pushed toward drone", "forward", +1, (0.0, 0.0, 0.15), 0.0),
    "farther": ("hand pulled back", "forward", -1, (0.0, 0.0, -0.18), 0.0),
    "yaw": ("wrist turned clockwise", "yaw_right", +1, (0.0, 0.0, 0.0), 25.0),
}


def _takeoff_near():
    return [Step(0.5, "like", palm=TAKEOFF_PALM, move_s=0.4), Step(1.3, "like"),
            Step(4.5, "one", palm=REST_PALM, move_s=1.0)]


def axes_script(axes=tuple(AXES)) -> Script:
    steps = _takeoff_near()
    for name in axes:
        _, _, _, off, roll = AXES[name]
        target = tuple(np.add(REST_PALM, off))
        steps += [
            Step(0.7, "fist", palm=REST_PALM, roll_deg=0.0),
            Step(2.0, "fist", palm=target, roll_deg=roll, move_s=0.5, tag=name),
            Step(1.5, "palm"),
            Step(0.6, "palm", palm=REST_PALM, roll_deg=0.0, move_s=0.4),
        ]
    steps += [Step(1.5, "ok"), Step(5.0, "one")]
    return Script(steps)


def far_script() -> Script:
    hidden = dict(show_hand=False)
    return Script([
        Step(2.0, "one", right_arm="up", left_arm="up", **hidden),
        Step(4.5, "one", right_arm="down", **hidden),
        Step(2.5, "one", right_arm="point_right", tag="point_right", **hidden),
        Step(1.0, "one", right_arm="t", left_arm="t", **hidden),
        Step(1.0, "one", right_arm="down", **hidden),
        Step(2.5, "one", right_arm="point_up", tag="point_up", **hidden),
        Step(1.0, "one", right_arm="t", left_arm="t", **hidden),
        Step(2.0, "one", right_arm="up", left_arm="up", **hidden),
        Step(8.0, "one", right_arm="down", **hidden),
    ])


SCRIPTS = {"axes": axes_script, "far": far_script}
# person to drone take-off point, m: near control needs the hand close, arm poses need the whole
# upper body in view from the ground
DISTANCE = {"axes": 1.6, "far": 3.0}
FAR_EXPECT = {"point_right": ("point right", "right", +1), "point_up": ("point up", "up", +1)}

CMD_MIN = {"forward": 0.05, "right": 0.05, "up": 0.05, "yaw_right": math.radians(2.0)}


def user_delta(a, b) -> dict:
    """Drone motion from record a to record b in the user frame."""
    d = np.asarray(b["pos"]) - np.asarray(a["pos"])
    return {"forward": float(d[0]), "right": float(-d[1]), "up": float(d[2]),
            "yaw_right": float(-(b["yaw"] - a["yaw"]))}


def user_cmd(rec) -> dict:
    f, r, u, y = rec["cmd"]
    return {"forward": f, "right": r, "up": u, "yaw_right": -y}


def evaluate(records, expect) -> list[dict]:
    """One row per tagged window: mean command and displacement on the expected axis."""
    rows = []
    for tag, (desc, axis, sign) in expect.items():
        win = [r for r in records if r["tag"] == tag]
        if not win:
            rows.append(dict(tag=tag, desc=desc, axis=axis, sign=sign, ok=False, cmd=0.0, moved=0.0,
                             modes="never shown"))
            continue
        cmd = float(np.mean([user_cmd(r)[axis] for r in win]))
        moved = user_delta(win[0], win[-1])[axis]
        modes = sorted({r["mode"] for r in win})
        ok = sign * cmd >= CMD_MIN[axis] and sign * moved > 0
        rows.append(dict(tag=tag, desc=desc, axis=axis, sign=sign, ok=ok, cmd=cmd, moved=moved,
                         modes="/".join(modes)))
    return rows


def expectations(name: str) -> dict:
    if name == "axes":
        return {k: (v[0], v[1], v[2]) for k, v in AXES.items()}
    if name == "far":
        return FAR_EXPECT
    raise ValueError(name)
