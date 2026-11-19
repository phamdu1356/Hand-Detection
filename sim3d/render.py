"""Drawing: the person as a stick figure in the PyBullet window, and the drone camera view."""
from __future__ import annotations

import numpy as np

BODY_BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
              (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
_TRAIL_MAX = 300


class SceneView:
    """Stick figure, camera ray and drone trail drawn as debug lines in the simulator window."""

    def __init__(self, client, camera):
        import pybullet as p
        self.pb, self.p, self.camera = client, p, camera
        pb = client
        pb.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        pb.resetDebugVisualizerCamera(cameraDistance=4.2, cameraYaw=55, cameraPitch=-22,
                                      cameraTargetPosition=[-0.7, 0.0, 1.0])
        head = pb.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0.95, 0.8, 0.65, 1])
        self.head = pb.createMultiBody(baseMass=0, baseVisualShapeIndex=head, basePosition=[0, 0, -5])
        self.ids: dict = {}
        self.trail: list = []
        self.last_pos = None
        self.frame = 0

    def _line(self, key, a, b, color, width):
        old = self.ids.get(key, -1)
        self.ids[key] = self.pb.addUserDebugLine(list(a), list(b), color, lineWidth=width,
                                                 replaceItemUniqueId=old)

    def draw(self, body_w, hand_w, state, obs, out):
        for k, (i, j) in enumerate(BODY_BONES):
            self._line(("b", k), body_w[i], body_w[j], [0.2, 0.2, 0.25], 4)
        neck = 0.5 * (body_w[5] + body_w[6])
        self._line("neck", neck, body_w[0], [0.2, 0.2, 0.25], 4)
        self.pb.resetBasePositionAndOrientation(self.head, list(0.5 * (body_w[3] + body_w[4]) + [0, 0, 0.03]),
                                                [0, 0, 0, 1])
        for k, (i, j) in enumerate(HAND_BONES):
            if hand_w is None:
                self._line(("h", k), [0, 0, -5], [0, 0, -5], [0, 0, 0], 1)
            else:
                seen = obs.hand is not None
                self._line(("h", k), hand_w[i], hand_w[j], [0.1, 0.7, 0.1] if seen else [0.85, 0.1, 0.1], 2)
        c, fwd, _, _ = self.camera.pose(state.pos, state.yaw)
        self._line("ray", c, c + 0.6 * fwd, [1.0, 0.75, 0.0], 3)
        self.frame += 1
        pos = np.asarray(state.pos, dtype=float)
        if self.last_pos is not None and self.frame % 3 == 0:
            self.trail.append(self.pb.addUserDebugLine(list(self.last_pos), list(pos), [0.2, 0.4, 1.0], 2))
            if len(self.trail) > _TRAIL_MAX:
                self.pb.removeUserDebugItem(self.trail.pop(0))
        if self.last_pos is None or self.frame % 3 == 0:
            self.last_pos = pos

    def close(self):
        pass


def drone_view(obs, out, camera, scale: float = 0.75):
    """What the controller sees from the drone camera, with a small HUD. Returns a BGR image."""
    import cv2
    w, h = camera.width, camera.height
    img = np.full((h, w, 3), 40, np.uint8)
    if obs.body is not None:
        k, cf = obs.body.kps, obs.body.kps_conf
        for i, j in BODY_BONES:
            if cf[i] > 0 and cf[j] > 0:
                cv2.line(img, tuple(int(v) for v in k[i]), tuple(int(v) for v in k[j]), (200, 200, 200), 2)
    if obs.hand is not None:
        k = obs.hand.kps
        for i, j in HAND_BONES:
            cv2.line(img, tuple(int(v) for v in k[i]), tuple(int(v) for v in k[j]), (80, 230, 80), 2)
    c = out.cmd
    lines = [
        f"mode {out.mode.value}   range {out.range_mode}   gesture {out.debug.get('label')}",
        f"cmd fwd {c.forward:+.2f}  right {c.right:+.2f}  up {c.up:+.2f}  yaw {np.degrees(c.yaw_rate):+.0f} deg/s",
        "hand: " + ("seen" if obs.hand is not None else "LOST") + "   person: "
        + ("seen" if obs.body is not None else "LOST"),
    ]
    for n, s in enumerate(lines):
        cv2.putText(img, s, (12, 28 + 26 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
    return cv2.resize(img, (int(w * scale), int(h * scale)))
