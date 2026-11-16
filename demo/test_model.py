"""Hệ thống điều khiển Flycam bằng tay Offline Tối Ưu (1 Model YOLO11n-Pose làm 3 việc + 2-Stage Adaptive ROI Zoom cho tầm xa 10m).

Kiến trúc xử lý:
  1. Định vị 3D & Khoảng cách: Tâm lòng bàn tay + độ dài xương bàn tay |kp9 - kp0| (không bị sai khi nắm tay).
  2. Cử chỉ tĩnh & Góc chỉ 360°: Đếm khớp ngón tay (palm, fist, ok, one, peace_inverted, three, four, like) + Debounce.
  3. Cử chỉ động (Vẫy tay): Tính vector vận tốc cổ tay kp0 qua 10 frame (SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN).
  4. Tầm xa 10m (2-Stage Adaptive ROI Zoom):
     - Stage 1 (ROI Locator): Khi full-frame confidence thấp hoặc ảnh độ phân giải cao (>= 1280px) / chế độ 10m,
       dùng yolo11n-pose.pt (COCO kp[9], kp[10] cổ tay + khuỷu tay/vai) hoặc lưới tile thích ứng trên ảnh gốc
       để cắt cửa sổ vuông quanh bàn tay mà không bị mất nét do resize.
     - Stage 2 (Hand Pose Zoom): Chạy hand_pose.pt / hand_pose.onnx (imgsz=640) trên vùng ROI đã cắt và
       chiếu ngược hộp [x, y, w, h] cùng 21 khớp (kx, ky) về tọa độ khung hình gốc.
  * Tự động fallback về hand_localization_model.h5 nếu chưa có hand_pose.pt / hand_pose.onnx.

Cách dùng:
  python test_model.py --image anh.jpg                  # chạy thử trên ảnh tĩnh (auto 2m/10m)
  python test_model.py --image anh_10m.jpg --far-field always
  python test_model.py --webcam                         # chạy trực tiếp từ camera (bấm 'c' chụp ảnh sắc nét, 'q' thoát)
"""
import argparse
from collections import Counter, deque
import math
import os
import time

import cv2
import numpy as np

IMG_SIZE = 224
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]
GESTURE_TO_CMD = {
    "palm": "FOLLOW_PALM (Bam theo tay 3D)",
    "fist": "HOVER (Dung yen / Phanh)",
    "ok": "LAND (Ha canh)",
    "one": "POINT_MOVE (Bay theo huong chi)",
    "peace_inverted": "BACKWARD (Bay lui)",
    "three": "UP (Bay len)",
    "four": "YAW_LEFT (Xoay trai)",
    "like": "TAKEOFF (Cat canh)",
}


def load_cnn_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def predict_box(model, frame_bgr):
    """Dự đoán hộp (x, y, w, h) theo pixel từ mô hình CNN gốc (fallback)."""
    import tensorflow as tf
    h_img, w_img = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = tf.image.resize(rgb, [IMG_SIZE, IMG_SIZE])[tf.newaxis, ...]
    bx, by, bw, bh = model(x, training=False).numpy()[0]
    return bx * w_img, by * h_img, bw * w_img, bh * h_img


def _is_valid_kp(pt):
    return (
        pt is not None
        and len(pt) >= 2
        and math.isfinite(float(pt[0]))
        and math.isfinite(float(pt[1]))
        and (float(pt[0]) > 0.0 or float(pt[1]) > 0.0)
    )


def analyze_hand_keypoints(kps):
    """Từ 21 khớp (x, y), trích xuất đồng thời: cử chỉ tĩnh, góc chỉ ngón trỏ và độ dài xương bàn tay."""
    kps = np.asarray(kps, dtype=np.float32)
    if kps.shape[0] < 21 or not _is_valid_kp(kps[0]) or not _is_valid_kp(kps[9]):
        return "UNKNOWN", 0.0, 0.0

    wrist = kps[0]
    middle_mcp = kps[9]
    palm_bone_len = float(np.linalg.norm(middle_mcp - wrist))
    if palm_bone_len < 1e-3:
        return "UNKNOWN", 0.0, 0.0

    # Kiểm tra trạng thái mở của 4 ngón dài: trỏ (8,6), giữa (12,10), áp út (16,14), út (20,18)
    fingers = []
    for tip_idx, pip_idx in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if _is_valid_kp(kps[tip_idx]) and _is_valid_kp(kps[pip_idx]):
            is_open = np.linalg.norm(kps[tip_idx] - wrist) > np.linalg.norm(kps[pip_idx] - wrist) * 1.15
            fingers.append(bool(is_open))
        else:
            fingers.append(False)

    if _is_valid_kp(kps[4]) and _is_valid_kp(kps[2]) and _is_valid_kp(kps[17]):
        thumb_open = bool(np.linalg.norm(kps[4] - kps[17]) > np.linalg.norm(kps[2] - kps[17]) * 1.18)
    else:
        thumb_open = False

    if _is_valid_kp(kps[4]) and _is_valid_kp(kps[8]):
        thumb_index_dist = float(np.linalg.norm(kps[4] - kps[8])) / palm_bone_len
    else:
        thumb_index_dist = 999.0

    # Góc chỉ của ngón trỏ (kp5 -> kp8) trong mặt phẳng camera (-180..180 độ)
    if _is_valid_kp(kps[5]) and _is_valid_kp(kps[8]):
        dx, dy = float(kps[8][0] - kps[5][0]), float(kps[5][1] - kps[8][1])
        point_angle = math.degrees(math.atan2(dy, dx))
    else:
        point_angle = 0.0

    count_4 = sum(fingers)
    if thumb_index_dist < 0.42 and sum(fingers[1:]) >= 2:
        gesture = "ok"
    elif count_4 == 0 and not thumb_open:
        gesture = "fist"
    elif count_4 == 0 and thumb_open:
        gesture = "like"
    elif fingers[0] and count_4 == 1:
        gesture = "one"
    elif fingers[0] and fingers[1] and count_4 == 2:
        gesture = "peace_inverted"
    elif count_4 == 3:
        gesture = "three"
    elif count_4 == 4 and not thumb_open:
        gesture = "four"
    elif count_4 == 4 and thumb_open:
        gesture = "palm"
    else:
        gesture = "palm" if count_4 >= 3 else "fist"

    return gesture, point_angle, palm_bone_len


def detect_swipe_from_wrist(wrist_history, min_disp=0.14):
    """Phát hiện cử chỉ vẫy tay động từ lịch sử tọa độ chuẩn hóa của khớp cổ tay kp0."""
    if len(wrist_history) < 8:
        return "STAND_STILL"
    dx = wrist_history[-1][0] - wrist_history[0][0]
    dy = wrist_history[-1][1] - wrist_history[0][1]
    if abs(dx) < min_disp and abs(dy) < min_disp:
        return "STAND_STILL"
    if abs(dx) > abs(dy) * 1.4:
        return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
    if abs(dy) > abs(dx) * 1.4:
        return "SWIPE_DOWN" if dy > 0 else "SWIPE_UP"
    return "STAND_STILL"


def compute_3d_velocity(palm_cx, palm_cy, palm_bone_len, w_img, h_img, deadzone=0.07):
    """Tính vận tốc 3D (Roll, Throttle, Pitch) dựa trên tâm bàn tay và độ dài xương lòng bàn tay."""
    err_x = (palm_cx / max(float(w_img), 1.0)) - 0.5
    err_y = 0.5 - (palm_cy / max(float(h_img), 1.0))
    norm_bone = palm_bone_len / max(float(h_img), 1.0)
    target_bone = 0.12  # Độ dài xương chuẩn khi đứng cách Flycam ~2m

    vx = 0.0 if abs(err_x) < deadzone else float(np.clip(err_x * 2.2, -1.0, 1.0))
    vy = 0.0 if abs(err_y) < deadzone else float(np.clip(err_y * 2.2, -1.0, 1.0))
    vz = 0.0 if abs(target_bone - norm_bone) < 0.015 else float(np.clip((target_bone - norm_bone) * 6.0, -1.0, 1.0))
    return vx, vy, vz


def extract_square_roi(frame, x1, y1, crop_size, pad_value=114):
    """Cắt cửa sổ vuông (crop_size x crop_size) theo pixel ảnh gốc, tự động pad biên nếu tràn khung hình."""
    h_img, w_img = frame.shape[:2]
    crop_size = max(int(round(crop_size)), 16)
    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    crop = np.full((crop_size, crop_size, 3), pad_value, dtype=frame.dtype)
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(w_img, x2)
    src_y2 = min(h_img, y2)

    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    return crop, (x1, y1, crop_size)


def project_roi_detection_to_frame(box_xywh_crop, kps_xy_crop, kpt_conf_crop, origin_x, origin_y, w_img, h_img):
    """Chiếu hộp [cx, cy, w, h] và 21 khớp (kx, ky) từ tọa độ ROI về hệ tọa độ khung hình gốc."""
    cx_c, cy_c, bw_c, bh_c = (float(v) for v in box_xywh_crop)
    x1 = float(np.clip(cx_c - bw_c / 2.0 + origin_x, 0.0, float(w_img)))
    y1 = float(np.clip(cy_c - bh_c / 2.0 + origin_y, 0.0, float(h_img)))
    x2 = float(np.clip(cx_c + bw_c / 2.0 + origin_x, 0.0, float(w_img)))
    y2 = float(np.clip(cy_c + bh_c / 2.0 + origin_y, 0.0, float(h_img)))
    box_tlwh = (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))

    kps_arr = np.asarray(kps_xy_crop, dtype=np.float32)
    kps_full = np.zeros_like(kps_arr, dtype=np.float32)
    if kps_arr.size > 0:
        valid = (kps_arr[:, 0] > 0.0) | (kps_arr[:, 1] > 0.0)
        if kpt_conf_crop is not None:
            conf_arr = np.asarray(kpt_conf_crop, dtype=np.float32)
            if conf_arr.shape[0] == kps_arr.shape[0]:
                valid = valid & (conf_arr > 0.0)
        kps_full[valid, 0] = np.clip(kps_arr[valid, 0] + float(origin_x), 0.0, float(w_img))
        kps_full[valid, 1] = np.clip(kps_arr[valid, 1] + float(origin_y), 0.0, float(h_img))

    return box_tlwh, kps_full


def _box_iou_xywh_tl(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(aw * ah + bw * bh - inter, 1e-6)
    return inter / union


def build_pose_candidate_rois(
    person_boxes,
    body_kpts,
    body_kpt_confs,
    w_img,
    h_img,
    min_crop=128,
    kpt_conf_thres=0.20,
    person_confs=None,
):
    """Tạo danh sách ROI vuông ưu tiên quanh cổ tay kp[9], kp[10] và ngữ cảnh khuỷu tay/vai từ yolo11n-pose."""
    candidates = []
    max_side = max(int(w_img), int(h_img))
    n_persons = len(person_boxes)

    order = list(range(n_persons))
    if person_confs is not None and len(person_confs) == n_persons:
        order.sort(key=lambda idx: float(person_confs[idx]), reverse=True)

    for p_idx in order:
        px1, py1, px2, py2 = [float(v) for v in person_boxes[p_idx][:4]]
        pw = max(px2 - px1, 1.0)
        ph = max(py2 - py1, 1.0)
        p_conf = float(person_confs[p_idx]) if person_confs is not None and p_idx < len(person_confs) else 1.0
        kps = np.asarray(body_kpts[p_idx], dtype=np.float32)
        kconf = np.asarray(body_kpt_confs[p_idx], dtype=np.float32) if body_kpt_confs is not None else None

        person_added_arm = False
        arm_specs = (
            ("left_wrist", COCO_LEFT_SHOULDER, COCO_LEFT_ELBOW, COCO_LEFT_WRIST),
            ("right_wrist", COCO_RIGHT_SHOULDER, COCO_RIGHT_ELBOW, COCO_RIGHT_WRIST),
        )
        for tag, s_idx, e_idx, w_idx in arm_specs:
            if kps.shape[0] <= max(s_idx, e_idx, w_idx):
                continue

            w_pt = kps[w_idx]
            e_pt = kps[e_idx]
            s_pt = kps[s_idx]
            w_c = float(kconf[w_idx]) if kconf is not None else 1.0
            e_c = float(kconf[e_idx]) if kconf is not None else 1.0
            s_c = float(kconf[s_idx]) if kconf is not None else 1.0

            if _is_valid_kp(w_pt) and w_c >= kpt_conf_thres:
                if _is_valid_kp(e_pt) and e_c >= kpt_conf_thres and float(np.linalg.norm(w_pt - e_pt)) > 4.0:
                    vec = w_pt - e_pt
                    forearm_len = float(np.linalg.norm(vec))
                    hx, hy = w_pt + 0.28 * vec
                    crop_size = max(float(min_crop), 2.8 * forearm_len, 0.38 * ph)
                    priority = 1.0 + p_conf + w_c
                else:
                    hx, hy = float(w_pt[0]), float(w_pt[1])
                    crop_size = max(float(min_crop), 0.42 * ph)
                    priority = 0.8 + p_conf + w_c

                crop_size = int(round(min(crop_size, float(max_side))))
                rx = int(round(float(hx) - crop_size / 2.0))
                ry = int(round(float(hy) - crop_size / 2.0))
                candidates.append((rx, ry, crop_size, priority, tag))
                person_added_arm = True
            elif _is_valid_kp(e_pt) and _is_valid_kp(s_pt) and e_c >= kpt_conf_thres and s_c >= kpt_conf_thres:
                vec_upper = e_pt - s_pt
                upper_len = float(np.linalg.norm(vec_upper))
                if upper_len > 4.0:
                    hx, hy = e_pt + 0.9 * vec_upper
                    crop_size = int(round(min(max(float(min_crop), 2.8 * upper_len, 0.42 * ph), float(max_side))))
                    rx = int(round(float(hx) - crop_size / 2.0))
                    ry = int(round(float(hy) - crop_size / 2.0))
                    candidates.append((rx, ry, crop_size, 0.45 + p_conf, f"{tag}_extrap"))
                    person_added_arm = True

        if not person_added_arm:
            upper_h = 0.72 * ph
            crop_size = int(round(min(max(float(min_crop), max(pw, upper_h) * 0.9), float(max_side))))
            cx = px1 + pw * 0.5
            cy = py1 + upper_h * 0.5
            candidates.append(
                (int(round(cx - crop_size / 2.0)), int(round(cy - crop_size / 2.0)), crop_size, 0.25 + p_conf, "upper_body")
            )

    candidates.sort(key=lambda item: item[3], reverse=True)
    deduped = []
    for cand in candidates:
        cand_box = (cand[0], cand[1], cand[2], cand[2])
        if any(_box_iou_xywh_tl(cand_box, (k[0], k[1], k[2], k[2])) > 0.78 for k in deduped):
            continue
        deduped.append(cand)
    return deduped


def build_adaptive_tile_rois(w_img, h_img, tile_size=None, overlap=0.25):
    """Sinh lưới cửa sổ vuông giao nhau (SAHI-style) cho Stage 1B khi không phát hiện được khung người."""
    w_img = int(w_img)
    h_img = int(h_img)
    if tile_size is None:
        tile_size = max(256, min(448, int(round(w_img / 4.5))))
    tile_size = max(128, min(tile_size, max(w_img, h_img)))
    stride = max(64, int(round(tile_size * (1.0 - overlap))))

    def _axis_starts(total_len):
        if total_len <= tile_size:
            return [0]
        starts = list(range(0, max(1, total_len - tile_size + 1), stride))
        last = total_len - tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    xs = _axis_starts(w_img)
    ys = _axis_starts(h_img)
    rois = []
    cx_frame, cy_frame = w_img / 2.0, h_img / 2.0
    for y in ys:
        for x in xs:
            dist = math.hypot((x + tile_size / 2.0) - cx_frame, (y + tile_size / 2.0) - cy_frame)
            rois.append((x, y, tile_size, -dist, "tile"))
    rois.sort(key=lambda item: item[3], reverse=True)
    return rois


class OptimalFlycamController:
    """Bộ điều khiển Flycam tối ưu dùng 1 model YOLO11n-Pose + 2-Stage Adaptive ROI Zoom (fallback CNN nếu chưa có)."""

    def __init__(
        self,
        base_dir,
        cnn_path,
        pose_model_path=None,
        body_pose_model_path=None,
        far_field_mode="auto",
        high_res_width=1280,
        body_conf_thres=0.25,
        enable_tile_fallback=True,
        use_temporal_roi=True,
        preload_body_pose=False,
    ):
        self.base_dir = base_dir
        self.cnn_path = cnn_path
        self.pose_model = None
        self.pose_model_path = None
        self.pose_is_onnx = False
        self.body_pose_model = None
        self.body_pose_model_path = None
        self.body_is_onnx = False
        self._body_pose_unavailable = False
        self.cnn_model = None

        self.far_field_mode = far_field_mode
        self.high_res_width = int(high_res_width)
        self.body_conf_thres = float(body_conf_thres)
        self.enable_tile_fallback = bool(enable_tile_fallback)
        self.use_temporal_roi = bool(use_temporal_roi)

        self.wrist_history = deque(maxlen=10)
        self.gesture_window = deque(maxlen=5)
        self.ema_box = None
        self.last_roi = None
        self._last_was_roi = False
        self.last_result = None

        if pose_model_path is None and cnn_path and str(cnn_path).lower().endswith((".pt", ".onnx")) and os.path.exists(cnn_path):
            pose_model_path = cnn_path

        resolved_pose = None
        if pose_model_path and os.path.exists(pose_model_path):
            resolved_pose = pose_model_path
        else:
            demo_dir = os.path.dirname(os.path.abspath(__file__))
            for search_dir in (base_dir, demo_dir):
                for candidate in ("hand_pose.pt", "hand_pose.onnx"):
                    p = os.path.join(search_dir, candidate)
                    if os.path.exists(p):
                        resolved_pose = p
                        break
                if resolved_pose:
                    break

        if resolved_pose is not None:
            from ultralytics import YOLO
            self.pose_model = YOLO(resolved_pose, task="pose")
            self.pose_model_path = resolved_pose
            self.pose_is_onnx = resolved_pose.lower().endswith(".onnx")
            print(f"Đã nạp model tối ưu: {resolved_pose}")

        if self.pose_model is None and cnn_path and os.path.exists(cnn_path) and not str(cnn_path).lower().endswith((".pt", ".onnx")):
            self.cnn_model = load_cnn_model(cnn_path)
            print(f"Chưa có hand_pose.pt, dùng tạm model CNN: {cnn_path}")

        demo_dir = os.path.dirname(os.path.abspath(__file__))
        self._body_pose_candidates = []
        if body_pose_model_path:
            self._body_pose_candidates.append(body_pose_model_path)
        for search_dir in (base_dir, demo_dir):
            self._body_pose_candidates.append(os.path.join(search_dir, "yolo11n-pose.pt"))
        self._body_pose_candidates.append("yolo11n-pose.pt")

        has_local_body = any(os.path.exists(p) for p in self._body_pose_candidates[:-1])
        if preload_body_pose or (self.pose_model is not None and (far_field_mode == "always" or has_local_body)):
            self._ensure_body_pose_model()

    def reset_state(self):
        """Xóa trạng thái làm mượt theo thời gian (dùng khi chuyển sang ảnh độc lập hoặc khởi động lại chuỗi)."""
        self.ema_box = None
        self.wrist_history.clear()
        self.gesture_window.clear()
        self.last_roi = None
        self._last_was_roi = False
        self.last_result = None

    def smooth_box(self, box, alpha=0.6):
        arr = np.array(box, dtype=np.float32)
        self.ema_box = arr if self.ema_box is None else (alpha * arr + (1.0 - alpha) * self.ema_box)
        return self.ema_box.tolist()

    def _ensure_body_pose_model(self):
        if self.body_pose_model is not None:
            return self.body_pose_model
        if self._body_pose_unavailable:
            return None

        from ultralytics import YOLO
        for candidate in self._body_pose_candidates:
            if candidate != "yolo11n-pose.pt" and not os.path.exists(candidate):
                continue
            try:
                self.body_pose_model = YOLO(candidate, task="pose")
                self.body_pose_model_path = candidate
                self.body_is_onnx = str(candidate).lower().endswith(".onnx")
                return self.body_pose_model
            except Exception as exc:
                print(f"[Cảnh báo] Không nạp được Stage-1 pose model ({candidate}): {exc}")

        self._body_pose_unavailable = True
        return None

    @staticmethod
    def _to_numpy(val):
        if hasattr(val, "cpu"):
            return val.cpu().numpy()
        return np.asarray(val)

    def _extract_top_hand_from_result(self, res, origin_x, origin_y, w_img, h_img, min_conf):
        if res is None or res.boxes is None or len(res.boxes) == 0:
            return None
        if res.keypoints is None or res.keypoints.xy is None or len(res.keypoints.xy) == 0:
            return None

        best = None
        kpts_all = self._to_numpy(res.keypoints.xy)
        if kpts_all.ndim < 3 or kpts_all.shape[0] == 0 or kpts_all.shape[1] == 0:
            return None
        kconf_all = self._to_numpy(res.keypoints.conf) if getattr(res.keypoints, "conf", None) is not None else None

        for idx in range(len(res.boxes)):
            box_obj = res.boxes[idx]
            conf_arr = self._to_numpy(box_obj.conf)
            conf = float(conf_arr.reshape(-1)[0])
            if conf < min_conf:
                continue
            if idx >= len(kpts_all):
                continue
            xywh = self._to_numpy(box_obj.xywh[0]).reshape(-1)
            kps_c = kpts_all[idx]
            kconf_c = kconf_all[idx] if kconf_all is not None and idx < len(kconf_all) else None
            box_full, kps_full = project_roi_detection_to_frame(
                xywh[:4], kps_c, kconf_c, origin_x, origin_y, w_img, h_img
            )
            if best is None or conf > best["conf"]:
                best = {
                    "box": box_full,
                    "keypoints": kps_full,
                    "kpt_conf": np.asarray(kconf_c, dtype=np.float32) if kconf_c is not None else None,
                    "conf": conf,
                }
        return best

    def _run_pose_on_roi(self, frame, origin_x, origin_y, crop_size, conf_thres):
        h_img, w_img = frame.shape[:2]
        crop, (ox, oy, csize) = extract_square_roi(frame, origin_x, origin_y, crop_size)
        res = self.pose_model.predict(crop, conf=conf_thres, imgsz=640, verbose=False)[0]
        det = self._extract_top_hand_from_result(res, ox, oy, w_img, h_img, min_conf=conf_thres)
        if det is not None:
            det["roi"] = (ox, oy, csize)
        return det

    def _run_stage1_pose_rois(self, frame, conf_thres, max_rois=6):
        h_img, w_img = frame.shape[:2]
        body_model = self._ensure_body_pose_model()
        if body_model is None:
            return None, 0

        body_imgsz = 640 if self.body_is_onnx else (960 if w_img >= 2560 else 640)
        body_res = body_model.predict(frame, conf=self.body_conf_thres, imgsz=body_imgsz, verbose=False)[0]
        if body_res is None or body_res.boxes is None or len(body_res.boxes) == 0:
            return None, 0
        if body_res.keypoints is None or body_res.keypoints.xy is None or len(body_res.keypoints.xy) == 0:
            return None, 0

        person_boxes = self._to_numpy(body_res.boxes.xyxy)
        person_confs = self._to_numpy(body_res.boxes.conf).reshape(-1)
        body_kpts = self._to_numpy(body_res.keypoints.xy)
        body_kpt_confs = (
            self._to_numpy(body_res.keypoints.conf)
            if getattr(body_res.keypoints, "conf", None) is not None
            else None
        )

        rois = build_pose_candidate_rois(
            person_boxes, body_kpts, body_kpt_confs, w_img, h_img, person_confs=person_confs
        )
        best_det = None
        checked = 0
        for rx, ry, rsize, _prio, tag in rois[:max_rois]:
            checked += 1
            det = self._run_pose_on_roi(frame, rx, ry, rsize, conf_thres=conf_thres)
            if det is not None and (best_det is None or det["conf"] > best_det["conf"]):
                det["roi_tag"] = tag
                best_det = det
        return best_det, checked

    def _run_stage1_tile_rois(self, frame, conf_thres):
        h_img, w_img = frame.shape[:2]
        rois = build_adaptive_tile_rois(w_img, h_img)
        coarse_conf = min(conf_thres, 0.30)
        best_det = None
        checked = 0

        for rx, ry, rsize, _prio, _tag in rois:
            checked += 1
            coarse = self._run_pose_on_roi(frame, rx, ry, rsize, conf_thres=coarse_conf)
            if coarse is None:
                continue

            bx, by, bw, bh = coarse["box"]
            refine_size = max(128, min(384, int(round(3.0 * max(bw, bh)))))
            if refine_size < rsize * 0.85:
                checked += 1
                refined = self._run_pose_on_roi(
                    frame,
                    bx + bw / 2.0 - refine_size / 2.0,
                    by + bh / 2.0 - refine_size / 2.0,
                    refine_size,
                    conf_thres=conf_thres,
                )
                cand = refined if refined is not None else (coarse if coarse["conf"] >= conf_thres else None)
            else:
                cand = coarse if coarse["conf"] >= conf_thres else None

            if cand is not None and (best_det is None or cand["conf"] > best_det["conf"]):
                best_det = cand
                if best_det["conf"] >= max(conf_thres, 0.72):
                    break

        return best_det, checked

    def _update_temporal_roi(self, box_tlwh, w_img, h_img):
        bx, by, bw, bh = box_tlwh
        crop_size = max(128, min(max(w_img, h_img), int(round(2.8 * max(bw, bh)))))
        rx = int(round(bx + bw / 2.0 - crop_size / 2.0))
        ry = int(round(by + bh / 2.0 - crop_size / 2.0))
        self.last_roi = (rx, ry, crop_size)

    def _detect_hand_pose(self, frame, conf_thres=0.5, far_field=None):
        h_img, w_img = frame.shape[:2]
        mode = far_field if far_field is not None else self.far_field_mode
        is_high_res = w_img >= self.high_res_width
        rois_checked = 0

        # 0) Temporal ROI tracking shortcut cho chuỗi video / webcam 10m
        if mode != "off" and self.use_temporal_roi and self.last_roi is not None and self._last_was_roi:
            rois_checked += 1
            det = self._run_pose_on_roi(frame, *self.last_roi, conf_thres=conf_thres)
            if det is not None:
                det["stage"] = "temporal_roi"
                det["rois_checked"] = rois_checked
                self._update_temporal_roi(det["box"], w_img, h_img)
                return det
            self.last_roi = None

        # 1) Full-frame pass (chỉ bỏ qua khi ép buộc chế độ 10m far_field='always')
        ff_candidate_roi = None
        if mode != "always":
            probe_conf = conf_thres if mode == "off" else min(conf_thres, 0.20)
            res = self.pose_model.predict(frame, conf=probe_conf, imgsz=640, verbose=False)[0]
            ff_any = self._extract_top_hand_from_result(res, 0, 0, w_img, h_img, min_conf=probe_conf)

            if ff_any is not None:
                bx, by, bw, bh = ff_any["box"]
                is_large_enough = (not is_high_res) or (max(bw, bh) >= 0.11 * max(w_img, h_img))
                if ff_any["conf"] >= conf_thres and (mode == "off" or is_large_enough):
                    ff_any["stage"] = "full_frame"
                    ff_any["roi"] = None
                    ff_any["rois_checked"] = rois_checked
                    self._last_was_roi = False
                    self.last_roi = None
                    return ff_any

                if mode != "off":
                    csize = max(128, min(max(w_img, h_img), int(round(3.0 * max(bw, bh)))))
                    ff_candidate_roi = (bx + bw / 2.0 - csize / 2.0, by + bh / 2.0 - csize / 2.0, csize)

            if mode == "off":
                self._last_was_roi = False
                self.last_roi = None
                return None

            if ff_candidate_roi is not None:
                rois_checked += 1
                zoom_det = self._run_pose_on_roi(frame, *ff_candidate_roi, conf_thres=conf_thres)
                if zoom_det is not None:
                    zoom_det["stage"] = "low_conf_roi"
                    zoom_det["rois_checked"] = rois_checked
                    self._last_was_roi = True
                    self._update_temporal_roi(zoom_det["box"], w_img, h_img)
                    return zoom_det

        # 2) Stage 1A: Human Pose Wrist ROI Locator (yolo11n-pose.pt) -> Stage 2 Hand Pose Zoom
        pose_det, checked_1a = self._run_stage1_pose_rois(frame, conf_thres=conf_thres)
        rois_checked += checked_1a
        if pose_det is not None:
            pose_det["stage"] = "pose_wrist_roi"
            pose_det["rois_checked"] = rois_checked
            self._last_was_roi = True
            self._update_temporal_roi(pose_det["box"], w_img, h_img)
            return pose_det

        # 3) Stage 1B: Adaptive High-Res Tile Fallback (chỉ khi không thấy người hoặc ảnh độ phân giải cao)
        if self.enable_tile_fallback and (is_high_res or mode == "always"):
            tile_det, checked_1b = self._run_stage1_tile_rois(frame, conf_thres=conf_thres)
            rois_checked += checked_1b
            if tile_det is not None:
                tile_det["stage"] = "tile_roi"
                tile_det["rois_checked"] = rois_checked
                self._last_was_roi = True
                self._update_temporal_roi(tile_det["box"], w_img, h_img)
                return tile_det

        # 4) Fallback cuối khi mode='always' nhưng bàn tay quá gần che khuất toàn thân
        if mode == "always":
            res = self.pose_model.predict(frame, conf=conf_thres, imgsz=640, verbose=False)[0]
            ff_det = self._extract_top_hand_from_result(res, 0, 0, w_img, h_img, min_conf=conf_thres)
            if ff_det is not None:
                ff_det["stage"] = "full_frame"
                ff_det["roi"] = None
                ff_det["rois_checked"] = rois_checked
                self._last_was_roi = False
                self.last_roi = None
                return ff_det

        self._last_was_roi = False
        self.last_roi = None
        return None

    def process_frame(self, frame, conf_thres=0.5, far_field=None):
        t0 = time.perf_counter()
        h_img, w_img = frame.shape[:2]

        if self.pose_model is not None:
            det = self._detect_hand_pose(frame, conf_thres=conf_thres, far_field=far_field)
            if det is None:
                self.ema_box = None
                self.wrist_history.clear()
                self.gesture_window.clear()
                latency_ms = (time.perf_counter() - t0) * 1000.0
                self.last_result = {
                    "detected": False,
                    "box": None,
                    "raw_box": None,
                    "keypoints": None,
                    "kpt_conf": None,
                    "conf": 0.0,
                    "gesture": "NONE",
                    "raw_gesture": "NONE",
                    "swipe": "STAND_STILL",
                    "angle": 0.0,
                    "bone_len": 0.0,
                    "velocity": (0.0, 0.0, 0.0),
                    "cmd": "HOVER",
                    "stage": "none",
                    "roi": None,
                    "rois_checked": 0,
                    "latency_ms": latency_ms,
                }
                cv2.putText(
                    frame,
                    "NO HAND -> CMD: HOVER (An toan)",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2,
                )
                return frame, None, "HOVER"

            raw_box = det["box"]
            sx, sy, sw, sh = self.smooth_box(raw_box)
            kps = det["keypoints"]
            roi = det.get("roi")
            stage = det.get("stage", "full_frame")

            # Vẽ khung ROI thích ứng nếu chạy Stage 2 Zoom
            if roi is not None:
                rx, ry, rsize = roi
                cv2.rectangle(
                    frame,
                    (max(0, int(rx)), max(0, int(ry))),
                    (min(w_img - 1, int(rx + rsize)), min(h_img - 1, int(ry + rsize))),
                    (255, 255, 0),
                    1,
                )

            # Vẽ khung xương 21 khớp bàn tay
            for i, j in HAND_CONNECTIONS:
                if i < len(kps) and j < len(kps) and _is_valid_kp(kps[i]) and _is_valid_kp(kps[j]):
                    cv2.line(frame, (int(kps[i][0]), int(kps[i][1])), (int(kps[j][0]), int(kps[j][1])), (255, 180, 0), 2)
            for kx, ky in kps:
                if kx > 0 or ky > 0:
                    cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 255), -1)

            raw_gesture, angle, bone_len = analyze_hand_keypoints(kps)
            self.gesture_window.append(raw_gesture)
            gesture = Counter(self.gesture_window).most_common(1)[0][0]  # Debounce chống nhiễu

            if _is_valid_kp(kps[0]):
                self.wrist_history.append((float(kps[0][0]) / max(float(w_img), 1.0), float(kps[0][1]) / max(float(h_img), 1.0)))
            bone_norm = bone_len / max(float(w_img), float(h_img), 1.0)
            swipe_min_disp = min(0.14, max(0.035, 1.8 * bone_norm)) if bone_norm < 0.06 else 0.14
            swipe = detect_swipe_from_wrist(self.wrist_history, min_disp=swipe_min_disp)

            if _is_valid_kp(kps[0]) and _is_valid_kp(kps[9]):
                palm_cx = (float(kps[0][0]) + float(kps[9][0])) / 2.0
                palm_cy = (float(kps[0][1]) + float(kps[9][1])) / 2.0
            else:
                palm_cx = sx + sw / 2.0
                palm_cy = sy + sh / 2.0
            vx, vy, vz = compute_3d_velocity(palm_cx, palm_cy, bone_len, w_img, h_img)

            cmd = GESTURE_TO_CMD.get(gesture, gesture)
            if gesture == "one":
                cmd = f"POINT {angle:+.0f}deg"
            if swipe != "STAND_STILL":
                cmd = f"{swipe} | {cmd}"

            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.last_result = {
                "detected": True,
                "box": (sx, sy, sw, sh),
                "raw_box": raw_box,
                "keypoints": kps,
                "kpt_conf": det.get("kpt_conf"),
                "conf": det["conf"],
                "gesture": gesture,
                "raw_gesture": raw_gesture,
                "swipe": swipe,
                "angle": angle,
                "bone_len": bone_len,
                "velocity": (vx, vy, vz),
                "cmd": cmd,
                "stage": stage,
                "roi": roi,
                "rois_checked": det.get("rois_checked", 0),
                "latency_ms": latency_ms,
            }

            stage_tag = "[ROI Zoom]" if stage != "full_frame" else "[Full]"
            info = f"CMD: {cmd} | 3D(R,T,P)=({vx:+.2f},{vy:+.2f},{vz:+.2f}) {stage_tag}"
            x_i, y_i, w_i, h_i = (int(round(v)) for v in (sx, sy, sw, sh))
            cv2.rectangle(frame, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 0, 255), 2)
            cv2.putText(frame, info, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            return frame, (sx, sy, sw, sh), cmd

        # Fallback CNN gốc khi chưa tải hand_pose.pt
        if self.cnn_model is not None:
            box = self.smooth_box(predict_box(self.cnn_model, frame))
            x_i, y_i, w_i, h_i = (int(round(v)) for v in box)
            cv2.rectangle(frame, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 0, 255), 2)
            cv2.putText(
                frame,
                "CNN Fallback (Hay tai hand_pose.pt de chay 21 khop)",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            self.last_result = {
                "detected": True,
                "box": tuple(box),
                "raw_box": tuple(box),
                "keypoints": None,
                "kpt_conf": None,
                "conf": 1.0,
                "gesture": "CNN_LOCALIZED",
                "raw_gesture": "CNN_LOCALIZED",
                "swipe": "STAND_STILL",
                "angle": 0.0,
                "bone_len": 0.0,
                "velocity": (0.0, 0.0, 0.0),
                "cmd": "CNN_LOCALIZED",
                "stage": "cnn_fallback",
                "roi": None,
                "rois_checked": 0,
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
            }
            return frame, box, "CNN_LOCALIZED"

        self.last_result = None
        return frame, None, "NO_MODEL"


def run_image(controller, path, conf_thres=0.5):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Không đọc được ảnh: {path}")
    controller.reset_state()
    frame, box, cmd = controller.process_frame(frame, conf_thres=conf_thres)
    stage = controller.last_result.get("stage", "none") if controller.last_result else "none"
    if box is not None:
        print(
            "Hộp dự đoán (x, y, w, h) pixel: "
            + ", ".join(f"{v:.1f}" for v in box)
            + f" | Stage: {stage} | Lệnh: {cmd}"
        )
    else:
        print(f"Không phát hiện tay | Stage: {stage} | Lệnh: {cmd}")
    out = os.path.splitext(path)[0] + "_pred.jpg"
    cv2.imwrite(out, frame)
    print(f"Đã lưu: {out}")


def run_webcam(controller, cam_index, save_dir, conf_thres=0.5):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam {cam_index}")
    os.makedirs(save_dir, exist_ok=True)
    print("Đang chạy Camera. Bấm 'c' để chụp lưu ảnh sắc nét, bấm 'q' để thoát.")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_frame = cv2.flip(frame, 1)
        vis_frame = raw_frame.copy()
        vis_frame, _, _ = controller.process_frame(vis_frame, conf_thres=conf_thres)
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        stage = controller.last_result.get("stage", "none") if controller.last_result else "none"
        cv2.putText(
            vis_frame,
            f"FPS: {fps:.1f} | Mode: {stage} | [c]: Chup anh | [q]: Thoat",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Optimal Flycam Hand Control", vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            fname = os.path.join(save_dir, f"flycam_{int(time.time() * 1000)}.jpg")
            cv2.imwrite(fname, raw_frame)
            print(f"Đã chụp ảnh sắc nét lưu tại: {fname}")
    cap.release()
    cv2.destroyAllWindows()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(base_dir, "hand_localization_model.h5"))
    ap.add_argument("--pose-model", default=None, help="đường dẫn tới hand_pose.pt hoặc hand_pose.onnx")
    ap.add_argument("--body-model", default=None, help="đường dẫn tới yolo11n-pose.pt (Stage 1 ROI Locator)")
    ap.add_argument(
        "--far-field",
        choices=("auto", "always", "off"),
        default="auto",
        help="chế độ 2-Stage Adaptive ROI Zoom tầm xa 10m (mặc định: auto)",
    )
    ap.add_argument("--conf", type=float, default=0.5, help="ngưỡng độ tin cậy phát hiện bàn tay")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="đường dẫn ảnh cần thử")
    g.add_argument("--webcam", action="store_true", help="chạy trực tiếp từ webcam / camera flycam")
    ap.add_argument("--camera", type=int, default=0, help="số thứ tự webcam (mặc định 0)")
    args = ap.parse_args()

    controller = OptimalFlycamController(
        base_dir,
        args.model,
        pose_model_path=args.pose_model,
        body_pose_model_path=args.body_model,
        far_field_mode=args.far_field,
    )
    if args.image:
        run_image(controller, args.image, conf_thres=args.conf)
    else:
        run_webcam(controller, args.camera, images_dir, conf_thres=args.conf)


if __name__ == "__main__":
    main()
