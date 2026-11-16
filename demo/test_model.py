"""Hệ thống điều khiển Flycam bằng tay Offline Tối Ưu (1 Model YOLO11n-Pose làm 3 việc).

1 Model duy nhất (hand_pose.pt / hand_pose.onnx) xử lý cả 3 tác vụ trong 1 lần chạy:
  1. Định vị 3D & Khoảng cách: Tâm lòng bàn tay + độ dài xương bàn tay |kp9 - kp0| (không bị sai khi nắm tay).
  2. Cử chỉ tĩnh & Góc chỉ 360°: Đếm khớp ngón tay (palm, fist, ok, one, peace, three, four, like) + chống nhiễu Debounce.
  3. Cử chỉ động (Vẫy tay): Tính vector vận tốc cổ tay kp0 qua 10 frame (SWIPE_LEFT, SWIPE_RIGHT, SWIPE_UP, SWIPE_DOWN).
  * Tự động fallback về hand_localization_model.h5 nếu chưa tải hand_pose.pt.

Cách dùng:
  python test_model.py --image anh.jpg          # chạy thử trên ảnh tĩnh
  python test_model.py --webcam                 # chạy trực tiếp từ camera (bấm 'c' chụp ảnh sắc nét, 'q' thoát)
"""
import argparse
from collections import Counter, deque
import math
import os
import time

import cv2
import numpy as np

IMG_SIZE = 224
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


def analyze_hand_keypoints(kps):
    """Từ 21 khớp (x, y), trích xuất đồng thời: cử chỉ tĩnh, góc chỉ ngón trỏ và độ dài xương bàn tay."""
    wrist = kps[0]
    middle_mcp = kps[9]
    palm_bone_len = float(np.linalg.norm(middle_mcp - wrist))
    if palm_bone_len < 1e-3:
        return "UNKNOWN", 0.0, 0.0

    # Kiểm tra trạng thái mở của 4 ngón dài: trỏ (8,6), giữa (12,10), áp út (16,14), út (20,18)
    fingers = [
        np.linalg.norm(kps[8] - wrist) > np.linalg.norm(kps[6] - wrist) * 1.15,
        np.linalg.norm(kps[12] - wrist) > np.linalg.norm(kps[10] - wrist) * 1.15,
        np.linalg.norm(kps[16] - wrist) > np.linalg.norm(kps[14] - wrist) * 1.15,
        np.linalg.norm(kps[20] - wrist) > np.linalg.norm(kps[18] - wrist) * 1.15,
    ]
    thumb_open = np.linalg.norm(kps[4] - kps[17]) > np.linalg.norm(kps[2] - kps[17]) * 1.18
    thumb_index_dist = np.linalg.norm(kps[4] - kps[8]) / palm_bone_len

    # Góc chỉ của ngón trỏ (kp5 -> kp8) trong mặt phẳng camera (-180..180 độ)
    dx, dy = kps[8][0] - kps[5][0], kps[5][1] - kps[8][1]
    point_angle = math.degrees(math.atan2(dy, dx))

    count_4 = sum(fingers)
    if thumb_index_dist < 0.38 and sum(fingers[1:]) >= 2:
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
    err_x = (palm_cx / w_img) - 0.5
    err_y = 0.5 - (palm_cy / h_img)
    norm_bone = palm_bone_len / max(float(h_img), 1.0)
    target_bone = 0.12  # Độ dài xương chuẩn khi đứng cách Flycam ~2m

    vx = 0.0 if abs(err_x) < deadzone else float(np.clip(err_x * 2.2, -1.0, 1.0))
    vy = 0.0 if abs(err_y) < deadzone else float(np.clip(err_y * 2.2, -1.0, 1.0))
    vz = 0.0 if abs(target_bone - norm_bone) < 0.015 else float(np.clip((target_bone - norm_bone) * 6.0, -1.0, 1.0))
    return vx, vy, vz


class OptimalFlycamController:
    """Bộ điều khiển Flycam tối ưu dùng 1 model YOLO11n-Pose duy nhất (fallback CNN nếu chưa có)."""

    def __init__(self, base_dir, cnn_path):
        self.pose_model = None
        self.cnn_model = None
        self.wrist_history = deque(maxlen=10)
        self.gesture_window = deque(maxlen=5)
        self.ema_box = None

        for candidate in ("hand_pose.pt", "hand_pose.onnx"):
            p = os.path.join(base_dir, candidate)
            if os.path.exists(p):
                from ultralytics import YOLO
                self.pose_model = YOLO(p)
                print(f"Đã nạp model tối ưu: {p}")
                break

        if self.pose_model is None and os.path.exists(cnn_path):
            self.cnn_model = load_cnn_model(cnn_path)
            print(f"Chưa có hand_pose.pt, dùng tạm model CNN: {cnn_path}")

    def smooth_box(self, box, alpha=0.6):
        arr = np.array(box, dtype=np.float32)
        self.ema_box = arr if self.ema_box is None else (alpha * arr + (1.0 - alpha) * self.ema_box)
        return self.ema_box.tolist()

    def process_frame(self, frame, conf_thres=0.5):
        h_img, w_img = frame.shape[:2]

        if self.pose_model is not None:
            res = self.pose_model.predict(frame, conf=conf_thres, imgsz=640, verbose=False)[0]
            if len(res.boxes) == 0 or res.keypoints is None or len(res.keypoints.xy) == 0:
                self.ema_box = None
                self.wrist_history.clear()
                self.gesture_window.clear()
                cv2.putText(frame, "NO HAND -> CMD: HOVER (An toan)", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                return frame, None, "HOVER"

            b = res.boxes[0]
            cx, cy, w, h = b.xywh[0].tolist()
            sx, sy, sw, sh = self.smooth_box([cx - w / 2, cy - h / 2, w, h])
            kps = res.keypoints.xy[0].cpu().numpy()

            # Vẽ khung xương 21 khớp bàn tay
            for i, j in HAND_CONNECTIONS:
                if i < len(kps) and j < len(kps) and kps[i][0] > 0 and kps[j][0] > 0:
                    cv2.line(frame, (int(kps[i][0]), int(kps[i][1])), (int(kps[j][0]), int(kps[j][1])), (255, 180, 0), 2)
            for kx, ky in kps:
                if kx > 0 and ky > 0:
                    cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 255), -1)

            raw_gesture, angle, bone_len = analyze_hand_keypoints(kps)
            self.gesture_window.append(raw_gesture)
            gesture = Counter(self.gesture_window).most_common(1)[0][0]  # Debounce chống nhiễu

            self.wrist_history.append((kps[0][0] / w_img, kps[0][1] / h_img))
            swipe = detect_swipe_from_wrist(self.wrist_history)

            palm_cx = (kps[0][0] + kps[9][0]) / 2.0
            palm_cy = (kps[0][1] + kps[9][1]) / 2.0
            vx, vy, vz = compute_3d_velocity(palm_cx, palm_cy, bone_len, w_img, h_img)

            cmd = GESTURE_TO_CMD.get(gesture, gesture)
            if gesture == "one":
                cmd = f"POINT {angle:+.0f}deg"
            if swipe != "STAND_STILL":
                cmd = f"{swipe} | {cmd}"

            info = f"CMD: {cmd} | 3D(R,T,P)=({vx:+.2f},{vy:+.2f},{vz:+.2f})"
            x_i, y_i, w_i, h_i = (int(round(v)) for v in (sx, sy, sw, sh))
            cv2.rectangle(frame, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 0, 255), 2)
            cv2.putText(frame, info, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            return frame, (sx, sy, sw, sh), cmd

        # Fallback CNN gốc khi chưa tải hand_pose.pt
        if self.cnn_model is not None:
            box = self.smooth_box(predict_box(self.cnn_model, frame))
            x_i, y_i, w_i, h_i = (int(round(v)) for v in box)
            cv2.rectangle(frame, (x_i, y_i), (x_i + w_i, y_i + h_i), (0, 0, 255), 2)
            cv2.putText(frame, "CNN Fallback (Hay tai hand_pose.pt de chay 21 khop)", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return frame, box, "CNN_LOCALIZED"

        return frame, None, "NO_MODEL"


def run_image(controller, path):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Không đọc được ảnh: {path}")
    frame, box, cmd = controller.process_frame(frame)
    if box is not None:
        print("Hộp dự đoán (x, y, w, h) pixel: " + ", ".join(f"{v:.1f}" for v in box) + f" | Lệnh: {cmd}")
    out = os.path.splitext(path)[0] + "_pred.jpg"
    cv2.imwrite(out, frame)
    print(f"Đã lưu: {out}")


def run_webcam(controller, cam_index, save_dir):
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
        vis_frame, _, _ = controller.process_frame(vis_frame)
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(vis_frame, f"FPS: {fps:.1f} | [c]: Chup anh | [q]: Thoat", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
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
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="đường dẫn ảnh cần thử")
    g.add_argument("--webcam", action="store_true", help="chạy trực tiếp từ webcam / camera flycam")
    ap.add_argument("--camera", type=int, default=0, help="số thứ tự webcam (mặc định 0)")
    args = ap.parse_args()

    controller = OptimalFlycamController(base_dir, args.model)
    if args.image:
        run_image(controller, args.image)
    else:
        run_webcam(controller, args.camera, images_dir)


if __name__ == "__main__":
    main()
