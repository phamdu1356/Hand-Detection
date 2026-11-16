"""Hệ thống định vị bàn tay & điều khiển Flycam Offline (hỗ trợ cả 3 mô hình + CNN gốc).

Các chế độ (--mode):
  1. pose   : YOLO11-Pose (hand_pose.pt) bắt 21 khớp bàn tay + suy luận lệnh bay 3D.
  2. detect : YOLO11-Detect (hand_gesture.pt) nhận diện 9 cử chỉ tĩnh HaGRID + vị trí tâm.
  3. lstm   : Kết hợp YOLO11-Pose (21 khớp) + mạng LSTM (hand_lstm.keras) nhận diện cử chỉ vẫy tay động.
  4. cnn    : Mô hình CNN hồi quy hộp giới hạn gốc (hand_localization_model.h5).
  5. auto   : Tự động chạy kết hợp các model hiện có trong thư mục.

Cách dùng:
  python test_model.py --image anh.jpg
  python test_model.py --webcam
  python test_model.py --webcam --mode pose
"""
import argparse
from collections import deque
import os
import time

import cv2
import numpy as np

IMG_SIZE = 224
SEQ_LEN = 15
DYNAMIC_ACTIONS = ["STAND_STILL", "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN"]
GESTURE_TO_CMD = {
    "palm": "FOLLOW_PALM (Bam theo tay)",
    "fist": "HOVER (Dung yen / Phanh)",
    "ok": "LAND (Ha canh)",
    "one": "FORWARD (Bay tien)",
    "peace_inverted": "BACKWARD (Bay lui)",
    "like": "UP (Bay len)",
    "call": "YAW_RIGHT (Xoay phai)",
    "four": "YAW_LEFT (Xoay trai)",
    "mute": "EMERGENCY_STOP (Dung khan cap)",
}


def load_cnn_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)


def load_yolo_model(path):
    from ultralytics import YOLO
    return YOLO(path)


def predict_box(model, frame_bgr):
    """Dự đoán hộp (x, y, w, h) theo pixel từ mô hình CNN gốc."""
    import tensorflow as tf
    h_img, w_img = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = tf.image.resize(rgb, [IMG_SIZE, IMG_SIZE])[tf.newaxis, ...]
    bx, by, bw, bh = model(x, training=False).numpy()[0]
    return bx * w_img, by * h_img, bw * w_img, bh * h_img


def classify_pose_keypoints(kps_xy):
    """Đếm ngón tay đang mở từ 21 khớp (x, y) để suy ra cử chỉ tĩnh khi dùng YOLO-Pose."""
    if kps_xy is None or len(kps_xy) < 21:
        return "UNKNOWN"
    wrist = kps_xy[0]
    # Đốt đầu ngón (tip) và đốt giữa (pip) của 4 ngón: trỏ (8,6), giữa (12,10), áp út (16,14), út (20,18)
    finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    extended = 0
    for tip, pip in finger_pairs:
        if np.linalg.norm(kps_xy[tip] - wrist) > np.linalg.norm(kps_xy[pip] - wrist) * 1.15:
            extended += 1
    # Ngón cái (4 so với khớp 2)
    if np.linalg.norm(kps_xy[4] - kps_xy[17]) > np.linalg.norm(kps_xy[2] - kps_xy[17]) * 1.15:
        extended += 1

    if extended == 0:
        return "fist"
    if extended == 1:
        return "one"
    if extended == 2:
        return "peace_inverted"
    if extended >= 4:
        return "palm"
    return f"{extended}_fingers"


def compute_drone_velocity(cx, cy, w_box, h_box, w_img, h_img, deadzone=0.08):
    """Tính lệnh điều khiển 3D (Roll, Throttle, Pitch) giữ tay ở tâm khung hình Camera Flycam."""
    err_x = (cx / w_img) - 0.5
    err_y = 0.5 - (cy / h_img)
    box_area_ratio = (w_box * h_box) / max(float(w_img * h_img), 1.0)
    target_area = 0.06  # Diện tích chuẩn ở khoảng cách ~2m

    vx = 0.0 if abs(err_x) < deadzone else float(np.clip(err_x * 2.0, -1.0, 1.0))
    vy = 0.0 if abs(err_y) < deadzone else float(np.clip(err_y * 2.0, -1.0, 1.0))
    vz = 0.0 if abs(target_area - box_area_ratio) < 0.02 else float(np.clip((target_area - box_area_ratio) * 8.0, -1.0, 1.0))
    return vx, vy, vz


def draw_box(frame, box, text=None, subtext=None):
    x, y, w, h = (int(round(v)) for v in box)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if text:
        cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if subtext:
        cv2.putText(frame, subtext, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return frame


class DroneHandController:
    """Tích hợp cả 3 mô hình (YOLO-Pose, YOLO-Detect, LSTM) và CNN fallback."""

    def __init__(self, mode, base_dir, cnn_path):
        self.mode = mode
        self.pose_model = None
        self.det_model = None
        self.lstm_model = None
        self.cnn_model = None
        self.kp_history = deque(maxlen=SEQ_LEN)
        self.ema_box = None

        pose_path = os.path.join(base_dir, "hand_pose.pt")
        det_path = os.path.join(base_dir, "hand_gesture.pt")
        lstm_path = os.path.join(base_dir, "hand_lstm.keras")

        if mode in ("pose", "lstm", "auto") and os.path.exists(pose_path):
            self.pose_model = load_yolo_model(pose_path)
        if mode in ("detect", "auto") and os.path.exists(det_path):
            self.det_model = load_yolo_model(det_path)
        if mode in ("lstm", "auto") and os.path.exists(lstm_path):
            self.lstm_model = load_cnn_model(lstm_path)
        if (mode == "cnn" or (self.pose_model is None and self.det_model is None)) and os.path.exists(cnn_path):
            self.cnn_model = load_cnn_model(cnn_path)

    def smooth_box(self, box, alpha=0.6):
        box_arr = np.array(box, dtype=np.float32)
        if self.ema_box is None:
            self.ema_box = box_arr
        else:
            self.ema_box = alpha * box_arr + (1.0 - alpha) * self.ema_box
        return self.ema_box.tolist()

    def process_frame(self, frame, conf_thres=0.5):
        h_img, w_img = frame.shape[:2]
        gesture = None
        dyn_action = "STAND_STILL"
        box_xywh = None

        # 1. Chạy YOLO-Detect (9 cử chỉ HaGRID) nếu có
        if self.det_model is not None:
            res = self.det_model.predict(frame, conf=conf_thres, verbose=False)[0]
            if len(res.boxes) > 0:
                b = res.boxes[0]
                cls_id = int(b.cls[0])
                gesture = self.det_model.names[cls_id]
                cx, cy, w, h = b.xywh[0].tolist()
                box_xywh = [cx - w / 2, cy - h / 2, w, h]

        # 2. Chạy YOLO-Pose (21 khớp bàn tay) + LSTM cử chỉ động nếu có
        if self.pose_model is not None:
            res_p = self.pose_model.predict(frame, conf=conf_thres, verbose=False)[0]
            if len(res_p.boxes) > 0:
                b = res_p.boxes[0]
                cx, cy, w, h = b.xywh[0].tolist()
                box_xywh = [cx - w / 2, cy - h / 2, w, h]
                if res_p.keypoints is not None and len(res_p.keypoints.xy) > 0:
                    kps = res_p.keypoints.xy[0].cpu().numpy()
                    for kx, ky in kps:
                        if kx > 0 and ky > 0:
                            cv2.circle(frame, (int(kx), int(ky)), 3, (255, 200, 0), -1)
                    if gesture is None:
                        gesture = classify_pose_keypoints(kps)
                    norm_kps = (kps / np.array([w_img, h_img], dtype=np.float32)).flatten()
                    self.kp_history.append(norm_kps)
                    if len(self.kp_history) == SEQ_LEN and self.lstm_model is not None:
                        seq = np.array(self.kp_history, dtype=np.float32)
                        seq = (seq - seq[0:1, :])[np.newaxis, ...]
                        probs = self.lstm_model(seq, training=False).numpy()[0]
                        dyn_action = DYNAMIC_ACTIONS[int(np.argmax(probs))]
            else:
                self.kp_history.clear()

        # 3. Nếu chỉ có model CNN gốc
        if self.pose_model is None and self.det_model is None and self.cnn_model is not None:
            box_xywh = list(predict_box(self.cnn_model, frame))
            gesture = "hand_localized"

        if box_xywh is None:
            self.ema_box = None
            cv2.putText(frame, "NO HAND -> CMD: HOVER (Dung yen)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return frame, None, "HOVER"

        sx, sy, sw, sh = self.smooth_box(box_xywh)
        cx, cy = sx + sw / 2.0, sy + sh / 2.0
        vx, vy, vz = compute_drone_velocity(cx, cy, sw, sh, w_img, h_img)

        cmd = GESTURE_TO_CMD.get(gesture, f"TRACK ({gesture})")
        if dyn_action != "STAND_STILL":
            cmd = f"{dyn_action} | {cmd}"
        vel_text = f"CMD: {cmd} | Vel(R,T,P)=({vx:+.2f}, {vy:+.2f}, {vz:+.2f})"
        draw_box(frame, (sx, sy, sw, sh), subtext=vel_text)
        return frame, (sx, sy, sw, sh), cmd


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


def run_webcam(controller, cam_index):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam {cam_index}")
    print("Đang chạy điều khiển Flycam qua Camera, bấm q để thoát.")
    prev = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        frame, _, _ = controller.process_frame(frame)
        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Drone Hand Control (Offline)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(base_dir, "hand_localization_model.h5"))
    ap.add_argument("--mode", choices=["auto", "pose", "detect", "lstm", "cnn"], default="auto")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="đường dẫn ảnh cần thử")
    g.add_argument("--webcam", action="store_true", help="chạy trực tiếp từ webcam / camera flycam")
    ap.add_argument("--camera", type=int, default=0, help="số thứ tự webcam (mặc định 0)")
    args = ap.parse_args()

    controller = DroneHandController(args.mode, base_dir, args.model)
    if args.image:
        run_image(controller, args.image)
    else:
        run_webcam(controller, args.camera)


if __name__ == "__main__":
    main()
