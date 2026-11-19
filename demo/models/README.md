# Mô hình cho camera RGB

**Tác giả dự án: phamdu1356**

Chế độ bàn tay dùng một mô hình YOLO Pose có đầu ra 21 điểm khớp. Khi chạy Python trực tiếp, đặt `hand_pose.pt` hoặc `hand_pose.onnx` ở thư mục gốc. Khi chạy dịch vụ Docker `fly`, đặt trọng số trong thư mục này. Docker gắn thư mục ở `/app/models` với quyền chỉ đọc.

Từ thư mục gốc, nếu đã có trọng số PyTorch:

```bash
cp hand_pose.pt demo/models/
docker compose -f demo/docker-compose.yml build fly
docker compose -f demo/docker-compose.yml run --rm fly
```

Cấu hình mặc định mở camera USB số 0, chạy PyBullet và ghi log tại `logs/run1.csv`. Để dùng ONNX, chép `hand_pose.onnx` vào cùng thư mục rồi chọn rõ tệp:

```bash
docker compose -f demo/docker-compose.yml run --rm fly --camera 0 --backend sim --pose-model /app/models/hand_pose.onnx
```

Mô hình `yolo11n-pose.pt` dành cho chế độ thử nghiệm cũ có nhận diện người. Hướng dẫn đọc luồng video, chạy mô phỏng thuần hình học và kết quả tinh chỉnh nằm trong [README dự án](../../README.md).
