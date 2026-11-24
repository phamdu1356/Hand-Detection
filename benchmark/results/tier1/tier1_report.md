# Báo cáo Benchmark Tầng 1 — Phát hiện Tay, 21 Khớp, Cổng Gate14 & Latency

- **Thời điểm chạy:** `2026-11-23 01:32:28`
- **Git commit:** `f90033bead94` (dirty=`True`)
- **Dataset nguồn:** `../datasets/hand-keypoints` (*in-distribution validation split*)
- **Làm sạch dữ liệu (SHA-256):** `val` gốc `7992` ảnh → loại **`38`** ảnh rò rỉ `train ∩ val` và **`1`** ảnh trùng nội bộ → chấm trên **`7953` ảnh sạch**.
- **Cảnh báo gần trùng nội bộ `val` (`dHash ≤ 4`):** phát hiện `19944` cặp (`6327` ảnh liên quan trong `val` sạch).
- **Cảnh báo gần trùng chéo `train ∩ val` (`dHash ≤ 2`):** `6095/7953` ảnh `val` có ảnh rất giống trong `train` (`31317` cặp).
- **Đặc điểm nhãn trên tập sạch:** `oob_box=991`, `oob_kp=184`, `missing_palm5=1092`, `all14_in_frame=6543/7953`, `multi_hand_imgs=0`.
- **Phân bố kích thước tay (`max(bw, bh)` px):** {'<40': 2, '40-80': 892, '80-160': 3682, '>=160': 3377} · percentiles={'10': 77.0, '25': 101.0, '50': 142.0, '75': 182.0, '90': 197.0}.
- **Chuẩn OKS của Ultralytics `.val()`:** `ultralytics_default_1_over_21` (`rect=False` công bằng cho `.pt` & `.onnx`).

> **Lưu ý phạm vi kết luận:** `hand-keypoints/val` dùng chung nguồn với `train` (và đã dùng để chọn `best.pt` qua 25 epoch), đồng thời chỉ có **2/7953 (0.03%) ảnh tay < 40 px** (trung vị `142.0 px`). Đây là điểm chuẩn **cận cảnh / nội bộ** để so trọng số, kiểm tra ONNX (`rect=False` công bằng), soi khớp yếu và `Gate14` — chưa thay thế đánh giá video thực tế ở 1,5–3 m (Tầng 2 & 3).

## 1. Bảng so sánh tổng hợp các trọng số (Tầng 1 — Ngưỡng vận hành `conf ≥ 0.50`)

| Model | `imgsz` | SHA-256 | Recall@0.5 | Precision@0.5 | Extra FP | AP@0.5 | Box mAP50–95 (Ultra) | Pose mAP50–95 (σ=1/21) | PCK@0.2 Box | PCK@0.2 Palm (14 kpt) | Gate14 (GT14 TP) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hand_pose.pt` *(≡ hand_pose_v3.pt)* | 512 | `093ae826a31a` | 97.37% | 97.81% | 35 | 98.97% | 84.35% | 66.75% | 95.52% | 89.09% | 94.88% |
| `hand_pose.pt@640` | 640 | `093ae826a31a` | 97.69% | 97.75% | 57 | 99.04% | 79.43% | 64.07% | 95.67% | 89.20% | 80.04% |
| `hand_pose_v2.pt` | 640 | `007b592edee3` | 96.28% | 97.47% | 31 | 98.64% | 83.88% | 66.69% | 93.55% | 86.90% | 94.75% |
| `hand_pose.onnx` *(≡ hand_pose_v3.onnx)* | 512 | `2da512c06d90` | 97.38% | 97.82% | 35 | 98.97% | 84.35% | 66.69% | 95.53% | 89.10% | 94.88% |
| `hand_pose_v2.onnx` | 640 | `7c7a7803e5de` | 96.28% | 97.47% | 31 | 98.64% | 83.89% | 66.68% | 93.55% | 86.90% | 94.76% |
| `runs/hand_only_trial/fit/weights/best.pt` | 640 | `901d7d519b7a` | 97.15% | 97.83% | 27 | 98.94% | 84.67% | 65.03% | 95.11% | 88.49% | 93.08% |

## 2. Chi tiết mẫu số `k/n`, 95% CI và Lát cắt loại trừ `val.txt`

### `hand_pose.pt` (`imgsz=512`, SHA-256 `093ae826a31a3014`)
- **Recall@0.5:** 97.37% (7744/7953, 95% CI 97.0–97.7%) · **Top-1 Conf Recall@0.5:** 97.11% · **Precision@0.5:** 97.81% (7744/7917, 95% CI 97.5–98.1%) · **Mean IoU (TP):** `0.9147`
- **PCK@0.2 Box:** Overall `95.52%` / on TP `98.09%` (trên `100%` ảnh `7953`; loại `437` khớp GT ngoài khung).
- **PCK@0.2 Palm (mẫu số `6861/7953` ảnh đủ 5 khớp lòng bàn tay):** 21 khớp Overall `89.29%` (on TP `92.04%`) · **14 khớp điều khiển** Overall `89.09%` (on TP `91.84%`).
- **Cổng `Gate14` (`all 14 kpts conf ≥ 0.5`):** All TP = 81.04% (6276/7744, 95% CI 80.2–81.9%) · **GT đủ 14 khớp TP** = 94.88% (6020/6345, 95% CI 94.3–95.4%).
- **Ultralytics `.val()` (`ultralytics_default_1_over_21`, mẫu số `7808` instances):** Box mAP50 `99.11%`, Box mAP50–95 `84.35%`, Pose mAP50 `87.09%`, Pose mAP50–95 `66.75%`.
- **Lát cắt loại trừ `runs/hand_only_trial/val.txt` (`n=7698`):** Recall@0.5 `97.39%`, AP@0.5 `98.95%`, PCK@0.2 Box `95.52%`, PCK@0.2 Palm 14-kpt `89.10%`, Gate14 (GT14 TP) `94.94%`.
- **Top-5 khớp sai lệch lớn nhất:** `thumb_tip` (11.3% palm / 5.8px), `pinky_tip` (10.4% palm / 5.3px), `index_tip` (10.0% palm / 5.3px), `wrist` (10.0% palm / 5.6px), `thumb_ip` (9.1% palm / 4.8px)
- **Top-5 khớp làm rớt Gate14 nhiều nhất:** `wrist` (314 lượt GT14 / 1382 tổng), `thumb_tip` (7 lượt GT14 / 101 tổng), `pinky_tip` (5 lượt GT14 / 84 tổng), `pinky_pip` (4 lượt GT14 / 89 tổng), `pinky_mcp` (3 lượt GT14 / 79 tổng)

### `hand_pose.pt@640` (`imgsz=640`, SHA-256 `093ae826a31a3014`)
- **Recall@0.5:** 97.69% (7769/7953, 95% CI 97.3–98.0%) · **Top-1 Conf Recall@0.5:** 97.43% · **Precision@0.5:** 97.75% (7769/7948, 95% CI 97.4–98.1%) · **Mean IoU (TP):** `0.9073`
- **PCK@0.2 Box:** Overall `95.67%` / on TP `97.93%` (trên `100%` ảnh `7953`; loại `437` khớp GT ngoài khung).
- **PCK@0.2 Palm (mẫu số `6861/7953` ảnh đủ 5 khớp lòng bàn tay):** 21 khớp Overall `89.38%` (on TP `91.80%`) · **14 khớp điều khiển** Overall `89.20%` (on TP `91.61%`).
- **Cổng `Gate14` (`all 14 kpts conf ≥ 0.5`):** All TP = 67.46% (5241/7769, 95% CI 66.4–68.5%) · **GT đủ 14 khớp TP** = 80.04% (5097/6368, 95% CI 79.0–81.0%).
- **Ultralytics `.val()` (`ultralytics_default_1_over_21`, mẫu số `7808` instances):** Box mAP50 `98.23%`, Box mAP50–95 `79.43%`, Pose mAP50 `86.19%`, Pose mAP50–95 `64.07%`.
- **Lát cắt loại trừ `runs/hand_only_trial/val.txt` (`n=7698`):** Recall@0.5 `97.67%`, AP@0.5 `99.03%`, PCK@0.2 Box `95.65%`, PCK@0.2 Palm 14-kpt `89.18%`, Gate14 (GT14 TP) `80.16%`.
- **Top-5 khớp sai lệch lớn nhất:** `thumb_tip` (11.7% palm / 6.0px), `wrist` (10.9% palm / 6.3px), `pinky_tip` (10.8% palm / 5.6px), `index_tip` (10.1% palm / 5.3px), `thumb_cmc` (9.8% palm / 5.7px)
- **Top-5 khớp làm rớt Gate14 nhiều nhất:** `wrist` (1224 lượt GT14 / 2420 tổng), `middle_tip` (57 lượt GT14 / 182 tổng), `thumb_tip` (38 lượt GT14 / 175 tổng), `index_tip` (16 lượt GT14 / 58 tổng), `pinky_pip` (14 lượt GT14 / 96 tổng)

### `hand_pose_v2.pt` (`imgsz=640`, SHA-256 `007b592edee3ef46`)
- **Recall@0.5:** 96.28% (7657/7953, 95% CI 95.8–96.7%) · **Top-1 Conf Recall@0.5:** 96.08% · **Precision@0.5:** 97.47% (7657/7856, 95% CI 97.1–97.8%) · **Mean IoU (TP):** `0.9143`
- **PCK@0.2 Box:** Overall `93.55%` / on TP `97.17%` (trên `100%` ảnh `7953`; loại `437` khớp GT ngoài khung).
- **PCK@0.2 Palm (mẫu số `6861/7953` ảnh đủ 5 khớp lòng bàn tay):** 21 khớp Overall `87.05%` (on TP `90.90%`) · **14 khớp điều khiển** Overall `86.90%` (on TP `90.75%`).
- **Cổng `Gate14` (`all 14 kpts conf ≥ 0.5`):** All TP = 82.70% (6332/7657, 95% CI 81.8–83.5%) · **GT đủ 14 khớp TP** = 94.75% (5936/6265, 95% CI 94.2–95.3%).
- **Ultralytics `.val()` (`ultralytics_default_1_over_21`, mẫu số `7808` instances):** Box mAP50 `98.94%`, Box mAP50–95 `83.88%`, Pose mAP50 `85.41%`, Pose mAP50–95 `66.69%`.
- **Lát cắt loại trừ `runs/hand_only_trial/val.txt` (`n=7698`):** Recall@0.5 `96.30%`, AP@0.5 `98.63%`, PCK@0.2 Box `93.56%`, PCK@0.2 Palm 14-kpt `86.88%`, Gate14 (GT14 TP) `94.73%`.
- **Top-5 khớp sai lệch lớn nhất:** `thumb_tip` (12.4% palm / 6.3px), `pinky_tip` (11.7% palm / 6.0px), `wrist` (11.0% palm / 6.2px), `ring_tip` (10.6% palm / 5.5px), `index_tip` (9.9% palm / 5.0px)
- **Top-5 khớp làm rớt Gate14 nhiều nhất:** `wrist` (314 lượt GT14 / 1224 tổng), `thumb_tip` (10 lượt GT14 / 77 tổng), `pinky_pip` (4 lượt GT14 / 61 tổng), `pinky_mcp` (3 lượt GT14 / 45 tổng), `pinky_tip` (3 lượt GT14 / 71 tổng)

### `hand_pose.onnx` (`imgsz=512`, SHA-256 `2da512c06d90869b`)
- **Recall@0.5:** 97.38% (7745/7953, 95% CI 97.0–97.7%) · **Top-1 Conf Recall@0.5:** 97.12% · **Precision@0.5:** 97.82% (7745/7918, 95% CI 97.5–98.1%) · **Mean IoU (TP):** `0.9147`
- **PCK@0.2 Box:** Overall `95.53%` / on TP `98.08%` (trên `100%` ảnh `7953`; loại `437` khớp GT ngoài khung).
- **PCK@0.2 Palm (mẫu số `6861/7953` ảnh đủ 5 khớp lòng bàn tay):** 21 khớp Overall `89.29%` (on TP `92.03%`) · **14 khớp điều khiển** Overall `89.10%` (on TP `91.83%`).
- **Cổng `Gate14` (`all 14 kpts conf ≥ 0.5`):** All TP = 81.05% (6277/7745, 95% CI 80.2–81.9%) · **GT đủ 14 khớp TP** = 94.88% (6021/6346, 95% CI 94.3–95.4%).
- **Ultralytics `.val()` (`ultralytics_default_1_over_21`, mẫu số `7808` instances):** Box mAP50 `99.11%`, Box mAP50–95 `84.35%`, Pose mAP50 `87.09%`, Pose mAP50–95 `66.69%`.
- **Lát cắt loại trừ `runs/hand_only_trial/val.txt` (`n=7698`):** Recall@0.5 `97.40%`, AP@0.5 `98.95%`, PCK@0.2 Box `95.53%`, PCK@0.2 Palm 14-kpt `89.11%`, Gate14 (GT14 TP) `94.94%`.
- **Top-5 khớp sai lệch lớn nhất:** `thumb_tip` (11.3% palm / 5.8px), `pinky_tip` (10.4% palm / 5.3px), `index_tip` (10.0% palm / 5.3px), `wrist` (10.0% palm / 5.6px), `thumb_ip` (9.2% palm / 4.8px)
- **Top-5 khớp làm rớt Gate14 nhiều nhất:** `wrist` (314 lượt GT14 / 1381 tổng), `thumb_tip` (7 lượt GT14 / 101 tổng), `pinky_tip` (5 lượt GT14 / 84 tổng), `pinky_pip` (4 lượt GT14 / 89 tổng), `pinky_mcp` (3 lượt GT14 / 79 tổng)

### `hand_pose_v2.onnx` (`imgsz=640`, SHA-256 `7c7a7803e5de30c3`)
- **Recall@0.5:** 96.28% (7657/7953, 95% CI 95.8–96.7%) · **Top-1 Conf Recall@0.5:** 96.08% · **Precision@0.5:** 97.47% (7657/7856, 95% CI 97.1–97.8%) · **Mean IoU (TP):** `0.9144`
- **PCK@0.2 Box:** Overall `93.55%` / on TP `97.17%` (trên `100%` ảnh `7953`; loại `437` khớp GT ngoài khung).
- **PCK@0.2 Palm (mẫu số `6861/7953` ảnh đủ 5 khớp lòng bàn tay):** 21 khớp Overall `87.05%` (on TP `90.90%`) · **14 khớp điều khiển** Overall `86.90%` (on TP `90.75%`).
- **Cổng `Gate14` (`all 14 kpts conf ≥ 0.5`):** All TP = 82.71% (6333/7657, 95% CI 81.8–83.5%) · **GT đủ 14 khớp TP** = 94.76% (5937/6265, 95% CI 94.2–95.3%).
- **Ultralytics `.val()` (`ultralytics_default_1_over_21`, mẫu số `7808` instances):** Box mAP50 `98.94%`, Box mAP50–95 `83.89%`, Pose mAP50 `85.35%`, Pose mAP50–95 `66.68%`.
- **Lát cắt loại trừ `runs/hand_only_trial/val.txt` (`n=7698`):** Recall@0.5 `96.30%`, AP@0.5 `98.63%`, PCK@0.2 Box `93.56%`, PCK@0.2 Palm 14-kpt `86.89%`, Gate14 (GT14 TP) `94.75%`.
- **Top-5 khớp sai lệch lớn nhất:** `thumb_tip` (12.4% palm / 6.3px), `pinky_tip` (11.7% palm / 6.0px), `wrist` (11.0% palm / 6.2px), `ring_tip` (10.6% palm / 5.5px), `index_tip` (9.9% palm / 5.0px)
- **Top-5 khớp làm rớt Gate14 nhiều nhất:** `wrist` (313 lượt GT14 / 1223 tổng), `thumb_tip` (10 lượt GT14 / 77 tổng), `pinky_pip` (4 lượt GT14 / 61 tổng), `pinky_mcp` (3 lượt GT14 / 45 tổng), `pinky_tip` (3 lượt GT14 / 71 tổng)

### `runs/hand_only_trial/fit/weights/best.pt` (`imgsz=640`, SHA-256 `901d7d519b7a19f2`)
- **Recall@0.5:** 97.15% (7726/7953, 95% CI 96.8–97.5%) · **Top-1 Conf Recall@0.5:** 96.93% · **Precision@0.5:** 97.83% (7726/7897, 95% CI 97.5–98.1%) · **Mean IoU (TP):** `0.9158`
- **PCK@0.2 Box:** Overall `95.11%` / on TP `97.90%` (trên `100%` ảnh `7953`; loại `437` khớp GT ngoài khung).
- **PCK@0.2 Palm (mẫu số `6861/7953` ảnh đủ 5 khớp lòng bàn tay):** 21 khớp Overall `88.69%` (on TP `91.68%`) · **14 khớp điều khiển** Overall `88.49%` (on TP `91.48%`).
- **Cổng `Gate14` (`all 14 kpts conf ≥ 0.5`):** All TP = 79.45% (6138/7726, 95% CI 78.5–80.3%) · **GT đủ 14 khớp TP** = 93.08% (5891/6329, 95% CI 92.4–93.7%).
- **Ultralytics `.val()` (`ultralytics_default_1_over_21`, mẫu số `7808` instances):** Box mAP50 `99.07%`, Box mAP50–95 `84.67%`, Pose mAP50 `86.72%`, Pose mAP50–95 `65.03%`.
- **Lát cắt loại trừ `runs/hand_only_trial/val.txt` (`n=7698`):** Recall@0.5 `97.13%`, AP@0.5 `98.93%`, PCK@0.2 Box `95.08%`, PCK@0.2 Palm 14-kpt `88.47%`, Gate14 (GT14 TP) `93.12%`.
- **Top-5 khớp sai lệch lớn nhất:** `thumb_tip` (12.1% palm / 6.3px), `wrist` (10.9% palm / 6.2px), `pinky_tip` (10.9% palm / 5.6px), `index_tip` (10.3% palm / 5.4px), `thumb_ip` (9.7% palm / 5.1px)
- **Top-5 khớp làm rớt Gate14 nhiều nhất:** `wrist` (397 lượt GT14 / 1471 tổng), `middle_tip` (29 lượt GT14 / 110 tổng), `thumb_tip` (19 lượt GT14 / 105 tổng), `pinky_mcp` (8 lượt GT14 / 64 tổng), `pinky_pip` (6 lượt GT14 / 68 tổng)

## 3. Phân tầng theo kích thước bàn tay (`px`) — Model tham chiếu `hand_pose.pt`

| Bucket (`max(bw,bh)`) | Số ảnh `n` | Recall@0.5 | PCK@0.2 Box | PCK@0.2 Palm (14 kpt) | Gate14 (GT14 TP) |
|---|---:|---:|---:|---:|---:|
| `<40 px` | 2 | 0.00% | 0.00% | 0.00% | N/A |
| `40-80 px` | 892 | 84.87% | 78.68% | 68.76% | 100.00% |
| `80-160 px` | 3682 | 98.13% | 95.88% | 87.47% | 98.96% |
| `>=160 px` | 3377 | 99.91% | 99.68% | 98.69% | 87.09% |

## 4. Chẩn đoán thu nhỏ bàn tay `[SCALED DIAGNOSTIC — KHÔNG PHẢI KHOẢNG CÁCH THẬT]`

| Model | Mức thu nhỏ | Số ảnh | Recall@0.5 | AP@0.5 | PCK@0.2 Box | PCK@0.2 Palm (14 kpt) | Gate14 (GT14 TP) |
|---|---|---:|---:|---:|---:|---:|---:|
| `hand_pose.pt` | `36px` | 300 | 76.67% | 81.31% | 69.05% | 54.52% | 100.00% |
| `hand_pose.pt` | `64px` | 300 | 93.33% | 96.06% | 87.70% | 74.43% | 97.86% |
| `hand_pose.pt@640` | `36px` | 300 | 85.00% | 92.02% | 78.43% | 65.36% | 100.00% |
| `hand_pose.pt@640` | `64px` | 300 | 97.33% | 98.95% | 92.78% | 80.60% | 100.00% |
| `hand_pose_v2.pt` | `36px` | 300 | 93.33% | 96.57% | 84.32% | 69.88% | 99.64% |
| `hand_pose_v2.pt` | `64px` | 300 | 97.00% | 98.76% | 92.76% | 84.24% | 99.66% |
| `hand_pose.onnx` | `36px` | 300 | 77.33% | 81.73% | 69.33% | 54.45% | 100.00% |
| `hand_pose.onnx` | `64px` | 300 | 93.33% | 96.31% | 87.57% | 74.43% | 98.21% |
| `hand_pose_v2.onnx` | `36px` | 300 | 93.33% | 96.44% | 84.46% | 69.57% | 99.64% |
| `hand_pose_v2.onnx` | `64px` | 300 | 97.00% | 98.79% | 92.67% | 84.05% | 99.66% |
| `runs/hand_only_trial/fit/weights/best.pt` | `36px` | 300 | 69.67% | 84.04% | 38.78% | 21.10% | 59.81% |
| `runs/hand_only_trial/fit/weights/best.pt` | `64px` | 300 | 97.00% | 98.24% | 90.70% | 65.36% | 88.66% |

## 5. Tốc độ suy luận thực tế trên khung webcam (20 warmup + 200 ảnh, tách GPU & CPU)

| Model | `imgsz` | Khung đo | GPU Mean (ms) | GPU P50 / P95 (ms) | GPU FPS | CPU Mean (ms) | CPU P50 / P95 (ms) | CPU FPS | ONNX Providers (GPU / CPU) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| `hand_pose.pt` | 512 | `640x480` | 6.44 | 6.374 / 7.032 | 155.29 | 58.76 | 57.178 / 64.825 | 17.02 | `-` |
| `hand_pose.pt@640` | 640 | `640x480` | 7.259 | 7.084 / 8.447 | 137.75 | 75.122 | 74.657 / 78.265 | 13.31 | `-` |
| `hand_pose_v2.pt` | 640 | `640x480` | 4.496 | 4.478 / 4.713 | 222.42 | 33.718 | 33.153 / 36.087 | 29.66 | `-` |
| `hand_pose.onnx` | 512 | `640x480` | 7.157 | 6.875 / 8.672 | 139.73 | 46.944 | 45.722 / 57.271 | 21.3 | `GPU: CUDAExecutionProvider, CPUExecutionProvider \| CPU: CPUExecutionProvider` |
| `hand_pose_v2.onnx` | 640 | `640x480` | 4.075 | 4.059 / 4.34 | 245.37 | 30.405 | 29.941 / 32.677 | 32.89 | `GPU: CUDAExecutionProvider, CPUExecutionProvider \| CPU: CPUExecutionProvider` |
| `runs/hand_only_trial/fit/weights/best.pt` | 640 | `640x480` | 7.246 | 7.047 / 8.446 | 138.02 | 75.177 | 75.031 / 77.745 | 13.3 | `-` |
| `hand_pose.pt` | 512 | `1280x720` | 5.838 | 5.805 / 6.067 | 171.3 | 45.825 | 45.224 / 49.978 | 21.82 | `-` |
| `hand_pose.pt@640` | 640 | `1280x720` | 6.512 | 6.497 / 6.832 | 153.57 | 59.82 | 59.39 / 61.539 | 16.72 | `-` |
| `hand_pose_v2.pt` | 640 | `1280x720` | 4.957 | 4.923 / 5.36 | 201.73 | 29.247 | 28.504 / 31.912 | 34.19 | `-` |
| `hand_pose.onnx` | 512 | `1280x720` | 7.7 | 7.723 / 8.452 | 129.87 | 46.332 | 44.61 / 54.993 | 21.58 | `GPU: CUDAExecutionProvider, CPUExecutionProvider \| CPU: CPUExecutionProvider` |
| `hand_pose_v2.onnx` | 640 | `1280x720` | 4.303 | 4.278 / 4.585 | 232.37 | 30.585 | 30.231 / 32.611 | 32.7 | `GPU: CUDAExecutionProvider, CPUExecutionProvider \| CPU: CPUExecutionProvider` |
| `runs/hand_only_trial/fit/weights/best.pt` | 640 | `1280x720` | 6.516 | 6.494 / 6.926 | 153.47 | 61.529 | 61.221 / 63.811 | 16.25 | `-` |

## 6. Bằng chứng trực quan kiểm chứng bằng mắt (`audit_top20_errors/`)

- Model tham chiếu: `hand_pose.pt`
- `audit_top20_errors/worst_pck/`: **20** ảnh TP lệch khớp nặng nhất.
- `audit_top20_errors/gate14_drops/`: **20** ảnh GT đủ 14 khớp nhưng rớt cổng `Gate14`.
- `audit_top20_errors/false_negatives/`: **20** ảnh model bỏ sót (10 bàn tay lớn nhất + 10 bàn tay nhỏ nhất).
- Quy ước màu: **Xanh lá = GT**, **Đỏ = Dự đoán khớp**, **Cam = Dự đoán lệch trên ca FN**, **Vòng vàng = khớp làm rớt `Gate14` kèm `conf`**.
