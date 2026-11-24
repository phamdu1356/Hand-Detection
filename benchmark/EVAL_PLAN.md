# Kế hoạch Tổng thể Đánh giá & Đo Điểm Thực tế (3 Tầng)

**Tác giả dự án:** `phamdu1356` · **Ngày lập:** `2026-11-22` · **Bổ trợ cho:** [benchmark/PLAN.md](PLAN.md)

Mục tiêu là xây dựng hệ thống đánh giá **3 tầng tách bạch, tái lập được và chống ảo điểm**, trả lời trung thực câu hỏi *"mô hình và hệ thống đang đạt bao nhiêu phần trăm trong thực tế"* mà không nén tất cả thành một con số `%` mơ hồ.

---

## 1. Lộ trình thực hiện từng bước (4 Bước)

| Bước | Phạm vi | Đầu ra chính | Trạng thái |
|---|---|---|---|
| **Bước 1** | **Tầng 1 — Benchmark ảnh tĩnh sạch + Schema nền tảng** | `benchmark/schema.py`, `tools/eval_tier1_static.py`, `tests/test_tier1_eval.py`, báo cáo `tier1_report.md` trên toàn bộ `7.953` ảnh sạch + `audit_top20_errors/` | **Hoàn thành & đã rà soát** (`2026-11-23`) |
| **Bước 2** | **Công cụ quay video chuẩn & duyệt nhãn mù** | `tools/record_eval_session.py`, `tools/review_labels.py`, test kiểm chứng schema `manifest.csv` / `labels.csv` | Sẵn sàng triển khai |
| **Bước 3** | **Tầng 2 & Tầng 3 — Bộ chấm độc lập + 24 ca kiểm thử** | `tools/score_eval.py` (Macro-F1, CM, Error Attribution 4 nhóm, Event Recall song song Decoupled & End-to-End, `FP/min`, Stop P50/P95, sai số đích, A vs B 95% CI, `failures.csv`) + `tests/test_eval_suite.py` | Chờ xong Bước 2 |
| **Bước 4** | **Chạy Pilot (`dev`), Sim3D (`122` lượt) & Đánh giá Chính thức (`test`)** | Bộ video pilot (`70` lượt + `4` phút trung tính), mô phỏng `122` lượt, và tập test chính thức (`210–350` lượt + `12–20` phút trung tính) | Chờ xong Bước 3 |

---

## 2. Tồn kho thông tin đã kiểm chứng (Baseline ngày 22/11/2026)

### 2.1. Trọng số trên máy
| Tệp | Kiến trúc / Params | `imgsz` | SHA-256 (16 ký tự đầu) | Ghi chú |
|---|---:|---:|---|---|
| `hand_pose.pt` | YOLO11s-pose (10,06M) | 512 | `093ae826a31a3014` | **Bản chuẩn mặc định** (`≡ hand_pose_v3.pt`) |
| `hand_pose.onnx` | ONNX static `[1,3,512,512]` | 512 | `2da512c06d90869b` | ONNX của bản chuẩn (`≡ hand_pose_v3.onnx`) |
| `hand_pose_v2.pt` | YOLO11n-pose (2,96M) | 640 | `007b592edee3ef46` | Bản nhẹ (6,0 MB) |
| `hand_pose_v2.onnx` | ONNX static `[1,3,640,640]` | 640 | `7c7a7803e5de30c3` | ONNX của bản nhẹ (12 MB) |
| `runs/hand_only_trial/fit/weights/best.pt` | YOLO11s-pose (10,06M) | 640 | `901d7d519b7a19f2` | Tinh chỉnh 4 epoch (`box 87,08% / pose 67,65%`) |

### 2.2. Phần cứng & Môi trường
- **GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (8.151 MiB VRAM), `torch 2.15.0+cu132`, CUDA sẵn sàng.
- **ONNX Runtime:** `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`.
- **Camera:** `/dev/video0` – `/dev/video3`.
- **Bộ kiểm thử hiện có:** `309 passed` (`pytest`, gồm `21` ca tất định của Bước 1), đã cài editable qua `pyproject.toml`.

### 2.3. Kiểm kê tập dữ liệu `../datasets/hand-keypoints`
- `train`: `18.776` ảnh (`52` trùng nội bộ → `18.724` duy nhất).
- `val`: `7.992` ảnh (toàn bộ kích thước `224×224`, `1` bàn tay/ảnh, 68 cột chuẩn, `vis ∈ {0, 2}`). Khi lọc `train` trước: **`38` tệp trùng byte với `train`** (`27` mã băm duy nhất) + **`1` tệp trùng nội bộ ngoài `train`** (`IMG_00019779.jpg`, tổng loại `39` tệp) → còn **`7.953` ảnh `val` sạch**.
- Quét gần trùng `dHash`: **`6.095/7.953` (`76,6%`)** ảnh `val` có ảnh rất giống trong `train` (`dHash ≤ 2`, `31.317` cặp; kiểm tra ngẫu nhiên 50 cặp cho `median MAD = 4,7/255`, `median corr = 0,975`), và `19.944` cặp gần trùng nội bộ `val` (`dHash ≤ 4`).
- Trên `7.953` ảnh sạch: `991` ảnh có cạnh box vượt nhẹ biên `[0, 1]`, `184` ảnh có khớp nhìn thấy vượt biên `[0, 1]` (`437` khớp bị loại khỏi PCK; Ultralytics `.val()` chấm `7.808` instances), **`1.092` ảnh (~13,7%) thiếu ít nhất 1 trong 5 khớp lòng bàn tay trong khung** (`1.005` `vis=0` + `87` ngoài biên), và **`6.543/7.953` (~82,3%) đủ cả 14 khớp điều khiển trong khung**.
- Kích thước bàn tay `max(bw, bh)` trên `val`: median **`142,0 px`**, **`2/7.953 (0,03%) < 40 px`** (`40–80 px`: `892`, `80–160 px`: `3.682`, `>=160 px`: `3.377`).

---

## 3. Số lượng test chuẩn hóa cho từng tầng

| Tầng / Loại test | Số lượng mẫu | Độ chính xác thống kê (95% CI) |
|---|---|---|
| **Tầng 1 — Ảnh tĩnh sạch** | **Toàn bộ `7.953` ảnh** × 5 trọng số độc nhất + `200` vòng đo latency (20 warmup) GPU & CPU | **±1,1%** trên toàn tập |
| **Tầng 2 & 3 — Pilot (`dev`)** | `2` phiên (`1,5 m`, `3,0 m`): **`70` lượt lệnh** + **`4` phút trung tính** | Chỉ dùng gỡ lỗi & khóa ngưỡng |
| **Tầng 2 & 3 — Chính thức (`test`)** | Tối thiểu 3 người × 2 cự ly: **`210` lượt lệnh** (`30`/lệnh) + **`12` phút trung tính**.<br>Mức vững (5 người × 2 cự ly): **`350` lượt lệnh** (`50`/lệnh) + **`20` phút trung tính**.<br>Full A vs B (5 người × 2 giao diện × 2 cự ly): **`700` lượt** + **`40` phút trung tính**. | **±4% tổng** / ±10% từng lệnh (`n=30`) |
| **Mô phỏng vòng kín 3D (`sim3d`)** | `30 seeds × 2 giao diện × 2 mức nhiễu` + `2` ca kiểm trục/đứng yên = **`122` lượt** | ±10% mỗi điều kiện |
| **Unit test kiểm chứng chính bộ chấm** | **`24+` ca kiểm thử tất định** (16 ca ở Bước 1 + 24 ca tổng ở Bước 3) | Tất định 100% |

---

## 4. 6 lớp bảo chứng để tin tưởng bộ test là đúng

1. **Ca tính tay chuẩn vàng (*Golden Hand-Calculated Cases*):** CSV nhỏ (15–30 dòng) tự tính bằng máy tính cầm tay; bộ chấm phải khớp tới chữ số thập phân thứ 4.
2. **3 phép thử đối nghịch (*Sanity Invariants*):** Oracle = `100%` Recall/F1 & `0` FP/min; Null/Blind = `0%` Recall & `100%` stop timeout (không được in `n/a`); Shuffled-label rớt về mức ngẫu nhiên (~12,5%).
3. **Bằng chứng trực quan mọi ca sai:** Tầng 1 xuất `audit_top20_errors/` (vẽ GT xanh lá / dự đoán đỏ); Tầng 2/3 xuất `failures.csv` + thumbnail đúng giây sai.
4. **Duyệt nhãn mù (*Blind Labeling*):** `review_labels.py` không chạy model hay hiện dự đoán khi đánh dấu `onset_s`/`offset_s`; kiểm chéo ngẫu nhiên 20% đoạn (`< 0,15 s`).
5. **Khóa tái lập & chống rò rỉ:** Mọi báo cáo ghi `git_commit`, `SHA-256` trọng số/dữ liệu, toàn bộ `ControlConfig`, mẫu số `k/n` + **95% CI**; chặn trùng hash & trùng người giữa `dev` và `test`.
6. **Đối chứng cảm giác lái trực tiếp (*Live Calibration*):** Lái thử `demo/fly_sim.py --camera 0 --backend sim` trong đúng điều kiện phòng/cự ly để kiểm tra thứ hạng lệnh khó/dễ khớp với bảng điểm.

---

## 5. Tóm tắt 9 lỗ hổng đã rà soát & biện pháp khóa

1. **`cue_s ≠ onset_s`:** tách 2 cột + duyệt nhãn mù bằng `review_labels.py` + bỏ biên `±0,1 s`.
2. **`val` có 0% tay `<40 px`:** ghi rõ Tầng 1 là *cận cảnh (median 139 px)*, tách riêng *scaled diagnostic*, đo theo mét chỉ công bố từ video thật.
3. **Video replay vòng mở thiên vị B (`hand_joystick`):** `rgb_replay` chỉ chấm nhận lệnh/FP/dừng; sai số đích & hồi tiếp camera dịch chuyển chấm bằng `sim3d` vòng kín / live.
4. **Lỗi dây chuyền FSM (*Cascade Failure*):** báo song song *Decoupled Command Recall* và *End-to-End Flight Recall*.
5. **Near-duplicate & thiếu `person_id`:** lọc SHA-256 + cảnh báo dHash (`Hamming ≤ 4`) và công bố rõ giới hạn dataset.
6. **Lẫn lỗi cổng `14-kpt >= 0.5` với lỗi YOLO:** tách 4 bucket nguyên nhân lỗi (`no_hand_detected` / `gate14_rejected` / `wrong_class_or_angle` / `temporal_hold_failed`).
7. **Ảo giác FPS trên RTX 5050:** đo latency bắt buộc cả GPU và CPU + ghi `timestamps.csv` thật cho webcam.
8. **Quá khớp vào `N = 1`:** tách `pilot_dev` khỏi `heldout_test` (`N >= 3` người mới), luôn in `k/n` + 95% CI.
9. **Chỉnh ngưỡng sau khi nhìn tập `test`:** khóa chặt `split=dev` vs `split=test` bằng mã hash ngay trong `benchmark/schema.py`.
