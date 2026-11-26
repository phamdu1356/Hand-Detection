# Báo cáo Benchmark Độc lập Ngoài Phân phối (Zero-Leakage OOD Suite)

- **Thời điểm chạy:** `2026-11-24 03:41:37`
- **Git commit:** `10be93648d61` (dirty=`True`)
- **Kiểm chứng chống rò rỉ (SHA-256 vs `hand-keypoints/train`, `18724` ảnh):**
  - `freihand_eval`: `0` tệp trùng byte với train · chi tiết: `{'source_total': 3960, 'intra_dups_removed': 0, 'n_images': 3960, 'n_gt_instances': 3960}`
  - `cmu_handdb_test`: `0` tệp trùng byte với train · chi tiết: `{'splits_loaded': ['manual_test'], 'total_json_files': 846, 'instances_by_split': {'manual_test': 846}, 'dup_hands_skipped': 0, 'n_images': 447, 'n_gt_instances': 846}`
  - `cmu_handdb_all`: `0` tệp trùng byte với train · chi tiết: `{'splits_loaded': ['manual_test', 'manual_train'], 'total_json_files': 2758, 'instances_by_split': {'manual_test': 846, 'manual_train': 1910}, 'dup_hands_skipped': 0, 'n_images': 1445, 'n_gt_instances': 2756}`
  - `coco_wholebody_val2017`: `0` tệp trùng byte với train · chi tiết: `{'pos_images': 1568, 'pos_gt_hands': 3956, 'ignore_hand_boxes': 0, 'neg_person_free_images': 2307}`
  - `hagrid_local_test`: `0` tệp trùng byte với train · chi tiết: `{'intra_dups_removed': 1, 'n_images': 479, 'counts_by_class': {'like': 120, 'none': 119, 'ok': 120, 'palm': 120}}`

## 1. Bảng tổng hợp OOD trên toàn bộ 3 tập 21 khớp gộp lại (Zero-Leakage Pooled + Macro)

| Model | `imgsz` | Pooled Recall@0.5 | Pooled Prec@0.5 | Pooled AP@0.5 | Pooled PCK@0.2 Box | Pooled PCK@0.2 Palm (14 kpt) | Pooled Gate14 (GT14 TP) | Macro Recall (3 bộ) | Macro PCK14 (3 bộ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hand_pose.pt` | 512 | 30.84% | 78.23% | 38.87% | 21.36% | 9.84% | 99.88% | 29.50% | 9.41% |
| `hand_pose.pt@640` | 640 | 33.32% | 74.96% | 41.47% | 22.38% | 10.27% | 99.49% | 32.02% | 9.92% |
| `hand_pose.onnx` | 512 | 30.95% | 80.15% | 39.54% | 21.47% | 9.87% | 99.94% | 29.65% | 9.45% |
| `hand_pose_v2.pt` | 640 | 35.65% | 82.63% | 45.43% | 24.24% | 11.76% | 99.82% | 35.14% | 11.55% |
| `runs/hand_only_trial/fit/weights/best.pt` | 640 | 31.43% | 80.28% | 40.32% | 21.10% | 9.73% | 99.04% | 30.00% | 9.28% |

> **Lưu ý phương pháp:** Trên `CMU HandDB` và `COCO-WholeBody val2017` (ảnh toàn cảnh nhiều người), một số bàn tay phụ trong nền không được gán nhãn (`valid=False` có box rỗng `[0,0,0,0]`), nên cột **Precision@0.5** của hai bộ này là **cận dưới bảo thủ**; tỷ lệ báo giả thực (FPR) được đo chính xác ở Mục 4 (`2.307` ảnh nền COCO không có người) và Mục 5 (`119` ảnh HaGRID `no_gesture`). Các chỉ số `Recall@0.5`, `PCK@0.2 Box/Palm (14 khớp)` và `Gate14 (GT14 TP)` là chính xác tuyệt đối.

## 2. Chi tiết theo từng bộ dataset 21 khớp OOD (tách riêng domain cận cảnh vs toàn cảnh)

| Dataset | Số tay GT | Model | Recall@0.5 | Precision@0.5 | AP@0.5 | PCK@0.2 Box | PCK@0.2 Palm (14 kpt) | PCK14 (on TP) | Gate14 (GT14 TP) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `freihand_eval` | 3960 | `hand_pose.pt` | 65.56% | 91.73% | 77.99% | 45.67% | 21.02% | 32.07% | 99.96% |
| `freihand_eval` | 3960 | `hand_pose.pt@640` | 67.47% | 91.82% | 80.82% | 45.60% | 20.73% | 30.72% | 99.51% |
| `freihand_eval` | 3960 | `hand_pose.onnx` | 65.56% | 91.73% | 78.02% | 45.67% | 21.03% | 32.07% | 99.96% |
| `freihand_eval` | 3960 | `hand_pose_v2.pt` | 65.88% | 93.18% | 80.35% | 45.86% | 22.46% | 34.09% | 99.89% |
| `freihand_eval` | 3960 | `runs/hand_only_trial/fit/weights/best.pt` | 66.09% | 94.27% | 81.86% | 44.79% | 20.73% | 31.37% | 99.08% |
| `cmu_handdb_test` | 846 | `hand_pose.pt` | 14.42% | 58.94% | 20.81% | 10.23% | 4.41% | 30.27% | 100.00% |
| `cmu_handdb_test` | 846 | `hand_pose.pt@640` | 18.09% | 63.75% | 24.20% | 13.11% | 6.03% | 33.01% | 100.00% |
| `cmu_handdb_test` | 846 | `hand_pose.onnx` | 15.72% | 65.20% | 24.30% | 11.14% | 4.81% | 30.29% | 100.00% |
| `cmu_handdb_test` | 846 | `hand_pose_v2.pt` | 21.75% | 79.65% | 31.70% | 16.03% | 7.45% | 33.89% | 100.00% |
| `cmu_handdb_test` | 846 | `runs/hand_only_trial/fit/weights/best.pt` | 15.25% | 58.11% | 20.42% | 10.47% | 4.37% | 28.35% | 99.22% |
| `cmu_handdb_all` | 2756 | `hand_pose.pt` | 17.71% | 66.85% | 27.12% | 13.17% | 5.98% | 33.55% | 99.59% |
| `cmu_handdb_all` | 2756 | `hand_pose.pt@640` | 20.61% | 68.68% | 30.39% | 15.41% | 7.16% | 34.49% | 99.30% |
| `cmu_handdb_all` | 2756 | `hand_pose.onnx` | 18.21% | 71.51% | 29.59% | 13.56% | 6.09% | 33.20% | 99.80% |
| `cmu_handdb_all` | 2756 | `hand_pose_v2.pt` | 30.66% | 80.63% | 42.06% | 21.60% | 10.01% | 32.40% | 99.64% |
| `cmu_handdb_all` | 2756 | `runs/hand_only_trial/fit/weights/best.pt` | 17.38% | 64.73% | 25.08% | 12.79% | 5.57% | 31.79% | 98.96% |
| `coco_wholebody_val2017` | 3956 | `hand_pose.pt` | 5.23% | 31.99% | 5.22% | 2.61% | 1.23% | 23.64% | 99.50% |
| `coco_wholebody_val2017` | 3956 | `hand_pose.pt@640` | 7.99% | 31.38% | 7.44% | 3.88% | 1.88% | 23.57% | 99.68% |
| `coco_wholebody_val2017` | 3956 | `hand_pose.onnx` | 5.18% | 34.80% | 5.29% | 2.65% | 1.23% | 23.77% | 100.00% |
| `coco_wholebody_val2017` | 3956 | `hand_pose_v2.pt` | 8.87% | 46.37% | 10.42% | 4.35% | 2.18% | 24.57% | 99.71% |
| `coco_wholebody_val2017` | 3956 | `runs/hand_only_trial/fit/weights/best.pt` | 6.52% | 38.97% | 6.85% | 3.06% | 1.53% | 23.62% | 98.80% |

## 3. Phân tầng kích thước bàn tay THỰC TẾ trên toàn bộ OOD (gồm tay nhỏ `<40 px` & `40–80 px` ngoài đời)

| Model | `<40 px` (Rec / PCK14) | `40-80 px` (Rec / PCK14) | `80-160 px` (Rec / PCK14) | `>=160 px` (Rec / PCK14) |
|---|---:|---:|---:|---:|
| `hand_pose.pt` | 1.39% / 0.20% (n=2659) | 11.20% / 2.98% (n=1812) | 50.62% / 16.42% (n=5988) | 9.39% / 2.61% (n=213) |
| `hand_pose.pt@640` | 3.61% / 0.69% (n=2659) | 14.74% / 3.89% (n=1812) | 52.92% / 16.69% (n=5988) | 11.27% / 2.75% (n=213) |
| `hand_pose.onnx` | 1.20% / 0.18% (n=2659) | 10.98% / 2.94% (n=1812) | 50.90% / 16.47% (n=5988) | 11.27% / 2.89% (n=213) |
| `hand_pose_v2.pt` | 4.32% / 0.86% (n=2659) | 18.71% / 5.66% (n=1812) | 55.44% / 18.63% (n=5988) | 14.55% / 5.44% (n=213) |
| `runs/hand_only_trial/fit/weights/best.pt` | 1.13% / 0.23% (n=2659) | 13.63% / 3.41% (n=1812) | 50.84% / 16.06% (n=5988) | 15.49% / 3.38% (n=213) |

## 4. Tỷ lệ báo giả trên ảnh nền đời thực không có người/tay (`COCO val2017` person-free)

| Model | Số ảnh nền | Box FPR (`conf ≥ 0.5`) | Command FPR (`hand_command != none`) | FP Boxes / 100 ảnh |
|---|---:|---:|---:|---:|
| `hand_pose.pt` | 2307 | 13.05% (301/2307, 95% CI 11.7–14.5%) | 11.83% (273/2307, 95% CI 10.6–13.2%) | 15.128 |
| `hand_pose.pt@640` | 2307 | 16.86% (389/2307, 95% CI 15.4–18.4%) | 15.30% (353/2307, 95% CI 13.9–16.8%) | 21.153 |
| `hand_pose.onnx` | 2307 | 10.01% (231/2307, 95% CI 8.9–11.3%) | 9.10% (210/2307, 95% CI 8.0–10.3%) | 11.443 |
| `hand_pose_v2.pt` | 2307 | 10.36% (239/2307, 95% CI 9.2–11.7%) | 9.54% (220/2307, 95% CI 8.4–10.8%) | 12.44 |
| `runs/hand_only_trial/fit/weights/best.pt` | 2307 | 8.11% (187/2307, 95% CI 7.1–9.3%) | 7.20% (166/2307, 95% CI 6.2–8.3%) | 9.45 |

## 5. Cử chỉ đời thực FullHD 0,5–4 m (`HaGRID Test`: `like`, `ok`, `palm`, `no_gesture`) qua `hand_command()`

| Model & Chế độ | Macro-F1 (3 cử chỉ) | Accuracy (360 cử chỉ) | `like` Rec | `ok` Rec | `palm` Rec | `no_gesture` Command FPR | Nguyên nhân sai (Mất tay / Rớt Gate14 / Sai cử chỉ) |
|---|---:|---:|---:|---:|---:|---:|---|
| `hand_pose.pt` (Raw YOLO) | 41.09% | 47.50% | 67.50% | 0.00% | 75.00% | 53.78% (64/119, 95% CI 44.8–62.5%) | 73 / 0 / 116 |
| `hand_pose.pt` (Perception ROI-Zoom) | 43.13% | 52.22% | 75.83% | 0.83% | 80.00% | 65.55% (78/119, 95% CI 56.6–73.5%) | 53 / 0 / 119 |
| `hand_pose.pt@640` (Raw YOLO) | 44.96% | 54.72% | 77.50% | 0.00% | 86.67% | 57.14% (68/119, 95% CI 48.2–65.7%) | 43 / 0 / 120 |
| `hand_pose.pt@640` (Perception ROI-Zoom) | 43.13% | 52.22% | 75.83% | 0.83% | 80.00% | 65.55% (78/119, 95% CI 56.6–73.5%) | 53 / 0 / 119 |
| `hand_pose.onnx` (Raw YOLO) | 41.97% | 48.61% | 70.00% | 0.83% | 75.00% | 59.66% (71/119, 95% CI 50.7–68.0%) | 69 / 0 / 116 |
| `hand_pose.onnx` (Perception ROI-Zoom) | 44.55% | 53.61% | 78.33% | 1.67% | 80.83% | 70.59% (84/119, 95% CI 61.9–78.0%) | 48 / 0 / 119 |
| `hand_pose_v2.pt` (Raw YOLO) | 46.47% | 58.89% | 80.00% | 0.83% | 95.83% | 69.75% (83/119, 95% CI 61.0–77.3%) | 37 / 0 / 111 |
| `hand_pose_v2.pt` (Perception ROI-Zoom) | 46.31% | 60.00% | 83.33% | 0.00% | 96.67% | 71.43% (85/119, 95% CI 62.7–78.8%) | 26 / 0 / 118 |
| `runs/hand_only_trial/fit/weights/best.pt` (Raw YOLO) | 43.63% | 50.83% | 75.00% | 0.00% | 77.50% | 51.26% (61/119, 95% CI 42.4–60.1%) | 41 / 0 / 136 |
| `runs/hand_only_trial/fit/weights/best.pt` (Perception ROI-Zoom) | 44.93% | 55.00% | 79.17% | 0.00% | 85.83% | 57.98% (69/119, 95% CI 49.0–66.5%) | 27 / 0 / 135 |
