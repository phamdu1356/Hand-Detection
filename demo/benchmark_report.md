# Flycam Hand Detection & 2-Stage 10m Benchmark Report

- **Timestamp**: `2026-11-19 10:33:06`
- **Mode**: `REAL_TRAINED_MODEL`
- **Gate (>= 95.0% 10m Accuracy)**: **FAIL (min 10m metric = 32.00% < 95.0%)**
- **PyTorch Model**: `hand_pose.pt`
- **ONNX Model**: `hand_pose.onnx`
- **Stage-2B Heatmap Refiner**: `N/A` (`active=False`)
- **Dataset**: `Real hand-keypoints (../datasets/hand-keypoints/data.yaml, split=val, n=50)`

## 1. Detection, Keypoints & Static Gesture Accuracy (2m vs 10m)

| Scenario | Pipeline Mode | Recall@0.5 (%) | mAP50 (%) | Mean IoU | PCK@0.2 Box (%) | PCK@0.2 Palm (%) | Gesture Acc (%) | Mean Latency (ms) | FPS |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Close 2m (640x480) | `full_frame_auto` | 100.00 | 100.00 | 0.9039 | 83.14 | 69.62 | 36.00 | 6.23 | 160.64 |
| 10m 1080p Person+Hand (Naive Full-Frame) | `naive_full_frame` | 2.00 | 0.15 | 0.0509 | 2.19 | 0.38 | 6.00 | 5.39 | 185.57 |
| 10m 1080p Person+Hand (2-Stage ROI Zoom) | `2stage_roi_zoom` | 98.00 | 97.96 | 0.7334 | 76.29 | 59.24 | 32.00 | 30.21 | 33.11 |
| 10m 4K Person+Hand (2-Stage ROI Zoom) | `2stage_roi_zoom` | 98.00 | 96.12 | 0.7948 | 81.43 | 66.57 | 40.00 | 39.90 | 25.06 |
| 10m 1080p ONNX Person+Hand (2-Stage ROI Zoom) | `onnx_2stage_roi_zoom` | 98.00 | 97.92 | 0.7352 | 76.10 | 58.00 | 32.00 | 34.32 | 29.14 |
| 10m 1080p Disembodied (Stage 1B Tile Fallback) | `stage1b_tile_fallback` | 93.33 | 90.22 | 0.7671 | 75.56 | 58.41 | 40.00 | 91.07 | 10.98 |

## 2. Per-Class Static Gesture Accuracy (10m 1080p 2-Stage vs Naive)

| Gesture | Close 2m (%) | 10m Naive (%) | 10m 2-Stage ROI Zoom (%) |
| :--- | ---: | ---: | ---: |
| `palm` | 75.0 | 0.0 | 75.0 |
| `fist` | 38.5 | 7.7 | 30.8 |
| `ok` | N/A | N/A | N/A |
| `one` | 0.0 | 0.0 | 0.0 |
| `peace_inverted` | 0.0 | 0.0 | 0.0 |
| `three` | 0.0 | 0.0 | 0.0 |
| `four` | 71.4 | 0.0 | 57.1 |
| `like` | 83.3 | 33.3 | 83.3 |

## 3. Dynamic Swipe Accuracy (2m vs 10m)

| Scenario | Overall Swipe Acc (%) | LEFT (%) | RIGHT (%) | UP (%) | DOWN (%) | STAND_STILL (%) | Mean Latency (ms) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Close 2m | 100.00 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 5.55 |
| 10m 1080p Naive | 20.00 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 5.31 |
| 10m 1080p 2-Stage ROI Zoom | 100.00 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 8.21 |

## 4. Inference Latency & FPS: PyTorch `.pt` vs `.onnx` (hand_pose)

| Benchmark Variant | Mean (ms) | P50 (ms) | P95 (ms) | FPS |
| :--- | ---: | ---: | ---: | ---: |
| PyTorch (.pt) | 640x640_ROI | 3.70 | 3.69 | 3.75 | 270.50 |
| PyTorch (.pt) | 1920x1080_Full | 4.29 | 4.29 | 4.44 | 233.18 |
| ONNX (.onnx) | 640x640_ROI | 3.44 | 3.43 | 3.50 | 290.68 |
| ONNX (.onnx) | 1920x1080_Full | 3.82 | 3.82 | 3.84 | 261.59 |

## 5. Real Stage-1 `yolo11n-pose.pt` 10m Person/Wrist Localization Check

| Source Image | Simulated 10m Scale | Persons Detected | Wrists Detected | Latency (ms) |
| :--- | :--- | ---: | ---: | ---: |
| `zidane.jpg` | 10m_1080p | 2 | 4 | 6.20 |
| `zidane.jpg` | 10m_4K | 2 | 4 | 9.05 |
| `bus.jpg` | 10m_1080p | 1 | 2 | 6.16 |
| `bus.jpg` | 10m_4K | 3 | 6 | 9.14 |
