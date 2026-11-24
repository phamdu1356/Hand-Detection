# Điều khiển drone bằng cử chỉ bàn tay

**Tác giả dự án: phamdu1356**

Người dùng đứng đối diện camera RGB gắn trên drone và ra lệnh bằng một bàn tay. Mô hình YOLO Pose tìm 21 điểm khớp tay trong từng khung hình. Bộ điều khiển đọc hình dạng bàn tay và hướng ngón trỏ, xác nhận cử chỉ qua nhiều khung hình rồi gửi lệnh đến drone mô phỏng. Bàn tay có thể đổi vị trí trên ảnh khi camera di chuyển. Tọa độ tâm bàn tay không được dùng để ra lệnh di chuyển trong chế độ mặc định.

**Phạm vi đã triển khai.** Chương trình nhận ảnh từ thiết bị video, luồng video tương thích OpenCV hoặc tệp ghi hình. Hai đầu ra hiện có là mô phỏng vật lý PyBullet và bộ mô phỏng đơn giản `mock`. Việc điều khiển động cơ trên drone thật còn phụ thuộc mẫu drone, giao thức điều khiển và thử nghiệm với phần cứng.

## Cử chỉ

Hướng trái/phải tính theo người đang đối diện camera. Đầu vào là ảnh gốc, chưa lật gương. Giữ tay trong khung hình cho đến khi màn hình xác nhận trạng thái.

| Cử chỉ | Thời điểm | Phản ứng |
|---|---|---|
| Giơ ngón cái, các ngón còn lại khép | Khi đang ở mặt đất, giữ khoảng 1 giây | Cất cánh đến độ cao đặt trước, mặc định 1,5 m |
| Duỗi riêng ngón trỏ, chỉ sang trái, phải, lên hoặc xuống | Khi đang bay tại chỗ, giữ hướng khoảng 0,25 giây | Di chuyển theo hướng chỉ trong khi còn thấy tín hiệu |
| Xòe bàn tay, thu ngón trỏ hoặc để tay ra khỏi hình | Trong lúc di chuyển theo ngón trỏ | Lệnh vận tốc về 0 ngay tại khung xử lý mất hướng. Drone mô phỏng cần thời gian để hãm |
| Chụm ngón cái và ngón trỏ thành dấu OK | Khi đang bay tại chỗ, giữ khoảng 1 giây | Hạ cánh |

Chương trình dùng tỷ lệ thời gian thấy đúng dấu trong cửa sổ xác nhận để chịu được vài khung mất khớp. Khi hình bị gián đoạn quá lâu, thời gian xác nhận bắt đầu lại. Các thông số nằm trong [cấu hình điều khiển](dronectl/config.py).

## Chạy thử

Chạy các lệnh từ thư mục gốc. Dự án đã được thử bằng Python 3.11. Nếu đã có môi trường `.venv` của dự án, kích hoạt nó. Để cài mới:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r demo/requirements.txt pytest
```

**Thử nhanh từ đầu đến cuối, không cần camera hay tệp mô hình.** Kịch bản tổng hợp ra hiệu cất cánh, dịch sang phải, dừng và hạ cánh. Lệnh thứ hai kiểm tra các lệnh tính lại có trùng với bản ghi hay không.

```bash
python demo/fly_sim.py --synthetic --backend mock --no-gui --log logs/hand.csv
python demo/fly_sim.py --replay logs/hand.csv --backend mock --no-gui
```

**Chạy mô phỏng camera gắn cố định trên drone.** Các điểm khớp 3D của bàn tay được chiếu qua camera ở vị trí thực tế của drone mô phỏng sau mỗi bước bay. Kịch bản kiểm tra bốn hướng, dừng và hạ cánh. Tọa độ khớp trong phép thử này được sinh từ hình học. Bỏ `--no-gui` để xem mô phỏng.

```bash
python tools/fly_3d.py --scenario hand --noise none --no-gui
python tools/fly_3d.py --scenario hand --noise webcam --seed 500 --repeat 30 --no-gui
```

**Chạy nhận diện trên ảnh RGB.** Đặt `hand_pose.pt` hoặc `hand_pose.onnx` trong thư mục gốc, hoặc truyền đường dẫn bằng `--pose-model`. Mặc định chỉ cần mô hình bàn tay. Với tệp đã ghi, bộ lọc dùng thời gian của video để một cử chỉ giữ 1 giây có cùng ý nghĩa khi máy xử lý nhanh hoặc chậm.

```bash
python demo/fly_sim.py --video ban_ghi_tu_drone.mp4 --backend sim --no-gui --log logs/flight.csv
python demo/fly_sim.py --camera 0 --backend sim
```

Nếu drone xuất luồng video, truyền địa chỉ luồng vào `--video`. Với thiết bị USB, thay `0` bằng số thứ tự thiết bị. Khi mở cửa sổ, nhấn `q` để thoát và `Space` để yêu cầu dừng. Cách đặt mô hình khi dùng Docker nằm trong [hướng dẫn mô hình](demo/models/README.md).

## Huấn luyện và đánh giá

Bộ **Hand Keypoints** cung cấp nhãn khung bao và 21 điểm khớp. Các nhãn này dùng để huấn luyện mô hình tư thế tay. Danh sách lệnh bay và cách xác nhận cử chỉ nằm trong bộ điều khiển. Notebook [huấn luyện trên Colab](train_hand_pose_colab.ipynb) và [notebook thí nghiệm ban đầu](CodeTrain.ipynb) lưu các bước huấn luyện trước đây.

Thử nghiệm tinh chỉnh cục bộ dùng cùng một trọng số khởi đầu, bốn epoch, ảnh 640 px và seed 42. Trong 2.048 ảnh train và 256 ảnh validation được chọn, trình nạp bỏ qua 52 và 6 ảnh có nhãn lỗi. Hai bản mô hình được chấm trên cùng **512 ảnh khác** thuộc tập validation gốc và tách riêng cho lần tinh chỉnh này.

| Mô hình | mAP50–95 khung bàn tay | mAP50–95 các khớp |
|---|---:|---:|
| Mô hình đang dùng | 77,46% | **67,74%** |
| Sau tinh chỉnh 4 epoch | **87,08%** | 67,65% |

Bản mặc định vẫn là `hand_pose.pt` vì việc đọc cử chỉ dựa vào các khớp. Mức chênh 0,09 điểm phần trăm chưa cho thấy bản mới cải thiện phần này. Có thể thử riêng trọng số mới tại `runs/hand_only_trial/fit/weights/best.pt`. Để lặp lại phép so sánh, dùng một thư mục kết quả mới:

```bash
python tools/finetune_hand.py --data ../datasets/hand-keypoints --weights hand_pose.pt --output runs/hand_trial_02
```

### Benchmark Tầng 1 — Phát hiện tay, 21 khớp, cổng `Gate14` và tốc độ

Bộ đánh giá Tầng 1 ([tools/eval_tier1_static.py](tools/eval_tier1_static.py)) chấm tất cả các trọng số độc nhất trên **toàn bộ `7.953` ảnh `val` sạch** của `../datasets/hand-keypoints` (đã loại bằng SHA-256 `38` tệp trùng byte với `train` — `27` mã băm duy nhất — và `1` tệp trùng nội bộ ngoài `train`), dùng chung cấu hình `rect=False` và chuẩn OKS mặc định của Ultralytics (`σ = 1/21`, đồng nhất với bảng tinh chỉnh phía trên). Chi tiết đầy đủ nằm tại [báo cáo Tầng 1](benchmark/results/tier1/tier1_report.md) và [kế hoạch đánh giá 3 tầng](benchmark/EVAL_PLAN.md).

| Mô hình | `imgsz` | Recall@0.5 | AP@0.5 | Pose mAP50–95 (`σ=1/21`) | PCK@0.2 Box | PCK@0.2 Palm (14 khớp) | Gate14 (GT14 TP) | Tay nhỏ 36 px (Recall / PCK14)* | GPU FPS (`640×480` / `1280×720`) | CPU FPS (`640×480` / `1280×720`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hand_pose.pt` *(≡ `v3.pt`)* | 512 | 97,37% | 98,97% | **66,75%** | 95,52% | 89,09% | **94,88%** | 76,7% / 54,5% | 155,3 / 171,3 | 17,0 / 21,8 |
| `hand_pose.onnx` *(≡ `v3.onnx`)* | 512 | **97,38%** | 98,97% | 66,69% | **95,53%** | **89,10%** | **94,88%** | 77,3% / 54,5% | 139,7 / 129,9 | 21,3 / 21,6 |
| `hand_pose.pt@640` | 640 | 97,69% | 99,04% | 64,07% | 95,67% | 89,20% | 80,04% | 85,0% / 65,4% | 137,8 / 153,6 | 13,3 / 16,7 |
| `hand_pose_v2.pt` (nano) | 640 | 96,28% | 98,64% | 66,69% | 93,55% | 86,90% | 94,75% | **93,3% / 69,9%** | 222,4 / 201,7 | 29,7 / **34,2** |
| `hand_pose_v2.onnx` (nano) | 640 | 96,28% | 98,64% | 66,68% | 93,55% | 86,90% | 94,76% | **93,3% / 69,6%** | **245,4 / 232,4** | **32,9** / 32,7 |
| `runs/hand_only_trial/.../best.pt` | 640 | 97,15% | 98,94% | 65,03% | 95,11% | 88,49% | 93,08% | 69,7% / 21,1% | 138,0 / 153,5 | 13,3 / 16,3 |

*\*Cột 36 px là bài chẩn đoán thu nhỏ thuần (`min_source >= 64 px`, 300 ảnh trên nền xám — `SCALED DIAGNOSTIC`), không thay thế phép đo cự ly thực bằng camera. Tốc độ đo trên RTX 5050 Laptop & CPU với 20 vòng warmup + 200 khung ở đúng kích thước webcam `640×480` và `1280×720`.*

**Kết luận rút ra từ Tầng 1 và giới hạn trung thực của tập `val`:**
- **Chọn trọng số mặc định:** `hand_pose.pt` (`imgsz=512`, Top-1 Conf Recall `97,11%`) dẫn đầu về chất lượng khớp cận cảnh (`66,75%` Pose mAP50–95, `89,09%` PCK14, `94,88%` Gate14). Trên khung hình webcam thật, `.pt` dùng letterbox chữ nhật (`512×384` ở 4:3, `512×288` ở 16:9) còn `.onnx` bị khóa vuông `[1,3,512,512]`, nên ở `1280×720` bản **`hand_pose.pt` nhanh hơn `.onnx` trên GPU (`171,3` vs `129,9 FPS`) và tương đương trên CPU (`21,8` vs `21,6 FPS`)**; chỉ ở `640×480` thuần CPU thì `hand_pose.onnx` nhỉnh hơn (`21,3` vs `17,0 FPS`). Ép `hand_pose.pt` lên `640` làm rớt `Gate14` xuống `80,04%`, còn `best.pt` (tinh chỉnh 4 epoch) giảm Pose mAP50–95 xuống `65,03%` và tụt mạnh ở tay nhỏ 36 px. Riêng trên máy chỉ có CPU cần `>30 FPS` hoặc bàn tay nhỏ (`36–64 px`), `hand_pose_v2.pt`/`.onnx` (nano, `640`) đạt `66,69%` Pose mAP50–95, **`93,3%` Recall ở 36 px** và **`29,7–34,2 FPS` trên CPU**.
- **Điểm nghẽn thực tế của `Gate14`:** Cổ tay (`wrist`, khớp `0`) chiếm tới **96% số ca bị loại bởi cổng `all(14 khớp >= 0,5)`** (`314/325` ca GT đủ 14 khớp), chủ yếu khi tay quá sát mép ảnh (`>=160 px`) làm cổ tay chạm biên.
- **Giới hạn của tập `hand-keypoints/val` (không dùng làm điểm thực tế ngoài đời):** Tập `val` này cùng nguồn phân bố với `train`, đã được dùng để chọn checkpoint trong 25 epoch train, toàn bộ `7.953` ảnh đều là vuông `224×224`, chỉ có **2/7.953 ảnh (`0,03%`) tay `< 40 px`** (trung vị `142 px`), và quét `dHash ≤ 2` phát hiện **`6.095/7.953` ảnh `val` có ảnh rất giống (cùng bối cảnh/chuỗi khung) trong `train`**. Do đó các số liệu Tầng 1 chỉ là **kiểm định nội bộ cận cảnh**; độ chính xác theo ý định người dùng và tỷ lệ kích hoạt nhầm ngoài thực tế cần đo ở Tầng 2 & Tầng 3 trên video độc lập.

Để chạy lại toàn bộ benchmark Tầng 1 (và xuất ảnh kiểm chứng bằng mắt tại `benchmark/results/tier1/audit_top20_errors/`):

```bash
python tools/eval_tier1_static.py \
  --data ../datasets/hand-keypoints \
  --models hand_pose.pt hand_pose.pt@640 hand_pose_v3.pt hand_pose_v2.pt hand_pose.onnx hand_pose_v3.onnx hand_pose_v2.onnx runs/hand_only_trial/fit/weights/best.pt \
  --exclude-audit-list runs/hand_only_trial/val.txt \
  --stress-scales 36,64 --latency-sizes 640x480,1280x720 \
  --output benchmark/results/tier1
```

## Kiểm chứng

- **Camera di chuyển, tay đứng yên.** Bài kiểm thử chiếu cùng bàn tay xòe hoặc nắm qua 91 vị trí và góc camera liên tiếp cho mỗi tư thế. Tâm tay dịch hơn 50 pixel trên ảnh. Bộ điều khiển giữ trạng thái bay tại chỗ, lệnh vận tốc bằng 0 trong toàn bộ phép thử.
- **Video qua mô hình thật.** Một clip 268 khung được dựng từ các ảnh tay đã chọn để kiểm tra đường truyền dữ liệu. Hệ thống phát lệnh cất cánh, bay lên theo ngón trỏ, dừng và hạ cánh. Khi phát lại log, cả 268 khung cho cùng lệnh. Clip này là thử nghiệm tích hợp có chủ đích. Độ chính xác theo ý định người dùng cần được đo trên các phiên video độc lập.
- **Mô phỏng ở 3 m.** Kịch bản không nhiễu hoàn thành đủ bốn hướng bay và hạ cánh. Với nhiễu kiểu webcam, **25/30 lượt** hoàn thành (seed 500–529). Ba lượt bỏ lỡ cử chỉ cất cánh, một lượt dịch phải quá ít, một lượt chưa hạ cánh khi kịch bản kết thúc. Đây là kết quả trên khớp tay mô phỏng. Cự ly và độ chính xác trên camera drone thật vẫn cần được đo.
- **Thử nghiệm 10 m trước đây.** [Báo cáo](demo/benchmark_report.md) dùng các ảnh ghép và có cấu hình định vị cổ tay bằng mô hình người. Số liệu trong báo cáo thuộc bài đo đó. Chế độ chỉ dùng bàn tay cần được đánh giá riêng ở từng cự ly.

```bash
python -m pytest tests -q
```

## Mã nguồn

| Thành phần | Vai trò |
|---|---|
| [demo/fly_sim.py](demo/fly_sim.py) và [demo/perception.py](demo/perception.py) | Đọc camera hoặc video, nhận diện, ghi và phát lại log |
| [dronectl](dronectl/) | Đọc cử chỉ, xác nhận lệnh theo thời gian và quản lý trạng thái bay |
| [sim3d](sim3d/) và [sim](sim/) | Camera, bàn tay 3D và mô phỏng chuyển động của drone |
| [tests](tests/) | Kiểm thử nhận diện, mất tín hiệu, chuyển trạng thái, hướng di chuyển và bộ benchmark (`309` ca `pytest`) |
| [benchmark](benchmark/) và [tools/eval_tier1_static.py](tools/eval_tier1_static.py) | Kế hoạch, schema kiểm toán và bộ benchmark Tầng 1 (lọc trùng SHA-256/dHash, PCK, Gate14, latency GPU/CPU) |
| [tools/finetune_hand.py](tools/finetune_hand.py) | Tinh chỉnh mô hình khớp tay và so sánh hai bản trọng số |

Để đối chiếu log từ bản cũ, `--replay` tự đọc chế độ đã ghi. Log đời đầu được phát bằng chế độ kết hợp tay và cơ thể. Ngoài chế độ mặc định `--input-mode hand` (chỉ ngón trỏ), có thể chọn `--input-mode hand_joystick` (cần lái 4 hướng chỉ dùng bàn tay) hoặc `--input-mode mixed` (chế độ thử nghiệm cũ).

Dữ liệu và phần mềm nền được dẫn riêng tại [Hand Keypoints của Ultralytics](https://docs.ultralytics.com/datasets/pose/hand-keypoints/), [HaGRID](https://github.com/hukenovs/hagrid) và [PyBullet](https://github.com/bulletphysics/bullet3).
