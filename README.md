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
| [tests](tests/) | Kiểm thử nhận diện, mất tín hiệu, chuyển trạng thái và hướng di chuyển |
| [tools/finetune_hand.py](tools/finetune_hand.py) | Tinh chỉnh mô hình khớp tay và so sánh hai bản trọng số |

Để đối chiếu log từ bản cũ, `--replay` tự đọc chế độ đã ghi. Log đời đầu được phát bằng chế độ kết hợp tay và cơ thể. Có thể chọn lại chế độ thử nghiệm cũ bằng `--input-mode mixed`.

Dữ liệu và phần mềm nền được dẫn riêng tại [Hand Keypoints của Ultralytics](https://docs.ultralytics.com/datasets/pose/hand-keypoints/), [HaGRID](https://github.com/hukenovs/hagrid) và [PyBullet](https://github.com/bulletphysics/bullet3).
