# Kế hoạch benchmark điều khiển drone bằng bàn tay

**Tác giả dự án: phamdu1356**
Bản thiết kế v0.1, ngày 22/11/2026. Các ngưỡng, số người và thời lượng dưới đây là đề xuất cho pilot. Chốt phiên bản trước khi mở tập test. Chưa có kết quả so sánh A/B theo giao thức này.

## Quyết định cần đưa ra

Với một camera RGB hướng về người dùng, khi nào chỉ ngón trỏ (A) hoặc dịch bàn tay như cần lái (B) giúp hoàn thành nhiệm vụ tốt hơn? Chỉ so sánh bốn hướng trái, phải, lên, xuống. Hai cách dùng cùng mô hình khớp tay, camera, khung hình gốc, giới hạn vận tốc, chỉ dẫn, phản hồi và lệnh cất cánh, dừng, hạ cánh. B lấy mốc khi người dùng nắm tay để kích hoạt. Chỉ dùng thông tin bàn tay cho cả A và B. Kiểm tra luồng B cũ vì nó có thể dùng vai hoặc tư thế người. Cố định thứ tự ưu tiên khi giải thích kết quả: hoàn thành nhiệm vụ, kích hoạt ngoài ý muốn và khả năng dừng, sau đó thời gian thực hiện và mức mỏi tay. Nếu các chỉ số đánh đổi hoặc bất định lớn, báo theo điều kiện.

## Phân công và bàn giao

| Chat | Đầu ra cần bàn giao | Phụ thuộc |
|---|---|---|
| Chức năng điều khiển | Hai bộ điều khiển dùng chung đầu vào và nhật ký, bảng ý định → thao tác cho A/B, cách đặt lại mốc, giới hạn vận tốc và ngoại lệ | Chốt tập lệnh với quản lý trước pilot |
| Dữ liệu | Danh mục phiên, video RGB liên tục, chú giải theo ý định, thời gian trung tính, danh sách người/phiên theo tập và kiểm tra trùng | Dùng cùng đồng hồ với nhật ký hệ thống |
| Huấn luyện | Bản chuẩn hiện có, dấu nhận dạng trọng số và cấu hình, đối chứng trên validation, quyết định giữ hoặc thay mô hình cho cả A/B | Chỉ dùng train/dev để chọn mô hình và ngưỡng |
| Kiểm thử | Bộ ca kiểm tra độc lập cho chấm sự kiện, dừng, quá hạn, log lỗi, rò rỉ dữ liệu; kết quả kỳ vọng tính tay | Dùng chung định nghĩa bên dưới |
| Benchmark | Chương trình chấm đọc nhãn và log độc lập, bảng theo lệnh/người/điều kiện, báo cáo sai lệch và lệnh tái lập | Chỉ chạy test cuối sau khi khóa scorer và cấu hình |
| Quản lý — chat hiện tại | Giải quyết khác biệt, duyệt giao diện và tiêu chí, khóa phiên bản, kiểm tra các đầu ra và tổng hợp kết luận | Theo dõi từ năm chat |

## Ba phép đo

1. **Phát lại video RGB.** Người gán nhãn đánh dấu thời điểm bắt đầu, kết thúc ý định và đoạn trung tính trước khi xem dự đoán. Khi có tín hiệu nhắc lệnh, phân biệt lúc nhắc và lúc tay thật sự bắt đầu ra dấu. Người thứ hai kiểm tra một phần nhãn, đặc biệt là dừng. A và B có thể cần động tác tay khác nhau cho cùng ý định nên quay phiên tương ứng cho mỗi cách. Cùng video chỉ dùng để so các phiên bản hiểu cùng một bộ cử chỉ.
2. **Mô phỏng vòng kín.** Ghép cặp A/B bằng cùng vị trí ban đầu, tình huống camera và chuỗi nhiễu ngoại sinh. Quỹ đạo sau khi điều khiển có thể khác. Thử camera tĩnh/đang dịch/đang nghiêng, tay đứng yên, mất khớp, dấu tay giữ xuyên nhiều khung và chuyển hướng. Phân biệt mô phỏng cấp khớp, RGB phát lại và RGB vòng kín khi công bố kết quả.
3. **Người dùng điều khiển trực tiếp.** Có thể dùng drone ảo để chấm mức dễ dùng của giao diện. Ghi rõ camera nào tạo ảnh và drone nào thực hiện lệnh. Tuyên bố về camera đang bay yêu cầu thử riêng với drone thật sau khi chọn thiết bị.

## Thiết kế lấy mẫu

- Pilot với 2–3 người để tìm lỗi cách quay, ngưỡng và hướng dẫn. Dành riêng một nhóm người cho đánh giá cuối. Ví dụ khả thi cho đồ án là 12 người mới, mỗi người làm 3 nhiệm vụ cho mỗi cách trong 2 điều kiện camera, tổng 144 lượt. Đảo thứ tự A/B giữa người, cân bằng thứ tự điều kiện, cho tập trước và nghỉ giữa các lượt. Dùng biến thiên của pilot để quyết định cỡ mẫu cuối.
- Mỗi người có ít nhất 2 phút hoạt động tự nhiên không có ý định ra lệnh cho mỗi cách. Ghi riêng đứng yên, chỉnh tóc, đưa tay vào/ra khung, che tay. Báo tổng phút thực tế và tách nhầm cất cánh, nhầm di chuyển.
- Chia tập theo người; tất cả phiên, clip cắt, ảnh tăng cường từ cùng người/video gốc ở cùng tập. Với nguồn ảnh thiếu danh tính, ghi rõ giới hạn và chỉ dùng cho phép đo thị giác phù hợp. Ghi khoảng cách đo được, độ phân giải, kích thước bàn tay theo pixel, điều kiện sáng và góc nhìn. Cự ly 10 m chỉ là một điều kiện nếu có mẫu tương ứng.
- Mất tín hiệu trong lúc thực hiện là kết quả của nhiệm vụ. Tệp hỏng hoặc mất đồng bộ phải giữ trong bảng kiểm kê, ghi lý do loại và báo mẫu số trước/sau. Không lấy khung hình kế cận làm lượt độc lập.

## Quy tắc chấm đề xuất trước pilot

| Chỉ số | Định nghĩa vận hành |
|---|---|
| Hoàn thành nhiệm vụ (chính) | Ví dụ cất cánh → dịch theo hướng yêu cầu 0,5 m → dừng trong bán kính 0,15 m quanh đích → hạ cánh trong 30 s. Lưu tọa độ đích trong hệ trục người dùng, đo cả lệch ngoài trục. Lấy mọi lượt bắt đầu làm mẫu số; ghi lại từng lần phải ra dấu lại. |
| Lệnh đúng, thiếu, nhầm, lặp | Gộp chuỗi khung phát cùng một lệnh liên tục thành một sự kiện. Ghép một-một, theo thứ tự với ý định trong cửa sổ từ lúc bắt đầu ra dấu tới lúc kết thúc cộng dung sai đồng bộ đã khóa. Sai hướng tính là bỏ lỡ lệnh mong muốn và phát sai lệnh quan sát, ghi vào ma trận nhầm. Lệnh chưa ghép là lệnh dư. |
| Kích hoạt ngoài ý muốn | Số sự kiện bắt đầu trong khoảng trung tính / tổng phút trung tính. Tách theo loại lệnh và trạng thái bay. |
| Dừng | Từ lúc người dùng bắt đầu ra hiệu dừng đến lúc lệnh vận tốc về 0; đo thêm thời gian và quãng đường đến khi tốc độ vật lý dưới 0,05 m/s liên tục 0,3 s. Chấm mọi lần có ý định dừng, tính cả khi mô hình không nhận ra bàn tay xòe. Báo số quá hạn, trung vị và P95 trên những lượt đo được. |
| Khả năng sử dụng | Thời gian hoàn thành, số lần ra dấu lại và đánh giá mức mỏi tay sau mỗi cách. Báo số lượt thất bại cùng thời gian của những lượt thành công. |
| Chẩn đoán | Độ chính xác theo lớp, số pixel của tay, độ tin cậy khớp, tần số khung hình và độ trễ xử lý phục vụ tìm nguyên nhân lỗi. Đo riêng trễ xử lý và trễ từ camera tới hành động. |

## Giao diện dữ liệu tối thiểu

- **Danh mục** — participant_id ẩn danh, session_id, trial_id, split, interface (A/B), source (rgb_replay/keypoint_sim/live), thiết bị, camera_condition, seed nếu có, đường dẫn và hash nguồn.
- **Nhãn độc lập** — trial_id, intent_id, expected_command, onset_s, offset_s, neutral_intervals, vị trí đích và timeout. Ý định lấy từ kịch bản hoặc người gán nhãn; ghi rõ nguồn.
- **Nhật ký hệ thống** — trial_id, timestamp_s, action/mode, vận tốc lệnh, vị trí và vận tốc đo được nếu có, cờ mất tín hiệu. Dùng chung mốc thời gian trong từng phiên. Trường vật lý thiếu ở video phát lại được đánh dấu không đo được.
- **Kết quả** — bảng ghép từng sự kiện, kết quả từng lượt và lý do thất bại. Khi báo A/B, tính chênh lệch ghép cặp theo người, trình bày khoảng tin cậy 95% bằng lấy mẫu lại theo người. Với số người ít, kèm bảng từng người. Phân tích mô phỏng ghép cặp theo kịch bản và seed riêng.

## Mốc nghiệm thu

1. Quản lý hợp nhất năm bản rà soát; khóa từ điển lệnh và giao diện nhãn/log v1.
2. Chạy pilot; kiểm tra ảnh → bộ điều khiển → log → bộ chấm. Ít nhất một ca đúng, sai hướng, lặp lệnh, dừng muộn, mất tay, timeout và không ra lệnh có kết quả tính tay khớp chương trình.
3. Khóa mã nguồn, hash trọng số, ngưỡng, danh sách người/test, seed, phiên bản bộ chấm và quy tắc loại mẫu. Mọi thay đổi sau đó phải ghi phiên bản và lý do.
4. Chạy đánh giá chính thức, lưu lệnh tái lập, toàn bộ kết quả từng lượt, lỗi và mẫu số. Báo riêng RGB, mô phỏng và người dùng; kết luận giới hạn ở phần cứng và điều kiện đã thử.

Kết quả cũ 25/30 là điểm tham chiếu của kịch bản khớp 3D trước đây. Bộ chấm mới phải tính sai số đến đích và dừng từ thời điểm người dùng ra hiệu. Đây là hai chỗ phép chấm cũ chưa kiểm tra đầy đủ.

## Quyết định sau rà soát của năm chat

Đã hoàn thành vòng rà soát kế hoạch ngày 22/11. Chưa mở pilot hoặc test chính thức.

- **Chức năng.** A đã có. B ở chế độ mixed hiện dùng được khung vai, có thêm trục tiến/lùi và xoay, và có thể giữ vận tốc cũ khi mất tay ngắn. Việc đầu tiên là tách B chỉ dùng khớp tay, chuẩn hóa độ dịch theo kích thước lòng bàn tay lúc lấy mốc và chọn trục trội. Dùng cùng tốc độ đích, giới hạn gia tốc và quy tắc dừng như A. Khi mất dấu, đưa lệnh về 0 ở lần cập nhật kế tiếp; B phải lấy lại mốc sau khi nắm lại.
- **Dữ liệu.** Clip 268 khung hiện tại là clip ghép. Chưa tìm thấy phiên RGB liên tục có nhãn ý định độc lập cho A/B. Đã xác minh hai ảnh IMG_00019779.jpg (validation) và IMG_00019778.jpg (test) của lần tinh chỉnh cũ giống từng byte. Kiểm hash giữa các tập, kiểm ảnh gần trùng và nhóm theo nguồn trước khi khóa danh mục mới.
- **Huấn luyện.** Dùng cùng hand_pose.pt cho pilot, SHA-256 `093ae826a31a3014bc17ddc725b3a0f50b522622f49f3d45e527850b4255f61d`. Ghi cấu hình suy luận thực dùng. Tập 512 ảnh đã dùng so sánh trọng số thuộc dữ liệu phát triển cho các quyết định tiếp theo. Chỉ mở một lượt train mới nếu dev chỉ ra lỗi nhận diện lặp lại.
- **Kiểm thử.** Bộ kiểm log cũ có thể cho qua CSV chỉ chứa tiêu đề hoặc thiếu trường. Bộ chấm v1 phải kiểm schema và tính đầy đủ trước khi chấm. Ca tính tay đầu tiên: đích dịch phải 0,50 m, thực tế dịch phải 0,01 m và lệch lên 0,30 m; sai số khoảng 0,575 m, phải trượt nếu dung sai là 0,15 m. Bổ sung ca tín hiệu dừng mà mô hình không nhận ra.
- **Benchmark.** Khóa bộ chấm độc lập với dự đoán. Phải có từng sự kiện, từng lượt, thời gian dừng lệnh và dừng vật lý, mẫu số các lượt thất bại, và đầu ra tái lập được. Bài 10 m cũ và 25/30 lượt mô phỏng tiếp tục là hai phép đo chẩn đoán riêng.

**Thứ tự triển khai.** Chức năng tách B và Dữ liệu tạo mẫu chú giải song song. Kiểm thử bàn giao các ca có đáp án tính tay để Benchmark xây bộ chấm. Quản lý đối chiếu cùng một lượt mẫu qua cả bốn phần trước khi tổ chức pilot. Huấn luyện giữ bản chuẩn trong thời gian này.
