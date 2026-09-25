# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lê Danh Trung

Công cụ gán nhãn đã dùng: CVAT Docker trên máy cá nhân

Sao chép file này thành `reports/REPORT.md` rồi vào các chỗ. Mọi con số phải truy được
từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc
`outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tại sao tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian, có vùng
đệm ở giữa, thay vì chia ngẫu nhiên? Nếu chia ngẫu nhiên, số đo trên tập kiểm thử sẽ bị lệch theo
hướng nào, và vì sao?

Tập dữ liệu được chia theo trục thời gian và có vùng đệm ở giữa để tránh hiện tượng rò rỉ dữ liệu (Data Leakage). Vì các frame trong một video liên tiếp nhau có hình ảnh gần như y hệt nhau, nếu chia ngẫu nhiên, các frame liền kề sẽ bị lọt vào cả tập train (pool) và tập test. Khi đó, mô hình chỉ đơn giản là "học thuộc lòng" bối cảnh của bức ảnh thay vì học cách nhận diện đặc trưng của xe. Số đo trên tập kiểm thử lúc này sẽ bị lệch theo hướng cao ảo tưởng (over-optimistic), nhưng thực chất mô hình sẽ thất bại khi gặp một đoạn video ở mốc thời gian khác.

## 2. Mô hình khởi đầu lạnh (cold start)

Chép dòng vòng 0 từ `rounds_table.md`. Dựa vào `outputs/compare_round0.jpg`, cho biết mô hình khởi
đầu lạnh không khớp nhãn tham chiếu ở những loại xe nào. Độ phủ (recall) theo kích thước xe cho
thấy điều gì? Một trường hợp nào cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai?

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào chỉ số Recall theo kích thước, mô hình cold start bỏ sót rất nhiều xe nhỏ (R small chỉ đạt 0.182, tức là trượt hơn 80%) và chỉ bắt được khoảng một nửa số xe trung bình/lớn (R medium 0.547, R large 0.561). Điều này cho thấy mô hình yếu trong việc nhận diện xe ở xa hoặc xe bị chìm vào bóng tối. Một trường hợp cần rà lại nhãn tham chiếu trước khi kết luận mô hình sai là khi mô hình đánh dấu một chiếc xe tối màu (có 2 đèn hậu) mà nhãn tham chiếu không có. Do nhãn tham chiếu của tập test cũng do AI tự tạo ban đầu và chưa được con người kiểm chứng, rất có thể mô hình cold start đã nhận diện đúng (True Positive) nhưng lại bị hệ thống chấm điểm oan thành nhận diện sai (False Positive).

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Công thức tính score giúp tổng hợp mức độ bối rối của mô hình: `U` ưu tiên ảnh có các box độ tin cậy thấp nhất, `A` đếm tỷ lệ các box thiếu chắc chắn trên toàn ảnh, và `D` là khoảng cách thời gian đến ảnh đã gán nhãn gần nhất. `MIN_GAP_S` đóng vai trò rào chắn, buộc công cụ phải nhảy cóc qua một khoảng thời gian nhất định để không chọn liên tiếp các ảnh sát nhau. Dựa vào bảng phân tích ở `SELECTION.md`, frame_0182.jpg và frame_0326.jpg được chọn vì có độ bất định rất cao. frame_0107.jpg dù điểm thấp hơn nhưng vẫn được đưa vào lô để đảm bảo sự đa dạng thời gian nhờ luật `MIN_GAP_S`. Ngược lại, frame_0372.jpg (Rank 6) điểm rất cao nhưng bị loại bỏ vì nó xuất hiện chưa tới 2 giây sau frame rank 2; việc gán nhãn ảnh gần trùng này vừa tốn công vừa vô ích. Điểm bất định (score) cao không chứng minh ảnh đó sẽ cải thiện mô hình. Nếu ảnh bị lóa sáng quá mức hoặc tối đen toàn tập, mô hình bất định là lẽ đương nhiên. Bắt mô hình học từ những ảnh quá nhiễu mà con người cũng khó gán nhãn chính xác có thể làm hỏng trọng số của mô hình.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 326 | 0.558 | -0.213 | 1.000 | 0.057 | 0.108 | 0.000 | 0.037 | 0.293 |

- Mức độ sửa nhãn (Vòng 1): Tổng cộng tôi đã thao tác trên 12 ảnh. Từ 169 box gợi ý, tôi giữ nguyên (accepted) 119 box, chỉnh sửa mép (edited) 30 box, xóa bỏ do nhận nhầm (deleted) 20 box, và vẽ bù (added) 177 box bị AI bỏ sót.
- Biến động AP50: AP50 giảm mạnh từ 0.771 xuống 0.558 (giảm 0.213 so với cold start).
- Nhóm xe: Độ chính xác (Precision) đạt tuyệt đối 1.0, nhưng Độ phủ (Recall) sụp đổ ở mọi phân khúc. Xe nhỏ từ 0.182 về 0.0, xe trung bình từ 0.547 về 0.037. Mô hình trở nên quá bảo thủ và gần như bỏ qua toàn bộ xe trên đường, trừ vài chiếc xe lớn.

Trong bước `BLIND_SCAN.md` (quan sát độc lập), tôi đếm được 26 xe trên frame_0187. Khi sửa nhãn (`REVIEW_LOG.csv`), tôi phát hiện AI bỏ sót rất nhiều xe đen và vẽ thêm hàng loạt box (chính là số liệu 177 box added trong `round1_diff.md`). Tuy nhiên, kết quả sau train lại cho thấy Recall giảm thê thảm (0.057). Điều này chứng tỏ việc ép mô hình nhỏ học thêm 177 box xe tối màu/mờ ảo từ vỏn vẹn 12 bức ảnh đã khiến nó bị overfitting. Mô hình học được cách né các ánh đèn đường nhưng lại sợ sai đến mức không dám dự đoán các xe tối màu nữa. Một ca khó tôi đã xử lý theo guideline: Xe tối màu đi sát lề, chỉ nhìn thấy 2 chấm đèn đỏ; thay vì chỉ khoanh 2 đèn, tôi ước lượng ranh giới thân xe chìm trong bóng tối và vẽ box ôm quanh cụm đèn đó.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

Kết quả vòng 1 đi lùi nghiêm trọng so với cold start (AP50 giảm sâu, Recall gần như biến mất). Tôi quyết định dừng lại để phân tích nguyên nhân thay vì mù quáng gán nhãn tiếp vòng 2. Đề xuất hai ca còn yếu cho vòng sau là: (1) Xe rất nhỏ ở xa tít tắp, (2) Xe tải tối màu bị che khuất một phần. Chi phí rà nhãn các ca này tốn thời gian và mỏi mắt, kèm theo nguy cơ bốc trúng ảnh gần trùng ở các mốc kẹt xe kéo dài là rất cao. Tập test 20 ảnh là quá nhỏ bé để đại diện cho toàn bộ bối cảnh đêm. Việc bỏ qua xe dưới 16px khiến công sức gán nhãn xe ở xa trong vòng 1 trở nên công cốc vì không được cộng điểm. Đặc biệt, nhãn tham chiếu tập test chưa được rà thủ công khiến điểm số bị méo mó (mô hình học xong vòng 1 có thể nhận diện đúng xe bóng tối nhưng bị trừ điểm vì nhãn gốc của tập test không có box đó). Vì AP50 giảm, trước khi train thêm, tôi cần kiểm tra: (1) Xuất nhãn test ra CVAT để soát lại xem có đang chấm sai cho mô hình không; (2) Xem xét giảm cường độ học (learning rate) hoặc tăng số lượng ảnh huấn luyện lên để tránh overfitting cục bộ vào 12 ảnh của vòng 1.