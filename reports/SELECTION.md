# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box: Tôi sẽ ưu tiên 5 frame sau:
frame_0182.jpg (Rank 1, Score: 0.9591, t=72.8s): Điểm bối rối cao nhất, mô hình dự đoán ra 28 boxes nhưng có tới 18 boxes không chắc chắn.
frame_0369.jpg (Rank 2, Score: 0.9324, t=147.6s): Điểm cao thứ hai, thời điểm cách xa frame 1 nên đảm bảo bối cảnh giao thông khác biệt.
frame_0380.jpg (Rank 3, Score: 0.9170, t=152.0s): Thời điểm cách frame trên 4.4 giây, đủ để các xe thay đổi vị trí.
frame_0099.jpg (Rank 8, Score: 0.9063, t=39.6s): Lựa chọn để bổ sung dữ liệu ở đoạn đầu video, giúp mô hình học được sự đa dạng.
frame_0270.jpg (Rank 13, Score: 0.8878, t=108.0s): Tôi quyết định chọn frame này và bỏ qua các frame hạng 4, 5, 6, 7 vì chúng có thời điểm quá sát nhau (ví dụ frame_0326 lúc 130.4s và frame_0331 lúc 132.4s chỉ cách nhau 2 giây). Sửa các "ảnh gần trùng" này rất lãng phí ngân sách.

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet: frame_0182.jpg: CSV ghi nhận selected=True. Bằng chứng là n_ambiguous=18 (rất cao). Trên contact sheet, đây là cảnh xe lóa đèn và chìm vào bóng tối khiến AI không tự tin.   frame_0326.jpg: CSV ghi nhận selected=True với score=0.9155, n_ambiguous=15. Trên contact sheet, mật độ xe đoạn này rất đông và che khuất nhau.   frame_0107.jpg: Dù rank 14 nhưng vẫn được chọn (selected=True). Bằng chứng CSV (t=42.8s) cho thấy nó được chọn để thay thế các frame điểm cao hơn (như rank 6, 9) bị loại do vi phạm khoảng cách thời gian tối thiểu (MIN_GAP_S).

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do: Ảnh điểm cao nhưng không chọn: frame_0372.jpg (Rank 6, score: 0.9101, selected=False).
Lý do: Trong CSV, t_sec = 148.8s. Nó xuất hiện chỉ sau frame_0369.jpg (Rank 2) vỏn vẹn 1.2 giây. Cảnh vật và vị trí xe gần như giữ nguyên (ảnh trùng lặp). Việc sửa nhãn sẽ tốn gấp đôi công sức nhưng không cung cấp thêm đặc trưng mới cho mô hình.

Điều phép chọn này chưa chứng minh về chất lượng mô hình: Phép tính điểm (Score) này chỉ tìm ra được những ảnh mà mô hình hiện tại đang dự đoán kém cỏi và bối rối nhất (uncertainty cao). Tuy nhiên, nó chưa chứng minh được rằng sau khi học thêm những ảnh này, chất lượng nhận diện của mô hình sẽ thực sự tăng lên. Nếu bức ảnh có điểm cao vì quá lóa hoặc nhòe đến mức người gán nhãn cũng chỉ có thể đoán mò, việc ép mô hình học theo các nhãn đó có thể gây nhiễu dữ liệu.
