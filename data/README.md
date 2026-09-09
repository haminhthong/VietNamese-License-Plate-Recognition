# Data card

Repository không phân phối ảnh hoặc biển số riêng tư. Dữ liệu đầu vào phải ở định dạng YOLO với `train`, `valid`, `test`, mỗi tập có thư mục `images` và `labels`.

Mỗi dòng nhãn gồm `class_id x_center y_center width height`; project hiện chỉ hỗ trợ class `0: license_plate`. Chạy `prepare_dataset.py` để kiểm tra schema, ảnh lỗi, nhãn thiếu, duplicate MD5 và tạo split cố định theo nhóm nguồn. `split_manifest.csv` là artifact cần lưu cùng mỗi thí nghiệm.

Trước khi công bố dữ liệu, phải ghi rõ nguồn, phiên bản, giấy phép, phạm vi đồng ý sử dụng và chính sách ẩn danh. Không đưa ảnh biển số thật lên repository nếu chưa có quyền. Metadata riêng cần có `capture_group`/`capture_session_id` và `plate_identity` hoặc `plate_identity_hash` để Protocol B hoạt động thật.

## Hợp đồng dữ liệu đánh giá E2E

Tách dữ liệu đánh giá thành hai file độc lập: `e2e/development.csv` để tune
padding/preprocessing/rectification/threshold và `e2e/test.csv` đã khóa để báo
cáo final. Mỗi file cần `image_path,x1,y1,x2,y2,plate_text,split`; giá trị
`split` tương ứng phải là `dev` hoặc `test` vì evaluator dùng đúng hai giá trị
này. Có thể thêm
`plate_identity`, `capture_group` và `layout` để bootstrap theo đơn vị độc lập.

`evaluate_ablation.py` chỉ đọc development. `evaluate_end_to_end.py` chỉ đọc
locked test. Cờ `--allow-unlocked-annotations` chỉ dành cho dữ liệu legacy.
