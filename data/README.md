# Hướng Dẫn Dữ Liệu (Data Guide)

Repository không phân phối ảnh hoặc biển số xe thực tế vì lý do bản quyền và quyền riêng tư. Dữ liệu huấn luyện và đánh giá cần được chuẩn bị theo cấu trúc sau:

## 1. Dữ liệu huấn luyện YOLOv8
Dữ liệu phát hiện biển số (Detector) tuân theo định dạng chuẩn của Ultralytics YOLO:
- Thư mục gốc chứa `train/`, `val/`, `test/`.
- Mỗi thư mục con chứa `images/` (ảnh JPG/PNG/WebP) và `labels/` (tệp TXT tương ứng).
- Mỗi dòng nhãn có định dạng: `class_id x_center y_center width height` (chuẩn hóa trong khoảng [0, 1]).
- Lớp đối tượng: `0: license_plate`.

Sử dụng script `python scripts/prepare_dataset.py --source data/raw --output dataset/grouped` để kiểm tra nhãn, phát hiện trùng lặp MD5 và chia tập an toàn theo nhóm camera/danh tính biển số nhằm tránh rò rỉ dữ liệu (Data Leakage).

## 2. Dữ liệu đánh giá OCR & End-to-End
- **Đánh giá OCR:** File CSV chứa các cột: `crop_path`, `plate_text`, và tùy chọn `layout` (`1_line` hoặc `2_line`). Xem file mẫu tại `data/ocr_annotations.example.csv`.
- **Đánh giá End-to-End:** File CSV chứa các cột: `image_path`, `x1`, `y1`, `x2`, `y2`, `plate_text`. Xem file mẫu tại `data/end_to_end_annotations.example.csv`.
