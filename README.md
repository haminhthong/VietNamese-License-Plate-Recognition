# 🚗 Vietnamese License Plate Recognition

[![CI](https://github.com/haminhthong/vietnamese-license-plate-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/haminhthong/vietnamese-license-plate-recognition/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8-111111)](https://github.com/ultralytics/ultralytics)
[![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-2E7D32)](https://github.com/JaidedAI/EasyOCR)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Runtime-Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hệ thống nhận diện biển số xe (ALPR) tự động cho **ảnh tĩnh từ camera/cổng kiểm soát tại Việt Nam**, kết hợp giữa **YOLOv8** (phát hiện biển số), **EasyOCR** (nhận dạng ký tự), xử lý hình học token đa dòng và kiểm tra cú pháp định dạng biển số xe Việt Nam.

---

## 💡 Điểm cốt lõi của dự án

1. **Phân định rõ vai trò mô hình:**
   - **Detector:** Huấn luyện mô hình YOLOv8 trên tập dữ liệu ảnh xe để định vị chính xác vị trí biển số (`bounding box`).
   - **Recognizer:** Tích hợp EasyOCR để nhận diện chuỗi ký tự trên vùng ảnh biển số đã crop và mở rộng lề đệm (`padding 5%`), giúp tránh mất nét viền hoặc ký tự sát mép.
2. **Xử lý hình học token 1 dòng & 2 dòng:**
   - Biển số Việt Nam gồm hai dạng chính: biển dài 1 dòng (ô tô) và biển vuông 2 dòng (xe máy, ô tô).
   - EasyOCR thường trả về nhiều bounding boxes nhỏ lẻ. Hệ thống phân tích tọa độ tâm (`center_x`, `center_y`) và chiều cao token để gom dòng theo trục thẳng đứng rồi sắp xếp từ trái sang phải.
3. **Tiền xử lý ảnh (Fast path & Fallback):**
   - **Fast path:** Chuyển xám và cân bằng tương phản cục bộ CLAHE để xử lý bóng râm, chói sáng.
   - **Fallback:** Nắn thẳng phối cảnh (Perspective Rectification) hoặc phân ngưỡng thích ứng khi ảnh chụp góc xiên hoặc tương phản yếu.
4. **Kiểm tra cú pháp biển số Việt Nam (Syntax Validation & Suggestion):**
   - Kiểm tra chuỗi ký tự thô theo các mẫu định dạng dân sự tiêu chuẩn (`DDLDDDD`, `DDLDDDDD`, `DDLLDDDDD`,...).
   - Đưa ra đề xuất sửa lỗi ký tự nhầm lẫn quang học thường gặp (ví dụ: `O` $\leftrightarrow$ `0`, `B` $\leftrightarrow$ `8`, `I` $\leftrightarrow$ `1`), nhưng **tuyệt đối không tự ý ghi đè** kết quả OCR thô nhằm đảm bảo tính minh bạch cho người vận hành.

---

## 🏗️ Kiến trúc Pipeline

```text
Ảnh phương tiện
      │
      ▼
YOLOv8 Plate Detector
      │
      ▼
Plate crop + 5% padding
      │
      ▼
Gray / CLAHE preprocessing
      │
      ▼
EasyOCR
      │
      ▼
Token geometry ordering (1-line / 2-line)
      │
      ▼
Chuỗi OCR thô
      │
      ▼
Vietnamese syntax validation
      │
      ├── Hợp lệ  ──► [51F12345] (format_valid = True)
      │
      └── Nhầm lẫn (O/0, B/8...) ──► Gợi ý sửa [51F12845] (needs_review = True)
      │
      ▼
Bounding box + Chuỗi biển số + Độ tin cậy
```

---

## 📊 Phương Pháp Đánh Giá (Evaluation)

Hệ thống đánh giá độc lập theo 3 tầng rõ ràng:

1. **Detector Evaluation (trên test split):**
   - Đo lường khả năng phát hiện bounding box của YOLOv8: **mAP@50**, **Recall@50**.
2. **OCR Evaluation (trên ảnh crop có nhãn):**
   - **Exact Plate Accuracy:** Tỷ lệ biển số nhận dạng chính xác 100%.
   - **Character Error Rate (CER):** Tỷ lệ sai lệch ở cấp độ từng ký tự theo khoảng cách Levenshtein.
3. **End-to-End Evaluation (toàn bộ pipeline):**
   - **End-to-End Exact Recall:** Tỷ lệ biển số trong ảnh thực tế vừa được phát hiện đúng vị trí (IoU $\ge 0.5$) vừa đọc chính xác toàn bộ chuỗi ký tự.
   - **Error Analysis:** Phân loại nguyên nhân lỗi thành: bỏ sót phát hiện (`detector_miss`), bounding box lệch (`iou_poor`), hoặc nhận dạng sai chữ (`ocr_wrong`).

---

## 🗂️ Cấu Trúc Dự Án

```text
├── configs/
│   ├── recognition.yaml      # Tham số nhận diện, crop padding, ngưỡng tin cậy
│   └── train.yaml            # Cấu hình huấn luyện YOLOv8
├── resources/
│   ├── plate_templates.yaml  # Mẫu định dạng biển số dân sự Việt Nam
│   └── ocr_confusions.yaml   # Bảng cặp ký tự dễ nhầm lẫn (O/0, B/8, etc.)
├── src/
│   ├── config.py             # Dataclass quản lý cấu hình
│   ├── dataset.py            # Kiểm soát leakage và chia tập train/val/test
│   ├── grammar.py            # Kiểm tra cú pháp và gợi ý ký tự
│   ├── io_utils.py           # Tiện ích đọc/ghi dữ liệu
│   ├── metrics.py            # IoU, CER, Exact Accuracy, Error Analysis
│   ├── ocr.py                # Tiền xử lý, EasyOCR và sắp xếp hình học token
│   ├── pipeline.py           # Điều phối pipeline nhận diện hoàn chỉnh
│   └── rectification.py      # Nắn phẳng góc nghiêng phối cảnh
├── app/
│   ├── api.py                # FastAPI REST API endpoints
│   ├── schemas.py            # Pydantic schemas kết quả nhận diện
│   └── ui.html               # Giao diện Web Dashboard upload ảnh trực quan
├── scripts/
│   ├── prepare_dataset.py    # Chia tập dữ liệu nhóm theo identity/burst
│   ├── train.py              # Huấn luyện mô hình YOLOv8
│   ├── predict.py            # Nhận diện biển số từ tệp ảnh qua CLI
│   ├── evaluate_detector.py  # Đánh giá mAP detector
│   ├── evaluate_ocr.py       # Đánh giá độ chính xác OCR
│   └── evaluate_end_to_end.py# Đánh giá pipeline end-to-end
├── tests/                    # Kiểm thử tự động (Unit & API tests)
├── Dockerfile                # Containerization cho ứng dụng
└── pyproject.toml            # Cấu hình package, dependencies và công cụ kiểm tra
```

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng (Quick Start)

### 1. Cài đặt môi trường

Yêu cầu **Python 3.11+**:

```bash
# Tạo và kích hoạt môi trường ảo
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .\.venv\Scripts\Activate.ps1

# Cài đặt package và dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Nhận diện biển số qua CLI

```bash
python scripts/predict.py --weights models/best.pt --source sample/car.jpg --output outputs/result.jpg
```

Kết quả trả về định dạng JSON:
```json
[
  {
    "box": [120, 150, 310, 220],
    "detection_confidence": 0.95,
    "text": "51F12345",
    "ocr_confidence": 0.88,
    "format_valid": true,
    "suggested_text": null,
    "needs_review": false,
    "layout": "1_line"
  }
]
```

### 3. Khởi chạy REST API & Web Dashboard

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

- **Giao diện Web UI:** Mở trình duyệt tại [http://localhost:8000/](http://localhost:8000/) để tải ảnh và xem khung bbox + biển số trực quan.
- **Swagger API Docs:** Truy cập [http://localhost:8000/docs](http://localhost:8000/docs).
- **Health Check:** `GET /health` trả về trạng thái hoạt động và khả dụng của mô hình.

---

## 🛠️ Huấn Luyện & Đánh Giá

### 1. Chuẩn bị dữ liệu và ngăn ngừa rò rỉ (Data Leakage)
Chia tập dữ liệu theo nhóm nguồn và danh tính biển số (`plate_identity` / `capture_group`):
```bash
python scripts/prepare_dataset.py --source data/raw --output dataset/grouped
```

### 2. Huấn luyện YOLOv8 Detector
```bash
python scripts/train.py --data dataset/grouped/data.yaml --config configs/train.yaml
```

### 3. Đánh giá kiểm thử
```bash
# Đánh giá riêng Detector
python scripts/evaluate_detector.py --weights models/best.pt --data dataset/grouped/data.yaml

# Đánh giá riêng OCR trên ảnh crop
python scripts/evaluate_ocr.py --annotations data/e2e/ocr_test.csv

# Đánh giá toàn diện End-to-End
python scripts/evaluate_end_to_end.py --weights models/best.pt --annotations data/e2e/test.csv
```

---

## 🧪 Kiểm Thử Tự Động & CI

Dự án duy trì kiểm thử tự động toàn diện qua GitHub Actions:

```bash
# Kiểm tra định dạng và quy chuẩn mã nguồn
python -m ruff format --check .
python -m ruff check .

# Chạy toàn bộ unit tests
python -m pytest -v

# Kiểm tra đóng gói build wheel
python -m build --wheel
```

---

## 📜 Giấy Phép (License)

Dự án được phát hành theo giấy phép [MIT License](LICENSE).
