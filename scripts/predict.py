"""Nhận diện biển số xe trên một tệp ảnh đầu vào bằng CLI."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Đảm bảo console Windows hỗ trợ UTF-8 không bị lỗi charmap cp1252
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Đảm bảo import được src khi chạy trực tiếp từ thư mục gốc
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import RecognitionConfig
from src.pipeline import LicensePlateRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Thực thi dự đoán biển số xe trên ảnh truyền qua CLI và xuất tệp kết quả."""
    parser = argparse.ArgumentParser(description="Nhận diện biển số xe từ một tệp ảnh.")
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn tới tệp trọng số YOLOv8 (.pt)")
    parser.add_argument("--source", type=Path, required=True, help="Đường dẫn tệp ảnh đầu vào")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/prediction.jpg"),
        help="Đường dẫn tệp ảnh kết quả minh họa",
    )
    parser.add_argument("--cpu", action="store_true", help="Ép buộc thực thi trên CPU")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/recognition.yaml"),
        help="Tệp cấu hình nhận diện dùng chung với API",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Ngưỡng tin cậy phát hiện của detector",
    )
    args = parser.parse_args()

    config = RecognitionConfig.from_yaml(args.config) if args.config.is_file() else RecognitionConfig()
    if args.confidence is not None:
        config = config.override(detection_confidence=args.confidence)

    recognizer = LicensePlateRecognizer(
        args.weights,
        gpu=False if args.cpu else None,
        config=config,
    )
    logger.info("Đang thực hiện nhận diện biển số cho tệp: %s", args.source)
    results = recognizer.predict_file(args.source, args.output)

    print(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info("Ảnh kết quả minh họa đã lưu tại: %s", args.output)


if __name__ == "__main__":
    main()
