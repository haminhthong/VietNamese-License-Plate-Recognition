"""Nhận diện biển số xe trên một tệp ảnh đầu vào."""

import argparse
import json
import logging
from pathlib import Path

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
        help="Tệp cấu hình nhận diện dùng chung với API và đánh giá",
    )
    parser.add_argument(
        "--candidate-confidence",
        type=float,
        default=None,
        help="Ngưỡng thấp để giữ candidate detector cho decision policy",
    )
    parser.add_argument(
        "--accept-confidence",
        type=float,
        default=None,
        help="Ngưỡng detector tối thiểu để AUTO_ACCEPT",
    )
    parser.add_argument("--ocr-threshold", type=float, default=None, help="Ngưỡng OCR để AUTO_ACCEPT")
    parser.add_argument(
        "--confidence",
        type=float,
        help="Alias legacy cho --candidate-confidence",
    )
    args = parser.parse_args()

    config = RecognitionConfig.from_yaml(args.config) if args.config.is_file() else RecognitionConfig()
    config = config.override(
        detector_candidate_threshold=(
            args.confidence if args.confidence is not None else args.candidate_confidence
        ),
        auto_accept_detector_threshold=args.accept_confidence,
        ocr_threshold=args.ocr_threshold,
    )
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
