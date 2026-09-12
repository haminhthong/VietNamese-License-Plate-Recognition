"""Đánh giá hiệu năng nhận dạng EasyOCR trên tập ảnh crop biển số kèm nhãn ground truth."""

import argparse
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
from src.io_utils import (
    read_image,
    require_columns,
    require_non_empty_text,
    resolve_relative_path,
    write_json,
)
from src.metrics import summarize_ocr
from src.ocr import read_plate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Đánh giá OCR-only trên các ảnh crop biển số kèm nhãn ground truth."""
    parser = argparse.ArgumentParser(description="Đánh giá mô hình OCR theo ground truth CSV.")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Tệp CSV chứa crop_path, plate_text và layout tùy chọn",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/ocr_metrics.json"), help="Tệp JSON lưu kết quả"
    )
    parser.add_argument("--cpu", action="store_true", help="Ép buộc chạy trên CPU")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/recognition.yaml"),
        help="Tệp cấu hình nhận diện",
    )
    args = parser.parse_args()

    import easyocr
    import pandas as pd
    import torch

    frame = pd.read_csv(args.annotations, dtype=str).fillna("")
    require_columns(frame, {"crop_path", "plate_text"}, "Ground truth OCR")
    require_non_empty_text(frame, "plate_text", "Ground truth OCR")

    logger.info("Khởi tạo EasyOCR Reader (GPU=%s)...", torch.cuda.is_available() and not args.cpu)
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available() and not args.cpu)
    config = RecognitionConfig.from_yaml(args.config) if args.config.is_file() else RecognitionConfig()
    pairs, rows = [], []
    base_directory = args.annotations.resolve().parent

    for row in frame.itertuples(index=False):
        crop_path = resolve_relative_path(row.crop_path, base_directory)
        image = read_image(crop_path, "tệp ảnh crop")
        result = read_plate(reader, image, getattr(row, "layout", "auto") or "auto", config=config)
        pairs.append((row.plate_text, result["text"]))
        rows.append({"crop_path": str(crop_path), "ground_truth": row.plate_text, **result})

    summary = summarize_ocr(pairs)
    payload = {
        "ocr_metrics": summary,
    }
    write_json(args.output, payload)
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".predictions.csv"), index=False)

    logger.info("Kết quả đánh giá OCR đã được ghi tại: %s", args.output)
    logger.info(
        "Chính xác 100%% biển: %.2f%% | CER: %.4f | Character Accuracy: %.2f%%",
        summary["exact_plate_accuracy"] * 100,
        summary["cer"],
        summary["character_accuracy"] * 100,
    )


if __name__ == "__main__":
    main()
