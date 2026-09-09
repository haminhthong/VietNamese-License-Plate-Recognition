"""Đánh giá hiệu năng nhận dạng OCR thô (raw) và OCR sau hậu xử lý (corrected) từ tệp CSV ground truth."""

import argparse
import logging
from pathlib import Path

import easyocr
import pandas as pd
import torch

from src.config import RecognitionConfig
from src.io_utils import (
    read_image,
    require_columns,
    require_manifest_split,
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
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình OCR và quy tắc hiệu chỉnh theo tệp CSV ground truth."
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Tệp CSV chứa crop_path, plate_text và layout tùy chọn",
    )
    parser.add_argument(
        "--allow-unlocked-annotations",
        action="store_true",
        help="Chỉ dùng legacy; bỏ qua bắt buộc development split",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/ocr_metrics.json"), help="Tệp JSON lưu kết quả chỉ số"
    )
    parser.add_argument("--cpu", action="store_true", help="Ép buộc chạy trên CPU")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/recognition.yaml"),
        help="Tệp cấu hình nhận diện dùng chung với API và CLI predict",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.annotations, dtype=str).fillna("")
    require_columns(frame, {"crop_path", "plate_text"}, "Ground truth OCR")
    require_non_empty_text(frame, "plate_text", "Ground truth OCR")
    require_manifest_split(
        frame,
        "dev",
        "Ground truth OCR",
        allow_unlocked=args.allow_unlocked_annotations,
    )

    logger.info("Khởi tạo EasyOCR Reader (GPU=%s)...", torch.cuda.is_available() and not args.cpu)
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available() and not args.cpu)
    config = RecognitionConfig.from_yaml(args.config) if args.config.is_file() else RecognitionConfig()
    raw_pairs, suggestion_pairs, rows = [], [], []
    base_directory = args.annotations.resolve().parent

    for row in frame.itertuples(index=False):
        crop_path = resolve_relative_path(row.crop_path, base_directory)
        image = read_image(crop_path, "tệp ảnh crop")
        result = read_plate(reader, image, getattr(row, "layout", "auto") or "auto", config=config)
        raw_pairs.append((row.plate_text, result["raw_text"]))
        suggestion_pairs.append((row.plate_text, result.get("correction_suggestion") or result["raw_text"]))
        rows.append({"crop_path": str(crop_path), "ground_truth": row.plate_text, **result})

    suggestion_summary = summarize_ocr(suggestion_pairs)
    payload = {
        "raw": summarize_ocr(raw_pairs),
        "suggestion": suggestion_summary,
        "corrected": suggestion_summary,  # Alias tương thích artifact cũ.
        "correction_mode": "suggest_only",
    }
    write_json(args.output, payload)
    pd.DataFrame(rows).to_csv(args.output.with_suffix(".predictions.csv"), index=False)

    logger.info("Kết quả đánh giá OCR đã được ghi tại: %s", args.output)
    logger.info(
        "Chính xác 100%% biển (Suggestion): %.2f%% | CER: %.4f",
        payload["suggestion"]["exact_plate_accuracy"] * 100,
        payload["suggestion"]["cer"],
    )


if __name__ == "__main__":
    main()
