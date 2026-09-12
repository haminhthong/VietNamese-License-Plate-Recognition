"""Đánh giá toàn diện end-to-end (YOLOv8 Detector + EasyOCR) trên ảnh thực tế kèm nhãn ground truth."""

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
from src.metrics import (
    classify_error,
    compute_confusion_matrix,
    compute_positional_accuracy,
    generate_error_analysis_report,
    match_ground_truth_boxes,
    summarize_latencies,
    summarize_ocr,
)
from src.pipeline import LicensePlateRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Đánh giá pipeline End-to-End tính toán Recall@IoU, Accuracy OCR, CER và Error Analysis."""
    parser = argparse.ArgumentParser(description="Đánh giá pipeline nhận diện biển số xe End-to-End.")
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn trọng số YOLOv8 (.pt)")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Tệp CSV ground truth (image_path, x1, y1, x2, y2, plate_text)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/end_to_end_metrics.json"), help="Tệp JSON xuất kết quả"
    )
    parser.add_argument(
        "--error-analysis-output",
        type=Path,
        default=Path("artifacts/error_analysis.csv"),
        help="Tệp CSV phân tích lỗi",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.5, help="Ngưỡng IoU coi như khớp bounding box"
    )
    parser.add_argument("--cpu", action="store_true", help="Ép buộc thực thi trên CPU")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/recognition.yaml"),
        help="Tệp cấu hình nhận diện",
    )
    args = parser.parse_args()

    import pandas as pd

    frame = pd.read_csv(args.annotations)
    require_columns(frame, {"image_path", "x1", "y1", "x2", "y2", "plate_text"}, "Annotation end-to-end")
    require_non_empty_text(frame, "plate_text", "Annotation end-to-end")

    logger.info("Khởi tạo pipeline LicensePlateRecognizer...")
    config = RecognitionConfig.from_yaml(args.config) if args.config.is_file() else RecognitionConfig()
    recognizer = LicensePlateRecognizer(args.weights, gpu=False if args.cpu else None, config=config)
    base_directory = args.annotations.resolve().parent
    records, latencies = [], []

    for image_name, ground_truths in frame.groupby("image_path", sort=False):
        image_path = resolve_relative_path(image_name, base_directory)
        image = read_image(image_path)
        try:
            predictions = recognizer.predict(image)
        except ValueError as error:
            logger.warning("Bỏ qua ảnh %s: %s", image_path, error)
            predictions = []
        latencies.append(recognizer.last_latency_ms)

        for match in match_ground_truth_boxes(predictions, ground_truths, iou_threshold=args.iou_threshold):
            truth, prediction, matched, best_iou = (
                match["truth"],
                match["prediction"],
                match["matched"],
                match["iou"],
            )
            error_type = classify_error(
                ground_truth=str(truth.plate_text),
                raw_pred=prediction.get("text", ""),
                corrected_pred=prediction.get("suggested_text", ""),
                detected=matched,
                candidate_found=match.get("candidate_found", matched),
                iou=best_iou,
                iou_threshold=args.iou_threshold,
            )
            records.append(
                {
                    "image_path": str(image_path),
                    "ground_truth": str(truth.plate_text),
                    "prediction": prediction.get("text", ""),
                    "suggested_text": prediction.get("suggested_text"),
                    "detector_confidence": prediction.get("detection_confidence", 0.0),
                    "ocr_confidence": prediction.get("ocr_confidence", 0.0),
                    "detected": matched,
                    "candidate_found": match.get("candidate_found", matched),
                    "iou": best_iou,
                    "error_type": error_type,
                }
            )

    predictions_frame = pd.DataFrame(records)
    latency_stats = summarize_latencies(latencies)

    pairs = list(zip(predictions_frame["ground_truth"], predictions_frame["prediction"], strict=True))
    ocr_summary = summarize_ocr(pairs)

    total_gt = len(predictions_frame)
    detected_count = int(predictions_frame["detected"].sum())
    e2e_exact_count = int(
        (
            predictions_frame["detected"]
            & (predictions_frame["ground_truth"] == predictions_frame["prediction"])
        ).sum()
    )

    metrics_payload = {
        "dataset": {
            "total_ground_truth_plates": total_gt,
            "detected_plates": detected_count,
            "detection_recall": detected_count / max(1, total_gt),
        },
        "end_to_end": {
            "exact_plate_recall": e2e_exact_count / max(1, total_gt),
            "exact_plate_count": e2e_exact_count,
        },
        "ocr_metrics": ocr_summary,
        "latency_ms": latency_stats,
        "positional_accuracy": compute_positional_accuracy(pairs),
        "confusion_matrix": compute_confusion_matrix(pairs),
    }

    write_json(args.output, metrics_payload)
    generate_error_analysis_report(records, args.error_analysis_output)

    logger.info("Kết quả đánh giá End-to-End đã lưu tại: %s", args.output)
    logger.info(
        "Detector Recall: %.2f%% | E2E Exact Recall: %.2f%% | CER: %.4f | Latency P50: %.1f ms",
        (detected_count / max(1, total_gt)) * 100,
        (e2e_exact_count / max(1, total_gt)) * 100,
        ocr_summary["cer"],
        latency_stats["p50_latency_ms"],
    )


if __name__ == "__main__":
    main()
