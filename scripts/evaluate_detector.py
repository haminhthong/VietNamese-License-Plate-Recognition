"""Đánh giá mô hình detector YOLOv8 trên tập dữ liệu kiểm thử (test split)."""

import argparse
import logging
import sys
from pathlib import Path

# Đảm bảo import được src khi chạy trực tiếp từ thư mục gốc
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from ultralytics import YOLO

from src.io_utils import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Đánh giá Precision, Recall, mAP50, mAP50-95 của detector YOLOv8."""
    parser = argparse.ArgumentParser(description="Đánh giá mô hình phát hiện biển số xe trên test split.")
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn tới tệp trọng số best.pt")
    parser.add_argument(
        "--data", type=Path, default=Path("data.yaml"), help="Đường dẫn tệp cấu hình data.yaml"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/detector_test_metrics.json"),
        help="Đường dẫn lưu kết quả JSON",
    )
    args = parser.parse_args()

    logger.info("Đang thực hiện đánh giá detector trên test split...")
    metrics = YOLO(str(args.weights)).val(data=str(args.data), split="test", imgsz=640, batch=16)

    class_ids = np.asarray(metrics.box.ap_class_index, dtype=int)
    precision = np.asarray(metrics.box.p, dtype=float)
    recall = np.asarray(metrics.box.r, dtype=float)
    ap50 = np.asarray(metrics.box.ap50, dtype=float)
    ap = np.asarray(metrics.box.ap, dtype=float)

    per_class = [
        {
            "class_id": int(class_id),
            "class": metrics.names[int(class_id)],
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "mAP50": float(ap50[index]),
            "mAP50_95": float(ap[index]),
        }
        for index, class_id in enumerate(class_ids)
    ]

    payload = {
        "weights": str(args.weights),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "per_class": per_class,
    }
    write_json(args.output, payload)
    logger.info("Kết quả đánh giá detector đã lưu tại: %s", args.output)
    logger.info(
        "mAP50: %.4f | mAP50-95: %.4f | Precision: %.4f | Recall: %.4f",
        payload["mAP50"],
        payload["mAP50_95"],
        payload["precision"],
        payload["recall"],
    )


if __name__ == "__main__":
    main()
