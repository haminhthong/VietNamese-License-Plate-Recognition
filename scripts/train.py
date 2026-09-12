"""Huấn luyện mô hình phát hiện biển số xe YOLOv8."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Đảm bảo import được src khi chạy trực tiếp từ thư mục gốc
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from ultralytics import YOLO

from src.config import TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Khởi chạy quá trình huấn luyện YOLOv8 với cấu hình từ YAML và CLI."""
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv8 cho nhận diện biển số xe.")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"), help="Đường dẫn tệp data.yaml")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train.yaml"), help="Đường dẫn tệp cấu hình train.yaml"
    )
    parser.add_argument("--model", help="Tên hoặc đường dẫn trọng số khởi tạo YOLO (mặc định: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, help="Số lượt huấn luyện tối đa")
    parser.add_argument("--batch", type=int, help="Kích thước batch")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="Thư mục lưu kết quả huấn luyện")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config) if args.config.is_file() else TrainingConfig()
    config = config.override(
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
    )

    run_name = f"yolov8n_plate_{datetime.now():%Y%m%d_%H%M%S}"
    logger.info("Khởi chạy huấn luyện YOLOv8: %s", run_name)
    logger.info(
        "Thông số: model=%s, epochs=%d, batch=%d, imgsz=%d",
        config.model,
        config.epochs,
        config.batch,
        config.image_size,
    )

    model = YOLO(config.model)
    model.train(
        data=str(args.data),
        epochs=config.epochs,
        patience=config.patience,
        imgsz=config.image_size,
        batch=config.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=config.workers,
        seed=config.seed,
        deterministic=True,
        project=str(args.runs),
        name=run_name,
        pretrained=True,
        optimizer="auto",
        close_mosaic=10,
        degrees=3.0,
        translate=0.10,
        scale=0.40,
        perspective=0.0005,
        fliplr=0.0,  # Không lật ngang vì biển số có chiều đọc chữ cố định
        flipud=0.0,
    )

    logger.info("Huấn luyện hoàn tất. Trọng số tốt nhất đã được lưu tại: %s", model.trainer.best)


if __name__ == "__main__":
    main()
