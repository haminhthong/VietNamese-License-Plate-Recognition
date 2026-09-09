"""Huấn luyện mô hình phát hiện biển số xe YOLOv8."""

import argparse
import hashlib
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

from src.config import TrainingConfig
from src.io_utils import sha256_file, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _git_commit() -> str | None:
    """Lấy commit hiện tại nếu đang chạy trong Git repository."""
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={Path.cwd()}", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _split_hashes(manifest_path: Path) -> dict[str, str | None]:
    """Hash độc lập cho train/val, giúp truy nguyên đúng dữ liệu đã huấn luyện."""
    if not manifest_path.is_file():
        return {"train_split_sha256": None, "val_split_sha256": None, "dataset_manifest_sha256": None}
    frame = pd.read_csv(manifest_path, dtype=str).fillna("")
    hashes: dict[str, str | None] = {"dataset_manifest_sha256": sha256_file(manifest_path)}
    for split, name in (("train", "train_split_sha256"), ("val", "val_split_sha256")):
        payload = frame[frame.get("split", pd.Series(index=frame.index, dtype=str)) == split]
        canonical = payload.to_csv(index=False).encode("utf-8")
        hashes[name] = hashlib.sha256(canonical).hexdigest()
    return hashes


def _resolve_manifest_path(data_yaml: Path) -> Path:
    """Tìm manifest theo `path` trong data.yaml, đúng với nơi prepare_dataset ghi ra."""
    if not data_yaml.is_file():
        return data_yaml.parent / "split_manifest.csv"
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return data_yaml.parent / "split_manifest.csv"
    configured_root = payload.get("path", ".")
    if not isinstance(configured_root, str | Path) or not str(configured_root).strip():
        return data_yaml.parent / "split_manifest.csv"
    dataset_root = Path(configured_root)
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    return dataset_root / "split_manifest.csv"


def main() -> None:
    """Khởi chạy quá trình huấn luyện YOLOv8 với cấu hình từ file YAML và đè CLI."""
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv8 cho nhận diện biển số xe.")
    parser.add_argument("--data", type=Path, default=Path("data.yaml"), help="Đường dẫn tệp data.yaml")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train.yaml"), help="Đường dẫn tệp cấu hình train.yaml"
    )
    parser.add_argument("--model", help="Tên hoặc đường dẫn trọng số khởi tạo YOLO")
    parser.add_argument("--epochs", type=int, help="Số lượt huấn luyện tối đa")
    parser.add_argument("--batch", type=int, help="Kích thước lô dữ liệu (batch size)")
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="Thư mục lưu kết quả huấn luyện")
    args = parser.parse_args()

    config = TrainingConfig.from_yaml(args.config).override(
        model=args.model,
        epochs=args.epochs,
        batch=args.batch,
    )

    run_name = f"yolov8n_grouped_{datetime.now():%Y%m%d_%H%M%S}"
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
        fliplr=0.0,
        flipud=0.0,
    )

    manifest_path = _resolve_manifest_path(args.data)
    lineage = _split_hashes(manifest_path)
    metadata = {
        "run_name": run_name,
        "data": str(args.data.resolve()),
        "model": config.model,
        "epochs": config.epochs,
        "batch": config.batch,
        "seed": config.seed,
        "best_weights": str(model.trainer.best),
        "best_weights_sha256": sha256_file(model.trainer.best),
        **lineage,
        "training_config_sha256": sha256_file(args.config),
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "ultralytics_version": getattr(__import__("ultralytics"), "__version__", "unknown"),
        "torch_version": torch.__version__,
    }
    write_json(Path(model.trainer.save_dir) / "experiment.json", metadata)
    logger.info("Huấn luyện hoàn tất. Trọng số tốt nhất đã được lưu tại: %s", model.trainer.best)


if __name__ == "__main__":
    main()
