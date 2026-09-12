"""Kiểm tra dữ liệu YOLOv8 và tạo tập phân chia Group-Safe Split chống rò rỉ dữ liệu."""

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

from src.dataset import audit_manifest, collect_manifest, find_group_safe_split, materialize_split
from src.io_utils import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Thu thập manifest dữ liệu, chia tập Group-Safe Split và tạo data.yaml."""
    parser = argparse.ArgumentParser(description="Chia tập dữ liệu YOLOv8 Group-Safe Split chống rò rỉ.")
    parser.add_argument(
        "--source", type=Path, required=True, help="Thư mục chứa dữ liệu gốc train/valid/test"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="CSV metadata có image key, capture_group và plate_identity (nếu có)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset/grouped"), help="Thư mục xuất dữ liệu split mới"
    )
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/dataset_audit.json"),
        help="Đường dẫn lưu báo cáo kiểm toán JSON",
    )
    args = parser.parse_args()

    logger.info("Thu thập manifest dữ liệu từ thư mục: %s", args.source)
    manifest = collect_manifest(args.source, metadata_path=args.metadata)
    audit = audit_manifest(manifest)

    logger.info("Đang thực hiện thuật toán tìm cách chia Group-Safe Split...")
    split = find_group_safe_split(manifest, seed=args.seed)
    audit["split_result"] = audit_manifest(split)
    write_json(args.audit_output, audit)
    logger.info("Báo cáo audit dữ liệu đã được lưu tại: %s", args.audit_output)

    yaml_path = materialize_split(split, args.output)
    summary = split.groupby("split").agg(images=("image_path", "count"), objects=("n_objects", "sum"))
    logger.info("Bảng phân bổ tập dữ liệu mới:\n%s", summary)
    logger.info("Tệp cấu hình dữ liệu YAML sẵn sàng: %s", yaml_path)


if __name__ == "__main__":
    main()
