"""Kiểm tra dữ liệu YOLOv8 và tạo tập phân chia Group-Safe Split không rò rỉ giữa các nhóm nguồn."""

import argparse
import logging
from pathlib import Path

from src.dataset import audit_manifest, collect_manifest, find_group_safe_split, materialize_split
from src.io_utils import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Thu thập manifest dữ liệu, kiểm toán trùng lặp, chia tập Group-Safe Split và tạo data.yaml."""
    parser = argparse.ArgumentParser(description="Chia tập dữ liệu YOLOv8 Group-Safe Split chống rò rỉ.")
    parser.add_argument(
        "--source", type=Path, required=True, help="Thư mục chứa dữ liệu gốc train/valid/test"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="CSV metadata có image key, capture_group/capture_session_id và plate_identity/plate_text",
    )
    parser.add_argument(
        "--allow-legacy-identity",
        action="store_true",
        help="Cho phép chạy legacy khi chưa có plate_identity; không dùng cho benchmark Protocol B",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dataset/grouped"), help="Thư mục xuất dữ liệu split mới"
    )
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("artifacts/dataset_audit.json"),
        help="Thư mục lưu báo cáo audit JSON",
    )
    args = parser.parse_args()

    logger.info("Thu thập manifest dữ liệu từ thư mục: %s", args.source)
    manifest = collect_manifest(args.source, metadata_path=args.metadata)
    audit = audit_manifest(manifest)
    identity_values = manifest["plate_identity"].where(
        manifest["plate_identity"].ne(""),
        manifest["plate_identity_hash"],
    )
    if not args.allow_legacy_identity and identity_values.eq("").any():
        raise ValueError(
            "Manifest còn plate_identity rỗng. Hãy cung cấp --metadata hoặc chỉ rõ "
            "--allow-legacy-identity cho dataset cũ."
        )
    logger.info("Đang thực hiện thuật toán tìm cách chia Group-Safe Split...")
    split = find_group_safe_split(
        manifest,
        seed=args.seed,
        require_identity=not args.allow_legacy_identity,
    )
    audit["split_result"] = audit_manifest(split)
    write_json(args.audit_output, audit)
    logger.info("Báo cáo audit dữ liệu đã được lưu tại: %s", args.audit_output)
    yaml_path = materialize_split(split, args.output)

    summary = split.groupby("split").agg(images=("image_path", "count"), objects=("n_objects", "sum"))
    logger.info("Bảng phân bổ tập dữ liệu mới:\n%s", summary)
    logger.info("Tệp cấu hình dữ liệu YAML sẵn sàng: %s", yaml_path)


if __name__ == "__main__":
    main()
