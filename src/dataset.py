"""Kiểm tra dữ liệu YOLOv8 và chia tập train/val/test theo nhóm để tránh rò rỉ dữ liệu (Data Leakage).

Module này:
1. Nhóm các ảnh có cùng chuỗi biển số (plate_identity) hoặc cùng lượt quay camera (capture_group) vào cùng một split.
2. Gộp các ảnh có cùng mã băm nội dung (MD5) để không bị phân tán giữa train và test.
3. Chia tập dữ liệu cân bằng theo tỷ lệ 70% Train / 15% Val / 15% Test sử dụng GroupShuffleSplit.
4. Tạo cấu trúc thư mục dữ liệu và file `data.yaml` tiêu chuẩn cho Ultralytics YOLOv8.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

CLASS_NAMES = {0: "license_plate"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def normalize_plate_identity(value: Any) -> str:
    """Chuẩn hóa identity để gom nhóm không phụ thuộc dấu cách hoặc dấu gạch nối."""
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Tính mã băm MD5 của tệp để phát hiện ảnh trùng lặp nội dung binary."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Phân tích tệp nhãn định dạng YOLO txt (class_id, x_center, y_center, width, height)."""
    rows = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Nhãn YOLO không hợp lệ tại {label_path}:{line_number}")
        class_value = float(parts[0])
        if not class_value.is_integer():
            raise ValueError(f"Class ID phải là số nguyên tại {label_path}:{line_number}")
        class_id = int(class_value)
        coordinates = tuple(map(float, parts[1:]))
        _, _, width, height = coordinates
        if (
            class_id not in CLASS_NAMES
            or not all(0 <= value <= 1 for value in coordinates)
            or width <= 0
            or height <= 0
        ):
            raise ValueError(f"Annotation không hợp lệ tại {label_path}:{line_number}")
        rows.append((class_id, *coordinates))
    return rows


def infer_source_group(stem: str) -> str:
    """Suy luận nhóm nguồn gốc từ tên tệp ảnh (ví dụ: 'cam01_001.jpg' -> 'cam01')."""
    match = re.match(r"^(.+?)[_-](\d+)$", stem)
    return match.group(1).lower() if match else f"standalone::{stem.lower()}"


def _load_metadata(root: Path, metadata_path: Path | None) -> dict[str, dict[str, Any]]:
    """Nạp metadata theo image key."""
    if metadata_path and not metadata_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp metadata: {metadata_path}")
    candidates = [metadata_path] if metadata_path else [root / "metadata.csv", root / "annotations.csv"]
    existing = next((path for path in candidates if path and path.is_file()), None)
    if existing is None:
        return {}

    metadata = pd.read_csv(existing, dtype=str).fillna("")
    key_column = next(
        (column for column in ("image_path", "image_name", "image_id") if column in metadata.columns),
        None,
    )
    if key_column is None:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for row in metadata.to_dict(orient="records"):
        key = str(row[key_column]).replace("\\", "/").strip()
        keys = {key, Path(key).name, Path(key).stem}
        for item in keys:
            lookup[item] = row
    return lookup


def _merge_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Gộp nhóm bằng Disjoint-Set Union (DSU) dựa trên plate_identity và MD5."""
    parent = {group_id: group_id for group_id in frame["group_id"].unique()}

    def find(group_id: str) -> str:
        while parent[group_id] != group_id:
            parent[group_id] = parent[parent[group_id]]
            group_id = parent[group_id]
        return group_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    # 1. Gộp theo MD5 exact duplicate
    for groups in frame.groupby("md5")["group_id"].unique():
        first = groups[0]
        for group_id in groups[1:]:
            union(first, group_id)

    # 2. Gộp theo plate_identity (Protocol: same plate -> same group)
    if "plate_identity" in frame.columns:
        for identity, group_rows in frame.groupby("plate_identity"):
            if identity:
                unique_groups = group_rows["group_id"].unique()
                first = unique_groups[0]
                for group_id in unique_groups[1:]:
                    union(first, group_id)

    result = frame.copy()
    result["group_id"] = result["group_id"].map(find)
    return result


def collect_manifest(root: Path, metadata_path: Path | None = None) -> pd.DataFrame:
    """Thu thập manifest ảnh và metadata phục vụ chia tập chống rò rỉ."""
    import cv2

    required_directories = [
        root / split / subdirectory
        for split in ("train", "valid", "test")
        for subdirectory in ("images", "labels")
    ]
    missing = [path for path in required_directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Thiếu thư mục dữ liệu bắt buộc: {missing[0]}")

    metadata = _load_metadata(root, metadata_path)
    records = []

    for original_split in ("train", "valid", "test"):
        image_dir, label_dir = root / original_split / "images", root / original_split / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise ValueError(f"Ảnh thiếu tệp nhãn: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Không thể đọc tệp ảnh: {image_path}")

            labels = parse_label_file(label_path)
            metadata_row = metadata.get(image_path.name, metadata.get(image_path.stem, {}))
            capture_group = metadata_row.get("capture_group") or infer_source_group(image_path.stem)
            plate_text = metadata_row.get("plate_text", "")
            plate_identity = normalize_plate_identity(metadata_row.get("plate_identity") or plate_text)

            records.append(
                {
                    "image_id": f"{original_split}/{image_path.name}",
                    "original_split": original_split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "image_name": image_path.name,
                    "capture_group": capture_group,
                    "group_id": capture_group,
                    "n_objects": len(labels),
                    "plate_identity": plate_identity,
                    "plate_text": plate_text,
                    "md5": file_md5(image_path),
                }
            )

    if not records:
        raise RuntimeError(f"Không tìm thấy dữ liệu YOLO hợp lệ trong: {root}")

    df = pd.DataFrame(records)
    return _merge_groups(df)


def find_group_safe_split(
    manifest: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> pd.DataFrame:
    """Chia tập train / val / test bằng GroupShuffleSplit đảm bảo không trùng group."""
    df = manifest.copy()
    groups = df["group_id"].values

    test_ratio = 1.0 - train_ratio
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=groups))

    temp_df = df.iloc[temp_idx]
    temp_groups = temp_df["group_id"].values
    val_share = val_ratio / test_ratio
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1.0 - val_share), random_state=seed)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_groups))

    val_idx = temp_df.iloc[val_idx_rel].index
    test_idx = temp_df.iloc[test_idx_rel].index

    df.loc[df.index[train_idx], "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def audit_manifest(frame: pd.DataFrame) -> dict[str, Any]:
    """Kiểm tra tổng quan dữ liệu và xác nhận không có leakage giữa các split."""
    split_col = "split" if "split" in frame.columns else "original_split"
    crossing_groups = int((frame.groupby("group_id")[split_col].nunique() > 1).sum())
    crossing_identities = 0
    if "plate_identity" in frame.columns:
        crossing_identities = int(
            (frame[frame["plate_identity"] != ""].groupby("plate_identity")[split_col].nunique() > 1).sum()
        )

    return {
        "total_images": len(frame),
        "total_objects": int(frame["n_objects"].sum()),
        "unique_groups": int(frame["group_id"].nunique()),
        "crossing_groups": crossing_groups,
        "crossing_identities": crossing_identities,
        "split_distribution": frame[split_col].value_counts().to_dict(),
    }


def materialize_split(split_df: pd.DataFrame, output_dir: Path | str) -> Path:
    """Tạo cấu trúc thư mục dataset mới và ghi file data.yaml cho YOLOv8."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    for _, row in split_df.iterrows():
        split = row["split"]
        src_img, src_lbl = Path(row["image_path"]), Path(row["label_path"])
        dst_img = out / split / "images" / src_img.name
        dst_lbl = out / split / "labels" / src_lbl.name
        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_lbl, dst_lbl)

    data_yaml = {
        "path": str(out.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": CLASS_NAMES,
    }
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml, sort_keys=False), encoding="utf-8")
    split_df.to_csv(out / "split_manifest.csv", index=False)
    return yaml_path
