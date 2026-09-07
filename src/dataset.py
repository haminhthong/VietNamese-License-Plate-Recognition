"""Kiểm tra dữ liệu YOLOv8 và chia tập train/val/test theo nhóm nguồn để tránh rò rỉ dữ liệu (Data Leakage).

Module này hỗ trợ:
1. Trích xuất tên nhóm nguồn (Video Frame Stem) từ tên tệp ảnh.
2. Tính mã băm MD5 để phát hiện ảnh trùng lặp nội dung.
3. Sử dụng cấu trúc dữ liệu Union-Find (Disjoint Set Union) để gộp các nhóm có chung ảnh trùng hash.
4. Thuật toán `GroupShuffleSplit` để phân chia dữ liệu cân bằng theo tỷ lệ 70% Train / 15% Val / 15% Test.
5. Tạo thư mục dữ liệu đích và file `data.yaml` tiêu chuẩn cho Ultralytics YOLOv8.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

# Định nghĩa danh sách các class và định dạng ảnh hỗ trợ
CLASS_NAMES = {0: "license_plate"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
METADATA_COLUMNS = {"image_name", "image_path", "image_id"}


def normalize_plate_identity(value: Any) -> str:
    """Chuẩn hóa identity để grouping không phụ thuộc dấu cách hay dấu phân cách."""
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def plate_identity_hash(value: Any) -> str:
    """Tạo hash ổn định cho artifact công khai, không phát tán biển số thật."""
    normalized = normalize_plate_identity(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16] if normalized else ""


def _load_metadata(root: Path, metadata_path: Path | None) -> dict[str, dict[str, Any]]:
    """Nạp metadata theo image key; tệp không bắt buộc để hỗ trợ dataset legacy."""
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
        raise ValueError(f"Metadata phải có một trong các cột: {sorted(METADATA_COLUMNS)}")
    lookup: dict[str, dict[str, Any]] = {}
    for row in metadata.to_dict(orient="records"):
        key = str(row[key_column]).replace("\\", "/").strip()
        keys = {key, Path(key).name, Path(key).stem}
        if key_column in {"image_path", "image_id"}:
            try:
                keys.add(str((root / key).resolve()).replace("\\", "/"))
            except OSError:
                pass
        for item in keys:
            if item in lookup and lookup[item] != row:
                raise ValueError(f"Metadata có nhiều dòng cho cùng ảnh: {key}")
            lookup[item] = row
    return lookup


def _metadata_for_image(metadata: dict[str, dict[str, Any]], image_path: Path, root: Path) -> dict[str, Any]:
    """Tìm metadata bằng đường dẫn tương đối, tên tệp hoặc stem."""
    relative = str(image_path.relative_to(root)).replace("\\", "/")
    absolute = str(image_path.resolve()).replace("\\", "/")
    for key in (relative, image_path.name, image_path.stem, absolute):
        if key in metadata:
            return metadata[key]
    return {}


def _first_metadata_value(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def infer_source_group(stem: str) -> str:
    """Suy luận nhóm nguồn gốc từ tên tệp ảnh (ví dụ: 'cam01_001.jpg' -> 'cam01').

    Args:
        stem (str): Tên tệp ảnh không kèm đuôi mở rộng.

    Returns:
        str: Nhận diện tiền tố làm mã nhóm, hoặc tạo nhóm đơn lẻ nếu không có mẫu tiền tố.
    """
    match = re.match(r"^(.+?)[_-](\d+)$", stem)
    return match.group(1).lower() if match else f"standalone::{stem.lower()}"


def compute_phash(path: Path) -> str:
    """Tính mã băm cảm nhận pHash (Perceptual Hash) 64-bit từ ảnh xám để tìm ảnh gần trùng lặp.

    Args:
        path (Path): Đường dẫn tệp ảnh.

    Returns:
        str: Chuỗi 16 ký tự hex đại diện pHash.
    """
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return ""
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    dct_low = dct[:8, :8]
    med = float(np.median(dct_low))
    bits = (dct_low > med).flatten()
    hash_int = 0
    for bit in bits:
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"


def phash_hamming_distance(hash1: str, hash2: str) -> int:
    """Tính khoảng cách Hamming giữa hai chuỗi pHash."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 64
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
    except ValueError:
        return 64
    return bin(val1 ^ val2).count("1")


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Tính mã băm MD5 của tệp để phát hiện ảnh bị trùng lặp nội dung binary.

    Args:
        path (Path): Đường dẫn tới tệp cần tính hash.
        chunk_size (int): Kích thước khối đọc dữ liệu (mặc định: 1 MB).

    Returns:
        str: Chuỗi hex digest đại diện cho mã MD5.
    """
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Phân tích tệp nhãn định dạng YOLO txt (class_id, x_center, y_center, width, height).

    Args:
        label_path (Path): Đường dẫn tệp nhãn txt.

    Returns:
        list[tuple[int, float, float, float, float]]: Danh sách các bounding box hợp lệ.

    Raises:
        ValueError: Nếu định dạng nhãn sai, tọa độ nằm ngoài [0, 1] hoặc class_id không hỗ trợ.
    """
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


class _BKTree:
    """BK-Tree tối giản cho tìm pHash trong bán kính Hamming."""

    def __init__(self) -> None:
        self.root: tuple[int, str, set[str], dict[int, _BKTree]] | None = None

    def add(self, value: int, group_id: str) -> None:
        if self.root is None:
            self.root = (value, group_id, {group_id}, {})
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance == 0:
                node[2].add(group_id)
                return
            child = node[3].get(distance)
            if child is None:
                node[3][distance] = _BKTree()
                node[3][distance].root = (value, group_id, {group_id}, {})
                return
            if child.root is None:
                child.root = (value, group_id, {group_id}, {})
                return
            node = child.root

    def search(self, value: int, radius: int) -> set[str]:
        """Trả các group có pHash cách ``value`` không quá ``radius``."""
        found: set[str] = set()

        def visit(node: tuple[int, str, set[str], dict[int, _BKTree]] | None) -> None:
            if node is None:
                return
            distance = (node[0] ^ value).bit_count()
            if distance <= radius:
                found.update(node[2])
            for edge, child in node[3].items():
                if distance - radius <= edge <= distance + radius:
                    visit(child.root)

        visit(self.root)
        return found


def collect_manifest(root: Path, metadata_path: Path | None = None) -> pd.DataFrame:
    """Thu thập manifest ảnh và metadata identity/capture cho Protocol B.

    Args:
        root (Path): Đường dẫn thư mục dữ liệu gốc chứa train/valid/test.

    Returns:
        pd.DataFrame: Bảng manifest chứa đầy đủ thuộc tính từng ảnh.

    Raises:
        FileNotFoundError: Nếu thiếu các thư mục con bắt buộc.
        RuntimeError: Nếu không tìm thấy cặp ảnh/nhãn hợp lệ.
        ValueError: Nếu phát hiện ảnh thiếu nhãn hoặc ảnh không đọc được.
    """
    import cv2

    required_directories = [
        root / split / subdirectory
        for split in ("train", "valid", "test")
        for subdirectory in ("images", "labels")
    ]
    missing_directories = [path for path in required_directories if not path.is_dir()]
    if missing_directories:
        raise FileNotFoundError(f"Thiếu thư mục dữ liệu bắt buộc: {missing_directories[0]}")

    metadata = _load_metadata(root, metadata_path)
    records = []
    missing_labels: list[Path] = []
    unreadable_images: list[Path] = []

    for original_split in ("train", "valid", "test"):
        image_dir, label_dir = root / original_split / "images", root / original_split / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(image_path)
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                unreadable_images.append(image_path)
                continue
            labels = parse_label_file(label_path)
            counts = Counter(item[0] for item in labels)
            metadata_row = _metadata_for_image(metadata, image_path, root)
            capture_group = _first_metadata_value(
                metadata_row,
                "capture_session_id",
                "capture_group",
                "session_id",
            ) or infer_source_group(image_path.stem)
            plate_text = _first_metadata_value(metadata_row, "plate_text")
            plate_identity = _first_metadata_value(metadata_row, "plate_identity") or plate_text
            identity_hash = _first_metadata_value(metadata_row, "plate_identity_hash") or plate_identity_hash(
                plate_identity
            )
            image_id = f"{original_split}/{image_path.name}"
            try:
                plate_count = int(metadata_row.get("plate_count", "") or len(labels))
            except (TypeError, ValueError) as error:
                raise ValueError(f"plate_count không hợp lệ trong metadata của {image_path}") from error
            records.append(
                {
                    "image_id": image_id,
                    "original_split": original_split,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "image_name": image_path.name,
                    "capture_group": capture_group,
                    "camera_id": _first_metadata_value(metadata_row, "camera_id"),
                    "group_id": capture_group,
                    "n_objects": len(labels),
                    "class_0_count": counts.get(0, 0),
                    "plate_count": max(0, plate_count),
                    "plate_identity": plate_identity,
                    "plate_identity_hash": identity_hash,
                    "plate_text": plate_text,
                    "plate_text_normalized": normalize_plate_identity(plate_text),
                    "layout": _first_metadata_value(metadata_row, "layout"),
                    "plate_annotation_id": f"{image_id}::plate-0",
                    "md5": file_md5(image_path),
                    "phash": compute_phash(image_path),
                }
            )

    if not records:
        raise RuntimeError(f"Không tìm thấy cặp ảnh/nhãn YOLO hợp lệ trong thư mục: {root}")
    if missing_labels:
        preview = ", ".join(str(path) for path in missing_labels[:3])
        raise ValueError(f"Có {len(missing_labels)} ảnh thiếu tệp nhãn. Ví dụ: {preview}")
    if unreadable_images:
        preview = ", ".join(str(path) for path in unreadable_images[:3])
        raise ValueError(f"Có {len(unreadable_images)} ảnh lỗi không đọc được. Ví dụ: {preview}")

    return _merge_groups_connected_by_hash(pd.DataFrame(records))


def _merge_groups_connected_by_hash(frame: pd.DataFrame) -> pd.DataFrame:
    """Sử dụng thuật toán Disjoint-Set Union (DSU) để gộp các nhóm nguồn có chung ảnh trùng MD5 hoặc pHash gần kề.

    Điều này đảm bảo hai ảnh giống hệt hoặc gần trùng không bao giờ bị rơi vào hai tập split khác nhau.
    """
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

    # Gộp theo trùng mã MD5
    for groups in frame.groupby("md5")["group_id"].unique():
        first = groups[0]
        for group_id in groups[1:]:
            union(first, group_id)

    # Gộp pHash bằng BK-Tree thay vì so sánh O(N²) toàn bộ cặp ảnh.
    phash_tree = _BKTree()
    phash_groups: dict[str, set[str]] = {}
    for row in frame[["group_id", "phash"]].drop_duplicates().itertuples(index=False):
        if not row.phash:
            continue
        try:
            phash_value = int(row.phash, 16)
        except ValueError:
            continue
        phash_tree.add(phash_value, row.group_id)
        phash_groups.setdefault(row.phash, set()).add(row.group_id)
    for phash, groups in phash_groups.items():
        for group_id in phash_tree.search(int(phash, 16), radius=4):
            for source_group in groups:
                union(source_group, group_id)

    # Protocol B: mỗi identity, capture session và chuỗi identity ghép phải ở một split.
    identity_columns = [
        column
        for column in ("plate_identity", "plate_identity_hash", "plate_text")
        if column in frame.columns
    ]
    identity_to_groups: dict[str, set[str]] = {}
    for row in frame[["group_id", *identity_columns]].to_dict(orient="records"):
        for column in identity_columns:
            values = re.split(r"[|;,]", str(row.get(column, "")))
            for value in values:
                identity = normalize_plate_identity(value)
                if identity:
                    identity_to_groups.setdefault(identity, set()).add(row["group_id"])
    for groups in identity_to_groups.values():
        first, *rest = sorted(groups)
        for group_id in rest:
            union(first, group_id)

    result = frame.copy()
    result["group_id"] = result["group_id"].map(find)
    return result


def audit_manifest(frame: pd.DataFrame) -> dict:
    """Thống kê chi tiết số lượng ảnh, đối tượng, nhóm nguồn và kiểm tra rò rỉ dữ liệu."""
    split_column = "split" if "split" in frame.columns else "original_split"
    crossing_groups = frame.groupby("group_id")[split_column].nunique()
    crossing_hashes = frame.groupby("md5")[split_column].nunique()

    identity_columns = [
        column
        for column in ("plate_identity", "plate_identity_hash", "plate_text")
        if column in frame.columns
    ]
    identity_pairs: dict[str, set[str]] = {}
    for row in frame[[split_column, *identity_columns]].to_dict(orient="records"):
        for column in identity_columns:
            for value in re.split(r"[|;,]", str(row.get(column, ""))):
                identity = normalize_plate_identity(value)
                if identity:
                    identity_pairs.setdefault(identity, set()).add(str(row[split_column]))
    crossing_plates = sum(len(splits) > 1 for splits in identity_pairs.values())

    # Đếm near-duplicate bằng BK-Tree; audit không còn bị nghẽn bởi O(N²).
    near_dup_pairs = 0
    if "phash" in frame.columns:
        tree = _BKTree()
        phashes = frame["phash"].dropna().astype(str).loc[lambda values: values != ""].unique()
        for phash in phashes:
            try:
                phash_value = int(phash, 16)
            except ValueError:
                continue
            neighbors = tree.search(phash_value, radius=4)
            near_dup_pairs += len(neighbors)
            tree.add(phash_value, phash)

    exact_duplicates = int(frame["md5"].duplicated().sum())
    split_values = frame["split"] if "split" in frame.columns else frame["original_split"]
    split_counts = split_values.value_counts().to_dict()

    return {
        "images": len(frame),
        "objects": int(frame["n_objects"].sum()),
        "groups": int(frame["group_id"].nunique()),
        "exact_duplicates": exact_duplicates,
        "near_duplicate_pairs": near_dup_pairs,
        "groups_crossing_splits": int((crossing_groups > 1).sum()),
        "duplicate_hashes_crossing_splits": int((crossing_hashes > 1).sum()),
        "plates_crossing_splits": int(crossing_plates),
        "train_images": int(split_counts.get("train", 0)),
        "validation_images": int(split_counts.get("val", split_counts.get("valid", 0))),
        "test_images": int(split_counts.get("test", 0)),
    }


def _score(frame: pd.DataFrame) -> float:
    """Đánh giá lệch split theo ảnh, biển, layout và camera/source."""
    if frame.empty:
        return float("inf")
    score = 0.0
    metrics = {
        "images": pd.Series(1, index=frame.index),
        "plates": pd.to_numeric(
            frame.get("plate_count", frame.get("n_objects", pd.Series(0, index=frame.index))),
            errors="coerce",
        ).fillna(0),
        "objects": pd.to_numeric(
            frame.get("n_objects", pd.Series(0, index=frame.index)),
            errors="coerce",
        ).fillna(0),
    }
    for column, weight in (("layout", 1.0), ("camera_id", 0.5)):
        if column not in frame.columns:
            continue
        for _, category_frame in frame.assign(_category=frame[column].replace("", "unknown")).groupby(
            "_category"
        ):
            if len(category_frame) < 3:
                continue
            for split, target in TARGET_RATIOS.items():
                actual = len(category_frame[category_frame["split"] == split]) / len(category_frame)
                score += weight * abs(actual - target)
    for split, target in TARGET_RATIOS.items():
        subset = frame[frame["split"] == split]
        score += 8.0 * abs(len(subset) / len(frame) - target)
        for name, values in metrics.items():
            total = float(values.sum())
            count = float(values.loc[subset.index].sum())
            if total <= 0:
                continue
            weight = {"images": 8.0, "plates": 4.0, "objects": 2.0}[name]
            score += weight * abs(count / total - target)
        if subset.empty:
            score += 1_000.0
    return score


def find_group_safe_split(
    frame: pd.DataFrame,
    seed: int = 42,
    trials: int = 1000,
    require_identity: bool = False,
) -> pd.DataFrame:
    """Tìm cách chia Group-Safe Split tối ưu nhất thông qua phương pháp thử nghiệm nhiều hạt giống (trials).

    Args:
        frame (pd.DataFrame): Manifest dữ liệu đã gộp nhóm.
        seed (int): Hạt giống ngẫu nhiên khởi tạo.
        trials (int): Số lần thử nghiệm tìm phương án tối ưu.

    Returns:
        pd.DataFrame: Bảng manifest chứa cột 'split' mới (train, val, test).
    """
    required = {"group_id", "image_path", "n_objects"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Manifest thiếu các cột bắt buộc để chia split: {sorted(missing)}")
    if trials <= 0:
        raise ValueError("Số lần thử split (trials) phải lớn hơn 0.")
    if require_identity:
        if "plate_identity" not in frame.columns and "plate_identity_hash" not in frame.columns:
            raise ValueError("Protocol B yêu cầu plate_identity hoặc plate_identity_hash trong manifest.")
        identities = pd.Series("", index=frame.index, dtype=str)
        if "plate_identity" in frame.columns:
            identities = frame["plate_identity"].map(normalize_plate_identity)
        if "plate_identity_hash" in frame.columns:
            hashes = frame["plate_identity_hash"].map(normalize_plate_identity)
            identities = identities.where(identities != "", hashes)
        if identities.eq("").any():
            raise ValueError("Protocol B không cho phép identity rỗng.")
    if frame["group_id"].nunique() < 3:
        raise RuntimeError("Cần ít nhất 3 group độc lập để tạo train/val/test.")

    best, best_score = None, float("inf")
    for trial in range(trials):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed + trial)
        _, temp_idx = next(outer.split(frame, groups=frame["group_id"]))
        temp = frame.iloc[temp_idx]
        if temp["group_id"].nunique() < 2:
            continue
        inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 10_000 + trial)
        val_rel, test_rel = next(inner.split(temp, groups=temp["group_id"]))
        candidate = frame.copy()
        candidate["split"] = "train"
        candidate.loc[temp.index[val_rel], "split"] = "val"
        candidate.loc[temp.index[test_rel], "split"] = "test"
        candidate_score = _score(candidate)
        if candidate_score < best_score:
            best, best_score = candidate, candidate_score

    if best is None or best_score >= 1_000:
        raise RuntimeError("Không thể tạo split theo nhóm mà vẫn bảo đảm đủ dữ liệu cho mỗi tập.")
    assert best.groupby("group_id")["split"].nunique().max() == 1
    return best


def materialize_split(frame: pd.DataFrame, destination: Path) -> Path:
    """Sao chép tệp ảnh và tệp nhãn vào thư mục phân chia mới và tạo tệp cấu hình data.yaml.

    Args:
        frame (pd.DataFrame): Manifest đã gán phân chia 'split'.
        destination (Path): Thư mục đích lưu dataset mới.

    Returns:
        Path: Đường dẫn tới tệp data.yaml vừa được khởi tạo.
    """
    if destination.exists():
        raise FileExistsError(f"Thư mục đích đã tồn tại: {destination}")

    duplicate_names = set(frame.loc[frame["image_name"].duplicated(keep=False), "image_name"])
    destination_names = []

    for row in frame.itertuples():
        source_image, source_label = Path(row.image_path), Path(row.label_path)
        name = (
            f"{row.original_split}__{source_image.name}"
            if row.image_name in duplicate_names
            else source_image.name
        )
        image_dir, label_dir = destination / row.split / "images", destination / row.split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_dir / name)
        shutil.copy2(source_label, label_dir / f"{Path(name).stem}.txt")
        destination_names.append(name)

    output = frame.copy()
    output["destination_name"] = destination_names
    output.to_csv(destination / "split_manifest.csv", index=False)

    yaml_path = destination.parent / "data.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(destination.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return yaml_path
