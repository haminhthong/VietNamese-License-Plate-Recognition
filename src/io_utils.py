"""Các hàm tiện ích đọc, kiểm tra và ghi tệp artifact dùng chung cho toàn bộ dự án.

Module này bao gồm các tiện ích xác thực cột DataFrame, giải quyết đường dẫn tương đối
và ghi tệp JSON theo chuẩn mã hóa UTF-8.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    """Tính SHA-256 theo stream để lưu lineage cho manifest/config/weights."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: set[str], source_name: str) -> None:
    """Kiểm tra DataFrame có chứa đầy đủ các cột bắt buộc hay không.

    Args:
        frame (pd.DataFrame): Bảng dữ liệu cần kiểm tra.
        required (set[str]): Tập hợp các tên cột bắt buộc.
        source_name (str): Tên mô tả nguồn dữ liệu (để hiển thị báo lỗi).

    Raises:
        ValueError: Nếu DataFrame thiếu một hoặc nhiều cột bắt buộc.
    """
    if missing := required - set(frame.columns):
        raise ValueError(f"Nguồn dữ liệu '{source_name}' thiếu các cột bắt buộc: {sorted(missing)}")


def require_non_empty_text(frame: pd.DataFrame, column: str, source_name: str) -> None:
    """Kiểm tra cột văn bản trong DataFrame không được rỗng hoặc chứa giá trị khoảng trắng.

    Args:
        frame (pd.DataFrame): Bảng dữ liệu cần kiểm tra.
        column (str): Tên cột văn bản.
        source_name (str): Tên nguồn dữ liệu.

    Raises:
        ValueError: Nếu DataFrame rỗng hoặc có giá trị trong cột bị rỗng.
    """
    if frame.empty:
        raise ValueError(f"Nguồn dữ liệu '{source_name}' không chứa dòng dữ liệu nào.")
    values = frame[column].fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"Cột '{column}' trong nguồn dữ liệu '{source_name}' không được để rỗng.")


def require_manifest_split(
    frame: pd.DataFrame,
    expected_split: str,
    source_name: str,
    *,
    allow_unlocked: bool = False,
) -> None:
    """Bắt buộc evaluator đọc đúng development hoặc locked-test manifest."""
    if "split" not in frame.columns:
        if allow_unlocked:
            return
        raise ValueError(
            f"Nguồn dữ liệu '{source_name}' phải có cột split={expected_split}; "
            "dùng cờ legacy chỉ khi chạy thử nghiệm cũ."
        )
    splits = frame["split"].fillna("").astype(str).str.strip().str.lower()
    aliases = {
        "valid": "dev",
        "validation": "dev",
        "development": "dev",
        "locked_test": "test",
        "final_test": "test",
    }
    normalized = splits.map(lambda value: aliases.get(value, value))
    expected = aliases.get(expected_split.lower(), expected_split.lower())
    if normalized.ne(expected).any():
        found = sorted(normalized.unique())
        raise ValueError(
            f"Nguồn dữ liệu '{source_name}' chứa split {found}, cần toàn bộ là '{expected_split}'."
        )


def resolve_relative_path(path: str | Path, base_directory: Path) -> Path:
    """Giải quyết đường dẫn: Giữ nguyên nếu là tuyệt đối, ghép với thư mục gốc nếu là tương đối.

    Args:
        path (str | Path): Đường dẫn cần kiểm tra.
        base_directory (Path): Thư mục gốc dùng để ghép đường dẫn tương đối.

    Returns:
        Path: Đường dẫn tuyệt đối đã được xử lý.
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_directory / candidate


def read_image(path: str | Path, description: str = "ảnh") -> np.ndarray:
    """Đọc ảnh màu từ đĩa và báo lỗi rõ ràng nếu tệp không hợp lệ.

    Args:
        path (str | Path): Đường dẫn tới tệp ảnh.
        description (str): Tên mô tả dùng trong thông báo lỗi.

    Returns:
        np.ndarray: Ảnh màu theo định dạng BGR của OpenCV.

    Raises:
        ValueError: Nếu OpenCV không thể đọc tệp ảnh.
    """
    image_path = Path(path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không đọc được {description}: {image_path}")
    return image


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Ghi dữ liệu dictionary ra tệp JSON mã hóa UTF-8 và tự động khởi tạo thư mục cha.

    Args:
        path (str | Path): Đường dẫn tệp JSON đầu ra.
        payload (dict[str, Any]): Dữ liệu dict cần ghi.

    Returns:
        Path: Đường dẫn tệp JSON đã ghi thành công.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
