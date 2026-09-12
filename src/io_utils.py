"""Các hàm tiện ích đọc, kiểm tra và ghi tệp artifact cho toàn bộ dự án."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def require_columns(frame: pd.DataFrame, required: set[str], source_name: str) -> None:
    """Kiểm tra DataFrame có chứa đầy đủ các cột bắt buộc hay không."""
    if missing := required - set(frame.columns):
        raise ValueError(f"Nguồn dữ liệu '{source_name}' thiếu các cột bắt buộc: {sorted(missing)}")


def require_non_empty_text(frame: pd.DataFrame, column: str, source_name: str) -> None:
    """Kiểm tra cột văn bản trong DataFrame không được rỗng hoặc chứa toàn khoảng trắng."""
    if frame.empty:
        raise ValueError(f"Nguồn dữ liệu '{source_name}' không chứa dòng dữ liệu nào.")
    values = frame[column].fillna("").astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"Cột '{column}' trong nguồn dữ liệu '{source_name}' không được để rỗng.")


def require_manifest_split(
    frame: pd.DataFrame,
    expected_split: str,
    source_name: str,
) -> None:
    """Kiểm tra DataFrame chứa đúng split mong muốn (train / val / test)."""
    if "split" not in frame.columns:
        return
    splits = frame["split"].fillna("").astype(str).str.strip().str.lower()
    aliases = {"valid": "val", "validation": "val", "dev": "val"}
    normalized = splits.map(lambda value: aliases.get(value, value))
    expected = aliases.get(expected_split.lower(), expected_split.lower())
    if normalized.ne(expected).any():
        found = sorted(normalized.unique())
        raise ValueError(
            f"Nguồn dữ liệu '{source_name}' chứa split {found}, cần toàn bộ là '{expected_split}'."
        )


def resolve_relative_path(path: str | Path, base_directory: Path) -> Path:
    """Ghép đường dẫn tương đối với thư mục gốc nếu chưa phải đường dẫn tuyệt đối."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_directory / candidate


def read_image(path: str | Path, description: str = "ảnh") -> np.ndarray:
    """Đọc ảnh từ đĩa cứng bằng OpenCV BGR."""
    image_path = Path(path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Không đọc được {description}: {image_path}")
    return image


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Ghi dữ liệu dictionary ra tệp JSON mã hóa UTF-8."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
