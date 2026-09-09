"""Kiểm tra mẫu biển số và tạo đề xuất sửa lỗi, không ghi đè kết quả OCR thô.

Grammar chỉ trả lời hai câu hỏi:

* Chuỗi OCR thô có khớp một mẫu được hỗ trợ hay không?
* Nếu chưa khớp, có đề xuất sửa ký tự nào có chi phí nhỏ hay không?

Module này không quyết định ``ACCEPT``/``REVIEW``/``REJECT`` và không tham gia
xếp hạng ứng viên OCR.
"""

from __future__ import annotations

from importlib import resources as package_resources
from pathlib import Path
from typing import Any

import yaml

ASCII_DIGITS = "0123456789"
ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SUPPORTED_PATTERNS_VERSION = "civilian-v1"

DEFAULT_DIGIT_SUBSTITUTIONS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "J": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "B": "8",
}
DEFAULT_LETTER_SUBSTITUTIONS = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "8": "B",
}
DEFAULT_PLATE_TEMPLATES = {
    7: ["DDLDDDD"],
    8: ["DDLDDDDD"],
    9: ["DDLDDDDDD", "DDLLDDDDD"],
    10: ["DDLLDDDDDD"],
}


def _load_yaml_resource(filename: str) -> tuple[Any | None, Any]:
    """Đọc resource từ source tree trước, sau đó thử resource đã đóng gói trong wheel."""
    source_path = Path(__file__).resolve().parent.parent / "resources" / filename
    if source_path.is_file():
        return source_path, yaml.safe_load(source_path.read_text(encoding="utf-8"))

    try:
        packaged = package_resources.files("resources").joinpath(filename)
        if packaged.is_file():
            return packaged, yaml.safe_load(packaged.read_text(encoding="utf-8"))
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    return None, None


def normalize_plate_text(text: str | None) -> str:
    """Viết hoa và giữ lại duy nhất chữ Latin ASCII cùng chữ số."""
    if text is None:
        return ""
    allowed = set(ASCII_DIGITS + ASCII_LETTERS)
    return "".join(character for character in str(text).upper() if character in allowed)


def load_plate_templates() -> dict[int, list[str]]:
    """Nạp mẫu biển số từ YAML; chỉ dùng mặc định khi tệp chưa tồn tại."""
    path, payload = _load_yaml_resource("plate_templates.yaml")
    if path is None:
        return DEFAULT_PLATE_TEMPLATES.copy()

    if not isinstance(payload, dict) or not isinstance(payload.get("templates"), dict):
        raise ValueError(f"Tệp mẫu biển số không hợp lệ: {path}")
    return {
        int(length): [str(template) for template in templates]
        for length, templates in payload["templates"].items()
    }


def load_ocr_substitutions() -> tuple[dict[str, str], dict[str, str]]:
    """Nạp bảng nhầm lẫn OCR từ YAML; chỉ dùng mặc định khi tệp chưa tồn tại."""
    path, payload = _load_yaml_resource("ocr_confusions.yaml")
    if path is None:
        return DEFAULT_DIGIT_SUBSTITUTIONS.copy(), DEFAULT_LETTER_SUBSTITUTIONS.copy()

    if not isinstance(payload, dict):
        raise ValueError(f"Tệp nhầm lẫn OCR không hợp lệ: {path}")
    digit_substitutions = payload.get("digit_substitutions", DEFAULT_DIGIT_SUBSTITUTIONS)
    letter_substitutions = payload.get("letter_substitutions", DEFAULT_LETTER_SUBSTITUTIONS)
    if not isinstance(digit_substitutions, dict) or not isinstance(letter_substitutions, dict):
        raise ValueError(f"Tệp nhầm lẫn OCR không hợp lệ: {path}")
    return dict(digit_substitutions), dict(letter_substitutions)


PLATE_TEMPLATES = load_plate_templates()
DIGIT_SUBSTITUTIONS, LETTER_SUBSTITUTIONS = load_ocr_substitutions()


def fit_plate_template(raw_text: str, template: str) -> dict[str, Any] | None:
    """Tính đề xuất sửa cho một template mà không thay đổi ``raw_text``."""
    normalized = normalize_plate_text(raw_text)
    if len(normalized) != len(template):
        return None

    suggestion: list[str] = []
    correction_cost = 0.0
    for character, expected_type in zip(normalized, template, strict=True):
        substitutions = DIGIT_SUBSTITUTIONS if expected_type == "D" else LETTER_SUBSTITUTIONS
        is_valid = character in ASCII_DIGITS if expected_type == "D" else character in ASCII_LETTERS
        if is_valid:
            suggestion.append(character)
            continue
        if character not in substitutions:
            return None
        suggestion.append(substitutions[character])
        correction_cost += 1.0

    return {
        "suggested_text": "".join(suggestion),
        "template": template,
        "correction_cost": correction_cost,
    }


def validate_and_correct_plate(
    raw_text: str,
    enable_correction: bool = True,
    max_cost: float = 1.0,
) -> dict[str, Any]:
    """Kiểm tra format và trả đề xuất sửa, tuyệt đối không auto-apply đề xuất.

    ``max_cost`` được giữ trong chữ ký để tương thích CLI cũ. Trong policy v1,
    chỉ cần có ``correction_suggestion`` là phải chuyển sang ``REVIEW``; không
    có trường hợp sửa một ký tự rồi tự động chấp nhận.
    """
    if max_cost < 0:
        raise ValueError("max_cost không được nhỏ hơn 0")

    normalized = normalize_plate_text(raw_text)
    candidates = [
        candidate
        for template in PLATE_TEMPLATES.get(len(normalized), [])
        if (candidate := fit_plate_template(normalized, template)) is not None
    ]
    best = min(candidates, key=lambda item: item["correction_cost"], default=None)
    format_valid = bool(best and best["correction_cost"] == 0.0)
    suggestion = None
    if enable_correction and best and not format_valid:
        suggestion = best["suggested_text"]

    return {
        "raw_text": normalized,
        "normalized_text": normalized,
        "text": normalized,
        "format_valid": format_valid,
        "template": best["template"] if best else None,
        "correction_cost": float(best["correction_cost"] if best else 0.0),
        "correction_suggestion": suggestion,
        "correction_applied": False,
        "needs_manual_review": not format_valid or suggestion is not None,
        "plate_pattern_version": SUPPORTED_PATTERNS_VERSION,
    }
