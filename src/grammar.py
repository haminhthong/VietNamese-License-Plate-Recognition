"""Kiểm tra cú pháp biển số Việt Nam và đề xuất sửa lỗi ký tự nhầm lẫn (O/0, B/8, etc.).

Module này:
1. Chuẩn hóa chuỗi ký tự sang chữ in hoa Latin ASCII và số.
2. Kiểm tra chuỗi có khớp với các mẫu định dạng biển số dân sự được hỗ trợ hay không.
3. Nếu chưa khớp, đưa ra gợi ý sửa ký tự (suggestion-only, tuyệt đối không tự động ghi đè).
"""

from __future__ import annotations

from importlib import resources as package_resources
from pathlib import Path
from typing import Any

import yaml

ASCII_DIGITS = "0123456789"
ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

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
    """Đọc resource từ source tree hoặc từ package."""
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
    """Viết hoa và giữ lại duy nhất chữ cái Latin ASCII cùng chữ số."""
    if text is None:
        return ""
    allowed = set(ASCII_DIGITS + ASCII_LETTERS)
    return "".join(character for character in str(text).upper() if character in allowed)


def load_plate_templates() -> dict[int, list[str]]:
    """Nạp mẫu định dạng biển số từ file YAML hoặc dùng danh sách mặc định."""
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
    """Nạp bảng quy tắc nhầm lẫn ký tự từ file YAML hoặc dùng bảng mặc định."""
    path, payload = _load_yaml_resource("ocr_confusions.yaml")
    if path is None:
        return DEFAULT_DIGIT_SUBSTITUTIONS.copy(), DEFAULT_LETTER_SUBSTITUTIONS.copy()

    if not isinstance(payload, dict):
        raise ValueError(f"Tệp nhầm lẫn OCR không hợp lệ: {path}")
    digit_subs = payload.get("digit_substitutions", DEFAULT_DIGIT_SUBSTITUTIONS)
    letter_subs = payload.get("letter_substitutions", DEFAULT_LETTER_SUBSTITUTIONS)
    return dict(digit_subs), dict(letter_subs)


PLATE_TEMPLATES = load_plate_templates()
DIGIT_SUBSTITUTIONS, LETTER_SUBSTITUTIONS = load_ocr_substitutions()


def fit_plate_template(raw_text: str, template: str) -> dict[str, Any] | None:
    """Khớp chuỗi ký tự với một template định dạng (D: Digit, L: Letter)."""
    normalized = normalize_plate_text(raw_text)
    if len(normalized) != len(template):
        return None

    suggestion: list[str] = []
    num_substitutions = 0
    for character, expected_type in zip(normalized, template, strict=True):
        substitutions = DIGIT_SUBSTITUTIONS if expected_type == "D" else LETTER_SUBSTITUTIONS
        is_valid = character in ASCII_DIGITS if expected_type == "D" else character in ASCII_LETTERS
        if is_valid:
            suggestion.append(character)
            continue
        if character not in substitutions:
            return None
        suggestion.append(substitutions[character])
        num_substitutions += 1

    return {
        "suggested_text": "".join(suggestion),
        "template": template,
        "num_substitutions": num_substitutions,
    }


def validate_and_correct_plate(
    raw_text: str,
    enable_correction: bool = True,
) -> dict[str, Any]:
    """Kiểm tra tính hợp lệ cú pháp và tạo đề xuất sửa ký tự (không tự động áp dụng).

    Returns:
        dict: Chứa text, format_valid, template, suggested_text, needs_review.
    """
    normalized = normalize_plate_text(raw_text)
    candidates = [
        candidate
        for template in PLATE_TEMPLATES.get(len(normalized), [])
        if (candidate := fit_plate_template(normalized, template)) is not None
    ]
    best = min(candidates, key=lambda item: item["num_substitutions"], default=None)
    format_valid = bool(best and best["num_substitutions"] == 0)

    suggestion = None
    if enable_correction and best and not format_valid:
        suggestion = best["suggested_text"]

    needs_review = (not format_valid) or (suggestion is not None)

    return {
        "text": normalized,
        "raw_text": normalized,
        "format_valid": format_valid,
        "template": best["template"] if best else None,
        "suggested_text": suggestion,
        "needs_review": needs_review,
    }
