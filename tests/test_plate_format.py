"""Kiểm thử cú pháp biển số xe Việt Nam và gợi ý sửa lỗi ký tự nhầm lẫn."""

from src.grammar import (
    load_ocr_substitutions,
    load_plate_templates,
    normalize_plate_text,
    validate_and_correct_plate,
)


def test_normalize_plate_text():
    """Kiểm tra hàm chuẩn hóa chuỗi biển số: loại bỏ dấu cách, gạch nối, dấu chấm và ký tự đặc biệt."""
    assert normalize_plate_text(" 51F-123.45 ") == "51F12345"
    assert normalize_plate_text("30a-999.99") == "30A99999"
    assert normalize_plate_text("43-B1 123.45") == "43B112345"
    assert normalize_plate_text(None) == ""
    assert normalize_plate_text("") == ""


def test_valid_civilian_plate_formats():
    """Kiểm tra các mẫu biển số dân sự Việt Nam hợp lệ."""
    # 8 ký tự: 2 số tỉnh + 1 chữ series + 5 số (DDLDDDDD)
    res_8 = validate_and_correct_plate("51F12345")
    assert res_8["format_valid"] is True
    assert res_8["text"] == "51F12345"
    assert res_8["suggested_text"] is None
    assert res_8["needs_review"] is False

    # 9 ký tự: 2 số tỉnh + 2 chữ series + 5 số (DDLLDDDDD)
    res_9 = validate_and_correct_plate("51MD12345")
    assert res_9["format_valid"] is True
    assert res_9["suggested_text"] is None


def test_confusion_correction_suggestion_without_overwriting():
    """Kiểm tra gợi ý sửa ký tự nhầm lẫn (I->1, B->8, O->0), không tự động ghi đè text thô."""
    # Ký tự I ở vị trí số -> gợi ý 1
    res_i = validate_and_correct_plate("51FI2345")
    assert res_i["text"] == "51FI2345"
    assert res_i["format_valid"] is False
    assert res_i["suggested_text"] == "51F12345"
    assert res_i["needs_review"] is True

    # Ký tự B ở vị trí số -> gợi ý 8
    res_b = validate_and_correct_plate("51F12B45")
    assert res_b["text"] == "51F12B45"
    assert res_b["format_valid"] is False
    assert res_b["suggested_text"] == "51F12845"
    assert res_b["needs_review"] is True


def test_resource_loaders():
    """Kiểm tra nạp từ điển quy tắc biển số và bảng nhầm lẫn ký tự từ YAML."""
    templates = load_plate_templates()
    assert 8 in templates
    assert "DDLDDDDD" in templates[8]

    digit_subs, letter_subs = load_ocr_substitutions()
    assert digit_subs["O"] == "0"
    assert digit_subs["B"] == "8"
    assert letter_subs["0"] == "O"
