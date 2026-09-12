"""Kiểm thử giải thuật hình học token OCR và sắp xếp thứ tự cho biển 1 dòng / 2 dòng."""

import numpy as np
import pytest

from src.ocr import (
    _token_geometry,
    infer_plate_layout,
    order_ocr_tokens,
)


def test_token_geometry_calculation():
    """Kiểm tra trích xuất tâm (center_x, center_y) và chiều cao (height) từ bbox 4 điểm."""
    bbox = [[10.0, 20.0], [50.0, 20.0], [50.0, 60.0], [10.0, 60.0]]
    geom = _token_geometry(bbox)
    assert geom["center_x"] == 30.0
    assert geom["center_y"] == 40.0
    assert geom["height"] == 40.0


def test_one_line_token_ordering():
    """Biển 1 dòng: các token được sắp xếp tuần tự theo chiều ngang từ trái sang phải."""
    # Giả lập kết quả EasyOCR: [bbox, text, conf]
    ocr_results = [
        ([[100, 10], [180, 10], [180, 50], [100, 50]], "12345", 0.90),
        ([[10, 10], [80, 10], [80, 50], [10, 50]], "51F", 0.95),
    ]
    text, conf, tokens = order_ocr_tokens(ocr_results, layout="1_line")
    assert text == "51F12345"
    assert round(conf, 2) > 0.90
    assert len(tokens) == 2


def test_two_line_token_ordering():
    """Biển 2 dòng: dòng trên (series tỉnh) xếp trước, dòng dưới (các số thứ tự) xếp sau."""
    ocr_results = [
        # Dòng dưới: số 12345 (y ở khoảng 60-100)
        ([[20, 60], [120, 60], [120, 100], [20, 100]], "12345", 0.88),
        # Dòng trên: 51F1 (y ở khoảng 10-50)
        ([[20, 10], [90, 10], [90, 50], [20, 50]], "51F1", 0.92),
    ]
    text, conf, tokens = order_ocr_tokens(ocr_results, layout="2_line")
    assert text == "51F112345"
    assert len(tokens) == 2


def test_infer_plate_layout():
    """Kiểm tra tự động suy luận layout 1 dòng hoặc 2 dòng dựa trên tỷ lệ khung hình."""
    wide_crop = np.zeros((50, 200, 3), dtype=np.uint8)  # ratio = 4.0
    assert infer_plate_layout(wide_crop) == "1_line"

    square_crop = np.zeros((100, 120, 3), dtype=np.uint8)  # ratio = 1.2
    assert infer_plate_layout(square_crop) == "2_line"

    with pytest.raises(ValueError):
        infer_plate_layout(wide_crop, wide_ratio_threshold=0)


def test_invalid_parameters_raise_error():
    """Kiểm tra xử lý ngoại lệ khi tham số layout hoặc confidence không hợp lệ."""
    with pytest.raises(ValueError):
        order_ocr_tokens([], "invalid_layout")
    with pytest.raises(ValueError):
        order_ocr_tokens([], "1_line", minimum_confidence=1.5)
