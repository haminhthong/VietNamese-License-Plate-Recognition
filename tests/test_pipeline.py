"""Kiểm thử quy trình pipeline, cắt ảnh có padding, nắn phối cảnh và metrics."""

import numpy as np
import pandas as pd
import pytest

from src.config import RecognitionConfig, TrainingConfig
from src.io_utils import read_image, resolve_relative_path, write_json
from src.metrics import box_iou, levenshtein_distance, match_ground_truth_boxes, summarize_ocr
from src.pipeline import crop_with_padding
from src.rectification import order_points, rectify_plate


def test_crop_with_padding():
    """Kiểm tra cắt vùng ảnh có đệm mở rộng và chặn biên ảnh."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    crop, padded_box = crop_with_padding(image, (10, 10, 50, 50), padding_ratio=0.10)
    # box w=40, h=40, pad_x=4, pad_y=4 -> padded (6, 6, 54, 54)
    assert padded_box == (6, 6, 54, 54)
    assert crop.shape[:2] == (48, 48)

    # Chặn biên dưới và phải
    _, boundary_box = crop_with_padding(image, (80, 80, 100, 100), padding_ratio=0.50)
    assert boundary_box[2] <= 100 and boundary_box[3] <= 100

    # Lỗi khi tọa độ box không hợp lệ
    with pytest.raises(ValueError):
        crop_with_padding(image, (50, 50, 20, 20))


def test_perspective_rectification_helpers():
    """Kiểm tra sắp xếp 4 điểm góc và thuật toán nắn phẳng phối cảnh."""
    points = np.array([[10, 10], [0, 0], [0, 10], [10, 0]])
    ordered = order_points(points)
    assert ordered.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]

    blank = np.zeros((50, 120, 3), dtype=np.uint8)
    output, changed = rectify_plate(blank)
    assert output.shape == blank.shape
    assert changed is False


def test_metrics_calculations():
    """Kiểm tra tính khoảng cách Levenshtein, IoU và tổng hợp chỉ số CER, Accuracy."""
    assert levenshtein_distance("51F12345", "51F12345") == 0
    assert levenshtein_distance("51F12345", "51F12346") == 1
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    ocr_sum = summarize_ocr([("51F12345", "51F12345"), ("30A12345", "30A12346")])
    assert ocr_sum["samples"] == 2
    assert ocr_sum["exact_plate_accuracy"] == 0.5
    assert ocr_sum["cer"] == 1 / 16


def test_match_ground_truth_boxes():
    """Kiểm tra thuật toán ghép cặp bounding box ground truth với predictions."""
    preds = [
        {"box": (0, 0, 10, 10), "text": "A"},
        {"box": (20, 0, 30, 10), "text": "B"},
    ]
    gt_df = pd.DataFrame(
        {
            "x1": [0, 20],
            "y1": [0, 0],
            "x2": [10, 30],
            "y2": [10, 10],
            "plate_text": ["A", "B"],
        }
    )
    matches = match_ground_truth_boxes(preds, gt_df, iou_threshold=0.5)
    assert len(matches) == 2
    assert all(m["matched"] for m in matches)


def test_config_loading_and_override(tmp_path):
    """Kiểm tra nạp file YAML cấu hình và ghi đè thông số."""
    rec_yaml = tmp_path / "rec.yaml"
    rec_yaml.write_text("detection_confidence: 0.30\nms_iou: 0.5\n", encoding="utf-8")
    cfg = RecognitionConfig.from_yaml(rec_yaml)
    assert cfg.detection_confidence == 0.30

    train_yaml = tmp_path / "train.yaml"
    train_yaml.write_text("model: yolov8s.pt\nepochs: 10\n", encoding="utf-8")
    t_cfg = TrainingConfig.from_yaml(train_yaml)
    assert t_cfg.model == "yolov8s.pt"
    assert t_cfg.epochs == 10
    assert t_cfg.override(epochs=20).epochs == 20


def test_io_utils(tmp_path):
    """Kiểm tra các hàm đọc/ghi dữ liệu JSON và giải quyết đường dẫn tương đối."""
    json_file = tmp_path / "test.json"
    write_json(json_file, {"key": "value"})
    assert json_file.is_file()

    resolved = resolve_relative_path("img.jpg", tmp_path)
    assert resolved == tmp_path / "img.jpg"

    with pytest.raises(ValueError):
        read_image(tmp_path / "not_found.jpg")
