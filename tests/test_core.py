"""Kiểm thử đơn vị tự động (Unit Tests) cho các module cốt lõi trong package src."""

import numpy as np
import pandas as pd
import pytest

from src.config import TrainingConfig
from src.io_utils import (
    read_image,
    require_columns,
    require_non_empty_text,
    resolve_relative_path,
    write_json,
)
from src.metrics import box_iou, levenshtein_distance, match_ground_truth_boxes, summarize_ocr
from src.ocr import infer_plate_layout, normalize_plate_text, order_ocr_tokens, validate_and_correct_plate
from src.pipeline import RecognitionConfig, crop_with_padding
from src.rectification import order_points, rectify_plate


def test_post_processing_keeps_raw_result_visible():
    """Kiểm tra raw OCR bất biến và correction chỉ là suggestion."""
    result = validate_and_correct_plate("51FI2345")
    assert result["raw_text"] == "51FI2345"
    assert result["normalized_text"] == "51FI2345"
    assert result["text"] == "51FI2345"
    assert result["format_valid"] is False
    assert result["correction_suggestion"] == "51F12345"
    assert result["correction_applied"] is False
    assert result["correction_cost"] == 1.0
    assert normalize_plate_text(" 51F-123.45 ") == "51F12345"
    assert normalize_plate_text("51À12345") == "5112345"


def test_levenshtein_distance_and_ocr_metrics():
    """Kiểm tra khoảng cách Levenshtein và các thống kê chỉ số OCR (CER, Accuracy)."""
    assert levenshtein_distance("51F12345", "51F12345") == 0
    assert levenshtein_distance("51F12345", "51F12346") == 1

    metrics = summarize_ocr([("51F12345", "51F12345"), ("30A12345", "30A12346")])
    assert metrics["samples"] == 2
    assert metrics["exact_plate_accuracy"] == 0.5
    assert metrics["cer"] == 1 / 16
    assert metrics["character_accuracy"] == 1.0 - (1 / 16)


def test_geometry_and_rectification_helpers():
    """Kiểm tra thuật toán sắp xếp 4 đỉnh góc, biến đổi phối cảnh và tính IoU."""
    points = order_points(np.array([[10, 10], [0, 0], [0, 10], [10, 0]]))
    assert points.tolist() == [[0, 0], [10, 0], [10, 10], [0, 10]]

    blank = np.zeros((50, 120, 3), dtype=np.uint8)
    output, changed = rectify_plate(blank)
    assert output.shape == blank.shape and changed is False
    assert infer_plate_layout(blank) == "1_line"
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_pipeline_configuration_and_crop_validation():
    """Kiểm tra xác thực tham số RecognitionConfig và cắt ảnh có khoảng đệm."""
    RecognitionConfig(detection_confidence=0.4, padding_ratio=0.1)
    with pytest.raises(ValueError):
        RecognitionConfig(detection_confidence=1.1)

    image = np.zeros((20, 30, 3), dtype=np.uint8)
    crop, padded_box = crop_with_padding(image, (5, 5, 15, 15), padding_ratio=0.5)
    assert crop.shape[:2] == (20, 20)
    assert padded_box == (0, 0, 20, 20)

    with pytest.raises(ValueError):
        crop_with_padding(image, (10, 10, 5, 5))


def test_rejects_invalid_ocr_options():
    """Kiểm tra xử lý ngoại lệ khi tham số OCR hoặc bố cục không hợp lệ."""
    with pytest.raises(ValueError):
        infer_plate_layout(np.zeros((10, 10, 3)), wide_ratio_threshold=0)
    with pytest.raises(ValueError):
        order_ocr_tokens([], "unknown")


def test_training_config_and_table_validation(tmp_path):
    """Kiểm tra nạp cấu hình từ tệp YAML và các hàm kiểm tra ràng buộc cột DataFrame."""
    config_path = tmp_path / "train.yaml"
    config_path.write_text("model: yolov8n.pt\nepochs: 5\nbatch: 2\nimgsz: 320\n", encoding="utf-8")
    config = TrainingConfig.from_yaml(config_path)
    assert config.image_size == 320
    assert config.override(epochs=10).epochs == 10

    frame = pd.DataFrame({"plate_text": ["51F12345"]})
    require_columns(frame, {"plate_text"}, "test")
    require_non_empty_text(frame, "plate_text", "test")

    with pytest.raises(ValueError):
        require_columns(frame, {"crop_path"}, "test")


def test_io_utils_helpers(tmp_path):
    """Kiểm tra các tiện ích I/O ghi JSON và giải quyết đường dẫn tương đối."""
    json_path = tmp_path / "sub" / "output.json"
    written = write_json(json_path, {"status": "success"})
    assert written.is_file()

    base_dir = tmp_path / "project"
    resolved = resolve_relative_path("data/test.jpg", base_dir)
    assert resolved == base_dir / "data" / "test.jpg"

    with pytest.raises(ValueError, match="Không đọc được ảnh kiểm thử"):
        read_image(tmp_path / "missing.jpg", "ảnh kiểm thử")


def test_error_analysis_classification():
    """Kiểm tra logic phân loại các nhóm lỗi trong module error_analysis."""
    from src.error_analysis import classify_error

    assert classify_error("51F12345", "51F12345", "51F12345", detected=True, iou=0.8) == "correct"
    assert classify_error("51F12345", "", "", detected=False, iou=0.0) == "detector_miss"
    assert classify_error("51F12345", "", "", detected=True, iou=0.3) == "iou_poor"
    assert classify_error("51F12345", "51F12345", "51F12345", detected=True, iou=0.3) == "iou_poor"
    assert classify_error("51F12345", "51F1234", "51F1234", detected=True, iou=0.8) == "ocr_wrong"
    assert (
        classify_error("51F12345", "51F12345", "51F12346", detected=True, iou=0.8)
        == "template_over_correction"
    )
    assert (
        classify_error(
            "51F12345",
            "51F12345",
            "",
            detected=True,
            candidate_found=True,
            iou=0.8,
        )
        == "decision_abstain"
    )


def test_box_matching_uses_global_iou_assignment():
    """Không để ghép cục bộ làm mất ghép IoU tốt hơn của box kế tiếp."""
    predictions = [
        {"box": (0, 0, 10, 10), "raw_text": "A"},
        {"box": (2, 0, 12, 10), "raw_text": "B"},
    ]
    truths = pd.DataFrame(
        {
            "x1": [1, 0],
            "y1": [0, 0],
            "x2": [11, 10],
            "y2": [10, 10],
            "plate_text": ["A", "B"],
        }
    )
    matches = match_ground_truth_boxes(predictions, truths, iou_threshold=0.5)
    assert [item["prediction"]["raw_text"] for item in matches] == ["B", "A"]
    assert all(item["matched"] for item in matches)
    # Candidate ở nơi khác không phải evidence định vị cho biển GT này.
    distant = [{"box": (100, 100, 110, 110), "raw_text": "A"}]
    missed = match_ground_truth_boxes(distant, truths)
    assert all(not item["candidate_found"] for item in missed)
    with pytest.raises(ValueError, match="iou_threshold"):
        match_ground_truth_boxes(predictions, truths, iou_threshold=0)


def test_phash_helpers_and_dataset_audit():
    """Kiểm tra thuật toán pHash và khoảng cách Hamming giữa hai chuỗi băm."""
    from src.dataset import phash_hamming_distance

    assert phash_hamming_distance("0000000000000000", "0000000000000000") == 0
    assert phash_hamming_distance("ffffffffffffffff", "0000000000000000") == 64
    assert phash_hamming_distance("000000000000000f", "0000000000000000") == 4


def test_recognition_config_from_yaml_and_validation(tmp_path):
    """Kiểm tra nạp RecognitionConfig từ tệp YAML và các giới hạn hợp lệ."""
    yaml_file = tmp_path / "rec.yaml"
    yaml_file.write_text("detection_confidence: 0.35\nnms_iou: 0.5\nimage_size: 320\n", encoding="utf-8")
    config = RecognitionConfig.from_yaml(yaml_file)
    assert config.detection_confidence == 0.35
    assert config.nms_iou == 0.5
    assert config.image_size == 320
    assert config.override(detector_candidate_threshold=0.2).detector_candidate_threshold == 0.2

    with pytest.raises(ValueError):
        RecognitionConfig(ocr_minimum_confidence=-0.1)
    with pytest.raises(ValueError):
        RecognitionConfig(correction_penalty=-0.1)
    with pytest.raises(ValueError):
        RecognitionConfig(single_variant_mode="khong_hop_le")


def test_reliability_policy_and_review_reasons():
    """Kiểm tra decision policy dùng gate rõ ràng và giữ score ở dạng diagnostic."""
    from src.ocr import evaluate_plate_reliability

    cfg = RecognitionConfig(
        detection_confidence=0.25, ocr_minimum_confidence=0.20, min_reliability_score=0.70
    )
    score, reasons, needs_review = evaluate_plate_reliability(0.95, 0.85, 0.75, True, 0.0, cfg)
    assert score > 0.70
    assert not needs_review
    assert reasons == []

    # Thử trường hợp format sai và consensus thấp
    score_low, reasons_low, needs_review_low = evaluate_plate_reliability(0.20, 0.15, 0.25, False, 2.0, cfg)
    assert needs_review_low is True
    assert "INVALID_FORMAT" in reasons_low
    assert "LOW_DETECTION_SCORE" in reasons_low
    assert "HIGH_CORRECTION_COST" in reasons_low


def test_correction_suggestion_always_requires_review():
    """Một suggestion một ký tự cũng không được auto-accept."""
    from src.decision import decide_plate

    policy = decide_plate(
        raw_text="51FI2345",
        detection_confidence=0.99,
        ocr_confidence=0.99,
        consensus_ratio=1.0,
        format_valid=False,
        correction_suggestion="51F12345",
        config=RecognitionConfig(),
    )
    assert policy["decision"] == "REVIEW"
    assert "CORRECTION_SUGGESTED" in policy["review_reasons"]


def test_decision_metrics_prioritize_auto_accept_precision():
    """Kiểm tra metric decision không nhầm coverage với độ chính xác."""
    from src.metrics import compute_decision_metrics

    metrics = compute_decision_metrics(
        [
            {
                "ground_truth": "51F12345",
                "accepted_prediction": "51F12345",
                "decision": "ACCEPT",
                "detected": True,
            },
            {
                "ground_truth": "30A12345",
                "accepted_prediction": "30A12346",
                "decision": "ACCEPT",
                "detected": True,
            },
            {"ground_truth": "43B11111", "decision": "REVIEW", "detected": True},
        ]
    )
    assert metrics["auto_accept_exact_precision"] == 0.5
    assert metrics["auto_accept_coverage"] == 2 / 3
    assert metrics["review_rate"] == 1 / 3


def test_advanced_metrics_and_confusion_matrix():
    """Kiểm tra ma trận nhầm lẫn, đo đạc positional accuracy, postprocessing gain/harm và bootstrap CI."""
    from src.metrics import (
        bootstrap_confidence_intervals,
        compute_confusion_matrix,
        compute_positional_accuracy,
        compute_postprocessing_gain_harm,
    )

    pairs = [("51F12345", "51F12345"), ("51F12345", "51F12845")]
    conf_mat = compute_confusion_matrix(pairs)
    assert conf_mat.get("3->8") == 1

    pos_acc = compute_positional_accuracy(pairs)
    assert "province_digits_accuracy" in pos_acc
    assert pos_acc["province_digits_accuracy"] == 1.0

    records = [
        {"ground_truth": "51F12345", "raw_prediction": "51FI2345", "corrected_prediction": "51F12345"},
        {"ground_truth": "51F12345", "raw_prediction": "51F12345", "corrected_prediction": "51F12345"},
    ]
    gain_harm = compute_postprocessing_gain_harm(records)
    assert gain_harm["correction_gain_count"] == 1
    assert gain_harm["correction_harm_count"] == 0

    cis = bootstrap_confidence_intervals(pairs, num_bootstraps=50)
    assert "exact_accuracy_ci95" in cis
    assert len(cis["exact_accuracy_ci95"]) == 2


def test_resource_loaders():
    """Kiểm tra nạp quy tắc biển số và từ điển thay thế từ tệp resources YAML."""
    from src.ocr import load_ocr_substitutions, load_plate_templates

    templates = load_plate_templates()
    assert 8 in templates
    assert "DDLDDDDD" in templates[8]

    d_subs, l_subs = load_ocr_substitutions()
    assert d_subs["O"] == "0"
    assert l_subs["0"] == "O"


def test_plate_identity_grouping_in_dsu():
    """Kiểm tra tính năng gộp theo plate_identity trong _merge_groups_connected_by_hash (Protocol B)."""
    from src.dataset import _merge_groups_connected_by_hash

    df = pd.DataFrame(
        {
            "group_id": ["cam1_01", "cam2_05"],
            "original_split": ["train", "val"],
            "md5": ["hash1", "hash2"],
            "phash": ["0000000000000000", "ffffffffffffffff"],
            "plate_identity": ["51F12345", "51F12345"],
        }
    )
    merged = _merge_groups_connected_by_hash(df)
    assert merged["group_id"].nunique() == 1
