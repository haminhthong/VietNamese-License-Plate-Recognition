"""Phân loại lỗi và xuất báo cáo phân tích lỗi (Error Analysis) cho pipeline nhận diện biển số."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .grammar import normalize_plate_text


def classify_error(
    ground_truth: str,
    raw_pred: str,
    corrected_pred: str,
    detected: bool,
    iou: float,
    iou_threshold: float = 0.5,
) -> str:
    """Phân loại nhóm nguyên nhân lỗi cho từng mẫu nhận diện theo Error Taxonomy tiêu chuẩn:
    - correct: Dự đoán đúng 100%
    - detector_miss: Detector bỏ sót hoàn toàn biển số
    - iou_poor: Bounding box có IoU thấp (< iou_threshold)
    - template_over_correction: Hậu xử lý template làm sai kết quả raw OCR vốn đúng
    - ocr_wrong: Lỗi nhận dạng OCR ký tự
    """
    gt_norm = normalize_plate_text(ground_truth)
    raw_norm = normalize_plate_text(raw_pred)
    corr_norm = normalize_plate_text(corrected_pred)

    if not detected:
        return "detector_miss"
    if iou < iou_threshold:
        return "iou_poor"
    if corr_norm == gt_norm:
        return "correct"
    if raw_norm == gt_norm and corr_norm != gt_norm:
        return "template_over_correction"
    return "ocr_wrong"


def generate_error_analysis_report(
    records: list[dict[str, Any]],
    output_csv: Path | str,
) -> pd.DataFrame:
    """Xuất DataFrame báo cáo phân loại lỗi và lưu ra tệp CSV."""
    df = pd.DataFrame(records)
    if "error_type" not in df.columns:
        df["error_type"] = [
            classify_error(
                row["ground_truth"],
                row["raw_prediction"],
                row.get("accepted_prediction", row.get("corrected_prediction", "")),
                row["detected"],
                row.get("iou", 1.0),
            )
            for _, row in df.iterrows()
        ]
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df
