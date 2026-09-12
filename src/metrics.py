"""Các công thức đo lường và đánh giá hiệu năng cho Detector, OCR và End-to-End ALPR.

Bao gồm:
- levenshtein_distance: Khoảng cách chỉnh sửa ký tự.
- box_iou: Giao trên Hợp giữa hai bounding box.
- summarize_ocr: Exact Plate Accuracy, CER (Character Error Rate), Character Accuracy.
- match_ground_truth_boxes: Ghép nối ground truth với predictions theo IoU.
- classify_error & generate_error_analysis_report: Phân loại nguyên nhân lỗi (Detection / IoU / OCR / Postprocessing).
- summarize_latencies: Mean, P50, P95 độ trễ xử lý.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .grammar import normalize_plate_text


def levenshtein_distance(source: str, target: str) -> int:
    """Tính khoảng cách chỉnh sửa Levenshtein giữa hai chuỗi sau khi chuẩn hóa."""
    src, tgt = normalize_plate_text(source), normalize_plate_text(target)
    previous = list(range(len(tgt) + 1))
    for src_idx, src_char in enumerate(src, 1):
        current = [src_idx]
        for tgt_idx, tgt_char in enumerate(tgt, 1):
            current.append(
                min(
                    current[tgt_idx - 1] + 1,
                    previous[tgt_idx] + 1,
                    previous[tgt_idx - 1] + (src_char != tgt_char),
                )
            )
        previous = current
    return previous[-1]


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Tính chỉ số IoU giữa hai Bounding Box dạng (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def summarize_ocr(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Tổng hợp chỉ số đánh giá OCR: Exact Plate Accuracy, CER và Character Accuracy."""
    normalized = [
        (normalize_plate_text(truth), normalize_plate_text(prediction)) for truth, prediction in pairs
    ]
    if not normalized:
        raise ValueError("Không có mẫu dữ liệu OCR nào để đánh giá.")
    total_characters = sum(len(truth) for truth, _ in normalized)
    if total_characters == 0:
        raise ValueError("Chuỗi nhãn OCR thực tế không được để rỗng hoàn toàn.")
    total_edits = sum(levenshtein_distance(truth, prediction) for truth, prediction in normalized)
    exact_matches = sum(truth == prediction for truth, prediction in normalized)
    cer = total_edits / total_characters
    return {
        "samples": len(normalized),
        "exact_plate_accuracy": exact_matches / len(normalized),
        "cer": cer,
        "character_accuracy": max(0.0, 1.0 - cer),
    }


def summarize_latencies(latencies_ms: Iterable[float]) -> dict[str, float]:
    """Tính các thống kê độ trễ xử lý (ms): Mean, P50, P95."""
    values = [float(item) for item in latencies_ms]
    if not values:
        return {"mean_latency_ms": 0.0, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0}
    return {
        "mean_latency_ms": float(np.mean(values)),
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p95_latency_ms": float(np.percentile(values, 95)),
    }


def match_ground_truth_boxes(
    predictions: list[dict[str, Any]],
    ground_truths: Any,
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Ghép nối bounding box ground truth với predictions theo IoU cao nhất."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold phải nằm trong khoảng (0, 1].")
    truth_rows = list(ground_truths.itertuples(index=False))
    if not truth_rows:
        return []

    edges = [
        (
            box_iou((truth.x1, truth.y1, truth.x2, truth.y2), prediction["box"]),
            truth_index,
            prediction_index,
        )
        for truth_index, truth in enumerate(truth_rows)
        for prediction_index, prediction in enumerate(predictions)
    ]
    best_iou_by_truth = {
        truth_index: max((edge[0] for edge in edges if edge[1] == truth_index), default=0.0)
        for truth_index in range(len(truth_rows))
    }
    assigned_truths: set[int] = set()
    assigned_predictions: set[int] = set()
    assignments: dict[int, tuple[int, float]] = {}
    for iou, truth_index, prediction_index in sorted(edges, reverse=True):
        if iou < iou_threshold:
            break
        if truth_index in assigned_truths or prediction_index in assigned_predictions:
            continue
        assigned_truths.add(truth_index)
        assigned_predictions.add(prediction_index)
        assignments[truth_index] = prediction_index, iou

    matched_results = []
    for truth_index, truth in enumerate(truth_rows):
        assignment = assignments.get(truth_index)
        matched = assignment is not None
        prediction_index, matched_iou = assignment if assignment else (None, best_iou_by_truth[truth_index])
        prediction = (
            predictions[prediction_index]
            if matched
            else {"text": "", "raw_text": "", "detection_confidence": 0.0, "ocr_confidence": 0.0}
        )
        matched_results.append(
            {
                "truth": truth,
                "prediction": prediction,
                "matched": matched,
                "candidate_found": best_iou_by_truth[truth_index] > 0,
                "iou": matched_iou,
            }
        )
    return matched_results


def classify_error(
    ground_truth: str,
    raw_pred: str,
    corrected_pred: str,
    detected: bool,
    iou: float,
    iou_threshold: float = 0.5,
    candidate_found: bool | None = None,
) -> str:
    """Phân loại nhóm nguyên nhân lỗi:
    - correct: Nhận diện chính xác 100%
    - detector_miss: YOLO không tìm thấy biển số
    - iou_poor: IoU dưới ngưỡng
    - ocr_wrong: Nhận dạng sai ký tự OCR
    - decision_abstain: Dự đoán bị bỏ qua
    - template_over_correction: Hậu xử lý làm sai kết quả thô vốn đúng
    """
    gt_norm = normalize_plate_text(ground_truth)
    raw_norm = normalize_plate_text(raw_pred)
    corr_norm = normalize_plate_text(corrected_pred)

    has_candidate = detected if candidate_found is None else candidate_found
    if not has_candidate or not detected:
        return "detector_miss"
    if iou < iou_threshold:
        return "iou_poor"
    target_pred = corr_norm if corr_norm else raw_norm
    if target_pred == gt_norm:
        return "correct"
    if raw_norm == gt_norm and corr_norm != gt_norm:
        return "template_over_correction"
    return "ocr_wrong"


def generate_error_analysis_report(
    records: list[dict[str, Any]],
    output_csv: Path | str,
) -> pd.DataFrame:
    """Xuất DataFrame báo cáo phân tích lỗi và lưu ra tệp CSV."""
    df = pd.DataFrame(records)
    if "error_type" not in df.columns:
        df["error_type"] = [
            classify_error(
                row["ground_truth"],
                row.get("raw_prediction", row.get("text", "")),
                row.get("accepted_prediction", row.get("corrected_prediction", "")),
                row["detected"],
                row.get("iou", 1.0),
                candidate_found=row.get("candidate_found"),
            )
            for _, row in df.iterrows()
        ]
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def compute_confusion_matrix(pairs: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Thống kê ma trận nhầm lẫn ký tự giữa Ground Truth và chuỗi dự đoán (ví dụ: '0->O': 5)."""
    counts: dict[str, int] = {}
    for truth, pred in pairs:
        gt_norm, pred_norm = normalize_plate_text(truth), normalize_plate_text(pred)
        if len(gt_norm) == len(pred_norm):
            for g_char, p_char in zip(gt_norm, pred_norm, strict=True):
                if g_char != p_char:
                    key = f"{g_char}->{p_char}"
                    counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def compute_positional_accuracy(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    """Đo độ chính xác theo vị trí: 2 chữ số tỉnh thành, chữ cái series, các chữ số thứ tự."""
    province_correct, province_total = 0, 0
    letter_correct, letter_total = 0, 0
    serial_correct, serial_total = 0, 0

    for truth, pred in pairs:
        gt = normalize_plate_text(truth)
        pr = normalize_plate_text(pred)
        if len(gt) >= 7 and len(pr) >= 7 and len(gt) == len(pr):
            province_total += 2
            province_correct += sum(gt[i] == pr[i] for i in range(2))

            letter_total += 1
            letter_correct += gt[2] == pr[2]

            serial_total += len(gt) - 3
            serial_correct += sum(gt[i] == pr[i] for i in range(3, len(gt)))

    return {
        "province_digits_accuracy": province_correct / max(1, province_total),
        "series_letter_accuracy": letter_correct / max(1, letter_total),
        "serial_digits_accuracy": serial_correct / max(1, serial_total),
    }


def compute_postprocessing_gain_harm(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Đo gain/harm của suggestion đối với các kết quả nhận diện."""
    gain_count, harm_count, raw_correct, total_corrections = 0, 0, 0, 0

    for r in records:
        gt = normalize_plate_text(r.get("ground_truth", ""))
        raw = normalize_plate_text(r.get("raw_prediction", r.get("text", "")))
        suggestion = normalize_plate_text(r.get("correction_suggestion", r.get("suggested_text", "")))

        if raw == gt:
            raw_correct += 1
        if suggestion and suggestion != raw:
            total_corrections += 1
            if raw != gt and suggestion == gt:
                gain_count += 1
            elif raw == gt and suggestion != gt:
                harm_count += 1

    total_samples = max(1, len(records))
    return {
        "correction_gain_count": gain_count,
        "correction_harm_count": harm_count,
        "suggestion_precision": gain_count / max(1, total_corrections),
        "raw_exact_accuracy": raw_correct / total_samples,
    }


def bootstrap_confidence_intervals(
    pairs: list[tuple[str, str]],
    num_bootstraps: int = 500,
    ci: float = 0.95,
) -> dict[str, list[float]]:
    """Tính 95% khoảng tin cậy Bootstrap cho Exact Accuracy và CER."""
    if not pairs:
        return {"exact_accuracy_ci95": [0.0, 0.0], "cer_ci95": [0.0, 0.0]}
    if num_bootstraps <= 0:
        raise ValueError("num_bootstraps phải lớn hơn 0.")
    if not 0 < ci < 1:
        raise ValueError("ci phải nằm trong khoảng (0, 1).")

    rng = np.random.default_rng(42)
    acc_bootstraps = []
    cer_bootstraps = []

    pairs_arr = np.array(pairs, dtype=object)
    n = len(pairs)
    for _ in range(num_bootstraps):
        indices = rng.integers(0, n, size=n)
        sample = pairs_arr[indices]
        summary = summarize_ocr([(p[0], p[1]) for p in sample])
        acc_bootstraps.append(summary["exact_plate_accuracy"])
        cer_bootstraps.append(summary["cer"])

    alpha = (1.0 - ci) / 2.0
    return {
        "exact_accuracy_ci95": [
            float(np.percentile(acc_bootstraps, alpha * 100)),
            float(np.percentile(acc_bootstraps, (1 - alpha) * 100)),
        ],
        "cer_ci95": [
            float(np.percentile(cer_bootstraps, alpha * 100)),
            float(np.percentile(cer_bootstraps, (1 - alpha) * 100)),
        ],
    }
