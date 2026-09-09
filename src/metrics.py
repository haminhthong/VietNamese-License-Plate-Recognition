"""Các công thức đo lường và đánh giá hiệu năng cho mô hình phát hiện (Detection) và nhận dạng ký tự (OCR).

Module này định nghĩa:
1. `levenshtein_distance`: Khoảng cách chỉnh sửa tối thiểu giữa hai chuỗi văn bản.
2. `box_iou`: Chỉ số Intersection-over-Union giữa hai khung hình chữ nhật.
3. `summarize_ocr`: Thống kê Exact Plate Accuracy, CER (Character Error Rate) và Character Accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .grammar import normalize_plate_text


def levenshtein_distance(source: str, target: str) -> int:
    """Tính khoảng cách chỉnh sửa Levenshtein giữa chuỗi nguồn và chuỗi đích sau khi chuẩn hóa.

    Args:
        source (str): Chuỗi ký tự biển số gốc (nhãn thực tế).
        target (str): Chuỗi ký tự biển số dự đoán.

    Returns:
        int: Số phép thay thế, thêm, hoặc xóa ký tự tối thiểu để biến chuỗi source thành target.
    """
    source, target = normalize_plate_text(source), normalize_plate_text(target)
    previous = list(range(len(target) + 1))
    for source_index, source_character in enumerate(source, 1):
        current = [source_index]
        for target_index, target_character in enumerate(target, 1):
            current.append(
                min(
                    current[target_index - 1] + 1,
                    previous[target_index] + 1,
                    previous[target_index - 1] + (source_character != target_character),
                )
            )
        previous = current
    return previous[-1]


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Tính chỉ số Giao trên Hợp (Intersection over Union - IoU) giữa hai Bounding Box dạng (x1, y1, x2, y2).

    Args:
        box_a: Bounding box thứ nhất.
        box_b: Bounding box thứ hai.

    Returns:
        float: Giá trị IoU nằm trong khoảng [0.0, 1.0].
    """
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def summarize_ocr(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Tổng hợp các chỉ số đánh giá độ chính xác nhận dạng OCR trên một tập cặp mẫu (Ground Truth, Prediction).

    Args:
        pairs (Iterable[tuple[str, str]]): Tập các cặp (nhãn_thực_tế, chuỗi_dự_đoán).

    Returns:
        dict[str, Any]: Dictionary chứa các chỉ số:
            - samples: Số lượng mẫu đánh giá.
            - exact_plate_accuracy: Tỷ lệ biển số đọc chính xác 100%.
            - cer: Character Error Rate (Tổng số phép sửa / Tổng số ký tự).
            - character_accuracy: Độ chính xác cấp ký tự max(0, 1 - CER).

    Raises:
        ValueError: Nếu tập mẫu rỗng hoặc chuỗi nhãn thực tế bị rỗng.
    """
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
    """Tính các chỉ số thống kê độ trễ xử lý (đơn vị: ms): Trung bình, P50, P95.

    Args:
        latencies_ms (Iterable[float]): Danh sách các giá trị độ trễ ms.

    Returns:
        dict[str, float]: Dictionary chứa mean_latency_ms, p50_latency_ms, p95_latency_ms.
    """
    import numpy as np

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
    """Ghép nối các bounding box ground truth với các dự đoán từ pipeline dựa theo chỉ số IoU cao nhất.

    Args:
        predictions (list[dict[str, Any]]): Danh sách các box dự đoán.
        ground_truths (Any): Iterative DataFrame các nhãn thực tế chứa x1, y1, x2, y2, plate_text.
        iou_threshold (float): Ngưỡng IoU tối thiểu để coi là khớp.

    Returns:
        list[dict[str, Any]]: Kết quả theo thứ tự ground truth, gồm prediction,
        matched, candidate_found và IoU tốt nhất/được ghép.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold phải nằm trong khoảng (0, 1].")
    truth_rows = list(ground_truths.itertuples(index=False))
    if not truth_rows:
        return []

    # Ghép theo toàn bộ cạnh IoU giảm dần. Cách này ổn định hơn việc duyệt
    # ground truth tuần tự, vì một box tốt không bị chiếm bởi một ghép cục bộ.
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
        prediction = predictions[prediction_index] if matched else {"raw_text": "", "text": ""}
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


def compute_confusion_matrix(pairs: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Thống kê ma trận nhầm lẫn ký tự giữa nhãn Ground Truth và chuỗi dự đoán (ví dụ: '0->O': 5)."""
    counts: dict[str, int] = {}
    for truth, pred in pairs:
        gt_norm, pred_norm = normalize_plate_text(truth), normalize_plate_text(pred)
        if len(gt_norm) == len(pred_norm):
            for g_char, p_char in zip(gt_norm, pred_norm, strict=True):
                if g_char != p_char:
                    key = f"{g_char}->{p_char}"
                    counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def compute_postprocessing_gain_harm(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Đo gain/harm của suggestion; suggestion không được coi là output đã apply."""
    gain_count = 0
    harm_count = 0
    raw_correct = 0
    corrected_correct = 0
    total_corrections = 0

    for r in records:
        gt = normalize_plate_text(r.get("ground_truth", ""))
        raw = normalize_plate_text(r.get("raw_prediction", ""))
        suggested_value = r.get("correction_suggestion")
        corr = normalize_plate_text(
            suggested_value
            if suggested_value not in (None, "")
            else r.get("corrected_prediction", r.get("prediction", r.get("raw_prediction", "")))
        )

        is_raw_match = raw == gt
        is_corr_match = corr == gt
        applied = raw != corr

        if is_raw_match:
            raw_correct += 1
        if is_corr_match:
            corrected_correct += 1
        if applied:
            total_corrections += 1
            if not is_raw_match and is_corr_match:
                gain_count += 1
            elif is_raw_match and not is_corr_match:
                harm_count += 1

    total_samples = max(1, len(records))
    raw_errors = total_samples - raw_correct

    return {
        "correction_gain_count": gain_count,
        "correction_harm_count": harm_count,
        "correction_gain_rate": gain_count / total_samples,
        "correction_harm_rate": harm_count / total_samples,
        "correction_precision": gain_count / max(1, total_corrections),
        "correction_recall": gain_count / max(1, raw_errors),
        "raw_exact_accuracy": raw_correct / total_samples,
        "corrected_exact_accuracy": corrected_correct / total_samples,
        "suggestion_gain_count": gain_count,
        "suggestion_harm_count": harm_count,
        "suggestion_precision": gain_count / max(1, total_corrections),
    }


def compute_decision_metrics(records: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """Tính Auto-Accept Precision, Coverage, Review Rate và Wrong Auto-Accept Rate."""
    rows = list(records)
    detected = [row for row in rows if row.get("detected", True)]
    accepted = [row for row in detected if row.get("decision") == "ACCEPT"]
    correct_accepted = [
        row
        for row in accepted
        if normalize_plate_text(row.get("ground_truth", ""))
        == normalize_plate_text(row.get("accepted_prediction", row.get("raw_prediction", "")))
    ]
    reviewed = [row for row in detected if row.get("decision") == "REVIEW"]
    wrong_accepted = len(accepted) - len(correct_accepted)
    return {
        "detected_plates": len(detected),
        "auto_accepted_plates": len(accepted),
        "auto_accept_exact_precision": len(correct_accepted) / max(1, len(accepted)),
        "auto_accept_coverage": len(accepted) / max(1, len(detected)),
        "review_rate": len(reviewed) / max(1, len(detected)),
        "wrong_auto_accept_count": wrong_accepted,
        "wrong_auto_accept_rate": wrong_accepted / max(1, len(accepted)),
    }


def compute_positional_accuracy(pairs: Iterable[tuple[str, str]]) -> dict[str, float]:
    """Đo độ chính xác OCR chia theo vị trí: 2 chữ số tỉnh/thành, chữ cái series, các chữ số thứ tự."""
    province_correct, province_total = 0, 0
    letter_correct, letter_total = 0, 0
    serial_correct, serial_total = 0, 0

    for truth, pred in pairs:
        gt = normalize_plate_text(truth)
        pr = normalize_plate_text(pred)
        if len(gt) >= 7 and len(pr) >= 7 and len(gt) == len(pr):
            # 2 chữ số tỉnh thành gốc
            province_total += 2
            province_correct += sum(gt[i] == pr[i] for i in range(2))

            # Chữ cái series (vị trí 2)
            letter_total += 1
            letter_correct += gt[2] == pr[2]

            # Các chữ số thứ tự đuôi (từ vị trí 3 trở đi)
            serial_total += len(gt) - 3
            serial_correct += sum(gt[i] == pr[i] for i in range(3, len(gt)))

    return {
        "province_digits_accuracy": province_correct / max(1, province_total),
        "series_letter_accuracy": letter_correct / max(1, letter_total),
        "serial_digits_accuracy": serial_correct / max(1, serial_total),
    }


def bootstrap_confidence_intervals(
    pairs: list[tuple[str, str]],
    num_bootstraps: int = 500,
    ci: float = 0.95,
    groups: list[str] | None = None,
) -> dict[str, list[float]]:
    """Tính bootstrap CI, có thể resample theo plate identity/capture group.

    Khi ``groups`` được cung cấp, toàn bộ dòng cùng group được lấy cùng nhau để
    các frame liên tiếp của một xe không bị coi là các quan sát độc lập.
    """
    import numpy as np

    if not pairs:
        return {"exact_accuracy_ci95": [0.0, 0.0], "cer_ci95": [0.0, 0.0]}
    if num_bootstraps <= 0:
        raise ValueError("num_bootstraps phải lớn hơn 0.")
    if not 0 < ci < 1:
        raise ValueError("ci phải nằm trong khoảng (0, 1).")
    if groups is not None and len(groups) != len(pairs):
        raise ValueError("groups phải có cùng số phần tử với pairs.")

    rng = np.random.default_rng(42)
    acc_bootstraps = []
    cer_bootstraps = []

    pairs_arr = np.array(pairs, dtype=object)
    group_values = np.array(groups if groups is not None else list(range(len(pairs))), dtype=object)
    unique_groups = np.unique(group_values)
    for _ in range(num_bootstraps):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([np.flatnonzero(group_values == group) for group in sampled_groups])
        boot_pairs = pairs_arr[indices]
        summary = summarize_ocr([(p[0], p[1]) for p in boot_pairs])
        acc_bootstraps.append(summary["exact_plate_accuracy"])
        cer_bootstraps.append(summary["cer"])

    alpha = (1.0 - ci) / 2.0
    acc_ci = [
        float(np.percentile(acc_bootstraps, alpha * 100)),
        float(np.percentile(acc_bootstraps, (1 - alpha) * 100)),
    ]
    cer_ci = [
        float(np.percentile(cer_bootstraps, alpha * 100)),
        float(np.percentile(cer_bootstraps, (1 - alpha) * 100)),
    ]

    return {
        "exact_accuracy_ci95": acc_ci,
        "cer_ci95": cer_ci,
    }
