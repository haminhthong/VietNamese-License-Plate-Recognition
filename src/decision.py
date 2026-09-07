"""Chính sách quyết định vận hành cho kết quả nhận diện biển số.

Decision layer chạy sau OCR và grammar. Các ngưỡng là các cổng độc lập nên
người vận hành có thể giải thích chính xác vì sao một biển bị ``REVIEW``.
``reliability_score`` chỉ là số chẩn đoán để quan sát, không phải xác suất và
không được dùng làm cổng quyết định.
"""

from __future__ import annotations

from typing import Any

from .config import RecognitionConfig

DECISION_ACCEPT = "ACCEPT"
DECISION_REVIEW = "REVIEW"
DECISION_REJECT = "REJECT"
DECISION_POLICY_VERSION = "1.0.0"


def diagnostic_reliability(
    detection_confidence: float,
    ocr_confidence: float,
    consensus_ratio: float,
    format_valid: bool,
) -> float:
    """Tính điểm quan sát đơn giản; không dùng làm điều kiện ``ACCEPT``."""
    format_signal = 1.0 if format_valid else 0.0
    values = [detection_confidence, ocr_confidence, consensus_ratio, format_signal]
    return max(0.0, min(1.0, sum(values) / len(values)))


def decide_plate(
    *,
    raw_text: str,
    detection_confidence: float,
    ocr_confidence: float,
    consensus_ratio: float,
    format_valid: bool,
    correction_suggestion: str | None,
    config: RecognitionConfig,
    reject_reason: str | None = None,
) -> dict[str, Any]:
    """Áp dụng các cổng rõ ràng để trả về quyết định và lý do kiểm duyệt."""
    reasons: list[str] = []
    if reject_reason:
        reasons.append(reject_reason)
    if not raw_text:
        reasons.append("UNREADABLE")
    if detection_confidence < config.auto_accept_detector_threshold:
        reasons.append("LOW_DETECTION_SCORE")
    if ocr_confidence < config.ocr_threshold:
        reasons.append("LOW_OCR_SCORE")
    if consensus_ratio < config.ocr_consensus_threshold:
        reasons.append("VARIANT_DISAGREEMENT")
    if not format_valid:
        reasons.append("FORMAT_INVALID")
    if correction_suggestion:
        reasons.append("CORRECTION_SUGGESTED")

    # Loại lý do trùng khi một ảnh vừa rỗng vừa bị đánh dấu unreadable.
    reasons = list(dict.fromkeys(reasons))
    if reject_reason or not raw_text:
        decision = DECISION_REJECT
    elif reasons:
        decision = DECISION_REVIEW
    else:
        decision = DECISION_ACCEPT

    return {
        "policy_version": DECISION_POLICY_VERSION,
        "decision": decision,
        "needs_manual_review": decision == DECISION_REVIEW,
        "review_reasons": reasons,
        "reliability_score": diagnostic_reliability(
            detection_confidence,
            ocr_confidence,
            consensus_ratio,
            format_valid,
        ),
        "evidence": {
            "detector_score": float(detection_confidence),
            "ocr_score": float(ocr_confidence),
            "variant_consensus": float(consensus_ratio),
        },
    }
