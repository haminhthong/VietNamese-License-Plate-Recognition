"""Tiền xử lý ảnh biển số, trích xuất raw OCR và chuẩn bị evidence cho policy.

Module này chịu trách nhiệm:
1. Chạy fast path Gray/CLAHE và chỉ fallback khi evidence chưa đủ.
2. Xác định bố cục biển 1 dòng (ô tô dài) hoặc 2 dòng (xe máy, ô tô ngắn) dựa trên aspect ratio.
3. Sắp xếp các token nhận dạng được theo đúng thứ tự hình học (từ trên xuống dưới, từ trái sang phải).
4. Chọn raw candidate theo consensus rồi mới gọi grammar để tạo suggestion.
5. Không tự động ghi đè raw OCR bằng kết quả grammar.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import RecognitionConfig
from .decision import decide_plate
from .grammar import (
    DEFAULT_DIGIT_SUBSTITUTIONS,
    DEFAULT_LETTER_SUBSTITUTIONS,
    DEFAULT_PLATE_TEMPLATES,
    DIGIT_SUBSTITUTIONS,
    LETTER_SUBSTITUTIONS,
    PLATE_TEMPLATES,
    load_ocr_substitutions,
    load_plate_templates,
    normalize_plate_text,
)
from .grammar import fit_plate_template as _fit_plate_template
from .grammar import validate_and_correct_plate as validate_plate_format
from .rectification import rectify_plate

# Bộ ký tự cho phép EasyOCR trả về.
ASCII_DIGITS = "0123456789"
ASCII_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
OCR_ALLOWLIST = f"{ASCII_DIGITS}{ASCII_LETTERS}-."
VALID_LAYOUTS = {"1_line", "2_line"}

# Các tên grammar được re-export để notebook/CLI cũ tiếp tục hoạt động.
__all__ = [
    "DEFAULT_DIGIT_SUBSTITUTIONS",
    "DEFAULT_LETTER_SUBSTITUTIONS",
    "DEFAULT_PLATE_TEMPLATES",
    "DIGIT_SUBSTITUTIONS",
    "LETTER_SUBSTITUTIONS",
    "PLATE_TEMPLATES",
    "fit_plate_template",
    "infer_plate_layout",
    "load_ocr_substitutions",
    "load_plate_templates",
    "normalize_plate_text",
    "order_ocr_tokens",
    "preprocess_plate_variants",
    "read_plate",
    "evaluate_plate_reliability",
    "validate_and_correct_plate",
]


def fit_plate_template(raw_text: str, template: str) -> dict[str, Any] | None:
    """Khớp template và giữ alias ``text`` cho code cũ."""
    result = _fit_plate_template(raw_text, template)
    if result is None:
        return None
    return {**result, "text": result["suggested_text"]}


def _token_geometry(bbox: list[list[float]]) -> dict[str, float]:
    """Trích xuất thông tin tọa độ tâm và chiều cao của bounding box token OCR."""
    points = np.asarray(bbox, dtype=float)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    return {
        "center_x": float((min_x + max_x) / 2),
        "center_y": float((min_y + max_y) / 2),
        "height": float(max(1.0, max_y - min_y)),
    }


def order_ocr_tokens(
    ocr_results: list, layout: str, minimum_confidence: float = 0.20
) -> tuple[str, float, list[dict[str, Any]]]:
    """Lọc các token nhiễu và sắp xếp theo đúng thứ tự đọc dựa trên bố cục 1 dòng hoặc 2 dòng.

    Returns:
        tuple[str, float, list[dict]]: (chuỗi_ký_tự, độ_tin_cậy, danh_sách_tokens).
    """
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"Bố cục không hợp lệ: {layout}")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence phải nằm trong khoảng [0, 1]")
    tokens = []
    for bbox, text, confidence in ocr_results:
        normalized = normalize_plate_text(text)
        if normalized and float(confidence) >= minimum_confidence:
            tokens.append(
                {
                    "text": normalized,
                    "confidence": float(confidence),
                    **_token_geometry(bbox),
                }
            )
    if not tokens:
        return "", 0.0, []

    median_height = float(np.median([token["height"] for token in tokens]))
    tokens = [token for token in tokens if token["height"] >= 0.45 * median_height]
    if not tokens:
        return "", 0.0, []

    if layout == "2_line" and len(tokens) >= 2:
        ordered = _order_two_line_tokens(tokens, median_height)
    else:
        ordered = sorted(tokens, key=lambda token: token["center_x"])

    text = "".join(token["text"] for token in ordered)
    confidence = float(
        np.average(
            [token["confidence"] for token in ordered],
            weights=[max(1, len(token["text"])) for token in ordered],
        )
    )
    return text, confidence, ordered


def _order_two_line_tokens(tokens: list[dict[str, Any]], median_height: float) -> list[dict[str, Any]]:
    """Phân tách các token thành 2 hàng (trên/dưới) và sắp xếp từng hàng từ trái sang phải."""
    sorted_by_y = sorted(tokens, key=lambda token: token["center_y"])
    gaps = [
        sorted_by_y[index + 1]["center_y"] - sorted_by_y[index]["center_y"]
        for index in range(len(sorted_by_y) - 1)
    ]
    largest_gap = max(gaps, default=0.0)
    if largest_gap >= 0.25 * median_height:
        split_index = int(np.argmax(gaps)) + 1
        rows = [sorted_by_y[:split_index], sorted_by_y[split_index:]]
    else:
        median_y = float(np.median([token["center_y"] for token in tokens]))
        rows = [
            [token for token in tokens if token["center_y"] <= median_y],
            [token for token in tokens if token["center_y"] > median_y],
        ]
    non_empty_rows = [row for row in rows if row]
    non_empty_rows.sort(key=lambda row: np.mean([token["center_y"] for token in row]))
    return [token for row in non_empty_rows for token in sorted(row, key=lambda token: token["center_x"])]


def preprocess_plate_variants(crop_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Tạo ra 4 biến thể ảnh tiền xử lý (Gray, CLAHE, Otsu, Adaptive Threshold) để tăng khả năng đọc OCR."""
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return {}
    height, width = crop_bgr.shape[:2]
    scale = float(np.clip(180 / max(height, width), 2.0, 4.0))
    enlarged = cv2.resize(crop_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    denoised = cv2.bilateralFilter(gray, 9, 55, 55)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
    return {"gray": gray, "clahe": clahe, "otsu": otsu, "adaptive": adaptive}


def infer_plate_layout(
    crop_bgr: np.ndarray,
    wide_ratio_threshold: float = 2.20,
    tokens: list[dict[str, Any]] | None = None,
) -> str:
    """Ước lượng tự động bố cục biển số dựa trên aspect ratio và phân bố tọa độ Y của tokens (nếu có)."""
    if wide_ratio_threshold <= 0:
        raise ValueError("wide_ratio_threshold phải lớn hơn 0")
    if crop_bgr is None or crop_bgr.size == 0:
        return "1_line"
    height, width = crop_bgr.shape[:2]
    initial_layout = "1_line" if width / max(1, height) >= wide_ratio_threshold else "2_line"

    # Tinh chỉnh dựa trên cụm tọa độ Y-center của token nếu có
    if tokens and len(tokens) >= 2:
        sorted_by_y = sorted(tokens, key=lambda t: t["center_y"])
        y_gaps = [
            sorted_by_y[i + 1]["center_y"] - sorted_by_y[i]["center_y"] for i in range(len(sorted_by_y) - 1)
        ]
        median_h = float(np.median([t["height"] for t in tokens]))
        if max(y_gaps, default=0.0) >= 0.25 * median_h:
            return "2_line"

    return initial_layout


def _variant_images(crop_bgr: np.ndarray, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Tạo đúng các biến thể được yêu cầu, tránh chạy thừa EasyOCR."""
    if "crop" in names:
        return {"crop": crop_bgr}
    all_variants = preprocess_plate_variants(crop_bgr)
    return {name: all_variants[name] for name in names if name in all_variants}


def _run_ocr_variant(
    reader: Any,
    image: np.ndarray,
    variant_name: str,
    config: RecognitionConfig,
    rectified: bool,
    layout_hint: str = "auto",
) -> dict[str, Any]:
    """Đọc một biến thể và thực hiện layout hai lượt dựa trên hình học token."""
    initial_layout = (
        layout_hint
        if layout_hint in VALID_LAYOUTS
        else infer_plate_layout(image, config.wide_ratio_threshold)
    )
    results = reader.readtext(image, detail=1, paragraph=False, allowlist=OCR_ALLOWLIST)

    # Lượt 1 dùng aspect ratio làm gợi ý; lượt 2 dùng Y-clustering của token.
    first_text, first_confidence, first_tokens = order_ocr_tokens(
        results,
        initial_layout,
        minimum_confidence=config.ocr_minimum_confidence,
    )
    final_layout = infer_plate_layout(
        image,
        config.wide_ratio_threshold,
        tokens=first_tokens,
    )
    if final_layout != initial_layout:
        raw_text, confidence, tokens = order_ocr_tokens(
            results,
            final_layout,
            minimum_confidence=config.ocr_minimum_confidence,
        )
    else:
        raw_text, confidence, tokens = first_text, first_confidence, first_tokens

    return {
        "raw_text": raw_text,
        "ocr_confidence": confidence,
        "layout": final_layout,
        "variant": variant_name,
        "rectified": rectified,
        "tokens": tokens,
    }


def _select_by_consensus(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    """Chọn raw text theo số phiếu, sau đó mới dùng OCR confidence để phá hòa."""
    non_empty = [candidate for candidate in candidates if candidate["raw_text"]]
    if not non_empty:
        return max(
            candidates,
            key=lambda item: item["ocr_confidence"],
            default={"raw_text": "", "ocr_confidence": 0.0, "layout": "1_line"},
        ), 0.0

    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in non_empty:
        groups.setdefault(candidate["raw_text"], []).append(candidate)
    chosen_text, chosen_group = max(
        groups.items(),
        key=lambda item: (
            len(item[1]),
            sum(candidate["ocr_confidence"] for candidate in item[1]) / len(item[1]),
            max(candidate["ocr_confidence"] for candidate in item[1]),
        ),
    )
    selected = max(chosen_group, key=lambda item: item["ocr_confidence"])
    # Mẫu không đọc được vẫn là một phiếu thất bại, nên mẫu số là tổng số pass.
    return selected, len(groups[chosen_text]) / len(candidates)


def _has_enough_evidence(candidates: list[dict[str, Any]], config: RecognitionConfig) -> bool:
    """Kiểm tra fast path đã đủ evidence để không cần fallback tốn thời gian."""
    if not candidates:
        return False
    selected, consensus = _select_by_consensus(candidates)
    return bool(
        selected.get("raw_text")
        and selected["ocr_confidence"] >= config.ocr_threshold
        and consensus >= config.ocr_consensus_threshold
    )


def validate_and_correct_plate(
    raw_text: str,
    enable_correction: bool = True,
    max_cost: float = 1.0,
) -> dict[str, Any]:
    """Alias tương thích tới grammar v1; sửa lỗi chỉ là đề xuất."""
    return validate_plate_format(raw_text, enable_correction=enable_correction, max_cost=max_cost)


def evaluate_plate_reliability(
    detection_confidence: float,
    ocr_confidence: float,
    ocr_consensus_ratio: float,
    format_valid: bool,
    correction_cost: float,
    config: RecognitionConfig,
) -> tuple[float, list[str], bool]:
    """API tương thích cũ; policy canonical không dùng điểm này làm gate."""
    policy = decide_plate(
        # API cũ không truyền raw text; giả định candidate đã có text để
        # phân biệt FORMAT_INVALID (REVIEW) với UNREADABLE (REJECT).
        raw_text="PLATE",
        detection_confidence=detection_confidence,
        ocr_confidence=ocr_confidence,
        consensus_ratio=ocr_consensus_ratio,
        format_valid=format_valid,
        correction_suggestion="suggestion" if correction_cost > 0 else None,
        config=config,
    )
    reasons = list(policy["review_reasons"])
    if "FORMAT_INVALID" in reasons and "INVALID_FORMAT" not in reasons:
        # Alias để artifact/test cũ vẫn đọc được; read_plate canonical chỉ trả tên mới.
        reasons.append("INVALID_FORMAT")
    if correction_cost > config.max_correction_cost and "HIGH_CORRECTION_COST" not in reasons:
        reasons.append("HIGH_CORRECTION_COST")
    return policy["reliability_score"], reasons, bool(reasons)


def read_plate(
    reader: Any,
    crop_bgr: np.ndarray,
    layout: str = "auto",
    config: RecognitionConfig | None = None,
    detection_confidence: float = 1.0,
) -> dict[str, Any]:
    """OCR cascade thích ứng: fast path trước, rectification chỉ khi evidence chưa đủ."""
    if layout != "auto" and layout not in VALID_LAYOUTS:
        raise ValueError(f"Bố cục không hợp lệ: {layout}")
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Crop biển số bị rỗng.")
    cfg = config or RecognitionConfig()

    if cfg.single_variant_mode:
        variant_names = (cfg.single_variant_mode,)
        candidates = [
            _run_ocr_variant(reader, image, name, cfg, False, layout)
            for name, image in _variant_images(crop_bgr, variant_names).items()
        ]
    elif not cfg.enable_preprocessing_variants:
        candidates = [_run_ocr_variant(reader, crop_bgr, "crop", cfg, False, layout)]
    else:
        # Fast path đúng theo policy: chỉ Gray + CLAHE.
        fast = _variant_images(crop_bgr, ("gray", "clahe"))
        candidates = [
            _run_ocr_variant(reader, image, name, cfg, False, layout) for name, image in fast.items()
        ]

        if not _has_enough_evidence(candidates, cfg):
            target_crop, rectified = crop_bgr, False
            if cfg.enable_rectification:
                target_crop, rectified = rectify_plate(crop_bgr)
            fallback = _variant_images(target_crop, ("gray", "otsu", "adaptive"))
            candidates.extend(
                _run_ocr_variant(reader, image, name, cfg, rectified, layout)
                for name, image in fallback.items()
                if name != "gray" or rectified
            )

    selected, consensus = _select_by_consensus(candidates)
    grammar = validate_and_correct_plate(
        selected.get("raw_text", ""),
        enable_correction=cfg.enable_template_correction,
        max_cost=cfg.max_correction_cost,
    )
    policy = decide_plate(
        raw_text=grammar["raw_text"],
        detection_confidence=detection_confidence,
        ocr_confidence=float(selected.get("ocr_confidence", 0.0)),
        consensus_ratio=consensus,
        format_valid=grammar["format_valid"],
        correction_suggestion=grammar["correction_suggestion"],
        config=cfg,
    )
    decision = policy["decision"]
    return {
        **grammar,
        "accepted_text": grammar["raw_text"] if decision == "ACCEPT" else None,
        "ocr_confidence": float(selected.get("ocr_confidence", 0.0)),
        "ocr_consensus_ratio": float(consensus),
        "consensus": float(consensus),
        "reliability_score": policy["reliability_score"],
        "evidence": policy["evidence"],
        "policy_version": policy["policy_version"],
        "decision": decision,
        "needs_manual_review": policy["needs_manual_review"],
        "review_reasons": policy["review_reasons"],
        "layout": selected.get("layout", layout if layout != "auto" else "1_line"),
        "variant": selected.get("variant"),
        "score": float(selected.get("ocr_confidence", 0.0)),
        "rectified": bool(selected.get("rectified", False)),
        "tokens": selected.get("tokens", []),
    }
