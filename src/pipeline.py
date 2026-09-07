"""Pipeline điều phối quy trình nhận diện biển số xe hoàn chỉnh (YOLOv8 + EasyOCR + Post-processing).

Module này kết nối các thành phần phát hiện đối tượng, cắt vùng ảnh có đệm, nắn góc phối cảnh,
nhận dạng ký tự và ghi đè kết quả lên ảnh minh họa.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .config import RecognitionConfig
from .io_utils import read_image
from .ocr import read_plate

Box = tuple[int, int, int, int]


def _rejected_plate_result(
    reason: str,
    detector_confidence: float = 0.0,
    rectified: bool = False,
) -> dict[str, Any]:
    """Tạo kết quả chuẩn cho candidate bị loại trước khi chạy OCR."""
    return {
        "raw_text": "",
        "normalized_text": "",
        "text": "",
        "accepted_text": None,
        "format_valid": False,
        "template": None,
        "correction_cost": 0.0,
        "correction_suggestion": None,
        "correction_applied": False,
        "ocr_confidence": 0.0,
        "ocr_consensus_ratio": 0.0,
        "consensus": 0.0,
        "reliability_score": 0.0,
        "evidence": {
            "detector_score": float(detector_confidence),
            "ocr_score": 0.0,
            "variant_consensus": 0.0,
        },
        "decision": "REJECT",
        "policy_version": "1.0.0",
        "plate_pattern_version": "civilian-v1",
        "needs_manual_review": False,
        "review_reasons": [reason],
        "layout": "1_line",
        "variant": None,
        "score": 0.0,
        "rectified": rectified,
        "tokens": [],
    }


def crop_with_padding(image: np.ndarray, box: Box, padding_ratio: float = 0.05) -> tuple[np.ndarray, Box]:
    """Cắt vùng ảnh chứa biển số từ Bounding Box và mở rộng lề đệm nhưng đảm bảo không vượt quá biên ảnh.

    Args:
        image (np.ndarray): Ảnh BGR gốc.
        box (Box): Tọa độ (x1, y1, x2, y2).
        padding_ratio (float): Tỷ lệ đệm (mặc định: 0.05).

    Returns:
        tuple[np.ndarray, Box]: Cặp (ảnh_crop_đã_đệm, tọa_độ_box_đã_đệm).

    Raises:
        ValueError: Nếu ảnh rỗng, padding_ratio âm hoặc tọa độ box không hợp lệ.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh đầu vào bị rỗng.")
    if padding_ratio < 0:
        raise ValueError("Tham số 'padding_ratio' không được nhỏ hơn 0.")
    x1, y1, x2, y2 = map(int, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Tọa độ Bounding box không hợp lệ: {box}")
    height, width = image.shape[:2]
    pad_x, pad_y = int((x2 - x1) * padding_ratio), int((y2 - y1) * padding_ratio)
    padded: Box = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(width, x2 + pad_x),
        min(height, y2 + pad_y),
    )
    px1, py1, px2, py2 = padded
    return image[py1:py2, px1:px2], padded


def validate_image_quality(image_bgr: np.ndarray, config: RecognitionConfig) -> dict[str, Any]:
    """Đánh giá nhanh độ phân giải, độ nét và phơi sáng trước khi chạy detector."""
    if image_bgr is None or image_bgr.size == 0:
        return {"valid": False, "reasons": ["IMAGE_EMPTY"]}
    height, width = image_bgr.shape[:2]
    reasons: list[str] = []
    if width < config.min_image_width or height < config.min_image_height:
        reasons.append("IMAGE_TOO_SMALL")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur_score < config.blur_threshold:
        reasons.append("IMAGE_TOO_BLURRY")

    dark_ratio = float(np.mean(gray <= 8))
    bright_ratio = float(np.mean(gray >= 247))
    if dark_ratio >= 0.98:
        reasons.append("IMAGE_UNDEREXPOSED")
    if bright_ratio >= 0.98:
        reasons.append("IMAGE_OVEREXPOSED")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "width": width,
        "height": height,
        "blur_score": blur_score,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
    }


class LicensePlateRecognizer:
    """Lớp điều phối chính thực thi pipeline end-to-end từ ảnh đầu vào đến kết quả biển số.

    Args:
        weights (str | Path): Đường dẫn đến tệp trọng số YOLOv8 (.pt).
        gpu (bool | None): Ép buộc sử dụng GPU (True), CPU (False) hoặc tự động phát hiện (None).
        config (RecognitionConfig | None): Cấu hình tùy chọn cho pipeline.
    """

    def __init__(
        self,
        weights: str | Path,
        gpu: bool | None = None,
        config: RecognitionConfig | None = None,
    ) -> None:
        import easyocr
        import torch
        from ultralytics import YOLO

        use_gpu = torch.cuda.is_available() if gpu is None else gpu
        self.config = config or RecognitionConfig()
        self.detector = YOLO(str(weights))
        self.reader = easyocr.Reader(["en"], gpu=use_gpu)
        self.last_latency_ms = 0.0

    def predict(self, image_bgr: np.ndarray, confidence: float | None = None) -> list[dict[str, Any]]:
        """Dự đoán và nhận diện toàn bộ biển số xe xuất hiện trong ảnh BGR.

        Args:
            image_bgr (np.ndarray): Mảng ảnh BGR (OpenCV format).
            confidence (float | None): Độ tin cậy đè tùy chọn.

        Returns:
            list[dict[str, Any]]: Danh sách kết quả nhận diện từng biển số.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào bị rỗng.")
        quality = validate_image_quality(image_bgr, self.config)
        if not quality["valid"]:
            self.last_latency_ms = 0.0
            raise ValueError(f"Ảnh không đạt quality gate: {', '.join(quality['reasons'])}")
        detection_confidence = self.config.detector_candidate_threshold if confidence is None else confidence
        if not 0 <= detection_confidence <= 1:
            raise ValueError("Tham số 'confidence' phải nằm trong khoảng [0, 1].")

        started_at = perf_counter()
        det_start = perf_counter()
        result = self.detector.predict(
            source=image_bgr,
            conf=detection_confidence,
            iou=self.config.nms_iou,
            imgsz=self.config.image_size,
            verbose=False,
        )[0]
        detector_latency_ms = (perf_counter() - det_start) * 1_000

        predictions: list[dict[str, Any]] = []
        boxes = [] if result.boxes is None else result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            detected_box: Box = tuple(map(int, xyxy))
            crop, padded_box = crop_with_padding(image_bgr, detected_box, self.config.padding_ratio)
            if crop.size == 0:
                continue
            class_id = int(box.cls.item())
            det_conf = float(box.conf.item())

            # Crop quá nhỏ không đủ thông tin cho OCR; trả REJECT có lý do rõ ràng.
            crop_height, crop_width = crop.shape[:2]
            if crop_width < self.config.min_plate_width or crop_height < self.config.min_plate_height:
                ocr_res = _rejected_plate_result(
                    "PLATE_TOO_SMALL",
                    detector_confidence=det_conf,
                )
                plate_ocr_latency_ms = 0.0
            else:
                plate_start = perf_counter()
                ocr_res = read_plate(
                    self.reader,
                    crop,
                    config=self.config,
                    detection_confidence=det_conf,
                )
                plate_ocr_latency_ms = (perf_counter() - plate_start) * 1_000

            predictions.append(
                {
                    "box": detected_box,
                    "padded_box": padded_box,
                    "class_id": class_id,
                    "detector_class": result.names.get(class_id, "unknown"),
                    "detection_confidence": det_conf,
                    "detector_latency_ms": detector_latency_ms,
                    "plate_ocr_latency_ms": plate_ocr_latency_ms,
                    **ocr_res,
                }
            )

        # Post-NMS deduplication: loại bỏ các bboxes trùng lặp có IoU cao và cùng kết quả text
        if len(predictions) > 1:
            from .metrics import box_iou

            filtered: list[dict[str, Any]] = []
            for p in sorted(predictions, key=lambda x: x["score"], reverse=True):
                duplicate = False
                for existing in filtered:
                    if (
                        box_iou(p["box"], existing["box"]) > 0.70
                        and p["text"] == existing["text"]
                        and p["text"] != ""
                    ):
                        duplicate = True
                        break
                if not duplicate:
                    filtered.append(p)
            predictions = filtered

        elapsed_ms = (perf_counter() - started_at) * 1_000
        self.last_latency_ms = elapsed_ms
        for prediction in predictions:
            prediction["image_pipeline_latency_ms"] = elapsed_ms
            prediction["pipeline_latency_ms"] = elapsed_ms  # Alias tương thích ngược
        return predictions

    def predict_file(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Đọc ảnh từ đĩa cứng, chạy dự đoán và tùy chọn lưu ảnh vẽ kết quả.

        Args:
            image_path (str | Path): Đường dẫn tệp ảnh nguồn.
            output_path (str | Path | None): Đường dẫn xuất ảnh kết quả minh họa (nếu có).

        Returns:
            list[dict[str, Any]]: Danh sách dự đoán biển số.
        """
        image = read_image(image_path, "tệp ảnh từ đĩa")
        predictions = self.predict(image)
        if output_path:
            annotated = draw_predictions(image, predictions)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), annotated):
                raise OSError(f"Không thể ghi ảnh kết quả ra tệp: {output_path}")
        return predictions


def draw_predictions(image: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    """Vẽ Bounding Box màu xanh lục và chuỗi ký tự biển số nhận dạng lên bản sao của ảnh gốc.

    Args:
        image (np.ndarray): Ảnh BGR gốc.
        predictions (list[dict[str, Any]]): Danh sách kết quả dự đoán từ pipeline.

    Returns:
        np.ndarray: Ảnh mới đã được vẽ trực quan kết quả.
    """
    annotated = image.copy()
    for prediction in predictions:
        x1, y1, x2, y2 = prediction["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = prediction.get("accepted_text") or prediction.get("raw_text") or "[không đọc được]"
        decision = prediction.get("decision", "REVIEW")
        label = f"{text} [{decision}] {prediction['detection_confidence']:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return annotated
