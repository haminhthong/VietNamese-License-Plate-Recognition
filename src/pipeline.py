"""Pipeline điều phối quy trình nhận diện biển số xe end-to-end:

Input image
   ↓
YOLOv8 plate detection
   ↓
Crop + padding
   ↓
OCR preprocessing (CLAHE / fallback)
   ↓
EasyOCR
   ↓
Token geometry ordering (1-line / 2-line)
   ↓
Vietnamese plate syntax validation
   ↓
Final plate result + optional correction suggestion
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


def crop_with_padding(
    image: np.ndarray,
    box: Box,
    padding_ratio: float = 0.05,
) -> tuple[np.ndarray, Box]:
    """Cắt vùng ảnh chứa biển số có mở rộng lề đệm (padding) để tránh mất mép ký tự."""
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


class LicensePlateRecognizer:
    """Điều phối toàn bộ quy trình nhận diện biển số từ ảnh xe đầu vào."""

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

    def predict(
        self,
        image_bgr: np.ndarray,
        confidence: float | None = None,
    ) -> list[dict[str, Any]]:
        """Nhận diện toàn bộ biển số xe xuất hiện trong ảnh BGR."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Ảnh đầu vào bị rỗng.")

        height, width = image_bgr.shape[:2]
        if width < self.config.min_image_width or height < self.config.min_image_height:
            raise ValueError(
                f"Kích thước ảnh quá nhỏ ({width}x{height}), tối thiểu cần "
                f"{self.config.min_image_width}x{self.config.min_image_height}."
            )

        det_conf = self.config.detection_confidence if confidence is None else confidence
        if not 0 <= det_conf <= 1:
            raise ValueError("Tham số 'confidence' phải nằm trong khoảng [0, 1].")

        started_at = perf_counter()

        # 1. Chạy YOLOv8 Detector
        result = self.detector.predict(
            source=image_bgr,
            conf=det_conf,
            iou=self.config.nms_iou,
            imgsz=self.config.image_size,
            verbose=False,
        )[0]

        predictions: list[dict[str, Any]] = []
        boxes = [] if result.boxes is None else result.boxes

        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            detected_box: Box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            box_conf = float(box.conf.item())

            # 2. Crop với padding
            crop, _ = crop_with_padding(image_bgr, detected_box, self.config.padding_ratio)
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            if crop_w < self.config.min_plate_width or crop_h < self.config.min_plate_height:
                continue

            # 3. Tiền xử lý + EasyOCR + Syntax validation
            ocr_res = read_plate(
                self.reader,
                crop,
                config=self.config,
                detection_confidence=box_conf,
            )

            predictions.append(
                {
                    "box": list(detected_box),
                    "detection_confidence": round(box_conf, 4),
                    "text": ocr_res["text"],
                    "raw_text": ocr_res["raw_text"],
                    "ocr_confidence": round(ocr_res["ocr_confidence"], 4),
                    "format_valid": ocr_res["format_valid"],
                    "suggested_text": ocr_res["suggested_text"],
                    "needs_review": ocr_res["needs_review"],
                    "layout": ocr_res["layout"],
                }
            )

        self.last_latency_ms = (perf_counter() - started_at) * 1_000
        return predictions

    def predict_file(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Nhận diện biển số từ tệp ảnh và lưu ảnh trực quan (nếu có yêu cầu)."""
        image = read_image(image_path, "tệp ảnh từ đĩa")
        predictions = self.predict(image)
        if output_path:
            annotated = draw_predictions(image, predictions)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), annotated):
                raise OSError(f"Không thể ghi ảnh kết quả ra: {output_path}")
        return predictions


def draw_predictions(image: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    """Vẽ bounding boxes và biển số nhận diện được lên ảnh."""
    annotated = image.copy()
    for pred in predictions:
        x1, y1, x2, y2 = pred["box"]
        color = (0, 255, 0) if not pred.get("needs_review") else (0, 165, 255)  # Green vs Orange
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = pred.get("text") or "[Không đọc được]"
        label = f"{text} ({pred['detection_confidence']:.2f})"
        cv2.putText(
            annotated,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return annotated
