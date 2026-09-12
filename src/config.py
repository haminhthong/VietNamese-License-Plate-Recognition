"""Quản lý và kiểm tra cấu hình huấn luyện YOLOv8 và nhận diện biển số xe."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    """Cấu hình huấn luyện mô hình phát hiện biển số YOLOv8."""

    model: str = "yolov8n.pt"
    epochs: int = 60
    batch: int = 16
    image_size: int = 640
    patience: int = 15
    seed: int = 42
    workers: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Tham số 'model' phải là chuỗi không rỗng.")

        for field_name in ("epochs", "batch", "image_size"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Tham số '{field_name}' phải là số nguyên dương (> 0).")

        if not isinstance(self.patience, int) or isinstance(self.patience, bool) or self.patience < 0:
            raise ValueError("Tham số 'patience' phải là số nguyên >= 0.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise TypeError("Tham số 'seed' phải là số nguyên (int).")
        if not isinstance(self.workers, int) or isinstance(self.workers, bool) or self.workers < 0:
            raise ValueError("Tham số 'workers' phải là số nguyên >= 0.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình YAML: {config_path}")

        payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Tệp cấu hình huấn luyện phải có định dạng YAML dictionary.")

        aliases = {"imgsz": "image_size"}
        normalized = {aliases.get(key, key): value for key, value in payload.items()}
        allowed = set(cls.__dataclass_fields__)
        if unknown := set(normalized) - allowed:
            raise ValueError(f"Tệp cấu hình chứa khóa không hợp lệ: {sorted(unknown)}")
        return cls(**normalized)

    def override(
        self,
        *,
        model: str | None = None,
        epochs: int | None = None,
        batch: int | None = None,
    ) -> TrainingConfig:
        return replace(
            self,
            model=self.model if model is None else model,
            epochs=self.epochs if epochs is None else epochs,
            batch=self.batch if batch is None else batch,
        )


@dataclass(frozen=True)
class RecognitionConfig:
    """Cấu hình cho detector, tiền xử lý và OCR nhận diện biển số."""

    detection_confidence: float = 0.25
    auto_accept_detector_threshold: float = 0.50
    nms_iou: float = 0.60
    image_size: int = 640
    padding_ratio: float = 0.05
    ocr_minimum_confidence: float = 0.20
    ocr_threshold: float = 0.50
    wide_ratio_threshold: float = 2.20

    enable_rectification: bool = True
    enable_preprocessing_variants: bool = True
    enable_template_correction: bool = True
    single_variant_mode: str | None = None

    min_plate_width: int = 20
    min_plate_height: int = 8
    min_image_width: int = 160
    min_image_height: int = 120

    @property
    def detector_candidate_threshold(self) -> float:
        return self.detection_confidence

    def __post_init__(self) -> None:
        for field_name in (
            "image_size",
            "min_plate_width",
            "min_plate_height",
            "min_image_width",
            "min_image_height",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Tham số '{field_name}' phải là số nguyên dương (> 0).")

        for field_name in (
            "detection_confidence",
            "auto_accept_detector_threshold",
            "nms_iou",
            "ocr_minimum_confidence",
            "ocr_threshold",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or isinstance(value, bool) or not (0 <= value <= 1):
                raise ValueError(f"Tham số '{field_name}' phải là số trong khoảng [0, 1].")

        if self.padding_ratio < 0:
            raise ValueError("Tham số 'padding_ratio' không được nhỏ hơn 0.")
        if self.wide_ratio_threshold <= 0:
            raise ValueError("Tham số 'wide_ratio_threshold' phải lớn hơn 0.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> RecognitionConfig:
        config_path = Path(path)
        if not config_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp cấu hình YAML: {config_path}")

        payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Tệp cấu hình nhận diện phải có định dạng YAML dictionary.")

        # Hỗ trợ alias detector_candidate_threshold -> detection_confidence
        normalized = dict(payload)
        if "detector_candidate_threshold" in normalized and "detection_confidence" not in normalized:
            normalized["detection_confidence"] = normalized.pop("detector_candidate_threshold")

        allowed = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in normalized.items() if k in allowed}
        return cls(**filtered)

    def override(self, **kwargs: Any) -> RecognitionConfig:
        if "detector_candidate_threshold" in kwargs and "detection_confidence" not in kwargs:
            kwargs["detection_confidence"] = kwargs.pop("detector_candidate_threshold")
        allowed = set(self.__dataclass_fields__)
        filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        return replace(self, **filtered)
