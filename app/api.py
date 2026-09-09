"""FastAPI dịch vụ API nhận diện biển số xe và giao diện Web UI trực quan.

Module này cung cấp các REST API endpoints:
- `GET /`: Trang chủ Web UI Dashboard trực quan.
- `GET /health`: Kiểm tra trạng thái sức khỏe ứng dụng và trọng số mô hình.
- `POST /predict`: Nhận tệp ảnh tải lên và trả về kết quả định vị & OCR biển số.
"""

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from src.config import RecognitionConfig
from src.pipeline import LicensePlateRecognizer

from .schemas import HealthResponse, LivenessResponse, PredictionResponse, ReadinessResponse

app = FastAPI(
    title="Vietnamese License Plate Recognition API",
    description="REST API và Web Dashboard giao diện cho hệ thống nhận diện biển số xe Việt Nam.",
    version="1.0.0",
)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UI_HTML_PATH = Path(__file__).parent / "ui.html"
MAX_CONCURRENT_INFERENCE = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCE", "1")))
INFERENCE_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)


def model_weights_path() -> Path:
    """Lấy đường dẫn tệp trọng số mô hình từ biến môi trường MODEL_WEIGHTS hoặc mặc định."""
    return Path(os.getenv("MODEL_WEIGHTS", "models/best.pt"))


@lru_cache(maxsize=1)
def get_recognizer() -> LicensePlateRecognizer:
    """Khởi tạo và lưu cache bộ đối tượng LicensePlateRecognizer."""
    weights = model_weights_path()
    if not weights.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp trọng số mô hình YOLOv8: {weights}")
    config_path = Path(os.getenv("RECOGNITION_CONFIG", "configs/recognition.yaml"))
    config = RecognitionConfig.from_yaml(config_path) if config_path.is_file() else RecognitionConfig()
    return LicensePlateRecognizer(weights, config=config)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> FileResponse:
    """Phục vụ trang giao diện Web UI trực quan cho trình duyệt."""
    if not UI_HTML_PATH.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy tệp giao diện Web UI (app/ui.html).")
    return FileResponse(UI_HTML_PATH)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Kiểm tra trạng thái sức khỏe dịch vụ và khả năng kết nối mô hình trọng số."""
    weights = model_weights_path()
    return HealthResponse(
        status="ok" if weights.is_file() else "model_missing",
        model_weights=str(weights),
        model_available=weights.is_file(),
    )


@app.get("/health/live", response_model=LivenessResponse)
def health_live() -> LivenessResponse:
    """Kiểm tra tiến trình dịch vụ API còn sống (Liveness Probe)."""
    return LivenessResponse(status="live")


@app.get("/health/ready", response_model=ReadinessResponse)
def health_ready() -> ReadinessResponse:
    """Kiểm tra tệp trọng số tồn tại để dịch vụ có thể khởi tạo model khi cần."""
    weights = model_weights_path()
    available = weights.is_file()
    if not available:
        raise HTTPException(status_code=503, detail="Dịch vụ chưa sẵn sàng do thiếu trọng số mô hình.")
    return ReadinessResponse(status="ready", model_available=True)


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: Annotated[UploadFile, File()]) -> PredictionResponse:
    """Nhận tệp ảnh tải lên (JPEG/PNG/WebP max 10MB) và thực hiện nhận diện biển số end-to-end."""
    if image.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ các định dạng ảnh JPEG, PNG hoặc WebP.")
    payload = await image.read(MAX_UPLOAD_BYTES + 1)
    if not payload:
        raise HTTPException(status_code=400, detail="Tệp ảnh tải lên bị rỗng.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Dung lượng ảnh vượt quá giới hạn tối đa 10 MB.")
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Dữ liệu tệp không phải là hình ảnh hợp lệ.")
    try:
        recognizer = get_recognizer()
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    try:
        async with INFERENCE_SEMAPHORE:
            raw_preds = await asyncio.to_thread(recognizer.predict, decoded)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail="Xảy ra lỗi trong quá trình thực thi suy luận mô hình."
        ) from error

    formatted_preds = []
    for item in raw_preds:
        recognition = {
            "raw_text": item.get("raw_text", ""),
            "normalized_text": item.get("normalized_text", item.get("raw_text", "")),
            "text": item.get("normalized_text", item.get("raw_text", "")),
            "format_valid": item.get("format_valid", False),
            "template": item.get("template"),
            "correction_cost": item.get("correction_cost", 0.0),
            "correction_suggestion": item.get("correction_suggestion"),
            "correction_applied": item.get("correction_applied", False),
            "plate_pattern_version": item.get("plate_pattern_version", "civilian-v1"),
        }
        scores = {
            "detector_confidence": item.get("detection_confidence", 0.0),
            "ocr_confidence": item.get("ocr_confidence", 0.0),
            "ocr_consensus_ratio": item.get("ocr_consensus_ratio", 1.0),
            "reliability_score": item.get("reliability_score", 1.0),
        }
        review = {
            "required": item.get("needs_manual_review", False),
            "reasons": item.get("review_reasons", []),
        }
        latencies = {
            "image_pipeline_latency_ms": item.get(
                "image_pipeline_latency_ms", item.get("pipeline_latency_ms", 0.0)
            ),
            "plate_ocr_latency_ms": item.get("plate_ocr_latency_ms", 0.0),
            "detector_latency_ms": item.get("detector_latency_ms", 0.0),
        }
        formatted_preds.append(
            {
                **item,
                "recognition": recognition,
                "scores": scores,
                "review": review,
                "latencies": latencies,
            }
        )

    return PredictionResponse(
        model_version=os.getenv("MODEL_VERSION", "unknown"),
        filename=image.filename,
        latency_ms=recognizer.last_latency_ms,
        predictions=formatted_preds,
    )
