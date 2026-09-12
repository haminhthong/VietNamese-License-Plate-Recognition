"""FastAPI REST API cho hệ thống nhận diện biển số xe và giao diện Web UI."""

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

from .schemas import HealthResponse, PredictionResponse

app = FastAPI(
    title="Vietnamese License Plate Recognition API",
    description="REST API và Web Dashboard cho hệ thống nhận diện biển số xe Việt Nam.",
    version="1.0.0",
)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
UI_HTML_PATH = Path(__file__).parent / "ui.html"


def model_weights_path() -> Path:
    """Đường dẫn tệp trọng số mô hình từ biến môi trường MODEL_WEIGHTS hoặc mặc định."""
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
    """Kiểm tra trạng thái hoạt động của dịch vụ và sự tồn tại của model weights."""
    weights = model_weights_path()
    return HealthResponse(
        status="ok" if weights.is_file() else "model_missing",
        model_weights=str(weights),
        model_available=weights.is_file(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(image: Annotated[UploadFile, File()]) -> PredictionResponse:
    """Nhận diện biển số xe từ tệp ảnh tải lên (JPEG, PNG, WebP tối đa 10MB)."""
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
        predictions = await asyncio.to_thread(recognizer.predict, decoded)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500, detail="Xảy ra lỗi trong quá trình thực thi suy luận mô hình."
        ) from error

    return PredictionResponse(
        filename=image.filename,
        latency_ms=round(recognizer.last_latency_ms, 2),
        predictions=predictions,
    )
