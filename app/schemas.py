"""Pydantic schemas cho FastAPI REST API của hệ thống nhận diện biển số xe."""

from pydantic import BaseModel, Field


class PlatePrediction(BaseModel):
    """Thông tin kết quả nhận diện cho một biển số xe."""

    box: list[int] = Field(description="Tọa độ Bounding Box dạng [x1, y1, x2, y2]")
    detection_confidence: float = Field(ge=0, le=1, description="Độ tin cậy phát hiện của YOLOv8")
    text: str = Field(description="Chuỗi ký tự biển số nhận dạng được")
    ocr_confidence: float = Field(ge=0, le=1, description="Độ tin cậy nhận dạng trung bình của OCR")
    format_valid: bool = Field(description="Chuỗi ký tự có khớp cú pháp biển số Việt Nam hay không")
    suggested_text: str | None = Field(default=None, description="Gợi ý sửa ký tự nhầm lẫn (nếu có)")
    needs_review: bool = Field(default=False, description="Cờ đánh dấu cần kiểm tra lại thủ công")
    layout: str = Field(default="1_line", description="Bố cục biển số ('1_line' hoặc '2_line')")


class PredictionResponse(BaseModel):
    """Kết quả phản hồi nhận diện ảnh end-to-end."""

    filename: str | None = Field(default=None, description="Tên tệp ảnh đã tải lên")
    latency_ms: float = Field(ge=0, description="Tổng thời gian xử lý ảnh (ms)")
    predictions: list[PlatePrediction] = Field(description="Danh sách các biển số nhận dạng được")


class HealthResponse(BaseModel):
    """Trạng thái sức khỏe và cấu hình mô hình của dịch vụ."""

    status: str = Field(description="Trạng thái dịch vụ ('ok' hoặc 'model_missing')")
    model_weights: str = Field(description="Đường dẫn tệp trọng số mô hình")
    model_available: bool = Field(description="Cờ báo tệp trọng số mô hình có sẵn sàng hay không")
