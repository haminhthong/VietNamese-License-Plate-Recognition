"""Định nghĩa Pydantic Schemas đầu ra cho FastAPI REST API của hệ thống nhận diện biển số xe.

Module này cung cấp các mô hình dữ liệu (Response Models):
- Model Layer: `RecognitionData`, `ScoreData`
- Decision Layer: `ReviewData`, `LatencyData`
- `PlatePrediction`: Thông tin chi tiết cho một biển số được phát hiện.
- `PredictionResponse`: Kết quả tổng hợp trả về cho một ảnh upload.
- `HealthResponse`: Trạng thái hoạt động và tình trạng model weight của dịch vụ.
"""

from pydantic import BaseModel, Field


class RecognitionData(BaseModel):
    """Thông tin nhận dạng chuỗi ký tự và chuẩn hóa mẫu (Model Layer - Recognition)."""

    raw_text: str = Field(description="Chuỗi ký tự nhận dạng thô từ OCR")
    normalized_text: str = Field(description="Chuỗi raw_text sau chuẩn hóa ASCII, không sửa theo grammar")
    text: str = Field(description="Alias tương thích ngược của normalized_text")
    format_valid: bool = Field(description="Chuỗi raw_text có khớp mẫu được hỗ trợ hay không")
    template: str | None = Field(default=None, description="Mẫu định dạng đã khớp (ví dụ: 'DDLDDDDD')")
    correction_cost: float = Field(ge=0, description="Chi phí hiệu chỉnh ký tự")
    correction_suggestion: str | None = Field(default=None, description="Đề xuất sửa để operator kiểm tra")
    plate_pattern_version: str = Field(default="civilian-v1", description="Phiên bản grammar đang dùng")
    correction_applied: bool = Field(
        default=False, description="Luôn false ở policy v1: không tự động ghi đè raw OCR"
    )


class ScoreData(BaseModel):
    """Thống kê độ tin cậy và điểm số của mô hình (Model Layer - Scores)."""

    detector_confidence: float = Field(ge=0, le=1, description="Độ tin cậy phát hiện của YOLOv8")
    ocr_confidence: float = Field(ge=0, le=1, description="Độ tin cậy nhận dạng trung bình của OCR")
    ocr_consensus_ratio: float = Field(
        ge=0, le=1, default=1.0, description="Tỷ lệ đồng thuận giữa các biến thể ảnh OCR"
    )
    reliability_score: float = Field(
        ge=0, le=1, default=1.0, description="Điểm diagnostic, không phải xác suất và không làm gate"
    )


class ReviewData(BaseModel):
    """Chính sách kiểm duyệt thủ công (Decision Layer - Human-in-the-loop Policy)."""

    required: bool = Field(default=False, description="Cờ báo kết quả cần được kiểm duyệt thủ công")
    reasons: list[str] = Field(
        default_factory=list, description="Danh sách lý do kích hoạt kiểm duyệt thủ công"
    )


class LatencyData(BaseModel):
    """Thống kê chi tiết độ trễ từng công đoạn (Stage Latency Profiling)."""

    image_pipeline_latency_ms: float = Field(ge=0, description="Tổng thời gian xử lý toàn bộ ảnh (ms)")
    plate_ocr_latency_ms: float = Field(
        ge=0, default=0.0, description="Thời gian thực thi OCR trên riêng vùng biển số này (ms)"
    )
    detector_latency_ms: float = Field(ge=0, default=0.0, description="Thời gian thực thi YOLO Detector (ms)")


class PlatePrediction(BaseModel):
    """Thông tin kết quả dự đoán hoàn chỉnh của một biển số xe được phát hiện."""

    box: tuple[int, int, int, int] = Field(description="Tọa độ Bounding Box dạng (x1, y1, x2, y2)")
    padded_box: tuple[int, int, int, int] = Field(description="Tọa độ Bounding Box sau khi mở rộng đệm")
    class_id: int = Field(description="Mã lớp định danh đối tượng (0: license_plate)")
    detector_class: str = Field(description="Tên lớp định danh đối tượng")

    # Phân tầng Model Layer & Decision Layer
    recognition: RecognitionData | None = Field(default=None, description="Tầng dữ liệu nhận dạng ký tự")
    scores: ScoreData | None = Field(default=None, description="Tầng dữ liệu điểm số & độ tin cậy")
    review: ReviewData | None = Field(default=None, description="Tầng dữ liệu chính sách kiểm duyệt thủ công")
    latencies: LatencyData | None = Field(default=None, description="Tầng dữ liệu độ trễ công đoạn")

    # Thuộc tính phẳng giữ tính tương thích ngược (Backward Compatibility)
    detection_confidence: float = Field(ge=0, le=1, description="Độ tin cậy phát hiện của YOLOv8")
    raw_text: str = Field(description="Chuỗi ký tự nhận dạng thô từ OCR")
    normalized_text: str = Field(description="Chuỗi raw_text sau chuẩn hóa ASCII")
    text: str = Field(description="Alias tương thích ngược của normalized_text")
    accepted_text: str | None = Field(
        default=None, description="Chuỗi chỉ được chấp nhận khi decision=ACCEPT"
    )
    format_valid: bool = Field(description="Chuỗi raw_text có khớp mẫu được hỗ trợ hay không")
    template: str | None = Field(default=None, description="Mẫu định dạng đã khớp (ví dụ: 'DDLDDDDD')")
    correction_cost: float = Field(ge=0, description="Chi phí sửa lỗi ký tự")
    correction_suggestion: str | None = Field(default=None, description="Đề xuất sửa, không tự động áp dụng")
    correction_applied: bool = Field(
        default=False, description="Cờ báo kết quả đã qua tự động hiệu chỉnh ký tự"
    )
    needs_manual_review: bool = Field(default=False, description="Cờ báo kết quả cần được kiểm tra thủ công")
    ocr_confidence: float = Field(ge=0, le=1, description="Độ tin cậy nhận dạng trung bình của EasyOCR")
    ocr_consensus_ratio: float = Field(
        ge=0, le=1, default=1.0, description="Tỷ lệ đồng thuận giữa các biến thể OCR"
    )
    reliability_score: float = Field(
        ge=0, le=1, default=1.0, description="Điểm tin cậy tổng hợp toàn hệ thống"
    )
    review_reasons: list[str] = Field(default_factory=list, description="Các lý do cần kiểm duyệt thủ công")
    decision: str = Field(description="Quyết định vận hành: ACCEPT, REVIEW hoặc REJECT")
    policy_version: str = Field(default="1.0.0", description="Phiên bản decision policy")
    layout: str = Field(description="Bố cục biển số suy luận ('1_line' hoặc '2_line')")
    rectified: bool = Field(description="Cờ báo ảnh crop đã được nắn góc phối cảnh thành công hay chưa")
    variant: str | None = Field(default=None, description="Tên biến thể tiền xử lý ảnh đạt điểm cao nhất")
    score: float = Field(description="Điểm số tổng hợp chọn kết quả tối ưu")
    pipeline_latency_ms: float = Field(ge=0, description="Thời gian thực thi pipeline trên ảnh (ms)")


class PredictionResponse(BaseModel):
    """Phản hồi JSON đầy đủ cho yêu cầu nhận diện ảnh."""

    model_version: str = Field(default="unknown", description="Phiên bản detector + OCR policy đang chạy")
    filename: str | None = Field(default=None, description="Tên tệp ảnh đã tải lên")
    latency_ms: float = Field(ge=0, description="Tổng thời gian xử lý ảnh (ms)")
    predictions: list[PlatePrediction] = Field(description="Danh sách các biển số nhận dạng được")


class HealthResponse(BaseModel):
    """Trạng thái sức khỏe và cấu hình mô hình của dịch vụ API."""

    status: str = Field(description="Trạng thái dịch vụ ('ok' hoặc 'model_missing')")
    model_weights: str = Field(description="Đường dẫn tệp trọng số mô hình")
    model_available: bool = Field(description="Cờ báo tệp trọng số mô hình có sẵn sàng hay không")


class LivenessResponse(BaseModel):
    """Trạng thái liveness (kiểm tra tiến trình còn sống)."""

    status: str = Field(default="live", description="Trạng thái dịch vụ đang chạy")


class ReadinessResponse(BaseModel):
    """Trạng thái readiness dựa trên việc tệp trọng số đã sẵn sàng."""

    status: str = Field(description="Trạng thái sẵn sàng ('ready' hoặc 'not_ready')")
    model_available: bool = Field(description="Cờ báo tệp trọng số tồn tại")
