"""Kiểm thử tự động cho FastAPI endpoints trong package app."""

from fastapi.testclient import TestClient

from app import api

client = TestClient(api.app)


def test_index_web_ui_endpoint():
    """Kiểm tra trang chủ / trả về giao diện Web UI HTML thành công."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Vietnamese License Plate Recognition" in response.text


def test_health_reports_missing_model(monkeypatch, tmp_path):
    """Kiểm tra API /health phản hồi khi chưa có tệp trọng số mô hình."""
    missing_weights = tmp_path / "missing.pt"
    monkeypatch.setenv("MODEL_WEIGHTS", str(missing_weights))
    response = api.health()
    assert response.status == "model_missing"
    assert response.model_available is False
    assert response.model_weights == str(missing_weights)


def test_health_reports_available_model(monkeypatch, tmp_path):
    """Kiểm tra API /health phản hồi khi tệp trọng số đã sẵn sàng."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"dummy")
    monkeypatch.setenv("MODEL_WEIGHTS", str(weights))
    response = api.health()
    assert response.status == "ok"
    assert response.model_available is True


def test_predict_rejects_unsupported_file_type():
    """Kiểm tra API /predict từ chối các định dạng không phải ảnh (415 Unsupported Media Type)."""
    response = client.post(
        "/predict",
        files={"image": ("test.txt", b"dummy text content", "text/plain")},
    )
    assert response.status_code == 415


def test_predict_rejects_empty_file():
    """Kiểm tra API /predict từ chối tệp rỗng (400 Bad Request)."""
    response = client.post(
        "/predict",
        files={"image": ("test.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
