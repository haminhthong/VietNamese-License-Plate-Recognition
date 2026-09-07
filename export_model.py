"""Xuất mô hình detector YOLOv8 sang định dạng ONNX phục vụ triển khai production và benchmark."""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Thực hiện export trọng số PyTorch (.pt) sang tệp đồ thị ONNX (.onnx)."""
    parser = argparse.ArgumentParser(description="Xuất mô hình YOLOv8 sang ONNX format.")
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn tới tệp trọng số best.pt")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh đầu vào (mặc định: 640)")
    parser.add_argument("--dynamic", action="store_true", help="Kích hoạt batch size và kích thước ảnh động")
    parser.add_argument(
        "--simplify", action="store_true", help="Tối ưu rút gọn đồ thị ONNX (yêu cầu onnxsim)"
    )
    args = parser.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp trọng số mô hình: {args.weights}")
    if args.imgsz <= 0:
        raise ValueError("Kích thước imgsz phải là số nguyên dương (> 0).")

    logger.info(
        "Đang thực hiện xuất mô hình YOLOv8 sang định dạng ONNX (imgsz=%d, dynamic=%s)...",
        args.imgsz,
        args.dynamic,
    )
    exported_path = YOLO(str(args.weights)).export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=17,
    )
    logger.info("Mô hình ONNX đã xuất thành công tại: %s", exported_path)


if __name__ == "__main__":
    main()
