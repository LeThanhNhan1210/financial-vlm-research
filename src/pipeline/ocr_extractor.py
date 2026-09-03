"""Extracts numerical bounds and axis coordinates to cross-validate VLM outputs."""
from typing import Dict, Any
from PIL import Image


class ChartOcrExtractor:
    """
    Module hỗ trợ trích xuất số liệu trên trục giá và thời gian của biểu đồ.
    Dùng để đối chiếu chéo (Cross-validation) chống ảo giác số liệu.
    """

    def __init__(self):
        pass

    def extract_bounds(self, image: Image.Image) -> Dict[str, Any]:
        """
        Trích xuất ngưỡng giá cao nhất, thấp nhất và khung thời gian hiển thị.
        Có thể tích hợp EasyOCR hoặc Tesseract khi cần thiết.
        """
        # Trả về placeholder cấu trúc bounds
        return {
            "min_price": None,
            "max_price": None,
            "timeframe": None,
            "status": "ready_for_ocr_integration",
        }
