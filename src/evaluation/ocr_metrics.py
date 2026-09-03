"""Calculates OCR extraction error and pattern recognition metrics."""
from typing import Dict, Any, List


def calculate_price_error(extracted_prices: List[float], true_prices: List[float]) -> Dict[str, float]:
    """
    Tính Mean Absolute Error (MAE) và Mean Absolute Percentage Error (MAPE)
    cho các mốc giá mô hình đọc được từ trục tọa độ.
    """
    if not extracted_prices or not true_prices or len(extracted_prices) != len(true_prices):
        return {"mae": 0.0, "mape": 0.0}

    mae = sum(abs(e - t) for e, t in zip(extracted_prices, true_prices)) / len(true_prices)
    mape = (sum(abs((e - t) / t) for e, t in zip(extracted_prices, true_prices) if t != 0) / len(true_prices)) * 100

    return {"mae": mae, "mape": mape}
