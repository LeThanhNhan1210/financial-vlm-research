"""Script chạy đánh giá 3 tầng (OCR -> LLM Judge -> Backtest)."""
from src.evaluation.llm_judge import LLMJudge


def main():
    print("=== ĐÁNH GIÁ MÔ HÌNH VLM THEO 3 TẦNG METRICS ===")
    judge = LLMJudge()
    sample_pred = "Xu hướng Uptrend mạnh, mô hình Bullish Engulfing tại hỗ trợ 1,200. Khuyến nghị BUY."
    res = judge.evaluate_prediction(sample_pred)
    print(f"[*] Điểm số chất lượng lập luận: {res['total_score']}/5.0")
    print(f"[*] Phản hồi: {res['feedback']}")


if __name__ == "__main__":
    main()
