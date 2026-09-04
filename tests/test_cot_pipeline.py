"""
Unit & Integration Tests for Financial CoT Labeling Pipeline (Step 2.4).
"""
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.anti_hallucination import CoTValidator
from src.pipeline.audit_sampler import AuditSampler
from src.pipeline.cot_generator import CoTLabelGenerator
from src.pipeline.prompt_engine import PromptEngine


class TestCoTPipeline(unittest.TestCase):

    def setUp(self):
        self.validator = CoTValidator()

    def test_prompt_engine_loads_few_shot(self):
        """Kiểm tra PromptEngine nạp đúng template và few-shot."""
        engine = PromptEngine("configs/prompt_templates.yaml")
        sys_prompt = engine.build_system_prompt()
        self.assertIn("CMT", sys_prompt)
        few_shots = engine.get_few_shot_examples()
        self.assertGreaterEqual(len(few_shots), 2)

    def test_validator_buy_logic_valid(self):
        """Kiểm tra logic lệnh BUY hợp lệ (SL < Entry < TP)."""
        valid_buy_text = """
        1. Xu hướng: Uptrend với mẫu Hammer.
        2. Khối lượng: Volume gia tăng vượt EMA20.
        3. Hỗ trợ: 100.0, rủi ro thủng đáy.
        4. Khuyến nghị:
        - Hành động: BUY
        - Vùng vào: 105.0
        - Cắt lỗ: 100.0
        - Chốt lời: 115.0
        - Tỷ lệ R:R: 2.0
        """
        eval_res = self.validator.evaluate_response(valid_buy_text)
        self.assertEqual(eval_res["action"], "BUY")
        self.assertTrue(eval_res["logic_valid"])
        self.assertTrue(eval_res["rr_valid"])
        self.assertTrue(eval_res["structure_valid"])
        self.assertTrue(eval_res["passed"])

    def test_validator_buy_logic_invalid_sl(self):
        """Kiểm tra phát hiện ảo giác: BUY nhưng Stop Loss > Entry."""
        invalid_buy_text = """
        1. Xu hướng: Uptrend.
        2. Khối lượng: Volume EMA.
        3. Hỗ trợ: Ngưỡng hỗ trợ và rủi ro.
        4. Khuyến nghị:
        - Hành động: BUY
        - Vùng vào: 105.0
        - Cắt lỗ: 110.0   # SAI LẦM: SL cao hơn Entry!
        - Chốt lời: 120.0
        """
        eval_res = self.validator.evaluate_response(invalid_buy_text)
        self.assertEqual(eval_res["action"], "BUY")
        self.assertFalse(eval_res["logic_valid"])
        self.assertIn("StopLoss", eval_res["logic_message"])

    def test_validator_sell_logic_valid(self):
        """Kiểm tra logic lệnh SELL hợp lệ (TP < Entry < SL)."""
        valid_sell_text = """
        1. Xu hướng: Downtrend nến Shooting Star.
        2. Khối lượng: Volume lớn cắt EMA20.
        3. Kháng cự: 90.0, rủi ro đảo chiều.
        4. Khuyến nghị:
        - Hành động: SELL
        - Vùng vào: 85.0
        - Cắt lỗ: 90.0
        - Chốt lời: 75.0
        - Tỷ lệ R:R: 2.0
        """
        eval_res = self.validator.evaluate_response(valid_sell_text)
        self.assertEqual(eval_res["action"], "SELL")
        self.assertTrue(eval_res["logic_valid"])
        self.assertTrue(eval_res["passed"])

    def test_audit_sampler_stratification(self):
        """Kiểm tra lấy mẫu phân tầng 30% đều trên 3 nhóm tài sản."""
        records = []
        for i in range(30):
            records.append({"id": f"vn_{i}", "asset_class": "VN30", "action": "BUY"})
        for i in range(30):
            records.append({"id": f"us_{i}", "asset_class": "US_Equities", "action": "SELL"})
        for i in range(30):
            records.append({"id": f"cr_{i}", "asset_class": "Crypto", "action": "HOLD"})

        sampler = AuditSampler(sample_rate=0.30, random_state=42)
        sampled = sampler.sample_records(records)

        # 30% của 90 mẫu = 27 mẫu
        self.assertEqual(len(sampled), 27)

        # Kiểm tra mỗi nhóm có đúng 9 mẫu
        from collections import Counter
        counts = Counter(r["asset_class"] for r in sampled)
        self.assertEqual(counts["VN30"], 9)
        self.assertEqual(counts["US_Equities"], 9)
        self.assertEqual(counts["Crypto"], 9)
        self.assertTrue(all(r["audit_status"] == "pending" for r in sampled))


    def test_generator_mock_pipeline(self):
        """Kiểm tra quy trình sinh nhãn mock cho sample records."""
        sample_record = {
            "id": "test_sample_001",
            "symbol": "VCB",
            "asset_class": "VN30",
            "timeframe": "1D",
            "detected_patterns": "hammer",
            "instruction": "Phân tích biểu đồ này.",
        }

        generator = CoTLabelGenerator(provider="mock")
        res = generator.generate_single(sample_record)

        self.assertEqual(res["action"], "BUY")
        self.assertIn("Cấu trúc xu hướng", res["cot_reasoning"])
        self.assertIn("Hành động: BUY", res["cot_reasoning"])
        self.assertTrue(res["validation_meta"]["passed"])



if __name__ == "__main__":
    unittest.main()
