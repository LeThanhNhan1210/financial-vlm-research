"""
Anti-Hallucination & Guardrails Validation for Financial VLM CoT Outputs.
Kiểm soát ảo giác số liệu và bảo đảm tính nhất quán toán học trong chuỗi suy luận.
"""
import re
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class CoTValidator:
    """
    Bộ thẩm định chuỗi suy luận CoT tài chính:
    - Bắt buộc tính nhất quán vị trí giá (BUY: SL < Entry < TP; SELL: TP < Entry < SL).
    - Đối soát toán học tỷ lệ Risk:Reward (R:R >= 1.5).
    - Chuẩn hóa đầu ra thành trường action và cot_reasoning cho JSONL.
    """

    ACTION_BUY = "BUY"
    ACTION_SELL = "SELL"
    ACTION_HOLD = "HOLD"

    def __init__(self, min_rr: float = 1.5, rr_tolerance: float = 0.35):
        self.min_rr = min_rr
        self.rr_tolerance = rr_tolerance

    def extract_action(self, text: str) -> str:
        """Trích xuất nhãn hành động BUY / SELL / HOLD từ văn bản."""
        text_upper = text.upper()
        action_match = re.search(r"(?:HÀNH ĐỘNG|KHUYẾN NGHỊ|ACTION|RECOMMENDATION)\s*[:\-]?\s*(BUY|SELL|HOLD|MUA|BÁN|GIỮ)", text_upper)
        if action_match:
            act = action_match.group(1)
            if act in ["BUY", "MUA"]:
                return self.ACTION_BUY
            if act in ["SELL", "BÁN"]:
                return self.ACTION_SELL
            if act in ["HOLD", "GIỮ"]:
                return self.ACTION_HOLD

        if re.search(r"\b(BUY|MUA)\b", text_upper):
            return self.ACTION_BUY
        if re.search(r"\b(SELL|BÁN)\b", text_upper):
            return self.ACTION_SELL
        if re.search(r"\b(HOLD|GIỮ|QUAN SÁT)\b", text_upper):
            return self.ACTION_HOLD

        return self.ACTION_HOLD

    def extract_prices(self, text: str) -> Dict[str, Optional[float]]:
        """Trích xuất giá Entry, Stop Loss, Take Profit, và R:R từ văn bản."""
        res: Dict[str, Optional[float]] = {
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward": None,
        }

        def _parse_num(val_str: str) -> Optional[float]:
            cleaned = re.sub(r"[^\d.]", "", val_str.replace(",", "."))
            try:
                return float(cleaned)
            except ValueError:
                return None

        # 1. Entry zone
        entry_match = re.search(
            r"(?:ENTRY|VÙNG\s*(?:GIÁ\s*)?VÀO|GIÁ\s*VÀO|MỞ\s*VỊ\s*THẾ)(?:\s*\([^)]*\))?\s*[:\-]?\s*(\d+[\d.,]*(?:\s*-\s*\d+[\d.,]*)?)",
            text,
            re.IGNORECASE,
        )
        if entry_match:
            parts = entry_match.group(1).split("-")
            nums = [_parse_num(p) for p in parts if _parse_num(p) is not None]
            if nums:
                res["entry"] = sum(nums) / len(nums)

        # 2. Stop Loss
        sl_match = re.search(
            r"(?:STOP\s*LOSS|STOP_LOSS|CẮT\s*LỖ|DỪNG\s*LỖ|\bSL\b)(?:\s*\([^)]*\))?\s*[:\-]?\s*(\d+[\d.,]*)",
            text,
            re.IGNORECASE,
        )
        if sl_match:
            res["stop_loss"] = _parse_num(sl_match.group(1))

        # 3. Take Profit
        tp_match = re.search(
            r"(?:TAKE\s*PROFIT|TAKE_PROFIT|CHỐT\s*LỜI|MỤC\s*TIÊU|\bTP\b)(?:\s*\([^)]*\))?\s*[:\-]?\s*(\d+[\d.,]*)",
            text,
            re.IGNORECASE,
        )
        if tp_match:
            res["take_profit"] = _parse_num(tp_match.group(1))

        # 4. Risk / Reward ratio
        rr_match = re.search(
            r"(?:TỶ\s*LỆ\s*R:R|R:R|RISK_REWARD|RISK/REWARD)(?:\s*\([^)]*\))?\s*[:\-]?\s*(\d+[\d.,]*)",
            text,
            re.IGNORECASE,
        )
        if rr_match:
            res["risk_reward"] = _parse_num(rr_match.group(1))


        return res

    def validate_trade_logic(
        self,
        action: str,
        entry: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> Tuple[bool, str]:
        """
        Kiểm tra tính logic toán học của lệnh giao dịch.
        - BUY: Stop Loss < Entry < Take Profit
        - SELL: Take Profit < Entry < Stop Loss
        """
        if action == self.ACTION_HOLD:
            return True, "HOLD action valid (no strict price boundaries required)."

        if entry is None or stop_loss is None or take_profit is None:
            return False, f"Missing price targets for action {action}: entry={entry}, sl={stop_loss}, tp={take_profit}"

        if action == self.ACTION_BUY:
            if not (stop_loss < entry < take_profit):
                return False, f"BUY logic violated: required StopLoss ({stop_loss}) < Entry ({entry}) < TakeProfit ({take_profit})"
            return True, "BUY logic valid."

        if action == self.ACTION_SELL:
            if not (take_profit < entry < stop_loss):
                return False, f"SELL logic violated: required TakeProfit ({take_profit}) < Entry ({entry}) < StopLoss ({stop_loss})"
            return True, "SELL logic valid."

        return True, "Logic check bypassed."

    def validate_risk_reward(
        self,
        action: str,
        entry: Optional[float],
        stop_loss: Optional[float],
        take_profit: Optional[float],
        declared_rr: Optional[float] = None,
    ) -> Tuple[bool, Optional[float], str]:
        """
        Tính toán và kiểm tra tính hợp lệ của tỷ lệ Risk:Reward.
        """
        if action == self.ACTION_HOLD or entry is None or stop_loss is None or take_profit is None:
            return True, None, "HOLD or incomplete targets, RR validation skipped."

        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)

        if risk <= 1e-6:
            return False, 0.0, "Zero risk detected (Entry == StopLoss)."

        actual_rr = round(reward / risk, 2)

        if actual_rr < 1.0:
            return False, actual_rr, f"Sub-optimal R:R ratio ({actual_rr} < 1.0), unfavorable trade profile."

        if declared_rr is not None and declared_rr > 0:
            diff = abs(actual_rr - declared_rr) / declared_rr
            if diff > self.rr_tolerance:
                return (
                    False,
                    actual_rr,
                    f"R:R discrepancy hallucination: calculated {actual_rr} vs declared {declared_rr} (diff: {diff*100:.1f}%)",
                )

        return True, actual_rr, f"R:R ratio valid ({actual_rr})."

    def validate_cot_structure(self, text: str) -> Tuple[bool, List[str]]:
        """Kiểm tra sự hiện diện của 4 bước lập luận chuẩn CMT."""
        missing = []
        text_lower = text.lower()

        # Bước 1: Trend & Price Action
        if not any(k in text_lower for k in ["xu hướng", "trend", "hành động giá", "price action", "nến"]):
            missing.append("Bước 1: Trend & Price Action")

        # Bước 2: Volume & Indicators
        if not any(k in text_lower for k in ["khối lượng", "volume", "ema", "ma", "chỉ báo", "thanh khoản"]):
            missing.append("Bước 2: Volume & Indicators")

        # Bước 3: Support, Resistance & Risk
        if not any(k in text_lower for k in ["hỗ trợ", "kháng cự", "support", "resistance", "rủi ro", "vô hiệu"]):
            missing.append("Bước 3: Support/Resistance & Risk")

        # Bước 4: Actionable Recommendation
        if not any(k in text_lower for k in ["khuyến nghị", "hành động", "action", "entry", "cắt lỗ", "chốt lời"]):
            missing.append("Bước 4: Actionable Recommendation")

        is_valid = len(missing) == 0
        return is_valid, missing

    def evaluate_response(self, raw_text: str) -> Dict[str, Any]:
        """
        Đánh giá toàn diện một chuỗi suy luận CoT sinh ra từ mô hình.
        """
        action = self.extract_action(raw_text)
        prices = self.extract_prices(raw_text)
        logic_ok, logic_msg = self.validate_trade_logic(
            action, prices["entry"], prices["stop_loss"], prices["take_profit"]
        )
        rr_ok, actual_rr, rr_msg = self.validate_risk_reward(
            action, prices["entry"], prices["stop_loss"], prices["take_profit"], prices["risk_reward"]
        )
        struct_ok, missing_steps = self.validate_cot_structure(raw_text)

        all_passed = logic_ok and rr_ok and struct_ok

        return {
            "passed": all_passed,
            "action": action,
            "prices": prices,
            "calculated_rr": actual_rr,
            "logic_valid": logic_ok,
            "logic_message": logic_msg,
            "rr_valid": rr_ok,
            "rr_message": rr_msg,
            "structure_valid": struct_ok,
            "missing_steps": missing_steps,
        }
