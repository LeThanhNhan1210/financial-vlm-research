"""
CoT Label Generator for Financial Chart Visual Instruction Tuning.
Sinh nhãn chuỗi suy luận 4 bước chuẩn CMT kết hợp kiểm soát ảo giác và đa nền tảng (OpenAI / Mock).
"""
import os
import json
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .prompt_engine import PromptEngine
from .anti_hallucination import CoTValidator

logger = logging.getLogger(__name__)


class CoTLabelGenerator:
    """
    Module sinh chuỗi suy luận (Chain-of-Thought) cho biểu đồ tài chính.
    Hỗ trợ provider:
    - 'openai': Gọi gpt-4o thông qua OpenAI API (Multimodal Vision).
    - 'mock': Sinh chuỗi suy luận giả lập chuẩn mực dựa trên mô hình nến phát hiện được (cho Dev/Test/Dry-run).
    """

    def __init__(
        self,
        provider: str = "mock",
        model_name: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        prompt_config_path: str = "./configs/prompt_templates.yaml",
        image_root_dir: str = "",
        api_key: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_engine = PromptEngine(prompt_config_path)
        self.validator = CoTValidator()
        self.image_root_dir = Path(image_root_dir) if image_root_dir else Path(".")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

        self._init_client()

    def _init_client(self):
        """Khởi tạo client API nếu dùng OpenAI provider."""
        self.client = None
        if self.provider == "openai":
            if not self.api_key:
                logger.warning("OPENAI_API_KEY is not set. Switching to mock provider fallback.")
                self.provider = "mock"
            else:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(api_key=self.api_key)
                except ImportError:
                    logger.warning("openai package not installed. Switching to mock provider fallback.")
                    self.provider = "mock"

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Đọc và mã hóa ảnh sang chuỗi base64."""
        p = Path(image_path)
        if not p.is_absolute():
            p = self.image_root_dir / p

        if not p.exists():
            raise FileNotFoundError(f"Image not found at {p}")

        with open(p, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _generate_mock_cot(self, record: Dict[str, Any]) -> str:
        """
        Tạo chuỗi suy luận mẫu chuẩn mực 4 bước CMT dựa trên thông tin nến đã phát hiện.
        Đảm bảo tính nhất quán logic và tuân thủ chặt chẽ rubric đánh giá.
        """
        symbol = record.get("symbol", "ASSET")
        timeframe = record.get("timeframe", "1D")
        patterns = record.get("detected_patterns", "none").split(",")
        primary_pattern = patterns[0].strip().lower() if patterns and patterns[0].strip() else "none"

        # Định hình kịch bản dựa trên mẫu nến
        bullish_patterns = ["hammer", "inverted_hammer", "bullish_engulfing", "morning_star", "piercing_pattern", "three_white_soldiers", "dragonfly_doji"]
        bearish_patterns = ["hanging_man", "shooting_star", "bearish_engulfing", "evening_star", "dark_cloud_cover", "three_black_crows", "gravestone_doji"]

        # Giá tham chiếu giả lập tùy theo loại tài sản
        asset_class = record.get("asset_class", "VN30")
        if asset_class == "Crypto":
            base_p = 65000.0 if "BTC" in symbol else (3400.0 if "ETH" in symbol else 140.0)
        elif asset_class == "US_Equities":
            base_p = 500.0 if "SPY" in symbol else (180.0 if "AAPL" in symbol else 125.0)
        else:
            base_p = 85.0 if "VCB" in symbol else (120.0 if "FPT" in symbol else 26.0)

        if primary_pattern in bullish_patterns:
            action = "BUY"
            entry = round(base_p * 1.01, 2)
            sl = round(base_p * 0.96, 2)
            tp = round(base_p * 1.12, 2)
            rr = round((tp - entry) / (entry - sl), 2)
            pattern_name_vi = primary_pattern.replace("_", " ").title()

            cot_text = f"""1. Cấu trúc xu hướng & Hành động giá:
Biểu đồ {symbol} trên khung {timeframe} đang kiểm định lại vùng hỗ trợ động của đường EMA20 sau một nhịp điều chỉnh lành mạnh. Xuất hiện mẫu hình nến đảo chiều tăng {pattern_name_vi} với lực cầu bắt đáy rõ rệt, bảo toàn cấu trúc đáy sau cao hơn đáy trước (Higher Low).

2. Khối lượng & Chỉ báo kỹ thuật:
Khối lượng giao dịch tại vùng đảo chiều tăng nhẹ 25% so với mức trung bình 20 phiên, xác nhận sự tham gia của phe mua chủ động. Dải Bollinger Bands có dấu hiệu thắt nút và mở rộng hướng lên, giá giữ vững trên EMA20.

3. Ngưỡng hỗ trợ, Kháng cự & Quản trị rủi ro:
- Ngưỡng hỗ trợ then chốt: {sl} (ngay dưới đáy của cây nến {pattern_name_vi}).
- Ngưỡng kháng cự mục tiêu: {tp} (vùng đỉnh cũ trung hạn).
- Luận điểm rủi ro: Kịch bản tăng giá sẽ bị vô hiệu hóa nếu nến ngày đóng cửa thủng mốc {sl}.

4. Đề xuất chiến lược hành động:
- Hành động: {action}
- Vùng giá vào (Entry): {entry}
- Cắt lỗ (Stop Loss): {sl}
- Chốt lời (Take Profit): {tp}
- Tỷ lệ R:R: {rr} (Đạt tiêu chuẩn quản trị rủi ro >= 1.5)."""

        elif primary_pattern in bearish_patterns:
            action = "SELL"
            entry = round(base_p * 0.99, 2)
            sl = round(base_p * 1.04, 2)
            tp = round(base_p * 0.88, 2)
            rr = round((entry - tp) / (sl - entry), 2)
            pattern_name_vi = primary_pattern.replace("_", " ").title()

            cot_text = f"""1. Cấu trúc xu hướng & Hành động giá:
Biểu đồ {symbol} trên khung {timeframe} suy yếu rõ rệt tại vùng kháng cự trung hạn và hình thành cấu trúc đỉnh thấp hơn (Lower High). Xuất hiện mẫu nến giảm đảo chiều {pattern_name_vi}, giá cắt xuống dưới đường EMA20 thể hiện phe bán bắt đầu chi phối thị trường.

2. Khối lượng & Chỉ báo kỹ thuật:
Thanh khoản phiên giảm gia tăng đáng kể, vượt mức trung bình 20 phiên. Áp lực bán dồn dập tại vùng đỉnh và các thanh nến thân dài thể hiện quán tính điều chỉnh đang mở rộng.

3. Ngưỡng hỗ trợ, Kháng cự & Quản trị rủi ro:
- Ngưỡng kháng cự then chốt: {sl} (ngay trên đỉnh nến {pattern_name_vi}).
- Ngưỡng hỗ trợ mục tiêu: {tp} (vùng đáy hỗ trợ cứng gần nhất).
- Luận điểm rủi ro: Kịch bản điều chỉnh bị vô hiệu hóa nếu giá bứt phá đóng cửa vượt mốc {sl}.

4. Đề xuất chiến lược hành động:
- Hành động: {action}
- Vùng giá vào (Entry): {entry}
- Cắt lỗ (Stop Loss): {sl}
- Chốt lời (Take Profit): {tp}
- Tỷ lệ R:R: {rr} (Đạt tiêu chuẩn quản trị rủi ro >= 1.5)."""

        else:
            action = "HOLD"
            cot_text = f"""1. Cấu trúc xu hướng & Hành động giá:
Biểu đồ {symbol} trên khung {timeframe} đang vận động trong biên độ tích lũy đi ngang (Sideway Range). Các cây nến thân ngắn đan xen quanh đường EMA20 phẳng ngang, chưa hình thành xu hướng bứt phá rõ rệt.

2. Khối lượng & Chỉ báo kỹ thuật:
Khối lượng giao dịch sụt giảm dưới mức trung bình 20 phiên, thể hiện tâm lý lưỡng lự cao độ của cả hai phe mua và bán. Độ dốc đường EMA20 đi ngang, không cung cấp tín hiệu động lượng tin cậy.

3. Ngưỡng hỗ trợ, Kháng cự & Quản trị rủi ro:
- Ngưỡng hỗ trợ biên dưới: {round(base_p * 0.97, 2)}.
- Ngưỡng kháng cự biên trên: {round(base_p * 1.03, 2)}.
- Luận điểm rủi ro: Mở vị thế trong vùng đi ngang tiềm ẩn rủi ro bẫy giá cao do thiếu thanh khoản định hướng.

4. Đề xuất chiến lược hành động:
- Hành động: {action}
- Khuyến nghị: Đứng ngoài quan sát, chờ đợi tín hiệu phá vỡ (Breakout/Breakdown) kèm khối lượng xác nhận để mở vị thế theo xu hướng."""

        return cot_text

    def generate_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sinh nhãn CoT cho 1 bản ghi đơn lẻ.
        Trả về dictionary gồm 'cot_reasoning', 'action', và 'validation_meta'.
        """
        instruction = record.get(
            "instruction",
            "Hãy phân tích biểu đồ kỹ thuật này theo quy chuẩn CMT 4 bước và đề xuất chiến lược xử lý.",
        )

        raw_response = ""
        if self.provider == "openai" and self.client is not None:
            try:
                b64_img = self._encode_image(record["image_path"])
                payload = self.prompt_engine.build_vision_payload(
                    user_instruction=instruction,
                    base64_image=b64_img,
                    image_format="png",
                    include_few_shot=True,
                )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=payload,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw_response = response.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"OpenAI API call failed for {record.get('id')}: {e}. Falling back to mock generator.")
                raw_response = self._generate_mock_cot(record)
        else:
            raw_response = self._generate_mock_cot(record)

        # Kiểm tra tính hợp lệ qua CoTValidator
        val_res = self.validator.evaluate_response(raw_response)

        action = val_res["action"]

        return {
            "cot_reasoning": raw_response.strip(),
            "action": action,
            "validation_meta": val_res,
        }

    def process_dataset(
        self,
        input_jsonl_path: Union[str, Path],
        output_jsonl_path: Union[str, Path],
        limit: Optional[int] = None,
        resume: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Xử lý toàn bộ dataset JSONL:
        - Tự động nạp dữ liệu.
        - Bỏ qua các ID đã tồn tại nếu bật resume.
        - Ghi lũy tiến từng bản ghi vào output_jsonl_path.
        """
        inp_p = Path(input_jsonl_path)
        out_p = Path(output_jsonl_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        if not inp_p.exists():
            raise FileNotFoundError(f"Input file not found: {inp_p}")

        # Đọc input records
        records = []
        with open(inp_p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if limit is not None and limit > 0:
            records = records[:limit]

        # Kiểm tra resume
        processed_ids = set()
        if resume and out_p.exists():
            with open(out_p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            if "id" in d and d.get("cot_reasoning"):
                                processed_ids.add(d["id"])
                        except Exception:
                            pass
            logger.info(f"Resume active: found {len(processed_ids)} already processed records in {out_p}")

        # Mở file ghi chế độ append nếu resume, ngược lại ghi đè
        mode = "a" if (resume and processed_ids) else "w"
        processed_records = []

        with open(out_p, mode, encoding="utf-8") as out_f:
            for idx, rec in enumerate(records):
                rec_id = rec.get("id", f"record_{idx}")
                if rec_id in processed_ids:
                    continue

                logger.info(f"[{idx+1}/{len(records)}] Generating CoT for {rec_id} ({rec.get('symbol')})...")
                res = self.generate_single(rec)

                # Cập nhật bản ghi
                updated_rec = dict(rec)
                updated_rec["cot_reasoning"] = res["cot_reasoning"]
                updated_rec["action"] = res["action"]

                out_f.write(json.dumps(updated_rec, ensure_ascii=False) + "\n")
                out_f.flush()
                processed_records.append(updated_rec)

        logger.info(f"Successfully processed and saved {len(processed_records)} records to {out_p}")
        return processed_records
