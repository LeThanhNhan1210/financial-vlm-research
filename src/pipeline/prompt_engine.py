import json
from pathlib import Path
from typing import List, Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class PromptEngine:
    def __init__(self, config_path: str = "./configs/prompt_templates.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        if HAS_YAML:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        # Fallback stdlib parser nếu chưa cài đặt pyyaml
        return self._fallback_yaml_parse()

    def _fallback_yaml_parse(self) -> Dict[str, Any]:
        """Bộ phân tích dự phòng nhẹ dùng stdlib khi không có pyyaml."""
        system_prompt = (
            "Bạn là một chuyên gia phân tích kỹ thuật tài chính cao cấp (CMT). "
            "Phân tích biểu đồ kỹ thuật theo quy trình 4 bước: "
            "1. Nhận diện cấu trúc xu hướng & hành động giá. "
            "2. Phân tích khối lượng và chỉ báo kỹ thuật. "
            "3. Lập luận rủi ro & xác định hỗ trợ/kháng cự. "
            "4. Đề xuất chiến lược hành động (BUY/SELL/HOLD, Entry, SL, TP, R:R)."
        )
        return {
            "system_prompt": system_prompt,
            "few_shot_examples": [
                {
                    "content": "Hãy phân tích biểu đồ kỹ thuật này.",
                    "assistant": "1. Xu hướng: Uptrend.\n2. Khối lượng: Volume lớn.\n3. Hỗ trợ: 100.\n4. Hành động: BUY, Entry: 105, SL: 100, TP: 115, R:R: 2.0.",
                },
                {
                    "content": "Hãy phân tích biểu đồ kỹ thuật này.",
                    "assistant": "1. Xu hướng: Downtrend.\n2. Khối lượng: Volume xả.\n3. Kháng cự: 85.\n4. Hành động: SELL, Entry: 81, SL: 85, TP: 73, R:R: 2.0.",
                }
            ],
        }

    def build_system_prompt(self) -> str:
        return self.config.get("system_prompt", "")

    def get_few_shot_examples(self) -> List[Dict[str, Any]]:
        """Lấy danh sách few-shot examples từ cấu hình."""
        return self.config.get("few_shot_examples", [])

    def build_chat_messages(
        self,
        user_instruction: str,
        few_shot_examples: List[Dict[str, Any]] = None,
        include_default_few_shot: bool = True,
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.build_system_prompt()}]

        examples = few_shot_examples
        if examples is None and include_default_few_shot:
            examples = self.get_few_shot_examples()

        if examples:
            for ex in examples:
                user_content = ex.get("content", ex.get("user", ""))
                assistant_content = ex.get("assistant", "")
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": assistant_content})

        messages.append({"role": "user", "content": user_instruction})
        return messages

    def build_vision_payload(
        self,
        user_instruction: str,
        base64_image: str,
        image_format: str = "png",
        include_few_shot: bool = True,
    ) -> List[Dict[str, Any]]:
        """Xây dựng payload tin nhắn đa phương thức (Vision) cho OpenAI API."""
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.build_system_prompt()}]

        if include_few_shot:
            for ex in self.get_few_shot_examples():
                user_content = ex.get("content", ex.get("user", ""))
                assistant_content = ex.get("assistant", "")
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": assistant_content})

        user_message_content = [
            {"type": "text", "text": user_instruction},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{base64_image}",
                    "detail": "high",
                },
            },
        ]
        messages.append({"role": "user", "content": user_message_content})
        return messages

