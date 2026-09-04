"""
Custom Data Collator for Qwen2.5-VL with Multi-modal Visual Tokens & Loss Masking.
Đảm bảo cơ chế Loss Masking: Chỉ tính loss trên câu trả lời của Trợ lý (CoT + Action).
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class QwenVLDataCollator:
    """
    Data Collator cho Qwen2.5-VL hỗ trợ:
    1. Đóng gói cả hình ảnh và văn bản qua processor của Qwen2.5-VL.
    2. Loss Masking: Gán label = -100 cho toàn bộ System prompt, User prompt và Visual tokens.
       Chỉ tính đạo hàm trên phần câu trả lời của Trợ lý (assistant CoT reasoning + action).
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        messages_list = [item["messages"] for item in batch]

        # 1. Trích xuất text qua Chat Template
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in messages_list
        ]

        # 2. Xử lý ảnh đầu vào
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages_list)
        except (ImportError, Exception):
            # Fallback nếu qwen_vl_utils không có sẵn hoặc ảnh rỗng
            image_inputs = []
            video_inputs = None
            for msg in messages_list:
                img_found = None
                for turn in msg:
                    if turn["role"] == "user":
                        for c in turn.get("content", []):
                            if isinstance(c, dict) and c.get("type") == "image":
                                img_found = c.get("image")
                                break
                if img_found is not None:
                    image_inputs.append(img_found)

            if not image_inputs:
                image_inputs = None

        # 3. Chạy qua Processor của Qwen2.5-VL
        batch_inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 4. Cơ chế Loss Masking: Chỉ tính loss trên Assistant tokens
        input_ids = batch_inputs["input_ids"]
        labels = input_ids.clone()

        # Token ID của pad
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # Tìm token phân tách assistant: <|im_start|>assistant
        assistant_token = "<|im_start|>assistant"
        assistant_ids = self.processor.tokenizer.encode(assistant_token, add_special_tokens=False)

        for i in range(len(input_ids)):
            seq = input_ids[i].tolist() if hasattr(input_ids[i], "tolist") else list(input_ids[i])
            # Tìm vị trí bắt đầu của assistant response

            assist_start = -1
            if assistant_ids:
                # Tìm chuỗi con assistant_ids
                target_len = len(assistant_ids)
                for pos in range(len(seq) - target_len + 1):
                    if seq[pos : pos + target_len] == assistant_ids:
                        # Vị trí bắt đầu tính loss là ngay sau thẻ assistant\n
                        assist_start = pos + target_len
                        # Bỏ qua token newline kế tiếp nếu có
                        if assist_start < len(seq) and seq[assist_start] in [198, 271]:  # \n token
                            assist_start += 1
                        break

            if assist_start != -1:
                # Toàn bộ prompt, system, visual tokens trước assistant đều gán -100
                labels[i, :assist_start] = -100
            else:
                # Nếu không tìm thấy thẻ phân cách, giữ nguyên (fallback)
                pass

        batch_inputs["labels"] = labels
        return batch_inputs
