"""Loads Vision-Language Models with 4-bit BitsAndBytes quantization."""
import torch
from transformers import BitsAndBytesConfig, AutoProcessor, AutoModelForVision2Seq


def load_quantized_vlm(model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
    """
    Nạp mô hình VLM với lượng hóa INT4 (NF4) để chạy trên Colab T4 (16GB VRAM).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_class = Qwen2_5_VLForConditionalGeneration
    except ImportError:
        try:
            from transformers import AutoModelForImageTextToText
            model_class = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_class = AutoModelForVision2Seq

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = model_class.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    return model, processor
