"""Configures and attaches LoRA adapters to VLM, freezing Vision Encoder."""
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def apply_qlora(model, r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    Chuẩn bị mô hình cho QLoRA:
    - Bật gradient checkpointing
    - Đóng băng Vision Encoder
    - Chỉ gán adapter lên các Linear layers của LLM backbone
    """
    model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model
