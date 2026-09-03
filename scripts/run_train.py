"""Script khởi chạy quá trình huấn luyện QLoRA."""
import yaml
from pathlib import Path
from src.models.vlm_loader import load_quantized_vlm
from src.models.lora_setup import apply_qlora


def main():
    print("=== KHỞI CHẠY QUÁ TRÌNH HUẤN LUYỆN QLORA TRÊN GOOGLE COLAB ===")
    config_path = Path("./configs/training_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["model_id"]
    print(f"[*] Đang nạp mô hình INT4: {model_id}...")
    model, processor = load_quantized_vlm(model_id)

    print("[*] Đang cấu hình LoRA Adapter (Freeze Vision, Train LLM)...")
    model = apply_qlora(model)

    print("[*] Sẵn sàng dữ liệu và tiến hành huấn luyện...")
    # Quá trình SFTTrainer sẽ được gọi tại đây


if __name__ == "__main__":
    main()
