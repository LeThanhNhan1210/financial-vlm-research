#!/usr/bin/env python3
"""
CLI Runner for Financial VLM QLoRA Fine-tuning on Google Colab T4 (Phase 2).
Usage:
    python scripts/run_train.py \
        --config configs/training_config.yaml \
        --train-file /content/drive/MyDrive/NCKH_AI/1_datasets/splits/train_cot.jsonl \
        --val-file /content/drive/MyDrive/NCKH_AI/1_datasets/splits/val.jsonl \
        --output-dir /content/drive/MyDrive/NCKH_AI/2_checkpoints/qlora_run \
        --epochs 3
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import FinancialChartDataset

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QLoRA instruction tuning for Qwen2.5-VL on financial chart CoT reasoning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--qlora-config",
        type=str,
        default="./configs/qlora_config.yaml",
        help="Path to QLoRA config YAML.",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Path to train JSONL file (overrides config).",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Path to validation JSONL file (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry-run check (verifies dataset loading and collator without full training).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    if HAS_YAML:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    args = parse_args()

    cfg_path = Path(args.config)
    qlora_cfg_path = Path(args.qlora_config)

    config = load_yaml(cfg_path)
    qlora_config = load_yaml(qlora_cfg_path)

    # Override config from CLI args if provided
    train_file = args.train_file or config.get("data", {}).get("train_split", "./data/splits/train.jsonl")
    val_file = args.val_file or config.get("data", {}).get("val_split", "./data/splits/val.jsonl")
    output_dir = args.output_dir or config.get("training", {}).get("output_dir", "./checkpoints/qlora_run")

    if "training" not in config:
        config["training"] = {}
    config["training"]["output_dir"] = output_dir

    if args.epochs is not None:
        config["training"]["num_train_epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    model_id = config.get("model", {}).get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")

    logger.info("=" * 65)
    logger.info("FINANCIAL VLM: QLoRA TRAINING ON COLAB T4 (PHASE 2)")
    logger.info("=" * 65)
    logger.info(f"Model ID      : {model_id}")
    logger.info(f"Train dataset : {train_file}")
    logger.info(f"Val dataset   : {val_file}")
    logger.info(f"Output dir    : {output_dir}")
    logger.info(f"Epochs        : {config['training'].get('num_train_epochs', 3)}")
    logger.info(f"Learning Rate : {config['training'].get('learning_rate', 2e-4)}")
    logger.info(f"Dry-run mode  : {args.dry_run}")
    logger.info("-" * 65)

    # 1. Nạp Dataset
    logger.info("[1/4] Loading financial chart datasets...")
    train_dataset = FinancialChartDataset(train_file)
    val_dataset = FinancialChartDataset(val_file) if Path(val_file).exists() else None

    logger.info(f"Train samples : {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Val samples   : {len(val_dataset)}")

    if len(train_dataset) == 0:
        logger.error(f"Train dataset is empty or file not found: {train_file}")
        sys.exit(1)

    # Chế độ Dry-run chỉ kiểm tra tính toàn vẹn của dataset và định dạng hội thoại
    if args.dry_run:
        logger.info("[Dry-run] Checking sample conversation structure...")
        sample_conv = train_dataset.get_conversation(0)
        logger.info(f"Sample conversation turn count: {len(sample_conv)}")
        logger.info(f"Sample user content keys: {[c.get('type') for c in sample_conv[0]['content']]}")
        logger.info(f"Sample assistant response snippet: {sample_conv[1]['content'][:120]}...")
        logger.info("[Dry-run] Dataset check PASSED! Ready for full training on GPU T4.")
        sys.exit(0)

    # 2. Nạp mô hình INT4
    logger.info(f"[2/4] Loading INT4 quantized model: {model_id}...")
    from src.models.vlm_loader import load_quantized_vlm
    from src.models.lora_setup import apply_qlora
    from src.training.trainer import VLMQwenTrainer

    model, processor = load_quantized_vlm(model_id)


    # 3. Gắn LoRA adapter (Freeze Vision, Train LLM)
    logger.info("[3/4] Attaching LoRA adapters to Language Model backbone...")
    lora_p = qlora_config.get("lora", {})
    model = apply_qlora(
        model=model,
        r=lora_p.get("r", 16),
        lora_alpha=lora_p.get("lora_alpha", 32),
        lora_dropout=lora_p.get("lora_dropout", 0.05),
    )

    # 4. Huấn luyện
    logger.info("[4/4] Starting VLM training run...")
    trainer = VLMQwenTrainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        training_config=config,
    )

    summary = trainer.train()

    logger.info("=" * 65)
    logger.info(f"[SUCCESS] QLoRA training finished! Final loss: {summary.get('train_loss')}")
    logger.info(f"Final adapter saved to: {summary.get('final_adapter_dir')}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
