"""
Trainer orchestrator for Qwen2.5-VL QLoRA instruction tuning on Google Colab T4.
Tối ưu hóa VRAM và hỗ trợ tự động đồng bộ Google Drive.
"""
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .collator import QwenVLDataCollator
from .callbacks import DriveSyncCallback

logger = logging.getLogger(__name__)


class VLMQwenTrainer:
    """
    Bộ quản lý huấn luyện QLoRA cho Qwen2.5-VL:
    - Bắt buộc các tham số tối ưu phần cứng Colab T4: batch_size=1, grad_accum=16, fp16=True, gradient_checkpointing=True.
    - Tích hợp QwenVLDataCollator và DriveSyncCallback.
    """

    def __init__(
        self,
        model,
        processor,
        train_dataset,
        eval_dataset=None,
        training_config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.config = training_config or {}

    def train(self) -> Dict[str, Any]:
        """
        Khởi chạy vòng lặp huấn luyện SFT / LoRA.
        """
        try:
            import torch
            from transformers import TrainingArguments, Trainer
        except ImportError as e:
            logger.error(f"Transformers or PyTorch not installed: {e}")
            raise

        cfg_train = self.config.get("training", {})
        output_dir = cfg_train.get("output_dir", "./checkpoints/qlora_run")
        drive_backup = cfg_train.get(
            "drive_backup_dir", "/content/drive/MyDrive/NCKH_AI/2_checkpoints"
        )

        # 1. Cấu hình TrainingArguments tối ưu T4
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=cfg_train.get("num_train_epochs", 3),
            per_device_train_batch_size=cfg_train.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=cfg_train.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=cfg_train.get("gradient_accumulation_steps", 16),
            learning_rate=float(cfg_train.get("learning_rate", 2e-4)),
            weight_decay=float(cfg_train.get("weight_decay", 0.01)),
            warmup_ratio=float(cfg_train.get("warmup_ratio", 0.05)),
            lr_scheduler_type=cfg_train.get("lr_scheduler_type", "cosine"),
            logging_steps=cfg_train.get("logging_steps", 5),
            save_strategy="steps",
            save_steps=cfg_train.get("save_steps", 50),
            save_total_limit=cfg_train.get("save_total_limit", 3),
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=cfg_train.get("eval_steps", 50) if self.eval_dataset else None,
            gradient_checkpointing=True,
            fp16=True,
            bf16=False,
            remove_unused_columns=False,  # Quan trọng với VLM đa phương thức
            report_to="none",  # Tránh lỗi wandb khi chưa cấu hình key
            seed=cfg_train.get("seed", 42),
        )

        # 2. Khởi tạo Data Collator và Callback
        collator = QwenVLDataCollator(self.processor)
        callbacks = [DriveSyncCallback(drive_backup_dir=drive_backup)]

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=collator,
            callbacks=callbacks,
        )

        logger.info("=== BẮT ĐẦU HUẤN LUYỆN QLORA TRÊN GPU T4 ===")
        start_time = time.time()

        train_result = trainer.train()

        total_time = time.time() - start_time
        logger.info(f"=== HUẤN LUYỆN HOÀN TẤT TRONG {total_time/60:.2f} PHÚT ===")

        # Đo đạc VRAM tiêu thụ đỉnh
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
            logger.info(f"Peak VRAM chiếm dụng: {peak_vram_gb} GB / 15.0 GB")

        # Lưu model và processor cuối cùng
        final_dir = Path(output_dir) / "final_adapter"
        trainer.save_model(str(final_dir))
        self.processor.save_pretrained(str(final_dir))
        logger.info(f"Đã lưu final adapter tại: {final_dir}")

        summary = {
            "train_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "total_time_seconds": round(total_time, 2),
            "peak_vram_gb": peak_vram_gb,
            "final_adapter_dir": str(final_dir),
        }
        return summary
