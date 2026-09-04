"""
Trainer orchestrator for Qwen2.5-VL QLoRA instruction tuning on Google Colab T4.
Tối ưu hóa VRAM và hỗ trợ tự động đồng bộ Google Drive.
"""
import time
import shutil
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

        # 3. Trích xuất log và vẽ biểu đồ Loss
        loss_curve_path = self._plot_and_save_metrics(
            log_history=trainer.state.log_history,
            output_dir=Path(output_dir),
            drive_backup_dir=Path(drive_backup) if drive_backup else None,
        )

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
            "loss_curve_path": str(loss_curve_path) if loss_curve_path else None,
        }
        return summary

    def _plot_and_save_metrics(
        self,
        log_history: list,
        output_dir: Path,
        drive_backup_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Tự động vẽ biểu đồ hội tụ Loss (Training & Validation Loss Curve)
        và lưu hình ảnh (PNG), tệp nhật ký (JSON) sang cả thư mục local và Google Drive.
        """
        import json

        train_steps, train_losses = [], []
        eval_steps, eval_losses = [], []

        for entry in log_history:
            step = entry.get("step")
            if "loss" in entry and step is not None:
                train_steps.append(step)
                train_losses.append(entry["loss"])
            if "eval_loss" in entry and step is not None:
                eval_steps.append(step)
                eval_losses.append(entry["eval_loss"])

        # Lưu log history ra JSON
        log_file = output_dir / "training_history.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_history, f, indent=2, ensure_ascii=False)

        # Đồng bộ log sang Drive nếu có
        if drive_backup_dir:
            drive_logs = drive_backup_dir.parent / "3_experiment_outputs" / "logs"
            try:
                drive_logs.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log_file, drive_logs / "training_history.json")
            except Exception as e:
                logger.warning(f"Could not sync logs to Drive: {e}")

        # Vẽ biểu đồ loss nếu có dữ liệu
        if not train_steps:
            return None

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(9, 5), dpi=150)
            plt.plot(
                train_steps,
                train_losses,
                label="Training Loss",
                color="#1f77b4",
                linewidth=2,
                marker="o",
                markersize=4,
            )
            if eval_steps:
                plt.plot(
                    eval_steps,
                    eval_losses,
                    label="Validation Loss",
                    color="#ff7f0e",
                    linewidth=2.5,
                    linestyle="--",
                    marker="s",
                    markersize=6,
                )

            plt.title(
                "Qwen2.5-VL-7B QLoRA Fine-Tuning Loss Curve (Google Colab T4)",
                fontsize=13,
                fontweight="bold",
                pad=12,
            )
            plt.xlabel("Optimizer Steps", fontsize=11)
            plt.ylabel("Cross-Entropy Loss", fontsize=11)
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend(fontsize=11)
            plt.tight_layout()

            # Lưu PNG vào thư mục checkpoint
            chart_file = output_dir / "loss_curve.png"
            plt.savefig(chart_file)
            logger.info(f"Đã lưu biểu đồ Loss Curve tại: {chart_file}")

            # Lưu vào Google Drive thư mục 3_experiment_outputs/figures/
            if drive_backup_dir:
                drive_figures = drive_backup_dir.parent / "3_experiment_outputs" / "figures"
                try:
                    drive_figures.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(chart_file, drive_figures / "qlora_loss_curve.png")
                    logger.info(f"Đã sao lưu biểu đồ sang Drive: {drive_figures / 'qlora_loss_curve.png'}")
                except Exception as e:
                    logger.warning(f"Could not copy loss curve to Drive: {e}")

            # Hiển thị trực quan trong Colab cell output
            try:
                plt.show()
            except Exception:
                pass

            plt.close()
            return chart_file
        except Exception as err:
            logger.warning(f"Không thể vẽ biểu đồ matplotlib (không ảnh hưởng đến weights): {err}")
            return None

