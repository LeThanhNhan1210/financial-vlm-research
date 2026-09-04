"""
VLMQwenTrainer – Orchestrator huấn luyện QLoRA cho Qwen2.5-VL trên Colab T4.

Thiết kế:
  - Đọc siêu tham số từ dict config (gốc YAML) → không hard-code.
  - Dùng inspect.signature(TrainingArguments.__init__) để chỉ truyền
    những kwargs mà phiên bản transformers đang cài thực sự hỗ trợ.
    Giải quyết triệt để lỗi eval_strategy / evaluation_strategy giữa các bản transformers.
  - Tự động vẽ loss curve và lưu lên Drive sau khi train xong.
"""
import inspect
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch

logger = logging.getLogger(__name__)

try:
    from transformers import Trainer, TrainingArguments
except ImportError:
    raise ImportError(
        "transformers chưa được cài đặt. "
        "Chạy: pip install transformers>=4.37"
    )

from .collator import QwenVLDataCollator
from .callbacks import DriveSyncCallback


class VLMQwenTrainer:
    """
    Lớp điều phối huấn luyện QLoRA cho Financial VLM.

    Parameters
    ----------
    model : PreTrainedModel đã gắn LoRA adapter (INT4 quantized).
    processor : Qwen2.5-VL processor (tokenizer + image processor).
    train_dataset : FinancialChartDataset cho tập train.
    eval_dataset : FinancialChartDataset cho tập val (tùy chọn).
    training_config : dict đọc từ configs/training_config.yaml.
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
        """Chạy huấn luyện và trả về dict tóm tắt kết quả."""
        cfg_train = self.config.get("training", {})
        output_dir = cfg_train.get("output_dir", "./checkpoints/qlora_run")
        drive_backup = cfg_train.get(
            "drive_backup_dir",
            "/content/drive/MyDrive/NCKH_AI/2_checkpoints",
        )

        # Lấy danh sách tham số mà TrainingArguments.__init__ thực sự nhận
        params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

        # 1. Xây dựng dict tham số cơ bản (luôn hợp lệ ở mọi phiên bản)
        args_dict = {
            "output_dir": output_dir,
            "num_train_epochs": cfg_train.get("num_train_epochs", 3),
            "per_device_train_batch_size": cfg_train.get("per_device_train_batch_size", 1),
            "per_device_eval_batch_size": cfg_train.get("per_device_eval_batch_size", 1),
            "gradient_accumulation_steps": cfg_train.get("gradient_accumulation_steps", 16),
            "learning_rate": float(cfg_train.get("learning_rate", 2e-4)),
            "weight_decay": float(cfg_train.get("weight_decay", 0.01)),
            "lr_scheduler_type": cfg_train.get("lr_scheduler_type", "cosine"),
            "logging_steps": cfg_train.get("logging_steps", 5),
            "save_strategy": "steps",
            "save_steps": cfg_train.get("save_steps", 50),
            "save_total_limit": cfg_train.get("save_total_limit", 3),
            "gradient_checkpointing": True,
            "fp16": True,
            "bf16": False,
            "remove_unused_columns": False,  # Bắt buộc False cho VLM đa phương thức
            "report_to": "none",             # Tránh lỗi cấu hình wandb
            "seed": cfg_train.get("seed", 2024),
        }


        # --- Xử lý tương thích eval_strategy vs evaluation_strategy ---
        # transformers >= 4.41 dùng eval_strategy, bản cũ dùng evaluation_strategy
        eval_mode = "steps" if self.eval_dataset else "no"
        eval_steps_val = cfg_train.get("eval_steps", 50) if self.eval_dataset else None

        if "eval_strategy" in params:
            args_dict["eval_strategy"] = eval_mode
        elif "evaluation_strategy" in params:
            args_dict["evaluation_strategy"] = eval_mode

        if eval_steps_val is not None and "eval_steps" in params:
            args_dict["eval_steps"] = eval_steps_val

        # Lọc chỉ truyền các tham số thực sự nằm trong signature
        valid_kwargs = {k: v for k, v in args_dict.items() if k in params}
        logger.info(f"[TrainingArguments] Truyền {len(valid_kwargs)} tham số hợp lệ")
        training_args = TrainingArguments(**valid_kwargs)

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

        logger.info("=" * 65)
        logger.info("BẮT ĐẦU HUẤN LUYỆN QLORA TRÊN GPU T4")
        logger.info("=" * 65)
        start_time = time.time()

        train_result = trainer.train()

        total_time = time.time() - start_time
        logger.info(f"HUẤN LUYỆN HOÀN TẤT TRONG {total_time / 60:.2f} PHÚT")

        # Đo đạc VRAM tiêu thụ đỉnh
        peak_vram_gb = 0.0
        if torch.cuda.is_available():
            peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)
            logger.info(f"Peak VRAM chiếm dụng: {peak_vram_gb} GB / 15.0 GB")

        # 3. Vẽ biểu đồ Loss curve
        loss_curve_path = self._plot_and_save_metrics(
            log_history=trainer.state.log_history,
            output_dir=Path(output_dir),
            drive_backup_dir=Path(drive_backup) if drive_backup else None,
        )

        # 4. Lưu adapter cuối cùng
        final_adapter_dir = Path(output_dir) / "final_adapter"
        self.model.save_pretrained(str(final_adapter_dir))
        logger.info(f"Adapter cuối cùng đã lưu tại: {final_adapter_dir}")

        return {
            "train_loss": train_result.training_loss,
            "train_runtime_min": round(total_time / 60, 2),
            "peak_vram_gb": peak_vram_gb,
            "final_adapter_dir": str(final_adapter_dir),
            "loss_curve_path": str(loss_curve_path) if loss_curve_path else None,
            "total_steps": trainer.state.global_step,
        }

    def _plot_and_save_metrics(
        self,
        log_history: list,
        output_dir: Path,
        drive_backup_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Vẽ biểu đồ train/eval loss và lưu dưới dạng PNG."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib chưa cài, bỏ qua vẽ biểu đồ loss.")
            return None

        train_steps, train_losses = [], []
        eval_steps, eval_losses = [], []

        for entry in log_history:
            if "loss" in entry and "step" in entry:
                train_steps.append(entry["step"])
                train_losses.append(entry["loss"])
            if "eval_loss" in entry and "step" in entry:
                eval_steps.append(entry["step"])
                eval_losses.append(entry["eval_loss"])

        if not train_steps:
            logger.info("Không có dữ liệu loss để vẽ biểu đồ.")
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_steps, train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
        if eval_steps:
            ax.plot(eval_steps, eval_losses, label="Eval Loss", color="#FF5722",
                    linewidth=1.5, linestyle="--", marker="o", markersize=4)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("QLoRA Fine-tuning Loss Curve (Qwen2.5-VL @ Colab T4)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "qlora_loss_curve.png"
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info(f"Biểu đồ loss curve đã lưu: {save_path}")

        # Sao lưu lên Drive nếu có đường dẫn
        if drive_backup_dir:
            try:
                import shutil
                drive_fig_dir = drive_backup_dir / "figures"
                drive_fig_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(save_path), str(drive_fig_dir / "qlora_loss_curve.png"))
                logger.info(f"Loss curve đã sao lưu lên Drive: {drive_fig_dir}")
            except Exception as e:
                logger.warning(f"Không thể sao lưu biểu đồ lên Drive: {e}")

        return save_path
