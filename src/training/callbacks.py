"""Training loops and callbacks for Financial VLM Research."""
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from transformers import TrainerCallback
except ImportError:
    # Dummy fallback nếu môi trường local chưa cài transformers
    class TrainerCallback:
        pass


class DriveSyncCallback(TrainerCallback):
    """
    Tự động đồng bộ checkpoint sang thư mục Google Drive để chống mất dữ liệu khi Colab timeout.
    Triển khai cơ chế copy an toàn các tệp tin Adapter và Trainer State sang Drive.
    """

    def __init__(
        self,
        drive_backup_dir: str = "/content/drive/MyDrive/NCKH_AI/2_checkpoints",
        sync_final_model: bool = True,
    ):
        self.drive_backup_dir = Path(drive_backup_dir)
        self.sync_final_model = sync_final_model
        self.synced_checkpoints = []

    def _sync_directory(self, src_dir: Path, dst_dir: Path) -> bool:
        """Thực hiện sao lưu an toàn thư mục từ local sang Drive."""
        try:
            dst_dir.parent.mkdir(parents=True, exist_ok=True)
            # Copy toàn bộ nội dung checkpoint
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            logger.info(f"[DriveSync] Sao lưu thành công: {src_dir.name} -> {dst_dir}")
            return True
        except Exception as e:
            logger.warning(f"[DriveSync] Lỗi sao lưu sang Drive (không làm dừng quá trình train): {e}")
            return False

    def on_save(self, args, state, control, **kwargs):
        """Được gọi tự động mỗi khi Trainer thực hiện lưu Checkpoint."""
        step = getattr(state, "global_step", 0)
        output_dir = Path(args.output_dir) if hasattr(args, "output_dir") else Path("./checkpoints")
        checkpoint_dir = output_dir / f"checkpoint-{step}"

        if not checkpoint_dir.exists():
            # Tìm thư mục checkpoint gần nhất
            existing = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
            if existing:
                checkpoint_dir = existing[-1]

        if checkpoint_dir.exists():
            target_dir = self.drive_backup_dir / checkpoint_dir.name
            logger.info(f"[DriveSync] Bắt đầu đồng bộ checkpoint bước {step} sang {target_dir}...")
            success = self._sync_directory(checkpoint_dir, target_dir)
            if success:
                self.synced_checkpoints.append(str(target_dir))
        else:
            logger.warning(f"[DriveSync] Không tìm thấy thư mục checkpoint tại {checkpoint_dir}")

    def on_train_end(self, args, state, control, **kwargs):
        """Được gọi khi toàn bộ quá trình huấn luyện hoàn tất."""
        if not self.sync_final_model:
            return

        output_dir = Path(args.output_dir) if hasattr(args, "output_dir") else Path("./checkpoints")
        if output_dir.exists():
            target_final = self.drive_backup_dir / "final_model"
            logger.info(f"[DriveSync] Đang sao lưu mô hình hoàn chỉnh sau train sang {target_final}...")
            self._sync_directory(output_dir, target_final)

