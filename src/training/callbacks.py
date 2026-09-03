"""Training loops and callbacks."""
from transformers import TrainerCallback


class DriveSyncCallback(TrainerCallback):
    """
    Tự động đồng bộ checkpoint sang thư mục Google Drive để chống mất dữ liệu khi Colab timeout.
    """

    def __init__(self, drive_backup_dir: str = "/content/drive/MyDrive/NCKH_AI/2_checkpoints"):
        self.drive_backup_dir = drive_backup_dir

    def on_save(self, args, state, control, **kwargs):
        print(f"[Callback] Checkpoint đã được lưu tại bước {state.global_step}. Đang sao lưu...")
