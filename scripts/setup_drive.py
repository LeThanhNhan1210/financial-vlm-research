"""Script tự động tạo và kiểm tra cấu trúc thư mục trên Google Drive (NCKH_AI)."""
import os
from pathlib import Path


def setup_drive_directories(drive_root: str = "/content/drive/MyDrive/NCKH_AI"):
    """
    Tự động khởi tạo toàn bộ các thư mục hiện vật bên trong thư mục NCKH_AI trên Google Drive.
    """
    folders = [
        f"{drive_root}/1_datasets/raw_images/vn30_daily",
        f"{drive_root}/1_datasets/raw_images/sp500_daily",
        f"{drive_root}/1_datasets/raw_images/crypto_4h",
        f"{drive_root}/1_datasets/processed_images",
        f"{drive_root}/1_datasets/dataset_archives",
        f"{drive_root}/1_datasets/splits",
        f"{drive_root}/2_checkpoints/best_model_adapter",

        f"{drive_root}/3_experiment_outputs/predictions",
        f"{drive_root}/3_experiment_outputs/evaluations",
        f"{drive_root}/3_experiment_outputs/figures",
        f"{drive_root}/3_experiment_outputs/logs",
        f"{drive_root}/4_backups",
    ]

    print(f"[*] Bắt đầu khởi tạo các thư mục lưu trữ bên trong: {drive_root}")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  ✔ {folder}")

    print("\n🎉 Đã cấu hình hoàn tất thư mục NCKH_AI trên Google Drive!")


if __name__ == "__main__":
    setup_drive_directories()
