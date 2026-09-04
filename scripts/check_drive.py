#!/usr/bin/env python3
"""
Diagnostic Script for Google Drive (NCKH_AI) Environment & Dataset Verification.
Chẩn đoán và phát hiện các bất thường trên Google Drive / Colab.
Usage:
    python scripts/check_drive.py
"""
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

POSSIBLE_DRIVE_ROOTS = [
    Path("/content/drive/MyDrive/NCKH_AI"),
    Path("/content/drive/My Drive/NCKH_AI"),
    Path("E:/financial-vlm-research/data"),
    Path("./data"),
]


def check_google_drive():
    print("=" * 70)
    print(" KIỂM TRA CHẨN ĐOÁN TOÀN DIỆN GOOGLE DRIVE (NCKH_AI)")
    print("=" * 70)

    # 1. Kiểm tra trạng thái mount Drive
    colab_drive_parent = Path("/content/drive")
    is_colab = Path("/content").exists()
    print(f"[*] Môi trường Colab phát hiện : {'CÓ (Google Colab)' if is_colab else 'KHÔNG (Máy Local / Khác)'}")

    if is_colab:
        if not colab_drive_parent.exists() or not list(colab_drive_parent.glob("*")):
            print("❌ [LỖI NGHIÊM TRỌNG] Google Drive chưa được mount!")
            print("   -> Cách khắc phục: Hãy chạy lệnh sau trong notebook:")
            print("      from google.colab import drive")
            print("      drive.mount('/content/drive')")
            return
        else:
            print("✔ Google Drive đã được mount tại /content/drive")

    # 2. Tìm thư mục gốc NCKH_AI
    active_root = None
    for r in POSSIBLE_DRIVE_ROOTS:
        if r.exists():
            active_root = r
            break

    if not active_root:
        print("❌ [BẤT THƯỜNG] Không tìm thấy thư mục NCKH_AI trên Drive!")
        print("   Các đường dẫn đã tìm kiếm:")
        for r in POSSIBLE_DRIVE_ROOTS:
            print(f"     - {r} (Tồn tại: {r.exists()})")
        print("\n   -> Hãy chạy: python scripts/setup_drive.py để khởi tạo cấu trúc.")
        return

    print(f"✔ Thư mục gốc hoạt động: {active_root}")
    print("-" * 70)

    # 3. Quét cấu trúc các thư mục con
    expected_subdirs = [
        "1_datasets",
        "1_datasets/raw_images",
        "1_datasets/splits",
        "2_checkpoints",
        "3_experiment_outputs",
        "4_backups",
    ]

    print("[*] Kiểm tra các thư mục con:")
    for sub in expected_subdirs:
        p = active_root / sub
        exists = p.exists()
        count_str = ""
        if exists:
            try:
                items = list(p.glob("*"))
                count_str = f"({len(items)} mục)"
            except Exception:
                pass
        status = "✔" if exists else "⚠️ CHƯA CÓ"
        print(f"   {status} {sub:<25} {count_str}")

    print("-" * 70)

    # 4. Kiểm tra các tệp tin dữ liệu quan trọng
    raw_images_dir = active_root / "1_datasets" / "raw_images"
    splits_dir = active_root / "1_datasets" / "splits"

    print("[*] Kiểm tra các tệp tin then chốt:")
    key_files = {
        "manifest.csv": raw_images_dir / "manifest.csv",
        "train.jsonl": splits_dir / "train.jsonl",
        "val.jsonl": splits_dir / "val.jsonl",
        "test.jsonl": splits_dir / "test.jsonl",
        "train_cot.jsonl": splits_dir / "train_cot.jsonl",
        "train_labeled.jsonl": splits_dir / "train_labeled.jsonl",
        "split_summary.json": splits_dir / "split_summary.json",
    }

    found_files = {}
    for name, fpath in key_files.items():
        exists = fpath.exists()
        found_files[name] = exists
        size_str = f"({fpath.stat().st_size / 1024:.1f} KB)" if exists else ""
        print(f"   {'✔' if exists else '❌'} {name:<22} : {'TỒN TẠI ' + size_str if exists else 'KHÔNG TÌM THẤY'}")

    print("-" * 70)

    # 5. Kiểm tra tính toàn vẹn của tệp Train JSONL và ảnh
    train_jsonl = None
    if (splits_dir / "train_cot.jsonl").exists():
        train_jsonl = splits_dir / "train_cot.jsonl"
    elif (splits_dir / "train_labeled.jsonl").exists():
        train_jsonl = splits_dir / "train_labeled.jsonl"
    elif (splits_dir / "train.jsonl").exists():
        train_jsonl = splits_dir / "train.jsonl"

    if train_jsonl:
        print(f"[*] Đối soát tệp Train [{train_jsonl.name}]:")
        records = []
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except Exception as je:
                        print(f"   ⚠️ Lỗi parse dòng JSON: {je}")

        print(f"   • Số bản ghi đọc được : {len(records)}")
        has_cot = sum(1 for r in records if r.get("cot_reasoning"))
        has_action = sum(1 for r in records if r.get("action"))
        print(f"   • Số bản ghi có CoT   : {has_cot}/{len(records)}")
        print(f"   • Số bản ghi có Action: {has_action}/{len(records)}")

        # Kiểm tra sự tồn tại của ảnh trong 3 bản ghi đầu
        print("   • Kiểm tra liên kết ảnh thực tế (Image Linkage):")
        missing_images = 0
        for i, r in enumerate(records[:5]):
            img_rel = r.get("image_path", "")
            # Thử các đường dẫn ảnh
            candidate_paths = [
                Path(img_rel),
                raw_images_dir / img_rel,
                raw_images_dir / Path(img_rel).name,
                active_root / "1_datasets" / img_rel,
            ]
            found = any(cp.exists() for cp in candidate_paths)
            if not found:
                missing_images += 1
            print(f"     [{i+1}] {r.get('id')}: {'✔ Tìm thấy ảnh' if found else '❌ KHÔNG TÌM THẤY ẢNH: ' + img_rel}")

        if missing_images > 0:
            print("\n   ⚠️ [CẢNH BÁO BẤT THƯỜNG] Ảnh trong JSONL không khớp đường dẫn trên Drive!")
            print(f"      Ảnh đang lưu tại: {raw_images_dir}")
            print(f"      Đường dẫn trong JSONL: {records[0].get('image_path')}")
            print("      -> Cần cấu hình image_root trỏ đúng vào raw_images.")
    else:
        print("❌ Chưa có tệp train.jsonl hoặc train_cot.jsonl trong splits/")

    print("=" * 70)
    print(" CHẨN ĐOÁN HOÀN TẤT!")
    print("=" * 70)


if __name__ == "__main__":
    check_google_drive()
