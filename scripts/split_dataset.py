"""CLI entry point for Time-series dataset splitting.

Usage:
    python scripts/split_dataset.py --manifest data/raw/manifest.csv --config configs/dataset_config.yaml --output-dir data/splits
    # Or on Google Colab:
    python scripts/split_dataset.py --manifest /content/drive/MyDrive/NCKH_AI/1_datasets/raw_images/manifest.csv --output-dir /content/drive/MyDrive/NCKH_AI/1_datasets/splits
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.splitter import split_manifest


def resolve_default_paths():
    """Auto-detect Colab Google Drive paths vs local repository paths."""
    drive_manifest = Path("/content/drive/MyDrive/NCKH_AI/1_datasets/raw_images/manifest.csv")
    drive_output = Path("/content/drive/MyDrive/NCKH_AI/1_datasets/splits")

    if drive_manifest.exists():
        return str(drive_manifest), str(drive_output)
    return "data/raw/manifest.csv", "data/splits"


def main():
    default_manifest, default_output = resolve_default_paths()

    parser = argparse.ArgumentParser(description="Strict Time-Series Dataset Splitter for Financial Charts")
    parser.add_argument("--manifest", type=str, default=default_manifest, help="Path to manifest.csv")
    parser.add_argument("--config", type=str, default="configs/dataset_config.yaml", help="Path to dataset_config.yaml")
    parser.add_argument("--output-dir", type=str, default=default_output, help="Target output directory for splits (.jsonl)")
    parser.add_argument("--global-sort", action="store_true", help="Perform global sort instead of per-asset grouping")
    args = parser.parse_args()

    print("=" * 70)
    print(" FINANCIAL VLM: TIME-SERIES DATASET SPLITTER (PHASE 1 - STEP 2.3)")
    print("=" * 70)
    print(f"Manifest source : {args.manifest}")
    print(f"Configuration   : {args.config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Grouping mode   : {'Global chronological' if args.global_sort else 'Per-asset chronological'}")
    print("-" * 70)

    try:
        stats = split_manifest(
            manifest_path=args.manifest,
            config_path=args.config,
            output_dir=args.output_dir,
            group_by_asset=not args.global_sort
        )
    except FileNotFoundError as fnf_err:
        print(f"\n[ERROR] {fnf_err}")
        print("Please check your manifest path or run scripts/generate_charts.py first.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Splitting failed: {e}")
        sys.exit(1)

    print("\n[RESULT] Split execution completed successfully!")
    print(f"  • Total samples   : {stats['total_manifest_samples']}")
    print(f"  • Train set       : {stats['train_samples']} ({stats['train_pct']}%) | Range: {stats['train_date_range']}")
    print(f"  • Val set         : {stats['val_samples']} ({stats['val_pct']}%) | Range: {stats['val_date_range']}")
    print(f"  • Test set        : {stats['test_samples']} ({stats['test_pct']}%) | Range: {stats['test_date_range']}")
    print(f"  • Purged (embargo): {stats['purged_samples']} ({stats['purged_pct']}%)")
    print("-" * 70)
    print("Generated files:")
    for split_name, path in stats["output_files"].items():
        print(f"  - {split_name.upper():<5}: {path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
