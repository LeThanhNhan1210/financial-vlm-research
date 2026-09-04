"""CLI entry point for chart generation pipeline.
Usage:
    python scripts/generate_charts.py --config configs/dataset_config.yaml --output-dir data/raw/
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.chart_generator import generate_all


def main():
    parser = argparse.ArgumentParser(description="Financial Candlestick Chart Generator")
    parser.add_argument("--config", type=str, default="configs/dataset_config.yaml", help="Path to dataset config YAML")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target output directory for rendered charts and manifest")
    args = parser.parse_args()

    manifest_path = generate_all(args.config, args.output_dir)
    print(f"\n[DONE] Successfully generated charts. Manifest located at: {manifest_path}")


if __name__ == "__main__":
    main()
