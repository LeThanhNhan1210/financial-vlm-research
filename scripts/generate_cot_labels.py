#!/usr/bin/env python3
"""
CLI Runner for Generating Financial Chain-of-Thought (CoT) Labels (Phase 1 - Step 2.4).
Usage:
    python scripts/generate_cot_labels.py \
        --input-file data/splits/train.jsonl \
        --output-file data/splits/train_labeled.jsonl \
        --provider mock \
        --limit 10 \
        --audit-export data/annotations/audit_samples.csv
"""
import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.cot_generator import CoTLabelGenerator
from src.pipeline.audit_sampler import AuditSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 4-step CMT CoT reasoning labels for financial chart datasets."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to the input split JSONL file (e.g., train.jsonl).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to output labeled JSONL file. Defaults to <input_name>_labeled.jsonl.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "openai"],
        help="Provider for CoT generation: 'openai' (GPT-4o) or 'mock' (offline/synthetic).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name for API provider (default: 'gpt-4o').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enable dry-run mode (forces provider='mock' and prints preview).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume; start generating from the first record.",
    )
    parser.add_argument(
        "--audit-export",
        type=str,
        default=None,
        help="Optional path to export 30%% HITL stratified audit CSV file.",
    )
    parser.add_argument(
        "--audit-rate",
        type=float,
        default=0.30,
        help="Audit sampling rate (default: 0.30 = 30%%).",
    )

    parser.add_argument(
        "--image-root",
        type=str,
        default="",
        help="Root directory where chart images are stored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    inp_path = Path(args.input_file)
    if not inp_path.exists():
        logger.error(f"Input file not found: {inp_path}")
        sys.exit(1)

    if args.output_file is None:
        out_path = inp_path.parent / f"{inp_path.stem}_labeled.jsonl"
    else:
        out_path = Path(args.output_file)

    provider = "mock" if args.dry_run else args.provider
    resume = not args.no_resume

    logger.info("=" * 65)
    logger.info("FINANCIAL VLM: CoT LABEL GENERATION (PHASE 1 - STEP 2.4)")
    logger.info("=" * 65)
    logger.info(f"Input file    : {inp_path}")
    logger.info(f"Output file   : {out_path}")
    logger.info(f"Provider      : {provider.upper()} (Model: {args.model})")
    logger.info(f"Dry-run mode  : {args.dry_run}")
    logger.info(f"Resume active : {resume}")
    logger.info(f"Sample limit  : {args.limit or 'ALL'}")
    logger.info("-" * 65)

    # Khởi tạo Generator
    generator = CoTLabelGenerator(
        provider=provider,
        model_name=args.model,
        image_root_dir=args.image_root,
    )

    # Chạy xử lý
    labeled_records = generator.process_dataset(
        input_jsonl_path=inp_path,
        output_jsonl_path=out_path,
        limit=args.limit,
        resume=resume,
    )

    # Lấy mẫu HITL audit nếu được yêu cầu
    if args.audit_export and labeled_records:
        sampler = AuditSampler(sample_rate=args.audit_rate)
        audit_csv = sampler.export_audit_csv(labeled_records, args.audit_export)
        logger.info(f"Exported HITL audit samples ({args.audit_rate*100:.0f}%) to: {audit_csv}")

    logger.info("=" * 65)
    logger.info(f"[SUCCESS] CoT labeling completed for {len(labeled_records)} records!")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
