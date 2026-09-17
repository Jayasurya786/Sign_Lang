#!/usr/bin/env python3
"""Convenient root-level dataset downloader and setup utility.

Usage:
    python download.py                 # Download full multi-source dataset (default)
    python download.py --quick         # Fast download (cap at 60 images/class for testing)
    python download.py --extract       # Download images AND extract MediaPipe landmarks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.download_hf_asl import DEFAULT_DATASETS, download_dataset, run_landmark_extraction


def main():
    parser = argparse.ArgumentParser(
        description="One-step ASL dataset downloader and setup utility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download.py                     # Standard download
    python download.py --quick             # Download 60 images per class for fast testing
    python download.py --extract           # Download images and build processed landmarks
    python download.py --max-per-class 150 # Custom limit per class
        """,
    )
    parser.add_argument(
        "--output",
        default="data/online_asl",
        help="Target folder for image dataset (default: data/online_asl)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup: download at most 60 images per letter class",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Maximum images per class (0 for all available)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Automatically extract landmarks and build master CSV after downloading",
    )
    args = parser.parse_args()

    max_per_class = 60 if args.quick else args.max_per_class
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("           AMERICAN SIGN LANGUAGE (ASL) DATASET SETUP")
    print("=" * 70)
    print(f"Target Output Folder:  {output_root.resolve()}")
    print(f"Mode:                  {'Quick (60 imgs/class)' if args.quick else ('Capped at ' + str(max_per_class) if max_per_class > 0 else 'Full Dataset (~11,000 images)')}")
    print("=" * 70)

    # Check status of preprocessed landmark files
    processed_csv = Path("data/processed/online_asl_landmarks.csv")
    if processed_csv.exists() and processed_csv.stat().st_size > 1000:
        print(f"[NOTE] Found bundled preprocessed landmark dataset ({processed_csv.stat().st_size / 1e6:.1f} MB).")
        print("       You can train models or run predictions immediately!")
    else:
        print("[INFO] Processed landmark dataset not found; it will be built during extraction.")

    print("\nStarting image downloads from Hugging Face...")
    total_added: dict[str, int] = {}
    for ds_id in DEFAULT_DATASETS:
        added = download_dataset(ds_id, output_root, max_per_class=max_per_class)
        for k, v in added.items():
            total_added[k] = total_added.get(k, 0) + v

    total_images = sum(total_added.values())
    print("\n" + "=" * 70)
    print(f"[SUMMARY] Successfully saved {total_images:,} images into {output_root}")
    print("=" * 70)

    if args.extract:
        run_landmark_extraction(output_root, processed_csv)

    print("\n[SUCCESS] ASL dataset is ready!")
    print("Next steps:")
    print("  1. Launch web application:  python app.py")
    print("  2. Audit dataset balance:   python scripts/check_dataset_balance.py")
    print("  3. Benchmark test images:   python scripts/evaluate_all_classes.py")
    print("  4. Retrain model:           python scripts/train_model.py\n")


if __name__ == "__main__":
    main()
