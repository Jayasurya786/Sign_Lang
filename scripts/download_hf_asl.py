#!/usr/bin/env python3
"""Download public ASL alphabet datasets from Hugging Face into labeled image folders.

Optimized with in-memory sample indexing, real-time progress logging,
and optional automated landmark extraction.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from datasets import load_dataset
except ImportError:
    print("Error: The 'datasets' library is required to download ASL datasets.")
    print("Please install it with: pip install datasets")
    sys.exit(1)

DEFAULT_DATASETS = [
    'Marxulia/asl_sign_languages_alphabets_v03',
    'Marxulia/asl_sign_languages_alphabets_v02',
]


def download_dataset(dataset_id: str, output_root: Path, max_per_class: int = 0) -> dict[str, int]:
    print(f"\n[+] Connecting to Hugging Face Hub: '{dataset_id}'...")
    start_time = time.time()
    try:
        dataset = load_dataset(dataset_id, split='train')
    except Exception as exc:
        print(f"[-] Failed to load '{dataset_id}': {exc}")
        print("    Check your internet connection or verify Hugging Face access.")
        return {}

    total_rows = len(dataset)
    print(f"[+] Loaded '{dataset_id}' with {total_rows:,} total images.")

    # Pre-index existing image counts per folder to eliminate repeated disk globs
    existing_counts: dict[str, int] = {}
    for letter_dir in output_root.glob('*'):
        if letter_dir.is_dir() and len(letter_dir.name) == 1 and letter_dir.name.isalpha():
            existing_counts[letter_dir.name.upper()] = len(list(letter_dir.glob('*.jpg')))

    added_counts: dict[str, int] = {}
    processed_count = 0

    print("[+] Extracting and saving images to disk...")
    for idx, row in enumerate(dataset):
        processed_count += 1
        if 'label' not in row or 'image' not in row:
            continue

        raw_label = row['label']
        if hasattr(dataset.features['label'], 'int2str'):
            label = dataset.features['label'].int2str(raw_label).upper()
        else:
            label = str(raw_label).strip().upper()

        if not label or len(label) != 1 or not label.isalpha():
            continue

        class_dir = output_root / label
        class_dir.mkdir(parents=True, exist_ok=True)

        current_count = existing_counts.get(label, 0)
        if max_per_class and current_count >= max_per_class:
            continue

        image_number = current_count + 1
        img_path = class_dir / f"{image_number:05d}.jpg"

        try:
            row['image'].convert('RGB').save(img_path, quality=95)
            existing_counts[label] = image_number
            added_counts[label] = added_counts.get(label, 0) + 1
        except Exception as img_exc:
            print(f"Warning: could not save image {img_path}: {img_exc}")

        # Progress reporting every 1,000 images
        if processed_count % 1000 == 0 or processed_count == total_rows:
            elapsed = time.time() - start_time
            rate = processed_count / max(0.1, elapsed)
            print(f"    Progress: {processed_count:,}/{total_rows:,} ({processed_count/total_rows*100:.1f}%) | {rate:.0f} img/s")

    elapsed_total = time.time() - start_time
    total_new = sum(added_counts.values())
    print(f"[OK] Finished '{dataset_id}': added {total_new:,} images in {elapsed_total:.1f}s.")
    return added_counts


def run_landmark_extraction(image_root: Path, output_csv: Path) -> None:
    print("\n[+] Triggering MediaPipe landmark extraction on newly downloaded images...")
    try:
        from scripts.build_combined_dataset import extract_from_folders, merge_archive_landmarks, CACHE_VERSION
        from src.data.landmarks import LandmarkExtractor
        import json

        extractor = LandmarkExtractor(static_image_mode=True, min_detection_confidence=0.30)
        archive_csv = Path('data/processed/archive_image_landmarks.csv')
        df_extracted = extract_from_folders(image_root, extractor, max_per_class=350)
        print(f"[+] Extracted {len(df_extracted)} landmarks from image folders.")

        combined_df = merge_archive_landmarks(df_extracted, archive_csv)
        print(f"[+] Combined dataset: {len(combined_df)} samples across {combined_df['label'].nunique()} classes.")

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_csv, index=False)
        meta_path = output_csv.with_suffix('.meta.json')
        meta_path.write_text(json.dumps({'version': CACHE_VERSION, 'num_samples': len(combined_df)}, indent=2), encoding='utf-8')
        print(f"[OK] Successfully built and cached landmarks at: {output_csv}")
    except Exception as exc:
        print(f"[-] Automated landmark extraction failed: {exc}")
        print("    You can run it manually with: python scripts/build_combined_dataset.py")


def main() -> None:
    parser = argparse.ArgumentParser(description='Download public ASL alphabet datasets into labeled image folders.')
    parser.add_argument('--output', default='data/online_asl', help='Output directory for labeled images (default: data/online_asl).')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS, help='HuggingFace dataset IDs to download.')
    parser.add_argument('--max-per-class', type=int, default=0, help='Optional limit per class (0 downloads all available images).')
    parser.add_argument('--build-landmarks', action='store_true', help='Automatically run landmark extraction after downloading.')
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("      ASL DATASET DOWNLOADER (HUGGING FACE HUB)")
    print("=" * 68)
    print(f"Target Directory: {output_root.resolve()}")
    print(f"Datasets:         {', '.join(args.datasets)}")
    print(f"Limit per class:  {args.max_per_class if args.max_per_class > 0 else 'Unlimited (All images)'}")
    print("=" * 68)

    total_added: dict[str, int] = {}
    for ds_id in args.datasets:
        added = download_dataset(ds_id, output_root, max_per_class=args.max_per_class)
        for k, v in added.items():
            total_added[k] = total_added.get(k, 0) + v

    print("\n" + "=" * 68)
    print(f"[SUMMARY] Total new images saved: {sum(total_added.values()):,}")
    print("=" * 68)

    if args.build_landmarks:
        output_csv = Path('data/processed/online_asl_landmarks.csv')
        run_landmark_extraction(output_root, output_csv)

    print("\n[OK] Dataset download complete.")
    print("     To audit balance:     python scripts/check_dataset_balance.py")
    print("     To extract landmarks: python scripts/build_combined_dataset.py")
    print("     To train model:       python scripts/train_model.py\n")


if __name__ == '__main__':
    main()
