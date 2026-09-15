#!/usr/bin/env python3
"""Build a unified, rich hand-landmark dataset for Sign Language Recognition.

This script extracts MediaPipe hand landmarks from image folders (e.g. data/online_asl),
combines them with existing supplementary archive landmarks, deduplicates/cleans,
and updates the landmark cache metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import pandas as pd

from src.data.dataset_config import ASL_FULL_CLASS_LIST, normalize_archive_label_name
from src.data.landmarks import LandmarkExtractor

CACHE_VERSION = 'v3-augmented-scale'


def extract_from_folders(
    root: Path,
    extractor: LandmarkExtractor,
    max_per_class: int = 350,
) -> pd.DataFrame:
    rows = []
    print(f'Extracting landmarks from image folders in {root} (up to {max_per_class} per class)...')

    for letter in sorted(ASL_FULL_CLASS_LIST):
        letter_dir = root / letter
        if not letter_dir.exists():
            continue

        images = sorted(letter_dir.glob('*.jpg')) + sorted(letter_dir.glob('*.png')) + sorted(letter_dir.glob('*.jpeg'))
        if max_per_class > 0:
            images = images[:max_per_class]

        extracted = 0
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            vector = extractor.extract_landmarks_from_frame(frame)
            if vector is None:
                continue

            record = {'label': letter}
            for i, val in enumerate(vector):
                record[f'f{i}'] = float(val)
            rows.append(record)
            extracted += 1

        print(f'  Class {letter}: extracted {extracted}/{len(images)} valid landmark sets.')

    return pd.DataFrame(rows)


def merge_archive_landmarks(df_extracted: pd.DataFrame, archive_csv: Path) -> pd.DataFrame:
    if not archive_csv.exists():
        return df_extracted

    try:
        archive_df = pd.read_csv(archive_csv)
        if 'label' not in archive_df.columns:
            return df_extracted

        archive_df['label'] = archive_df['label'].map(normalize_archive_label_name)
        archive_df = archive_df[archive_df['label'].isin(set(ASL_FULL_CLASS_LIST))]

        print(f'Merging {len(archive_df)} archive landmark samples from {archive_csv}...')
        combined = pd.concat([df_extracted, archive_df], ignore_index=True)
        return combined
    except Exception as exc:
        print(f'Warning: could not merge archive csv: {exc}')
        return df_extracted


def main() -> None:
    parser = argparse.ArgumentParser(description='Build combined ASL landmarks dataset.')
    parser.add_argument('--input', type=Path, default=Path('data/online_asl'), help='Root directory with letter subfolders.')
    parser.add_argument('--archive-csv', type=Path, default=Path('data/processed/archive_image_landmarks.csv'), help='Optional archive CSV.')
    parser.add_argument('--output-csv', type=Path, default=Path('data/processed/online_asl_landmarks.csv'), help='Output CSV path.')
    parser.add_argument('--max-per-class', type=int, default=320, help='Maximum images to process per class (0 for all).')
    args = parser.parse_args()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    extractor = LandmarkExtractor(static_image_mode=True, min_detection_confidence=0.30)

    df_extracted = extract_from_folders(args.input, extractor, max_per_class=args.max_per_class)
    print(f'Extracted {len(df_extracted)} samples from image folders.')

    combined_df = merge_archive_landmarks(df_extracted, args.archive_csv)
    print(f'Total combined dataset size: {len(combined_df)} samples across {combined_df["label"].nunique()} classes.')

    combined_df.to_csv(args.output_csv, index=False)
    meta_path = args.output_csv.with_suffix('.meta.json')
    meta_path.write_text(json.dumps({'version': CACHE_VERSION, 'num_samples': len(combined_df)}, indent=2), encoding='utf-8')
    print(f'Successfully saved dataset to {args.output_csv} with cache version {CACHE_VERSION}')


if __name__ == '__main__':
    main()

