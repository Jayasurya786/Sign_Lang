from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

DEFAULT_DATASETS = [
    'Marxulia/asl_sign_languages_alphabets_v03',
    'Marxulia/asl_sign_languages_alphabets_v02',
]


def download_dataset(dataset_id: str, output_root: Path, max_per_class: int = 0) -> dict[str, int]:
    print(f'Loading dataset: {dataset_id}...')
    dataset = load_dataset(dataset_id, split='train')
    counts: dict[str, int] = {}

    for row in dataset:
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

        existing_count = sum(1 for f in class_dir.glob('*.jpg'))
        if max_per_class and existing_count >= max_per_class:
            continue

        image_number = existing_count + 1
        img_path = class_dir / f'{image_number:05d}.jpg'
        row['image'].convert('RGB').save(img_path, quality=95)
        counts[label] = counts.get(label, 0) + 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description='Download public ASL alphabet datasets into labeled image folders.')
    parser.add_argument('--output', default='data/online_asl', help='Output directory for labeled images.')
    parser.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS, help='HuggingFace dataset IDs to download.')
    parser.add_argument('--max-per-class', type=int, default=0, help='Optional limit per class; 0 downloads every image.')
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    total_added: dict[str, int] = {}
    for ds_id in args.datasets:
        try:
            added = download_dataset(ds_id, output_root, max_per_class=args.max_per_class)
            for k, v in added.items():
                total_added[k] = total_added.get(k, 0) + v
            print(f'Done with {ds_id}: added {sum(added.values())} images.')
        except Exception as exc:
            print(f'Failed to download {ds_id}: {exc}')

    print(f'Total new images added: {sum(total_added.values())} into {output_root}')


if __name__ == '__main__':
    main()

