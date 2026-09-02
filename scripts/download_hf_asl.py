from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

DATASET_ID = 'Marxulia/asl_sign_languages_alphabets_v03'


def main() -> None:
    parser = argparse.ArgumentParser(description='Download a public ASL alphabet dataset into labeled image folders.')
    parser.add_argument('--output', default='data/online_asl', help='Output directory for labeled images.')
    parser.add_argument('--max-per-class', type=int, default=0, help='Optional limit per class; 0 downloads every image.')
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_ID, split='train')
    counts: dict[str, int] = {}

    for row in dataset:
        label = dataset.features['label'].int2str(row['label'])
        if args.max_per_class and counts.get(label, 0) >= args.max_per_class:
            continue
        class_dir = output_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        image_number = counts.get(label, 0) + 1
        row['image'].convert('RGB').save(class_dir / f'{image_number:05d}.jpg', quality=95)
        counts[label] = image_number

    print(f'Downloaded {sum(counts.values())} images into {output_root}')
    print('Classes:', ', '.join(sorted(counts)))


if __name__ == '__main__':
    main()
