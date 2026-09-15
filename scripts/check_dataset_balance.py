#!/usr/bin/env python3
"""Audit a Sign Language dataset for class balance and completeness.

This script checks whether the dataset folders for ASL letters A-Z are present,
contain enough samples, and are balanced enough for model training.

Example:
    python scripts/check_dataset_balance.py --root data/online_asl --min-per-class 300
    python scripts/check_dataset_balance.py --root data/online_asl --min-per-class 300 --init
"""

from __future__ import annotations

import argparse
from pathlib import Path

ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def count_files(folder: Path) -> int:
    if not folder.exists() or not folder.is_dir():
        return 0
    return sum(
        1
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_letter_dirs(root: Path) -> list[str]:
    created: list[str] = []
    for letter in ALPHABET:
        folder = root / letter
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            created.append(letter)
    return created


def analyze_dataset(root: Path, min_per_class: int) -> tuple[dict[str, int], list[str], list[str]]:
    counts: dict[str, int] = {}
    for letter in ALPHABET:
        counts[letter] = count_files(root / letter)

    missing = [letter for letter, count in counts.items() if count == 0]
    underfilled = [letter for letter, count in counts.items() if 0 < count < min_per_class]
    return counts, missing, underfilled


def format_report(counts: dict[str, int], missing: list[str], underfilled: list[str], min_per_class: int) -> str:
    lines: list[str] = []
    lines.append(f"Dataset root: {counts}")
    lines.append(f"Minimum samples per class: {min_per_class}")
    lines.append("")
    lines.append("Class counts:")
    for letter in ALPHABET:
        lines.append(f"  {letter}: {counts[letter]}")

    lines.append("")
    if missing:
        lines.append(f"Missing folders: {', '.join(missing)}")
    else:
        lines.append("Missing folders: none")

    if underfilled:
        lines.append(f"Underfilled classes (< {min_per_class}): {', '.join(underfilled)}")
    else:
        lines.append(f"Underfilled classes (< {min_per_class}): none")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ASL dataset balance and completeness.")
    parser.add_argument("--root", type=Path, default=Path("data/online_asl"), help="Dataset root (e.g. data/online_asl)")
    parser.add_argument("--min-per-class", type=int, default=300, help="Minimum recommended sample count per class.")
    parser.add_argument("--init", action="store_true", help="Create missing A-Z directories under the dataset root.")
    args = parser.parse_args()

    if args.init:
        created = ensure_letter_dirs(args.root)
        if created:
            print(f"Created missing folders: {', '.join(created)}")
        else:
            print("All ASL letter folders already exist.")

    counts, missing, underfilled = analyze_dataset(args.root, args.min_per_class)
    print(format_report(counts, missing, underfilled, args.min_per_class))

    if missing or underfilled:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
