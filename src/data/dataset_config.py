from __future__ import annotations

from typing import Iterable


# Canonical labels provided by the online ASL alphabet dataset.
ASL_FULL_CLASS_LIST = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


def normalize_label_name(value) -> str:
    if value is None:
        return ''

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip().upper()

    text = str(value).strip()
    if not text:
        return ''

    if text.lower().startswith('label_'):
        text = text.split('_', 1)[1]

    return text.upper()


def normalize_archive_label_name(value) -> str:
    if value is None:
        return ''

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value).is_integer():
            return str(int(value))
        return str(value).strip().upper()

    text = str(value).strip()
    if not text:
        return ''

    if text.lower().startswith('label_'):
        text = text.split('_', 1)[1]

    text = text.upper()
    if text in {'', '_', '-'}:
        return ''

    if text.isalpha() or text.isdigit():
        return text

    return ''


def validate_dataset_classes(dataset_classes: Iterable[str], target_classes: Iterable[str] | None = None) -> list[str]:
    normalized_dataset = {normalize_label_name(item) for item in dataset_classes if normalize_label_name(item)}
    target = list(target_classes) if target_classes is not None else ASL_FULL_CLASS_LIST
    normalized_target = [normalize_label_name(item) for item in target]
    missing = [label for label in normalized_target if label not in normalized_dataset]
    return missing


def build_class_mapping(class_names: Iterable[str]) -> dict[str, int]:
    normalized = [normalize_label_name(item) for item in class_names if normalize_label_name(item)]
    unique = list(dict.fromkeys(normalized))
    return {name: index for index, name in enumerate(unique)}
