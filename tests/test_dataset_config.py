from src.data.dataset_config import ASL_FULL_CLASS_LIST, normalize_label_name, validate_dataset_classes


def test_asl_full_class_list_contains_all_supported_signs():
    assert ASL_FULL_CLASS_LIST[0] == 'A'
    assert ASL_FULL_CLASS_LIST[-1] == 'Z'
    assert len(ASL_FULL_CLASS_LIST) == 26


def test_validate_dataset_classes_reports_missing_signs():
    classes = ['A', 'B', 'C', 'D']
    missing = validate_dataset_classes(classes, target_classes=['A', 'B', 'C', 'D', 'E'])
    assert missing == ['E']


def test_normalize_label_name_handles_numeric_and_string_labels():
    assert normalize_label_name('A') == 'A'
    assert normalize_label_name(0) == '0'
    assert normalize_label_name(' a ') == 'A'
