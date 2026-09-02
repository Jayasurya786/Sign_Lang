import numpy as np
import pandas as pd


def handle_missing_values(df: pd.DataFrame, feature_cols: list[str] | None = None):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != 'label']

    df = df.copy()
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
    return df


def remove_outliers_iqr(df: pd.DataFrame, feature_cols: list[str] | None = None, iqr_multiplier: float = 1.5):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != 'label']

    df = df.copy()
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df.reset_index(drop=True)


def normalize_landmarks(df: pd.DataFrame, feature_cols: list[str] | None = None):
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != 'label']

    df = df.copy()
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val == min_val:
            df[col] = 0.0
        else:
            df[col] = (df[col] - min_val) / (max_val - min_val)
    return df


def clean_landmark_dataset(df: pd.DataFrame):
    feature_cols = [col for col in df.columns if col != 'label']
    cleaned = handle_missing_values(df, feature_cols)
    return cleaned


def save_cleaned_dataset(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    cleaned = clean_landmark_dataset(df)
    cleaned.to_csv(output_path, index=False)
    return cleaned
