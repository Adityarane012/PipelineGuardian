"""Apply automatic cleaning steps to a dataframe."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object or pd.api.types.is_string_dtype(out[col]):
            coerced = pd.to_numeric(out[col], errors="coerce")
            if coerced.notna().sum() >= max(1, int(0.3 * len(out))):
                out[col] = coerced
    return out


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].isna().all():
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            med = out[col].median()
            fill = 0.0 if pd.isna(med) else med
            out[col] = out[col].fillna(fill)
        else:
            mode = out[col].mode()
            fill = mode.iloc[0] if len(mode) else ""
            out[col] = out[col].fillna(fill)
    return out


def _drop_outlier_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols: List[str] = [
        c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])
    ]
    if not numeric_cols or len(out) < 2:
        return out
    sub = out[numeric_cols].copy()
    filled = sub.fillna(sub.median())
    scaler = StandardScaler()
    z = scaler.fit_transform(filled)
    keep = ~np.any(np.abs(z) > 3, axis=1)
    return out.loc[keep].copy()


def _drop_corrupted_rows(df: pd.DataFrame, nan_threshold_ratio: float = 0.5) -> pd.DataFrame:
    thresh = max(1, int(nan_threshold_ratio * len(df.columns)))
    ok = df.isna().sum(axis=1) <= thresh
    return df.loc[ok].copy()


def repair_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply fixes in order:
    1. Coerce object columns that are mostly numeric via to_numeric(errors='coerce')
    2. Drop rows with excessive NaN (corrupted/partial rows)
    3. Fill remaining missing: median (numeric), mode (non-numeric)
    4. Remove duplicate rows
    5. Remove rows with |z| > 3 on any numeric column
    """
    cleaned = df.copy()
    cleaned = _coerce_numeric_columns(cleaned)
    cleaned = _drop_corrupted_rows(cleaned, nan_threshold_ratio=0.5)
    cleaned = _fill_missing(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    cleaned = _drop_outlier_rows(cleaned)
    cleaned = cleaned.reset_index(drop=True)
    return cleaned
