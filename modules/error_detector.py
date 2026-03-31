"""Detect common tabular data quality issues."""

from __future__ import annotations

from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _numeric_coerced_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _mixed_type_columns(df: pd.DataFrame) -> List[str]:
    """
    Columns where values do not share one consistent logical type:
    - object/string columns where some values parse as numbers and others do not (typical CSV quirk)
    - object columns mixing native float/int and str instances
    """
    mixed: List[str] = []
    for col in df.columns:
        s = df[col]
        if s.dtype != object and not pd.api.types.is_string_dtype(s):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        as_str = non_null.astype(str)
        parses = pd.to_numeric(as_str, errors="coerce").notna()
        if parses.any() and (~parses).any():
            mixed.append(str(col))
            continue
        types: Set[str] = set()
        for v in non_null.head(500):
            if isinstance(v, bool):
                types.add("bool")
            elif isinstance(v, (int, np.integer)):
                types.add("int")
            elif isinstance(v, (float, np.floating)):
                if pd.isna(v):
                    continue
                types.add("float")
            else:
                types.add("str")
        if len(types) > 1:
            mixed.append(str(col))
    return mixed


def _incorrect_datatype_columns(df: pd.DataFrame) -> List[str]:
    """
    Object columns where many values parse as numeric but column is not numeric dtype,
    or object column with at least one non-null that fails numeric parse while others succeed.
    """
    issues: List[str] = []
    for col in df.columns:
        s = df[col]
        if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        coerced = _numeric_coerced_series(s)
        parsed_ok = coerced.notna()
        # Heuristic: mostly parseable as number but stored as object → datatype issue
        if parsed_ok.sum() >= max(1, int(0.3 * len(s))) and parsed_ok.sum() < len(s):
            issues.append(str(col))
        elif parsed_ok.any() and (~parsed_ok & s.notna()).any():
            issues.append(str(col))
    return list(dict.fromkeys(issues))


def _outlier_row_mask(df: pd.DataFrame, numeric_cols: List[str]) -> pd.Series:
    """True where any numeric column has |z| > 3 (StandardScaler z-scores)."""
    if not numeric_cols:
        return pd.Series(False, index=df.index)
    sub = df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    filled = sub.fillna(sub.median())
    filled = filled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if filled.empty or len(filled) < 2:
        return pd.Series(False, index=df.index)
    scaler = StandardScaler()
    z = scaler.fit_transform(filled)
    bad = np.abs(z) > 3
    row_bad = bad.any(axis=1)
    return pd.Series(row_bad, index=df.index)


def _detect_errors_core(df: pd.DataFrame) -> Dict[str, Any]:
    missing_per_col = df.isna().sum().to_dict()
    missing_per_col = {str(k): int(v) for k, v in missing_per_col.items()}

    dup_count = int(df.duplicated().sum())

    mixed_cols = _mixed_type_columns(df)
    dtype_issue_cols = _incorrect_datatype_columns(df)

    # Numeric columns for z-score: already numeric or successfully coercible majority
    numeric_cols: List[str] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(str(c))
        else:
            coerced = _numeric_coerced_series(df[c])
            if coerced.notna().sum() >= max(1, int(0.5 * len(df))):
                numeric_cols.append(str(c))

    outlier_mask = _outlier_row_mask(df, numeric_cols)
    outlier_rows = int(outlier_mask.sum())

    outlier_by_col: Dict[str, int] = {}
    if numeric_cols:
        sub_oc = df[numeric_cols].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        filled_oc = sub_oc.fillna(sub_oc.median())
        filled_oc = filled_oc.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(filled_oc) >= 2:
            z_mat = StandardScaler().fit_transform(filled_oc)
            for i, col in enumerate(numeric_cols):
                outlier_by_col[col] = int(np.sum(np.abs(z_mat[:, i]) > 3))

    # Corrupted rows: excessive NaN (e.g. > 50% of columns)
    thresh = max(1, int(0.5 * len(df.columns)))
    excessive_nan_mask = df.isna().sum(axis=1) > thresh
    corrupted_rows = int(excessive_nan_mask.sum())

    return {
        "missing_values": missing_per_col,
        "duplicate_rows": dup_count,
        "incorrect_datatypes": dtype_issue_cols,
        "outliers": {
            "rows_flagged": outlier_rows,
            "by_column": outlier_by_col,
        },
        "mixed_type_columns": mixed_cols,
        "corrupted_rows": corrupted_rows,
        "corrupted_row_threshold_nan_count": thresh,
    }


def detect_errors(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Scan dataframe for missing values, duplicates, datatype problems, outliers, mixed types.

    Returns
    -------
    dict with keys suitable for UI and repair_engine (see app.py).
    If scanning fails, returns the same shape with ``scan_failed_message`` set and conservative defaults.
    """
    try:
        return _detect_errors_core(df)
    except Exception:
        missing_per_col: Dict[str, int] = {}
        try:
            missing_per_col = {str(k): int(v) for k, v in df.isna().sum().to_dict().items()}
        except Exception:
            pass
        ncols = len(df.columns)
        thresh = max(1, int(0.5 * ncols)) if ncols else 1
        return {
            "missing_values": missing_per_col,
            "duplicate_rows": 0,
            "incorrect_datatypes": [],
            "outliers": {"rows_flagged": 0, "by_column": {}},
            "mixed_type_columns": [],
            "corrupted_rows": 0,
            "corrupted_row_threshold_nan_count": thresh,
            "scan_failed_message": (
                "The full issue scan could not finish (unusual structure or cell values). "
                "Basic missing counts are shown where possible; you can still try repair."
            ),
        }
