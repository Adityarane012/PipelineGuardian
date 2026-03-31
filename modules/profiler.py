"""Dataset profiling: shape, dtypes, missing values."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize a dataframe for display and downstream checks.

    Returns
    -------
    dict with keys:
        row_count, column_count, dtypes, missing_values (per column),
        missing_total
    """
    missing = df.isna().sum().to_dict()
    missing = {str(k): int(v) for k, v in missing.items()}
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "missing_values": missing,
        "missing_total": int(df.isna().sum().sum()),
    }
