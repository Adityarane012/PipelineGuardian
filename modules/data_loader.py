"""Load CSV datasets from file-like objects or paths."""

from __future__ import annotations

import io
from typing import BinaryIO, Union

import pandas as pd


def load_dataset(file: Union[str, BinaryIO, io.BytesIO]) -> pd.DataFrame:
    """
    Load a CSV dataset from a path or uploaded file-like object.

    Parameters
    ----------
    file : str, pathlib.Path, or file-like (e.g. Streamlit UploadedFile)

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(file)
