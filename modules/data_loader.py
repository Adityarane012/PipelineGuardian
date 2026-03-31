"""Load CSV datasets from file-like objects or paths."""

from __future__ import annotations

import io
import os
from typing import BinaryIO, Union

import pandas as pd

from modules.errors import DatasetLoadError


def _empty_file_message() -> str:
    return (
        "This file is empty. Please upload a CSV that includes a header row and at least one data row."
    )


def load_dataset(file: Union[str, os.PathLike[str], BinaryIO, io.BytesIO]) -> pd.DataFrame:
    """
    Load a CSV dataset from a path or uploaded file-like object.

    Raises
    ------
    DatasetLoadError
        Empty file, unreadable CSV, wrong encoding, or no columns parsed.
    """
    path_str: str | None = None
    if isinstance(file, (str, os.PathLike)):
        path_str = os.fspath(file)
        if not os.path.isfile(path_str):
            raise DatasetLoadError(f"File not found: {path_str}")
        if os.path.getsize(path_str) == 0:
            raise DatasetLoadError(_empty_file_message())

    if hasattr(file, "read") and not isinstance(file, (str, os.PathLike)):
        sz = getattr(file, "size", None)
        if sz == 0:
            raise DatasetLoadError(_empty_file_message())

    try:
        if path_str is not None:
            df = pd.read_csv(path_str, encoding="utf-8")
        else:
            if hasattr(file, "seek"):
                file.seek(0)
            df = pd.read_csv(file, encoding="utf-8")
    except pd.errors.EmptyDataError:
        raise DatasetLoadError(
            "No tabular data was found. Add a header row and at least one row of values, "
            "or check that the file is really a CSV."
        ) from None
    except UnicodeDecodeError as exc:
        raise DatasetLoadError(
            "Could not read the file as UTF-8 text. Save the CSV as UTF-8, or remove binary / "
            "non-text content and try again."
        ) from exc
    except pd.errors.ParserError as exc:
        raise DatasetLoadError(
            "This file is not valid CSV (parsing failed). Use a comma-separated table with a header row, "
            "or export again from Excel/Google Sheets as CSV."
        ) from exc
    except OSError as exc:
        raise DatasetLoadError("Could not read the file from disk. Check permissions and try again.") from exc
    except Exception as exc:
        raise DatasetLoadError(
            "Unexpected error while reading the file. Confirm it is a plain CSV and try again."
        ) from exc

    if df.shape[1] == 0:
        raise DatasetLoadError(
            "The CSV has no columns. Include a header row with at least one column name."
        )

    return df
