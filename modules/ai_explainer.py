"""Template-based natural-language explanations (no external API)."""

from __future__ import annotations

from typing import Any


def explain_issue(issue_name: str, issue_value: Any) -> str:
    """
    Return a short, human-readable explanation for a detected data issue.

    Simulates concise "AI analyst" reasoning without calling external models.
    """
    key = (issue_name or "").strip().lower().replace(" ", "_")

    if key == "missing_values":
        return (
            "Missing values break many ML pipelines because algorithms expect complete "
            "feature vectors. Gaps often come from optional survey fields, sensor dropouts, "
            "or merge mismatches. Imputation (median for numbers, mode for categories) "
            "restores usable rows while staying close to the bulk of your data."
        )

    if key == "duplicate_rows":
        return (
            "Duplicate rows inflate sample size without adding information and can bias "
            "metrics or overfit simple models. Removing exact duplicates keeps one "
            "representative record per unique observation."
        )

    if key == "incorrect_datatypes":
        return (
            "Columns stored as text but meant to be numeric cannot be averaged, scaled, or "
            "used in regression. Coercing with `to_numeric(..., errors='coerce')` turns "
            "invalid tokens into NaN so you can fix or drop them in a controlled way."
        )

    if key == "outliers":
        return (
            "Extreme values (here flagged by |z-score| > 3) may be data-entry errors or "
            "rare events. For a quick hackathon baseline, dropping those rows reduces "
            "leverage from bad measurements; domain review would refine this in production."
        )

    if key == "mixed_type_columns":
        return (
            "Mixed-type columns mean some cells look like numbers and others like labels "
            "in the same field—often a sign of inconsistent entry or merged sources. "
            "Normalizing types (or splitting the column) improves consistency for modeling."
        )

    if key == "corrupted_rows":
        return (
            "Rows with many missing cells are often partial imports or broken records. "
            "Dropping rows that exceed a NaN threshold removes fragments that would "
            "otherwise add noise after imputation."
        )

    return (
        f"Issue `{issue_name}` was observed with detail: `{issue_value}`. "
        "Review the metric and consider whether it reflects collection bugs, "
        "schema drift, or legitimate edge cases before modeling."
    )
