"""
AI Data Pipeline Auto-Repair — Streamlit entrypoint.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on path when the process cwd differs from this folder.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from modules.ai_explainer import explain_issue
from modules.data_loader import load_dataset
from modules.errors import DatasetLoadError
from modules.error_detector import detect_errors
from modules.profiler import profile_dataset
from modules.repair_engine import repair_dataset

st.set_page_config(
    page_title="AI Data Pipeline Auto-Repair",
    layout="wide",
)

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "data", "sample_broken_dataset.csv")
# Shown in the UI (portable; not tied to any machine’s absolute path)
SAMPLE_PATH_DISPLAY = "data/sample_broken_dataset.csv"


def _issue_counts(profile: Dict[str, Any], issues: Dict[str, Any]) -> Dict[str, int]:
    """Scalar counts for metric tiles."""
    outlier_block = issues.get("outliers") or {}
    rows_flagged = outlier_block.get("rows_flagged", 0) if isinstance(outlier_block, dict) else 0
    incorrect = issues.get("incorrect_datatypes") or []
    mixed = issues.get("mixed_type_columns") or []
    return {
        "missing_cells": int(profile.get("missing_total", 0)),
        "duplicate_rows": int(issues.get("duplicate_rows", 0)),
        "corrupted_rows": int(issues.get("corrupted_rows", 0)),
        "outlier_rows": int(rows_flagged),
        "datatype_issue_cols": len(incorrect) if isinstance(incorrect, list) else 0,
        "mixed_type_cols": len(mixed) if isinstance(mixed, list) else 0,
    }


def _render_issue_metrics(counts: Dict[str, int], *, prefix: str = "") -> None:
    """Two rows of metrics for a consistent dashboard layout."""
    label = lambda k: f"{prefix}{k}" if prefix else k
    r1 = st.columns(3)
    r1[0].metric(label("Missing cells"), counts["missing_cells"])
    r1[1].metric(label("Duplicate rows"), counts["duplicate_rows"])
    r1[2].metric(label("Corrupted rows"), counts["corrupted_rows"])
    r2 = st.columns(3)
    r2[0].metric(label("Outlier rows (|z|>3)"), counts["outlier_rows"])
    r2[1].metric(label("Datatype issue cols"), counts["datatype_issue_cols"])
    r2[2].metric(label("Mixed-type cols"), counts["mixed_type_cols"])


def main() -> None:
    st.title("Pipeline Guardian")
    st.caption("AI-assisted data quality scan, repair, and export")
    st.markdown(
        "Load a CSV, review **metrics and detected issues**, run **one-click repair**, then compare "
        "**before vs after** and download the cleaned file."
    )

    st.divider()
    st.header("1. Load data", anchor=False)
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    col_a, col_b = st.columns(2)
    with col_a:
        use_sample = st.button("Load bundled sample dataset")
    with col_b:
        st.caption(f"Bundled sample (relative to app folder): `{SAMPLE_PATH_DISPLAY}`")

    if uploaded is not None:
        new_id = f"upload:{getattr(uploaded, 'name', 'file')}:{getattr(uploaded, 'size', 0)}"
        try:
            if getattr(uploaded, "size", None) == 0:
                raise DatasetLoadError(
                    "This upload is empty. Choose a non-empty CSV file, or use the bundled sample."
                )
            new_df = load_dataset(uploaded)
        except DatasetLoadError as exc:
            st.error(str(exc))
        else:
            if st.session_state.get("data_source_id") != new_id:
                st.session_state["cleaned_df"] = None
            st.session_state["data_source_id"] = new_id
            st.session_state["raw_df"] = new_df
    elif use_sample and os.path.isfile(SAMPLE_CSV):
        new_id = "sample:sample_broken_dataset.csv"
        try:
            with open(SAMPLE_CSV, "rb") as f:
                new_df = load_dataset(f)
        except DatasetLoadError as exc:
            st.error(str(exc))
            st.caption(f"Expected sample file: `{SAMPLE_PATH_DISPLAY}`")
        else:
            if st.session_state.get("data_source_id") != new_id:
                st.session_state["cleaned_df"] = None
            st.session_state["data_source_id"] = new_id
            st.session_state["raw_df"] = new_df

    df = st.session_state.get("raw_df")

    if df is None:
        st.info("Upload a CSV or click **Load bundled sample dataset** to begin.")
        return

    st.success("Dataset loaded.Profiling and issue scan run below on each run.")

    if len(df) > 0 and int(df.notna().sum().sum()) == 0:
        st.warning(
            "Every cell in this dataset is missing (NaN). Profiling and repair still run, but results will "
            "mostly reflect empty or default-filled data."
        )

    st.divider()
    st.header("2. Profile & preview", anchor=False)
    try:
        profile = profile_dataset(df)
    except Exception as exc:
        st.error(
            "Could not build a profile for this table. Check for unusual nested values or broken rows, "
            "then try another export format."
        )
        with st.expander("Technical detail"):
            st.code(str(exc))
        return
    p1, p2, p3 = st.columns(3)
    p1.metric("Rows", profile["row_count"])
    p2.metric("Columns", profile["column_count"])
    p3.metric("Missing cells (total)", profile["missing_total"])
    with st.expander("Preview (first 50 rows)", expanded=True):
        st.dataframe(df.head(50), width="stretch")
    with st.expander("Column dtypes & missing per column"):
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Dtypes")
            st.json(profile["dtypes"])
        with c2:
            st.caption("Missing per column")
            st.json(profile["missing_values"])

    st.divider()
    st.header("3. Detected issues", anchor=False)
    st.caption("Counts summarize what the detectors found; open JSON for full detail.")
    issues = detect_errors(df)
    scan_note = issues.pop("scan_failed_message", None)
    if scan_note:
        st.warning(scan_note)
    counts_before = _issue_counts(profile, issues)
    _render_issue_metrics(counts_before)
    with st.expander("Raw detection payload (JSON)"):
        st.json(issues)

    st.divider()
    st.header("4. AI-style explanations", anchor=False)
    explanation_keys = [
        ("missing_values", profile["missing_total"]),
        ("duplicate_rows", issues["duplicate_rows"]),
        ("incorrect_datatypes", issues["incorrect_datatypes"]),
        ("outliers", issues["outliers"]),
        ("mixed_type_columns", issues["mixed_type_columns"]),
        ("corrupted_rows", issues["corrupted_rows"]),
    ]
    for name, val in explanation_keys:
        with st.expander(name.replace("_", " ").title(), expanded=False):
            st.write(explain_issue(name, val))

    st.divider()
    st.header("5. Repair & export", anchor=False)
    if st.button("Run auto-repair", type="primary"):
        with st.status("Running repair pipeline…", expanded=True) as status:
            status.write("Coercing mostly-numeric text columns to numbers…")
            status.write("Dropping heavily incomplete rows, imputing missing values…")
            status.write("Removing exact duplicates and extreme outlier rows (|z| > 3)…")
            try:
                cleaned = repair_dataset(df)
            except Exception as exc:
                status.update(label="Repair failed", state="error")
                st.error(
                    "Repair could not finish. Unusual cell values, an all-empty result after row drops, "
                    "or a very wide mixed-type column can trigger this. Try simplifying the CSV and upload again."
                )
                with st.expander("Technical detail"):
                    st.code(str(exc))
            else:
                st.session_state["cleaned_df"] = cleaned
                status.update(label="Repair finished", state="complete")
                st.session_state["repair_notice"] = True

    if st.session_state.pop("repair_notice", False):
        st.toast("Repair complete — see before vs after below.", icon="✅")

    cleaned_df = st.session_state.get("cleaned_df")
    if cleaned_df is not None:
        st.subheader("Before vs after", anchor=False)
        comparison_ok = True
        try:
            profile_after = profile_dataset(cleaned_df)
            issues_after = detect_errors(cleaned_df)
            issues_after.pop("scan_failed_message", None)
            counts_after = _issue_counts(profile_after, issues_after)
        except Exception as exc:
            comparison_ok = False
            st.warning(
                "The cleaned dataset is ready, but the “after” metrics could not be computed. "
                "You can still preview and download the file below."
            )
            with st.expander("Technical detail"):
                st.code(str(exc))

        if comparison_ok:
            bcol, acol = st.columns(2)
            with bcol:
                st.markdown("##### Before repair")
                st.metric("Rows", profile["row_count"], delta=None)
                st.metric("Missing cells", counts_before["missing_cells"])
                st.metric("Duplicate rows", counts_before["duplicate_rows"])
                st.metric("Outlier rows", counts_before["outlier_rows"])
            with acol:
                st.markdown("##### After repair")
                d_rows = int(profile_after["row_count"] - profile["row_count"])
                d_miss = int(counts_after["missing_cells"] - counts_before["missing_cells"])
                d_dup = int(counts_after["duplicate_rows"] - counts_before["duplicate_rows"])
                d_out = int(counts_after["outlier_rows"] - counts_before["outlier_rows"])
                st.metric("Rows", profile_after["row_count"], delta=d_rows, delta_color="off")
                st.metric("Missing cells", counts_after["missing_cells"], delta=d_miss, delta_color="inverse")
                st.metric("Duplicate rows", counts_after["duplicate_rows"], delta=d_dup, delta_color="inverse")
                st.metric("Outlier rows", counts_after["outlier_rows"], delta=d_out, delta_color="inverse")
            st.caption(
                "Deltas show change (after − before). Fewer rows can mean deduplication and outlier drops; "
                "missing cells and duplicates should trend down."
            )

        prev, nxt = st.columns(2)
        with prev:
            st.markdown("**Preview — before** (50 rows)")
            st.dataframe(df.head(50), width="stretch")
        with nxt:
            st.markdown("**Preview — after** (50 rows)")
            st.dataframe(cleaned_df.head(50), width="stretch")

        if comparison_ok:
            with st.expander("After repair — detection JSON (sanity check)"):
                st.json(issues_after)

        buf = io.StringIO()
        cleaned_df.to_csv(buf, index=False)
        st.download_button(
            label="Download cleaned CSV",
            data=buf.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            type="primary",
        )


if __name__ == "__main__":
    main()
