"""
AI Data Pipeline Auto-Repair — Streamlit entrypoint.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

# Ensure project root is on path when the process cwd differs from this folder.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from modules.ai_explainer import explain_issue
from modules.data_loader import load_dataset
from modules.error_detector import detect_errors
from modules.profiler import profile_dataset
from modules.repair_engine import repair_dataset

st.set_page_config(
    page_title="AI Data Pipeline Auto-Repair",
    layout="wide",
)

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "data", "sample_broken_dataset.csv")


def main() -> None:
    st.title("AI Data Pipeline Auto-Repair")
    st.markdown(
        "Upload a CSV, review **profiling** and **detected issues**, then run **automatic repairs** "
        "and download the cleaned file."
    )

    # --- Upload dataset ---
    st.subheader("Upload dataset")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    col_a, col_b = st.columns(2)
    with col_a:
        use_sample = st.button("Load bundled sample dataset")
    with col_b:
        st.caption(f"Sample path: `{SAMPLE_CSV}`")

    if uploaded is not None:
        new_df = load_dataset(uploaded)
        new_id = f"upload:{getattr(uploaded, 'name', 'file')}:{getattr(uploaded, 'size', 0)}"
        if st.session_state.get("data_source_id") != new_id:
            st.session_state["cleaned_df"] = None
        st.session_state["data_source_id"] = new_id
        st.session_state["raw_df"] = new_df
    elif use_sample and os.path.isfile(SAMPLE_CSV):
        with open(SAMPLE_CSV, "rb") as f:
            new_df = load_dataset(f)
        new_id = "sample:sample_broken_dataset.csv"
        if st.session_state.get("data_source_id") != new_id:
            st.session_state["cleaned_df"] = None
        st.session_state["data_source_id"] = new_id
        st.session_state["raw_df"] = new_df

    df = st.session_state.get("raw_df")

    if df is None:
        st.info("Upload a CSV or click **Load bundled sample dataset** to begin.")
        return

    # --- Preview ---
    st.subheader("Dataset preview")
    st.dataframe(df.head(50), width="stretch")

    # --- Profiling ---
    st.subheader("Data profiling summary")
    profile = profile_dataset(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", profile["row_count"])
    c2.metric("Columns", profile["column_count"])
    c3.metric("Total missing cells", profile["missing_total"])
    st.write("**Column dtypes**")
    st.json(profile["dtypes"])
    st.write("**Missing values per column**")
    st.json(profile["missing_values"])

    # --- Detected issues ---
    st.subheader("Detected issues")
    issues = detect_errors(df)
    st.json(issues)

    # --- AI explanations ---
    st.subheader("AI explanations")
    explanation_keys = [
        ("missing_values", profile["missing_total"]),
        ("duplicate_rows", issues["duplicate_rows"]),
        ("incorrect_datatypes", issues["incorrect_datatypes"]),
        ("outliers", issues["outliers"]),
        ("mixed_type_columns", issues["mixed_type_columns"]),
        ("corrupted_rows", issues["corrupted_rows"]),
    ]
    for name, val in explanation_keys:
        st.markdown(f"**{name.replace('_', ' ').title()}**")
        st.write(explain_issue(name, val))

    # --- Repair ---
    st.subheader("Repair & export")
    if st.button("Repair Dataset", type="primary"):
        cleaned = repair_dataset(df)
        st.session_state["cleaned_df"] = cleaned
        st.success("Repair complete. Preview and download below.")

    cleaned_df = st.session_state.get("cleaned_df")
    if cleaned_df is not None:
        st.write("**Cleaned dataset preview**")
        st.dataframe(cleaned_df.head(50), width="stretch")
        buf = io.StringIO()
        cleaned_df.to_csv(buf, index=False)
        st.download_button(
            label="Download cleaned CSV",
            data=buf.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
