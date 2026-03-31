# AI Data Pipeline Auto-Repair

## Problem description

Real-world CSV datasets often arrive with **missing values**, **duplicate rows**, **wrong types**, **outliers**, and **inconsistent mixing of types in one column**. These issues break exploratory analysis and break many ML training pipelines. Teams spend valuable hackathon time on manual cleaning instead of modeling.

## Solution overview

This project is a **Streamlit** app that:

1. **Loads** a user-uploaded CSV (or a bundled sample file).
2. **Profiles** the table (shape, dtypes, missing counts).
3. **Detects** common quality problems using lightweight heuristics and **z-scores** via **scikit-learn** `StandardScaler`.
4. Shows **AI-style explanations** (template-based, no external API) so judges can read *why* each issue matters.
5. **Repairs** the data automatically: numeric coercion, dropping overly empty rows, median/mode imputation, deduplication, and outlier row removal (`|z| > 3`).
6. Lets the user **preview** and **download** the cleaned CSV.

## How to run locally

From the project root (`ai-data-pipeline-auto-repair/`):

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

## Example demo steps for judges

1. Start the app with `streamlit run app.py`.
2. Click **Load bundled sample dataset** (broken dataset is given in the data folder or upload your own CSV).
3. Scroll through **Dataset preview** and **Data profiling summary** (rows, columns, dtypes, missing counts).
4. Review **Detected issues** (JSON summary) and read the **AI explanations** for each category.
5. Click **Repair Dataset** and confirm the **Cleaned dataset preview** shows fewer problems (e.g. duplicates and extreme outliers removed, types coerced where applicable).
6. Click **Download cleaned CSV** to save the repaired file.

## Project layout

- `app.py` — Streamlit UI
- `modules/__init__.py` — package marker (reliable imports)
- `modules/data_loader.py` — CSV loading
- `modules/profiler.py` — profiling summary
- `modules/error_detector.py` — issue detection
- `modules/repair_engine.py` — automatic repairs
- `modules/ai_explainer.py` — natural-language explanations
- `data/sample_broken_dataset.csv` — intentionally messy demo data
