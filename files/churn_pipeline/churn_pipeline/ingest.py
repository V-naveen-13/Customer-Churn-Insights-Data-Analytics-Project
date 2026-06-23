"""
ingest.py — Data ingestion, validation, and cleaning for the churn pipeline.

Usage:
    from ingest import load_and_validate

Accepts any CSV with a churn flag column. Column name mapping is configured
in config.py so the pipeline works against your actual column names.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import COL, REQUIRED_COLS, OPTIONAL_COLS, CATEGORICAL_COLS, NUMERIC_COLS


# ── helpers ──────────────────────────────────────────────────────────────────

def _coerce_churn_flag(series: pd.Series) -> pd.Series:
    """Accept True/False, 1/0, 'Yes'/'No', 'Churned'/'Active', etc."""
    s = series.copy()
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int)
    mapping = {
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        'churned': 1, 'active': 0,
        'churn': 1, 'retained': 0,
        '1': 1, '0': 0,
    }
    return s.str.strip().str.lower().map(mapping).fillna(0).astype(int)


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse any date columns found in the dataframe."""
    date_cols = [COL.get('start_date'), COL.get('churn_date')]
    for col in date_cols:
        if col and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def _derive_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Derive tenure_months if not present but start/churn dates are."""
    tenure_col = COL.get('tenure')
    start_col = COL.get('start_date')
    churn_col = COL.get('churn_date')

    if tenure_col and tenure_col in df.columns:
        return df

    if start_col and start_col in df.columns:
        reference_date = pd.Timestamp.now()
        if churn_col and churn_col in df.columns:
            end_date = df[churn_col].fillna(reference_date)
        else:
            end_date = reference_date
        derived = ((end_date - df[start_col]).dt.days / 30.44).round(1)
        derived_name = tenure_col if tenure_col else 'tenure_months'
        df[derived_name] = derived.clip(lower=0)
        if not tenure_col:
            COL['tenure'] = derived_name

    return df


# ── validation ───────────────────────────────────────────────────────────────

class DataValidationReport:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def error(self, msg): self.errors.append(msg)
    def warn(self, msg): self.warnings.append(msg)
    def note(self, msg): self.info.append(msg)

    @property
    def passed(self): return len(self.errors) == 0

    def summary(self) -> str:
        lines = ['=== Data Validation Report ===']
        if self.errors:
            lines.append(f'\n[ERRORS] {len(self.errors)} critical issue(s):')
            for e in self.errors: lines.append(f'  ✗ {e}')
        if self.warnings:
            lines.append(f'\n[WARNINGS] {len(self.warnings)} issue(s):')
            for w in self.warnings: lines.append(f'  ⚠ {w}')
        if self.info:
            lines.append(f'\n[INFO]')
            for i in self.info: lines.append(f'  ✓ {i}')
        lines.append('\n' + ('PASSED' if self.passed else 'FAILED'))
        return '\n'.join(lines)


def validate(df: pd.DataFrame) -> DataValidationReport:
    report = DataValidationReport()

    # Required columns
    churn_col = COL.get('churn')
    if not churn_col or churn_col not in df.columns:
        report.error(f"Churn flag column '{churn_col}' not found. Set COL['churn'] in config.py.")
    
    for col in REQUIRED_COLS:
        mapped = COL.get(col)
        if mapped and mapped not in df.columns:
            report.warn(f"Expected column '{mapped}' (mapped from '{col}') not found — some analyses will be skipped.")

    # Row count
    if len(df) < 100:
        report.warn(f"Only {len(df)} rows — model reliability may be low. 500+ recommended.")
    else:
        report.note(f"{len(df):,} rows loaded.")

    # Churn balance
    if churn_col and churn_col in df.columns:
        churn_rate = df[churn_col].mean()
        report.note(f"Churn rate: {churn_rate:.1%}")
        if churn_rate < 0.02:
            report.warn("Churn rate < 2% — severe class imbalance. SMOTE will be applied.")
        if churn_rate > 0.5:
            report.warn("Churn rate > 50% — verify churn flag encoding.")

    # Missing data
    missing = df.isnull().mean()
    high_missing = missing[missing > 0.3]
    if not high_missing.empty:
        for col, pct in high_missing.items():
            report.warn(f"'{col}' has {pct:.0%} missing values — consider excluding.")

    # Numeric columns
    for col in NUMERIC_COLS:
        mapped = COL.get(col)
        if mapped and mapped in df.columns:
            if df[mapped].dtype == object:
                report.warn(f"'{mapped}' appears numeric but is stored as string — will attempt coercion.")

    return report


# ── cleaning ─────────────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
    1. Rename columns to internal standard names
    2. Coerce types
    3. Handle missing values
    4. Remove duplicates
    5. Clip outliers on key numeric fields
    """
    df = df.copy()

    # Build reverse map: original_name → internal_name
    rename_map = {}
    for internal, original in COL.items():
        if original and original in df.columns and original != internal:
            rename_map[original] = internal
    df = df.rename(columns=rename_map)

    # Coerce churn flag
    if 'churn' in df.columns:
        df['churn'] = _coerce_churn_flag(df['churn'])

    # Parse dates and derive tenure
    df = _parse_dates(df)
    df = _derive_tenure(df)

    # Coerce numeric columns
    numeric_internals = [k for k in NUMERIC_COLS if k in df.columns]
    for col in numeric_internals:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Coerce categorical columns
    cat_internals = [k for k in CATEGORICAL_COLS if k in df.columns]
    for col in cat_internals:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    # Remove full duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"[ingest] Removed {n_dupes} duplicate rows.")

    # Fill missing values
    for col in numeric_internals:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    for col in cat_internals:
        if col in df.columns:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else 'Unknown')

    # Clip obvious outliers: cap ARR/MRR at 99.5th percentile
    for col in ['arr', 'mrr', 'monthly_revenue']:
        if col in df.columns:
            cap = df[col].quantile(0.995)
            df[col] = df[col].clip(upper=cap)

    # Clip tenure at 0
    if 'tenure' in df.columns:
        df['tenure'] = df['tenure'].clip(lower=0)

    return df


# ── entry point ───────────────────────────────────────────────────────────────

def load_and_validate(path: str) -> tuple[pd.DataFrame, DataValidationReport]:
    """
    Main entry point. Load CSV, validate, and clean.

    Returns:
        df      — cleaned DataFrame with standardised column names
        report  — DataValidationReport (print report.summary() for details)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    ext = path.suffix.lower()
    if ext == '.csv':
        df_raw = pd.read_csv(path, low_memory=False)
    elif ext in ('.xlsx', '.xls'):
        df_raw = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .csv or .xlsx")

    print(f"[ingest] Loaded {len(df_raw):,} rows × {len(df_raw.columns)} columns from {path.name}")

    report = validate(df_raw)
    print(report.summary())

    if not report.passed:
        raise ValueError("Data validation failed. Fix errors above before proceeding.")

    df_clean = clean(df_raw)
    print(f"[ingest] Cleaning complete. Shape: {df_clean.shape}")

    return df_clean, report
