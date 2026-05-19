from __future__ import annotations

import pandas as pd


def _series_sample(series: pd.Series, limit: int = 12) -> list:
    values = series.dropna().head(limit).tolist()
    return [v.item() if hasattr(v, "item") else v for v in values]


def profile_dataframe(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include="number")
    categorical = df.select_dtypes(exclude="number")
    missing = df.isna().sum().sort_values(ascending=False)

    correlations = []
    if numeric.shape[1] > 1:
        corr = numeric.corr(numeric_only=True).fillna(0)
        for col in corr.columns:
            for row in corr.index:
                if col != row:
                    correlations.append({"x": col, "y": row, "value": round(float(corr.loc[row, col]), 3)})

    imbalance = {}
    for col in df.columns:
        unique = df[col].nunique(dropna=True)
        if 1 < unique <= 12:
            counts = df[col].value_counts(dropna=False).head(12)
            imbalance[col] = {str(k): int(v) for k, v in counts.items()}

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_columns": numeric.columns.tolist(),
        "categorical_columns": categorical.columns.tolist(),
        "datetime_candidates": [
            col for col in df.columns
            if "time" in col.lower() or "date" in col.lower() or "timestamp" in col.lower()
        ],
        "missing_values": {str(k): int(v) for k, v in missing.items() if int(v) > 0},
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_samples": {col: _series_sample(df[col]) for col in df.columns[:20]},
        "summary": numeric.describe().round(3).fillna(0).to_dict(),
        "correlations": correlations[:300],
        "imbalance": imbalance,
    }

