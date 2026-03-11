import pandas as pd


def detect_outliers(df: pd.DataFrame) -> dict:
    numeric_df = df.select_dtypes(include="number")
    results = {}

    for col in numeric_df.columns:
        series = numeric_df[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            results[col] = 0
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = series[(series < lower) | (series > upper)]
        results[col] = int(len(outliers))

    return results


def assess_readiness(df: pd.DataFrame):
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_ratio = missing_cells / total_cells
    duplicate_ratio = (duplicate_rows / rows) if rows else 0

    numeric_cols = int(df.select_dtypes(include="number").shape[1])
    text_cols = int(df.select_dtypes(include=["object", "category"]).shape[1])

    outlier_map = detect_outliers(df)
    total_outliers = int(sum(outlier_map.values()))

    missing_penalty = min(40, missing_ratio * 100)
    duplicate_penalty = min(20, duplicate_ratio * 100)
    outlier_penalty = min(15, (total_outliers / max(rows, 1)) * 100)

    score = int(max(0, min(100, 100 - missing_penalty - duplicate_penalty - outlier_penalty)))

    if score >= 85:
        status = "High data quality"
    elif score >= 65:
        status = "Moderate data quality"
    else:
        status = "Low data quality"

    summary = {
        "rows": rows,
        "columns": cols,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_cols,
        "text_columns": text_cols,
        "outlier_count": total_outliers,
        "score": score,
    }

    return score, status, summary
