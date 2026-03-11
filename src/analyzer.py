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


def assess_readiness(df: pd.DataFrame, goal_key: str):
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

    base_score = 100 - missing_penalty - duplicate_penalty - outlier_penalty

    if goal_key == "report":
        score = int(max(0, min(100, base_score + 8)))
        if score >= 75:
            status = "Suitable for report building"
        elif score >= 55:
            status = "Needs minor cleaning before report building"
        else:
            status = "Not suitable for report building yet"

    elif goal_key == "analysis":
        score = int(max(0, min(100, base_score)))
        if score >= 80:
            status = "Suitable for data analysis"
        elif score >= 60:
            status = "Needs cleaning before data analysis"
        else:
            status = "Low data quality for analysis"

    elif goal_key == "model":
        model_penalty = 0

        if numeric_cols == 0:
            model_penalty += 20
        if missing_ratio > 0.10:
            model_penalty += 10
        if duplicate_ratio > 0.05:
            model_penalty += 8

        score = int(max(0, min(100, base_score - 8 - model_penalty)))

        if score >= 85:
            status = "Suitable for model building"
        elif score >= 65:
            status = "Needs preprocessing before model building"
        else:
            status = "Not suitable for model building yet"

    else:
        score = int(max(0, min(100, base_score)))
        status = "Goal not selected"

    summary = {
        "rows": rows,
        "columns": cols,
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_cols,
        "text_columns": text_cols,
        "outlier_count": total_outliers,
        "score": score,
        "status": status,
    }

    return score, status, summary
