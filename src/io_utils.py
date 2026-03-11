from pathlib import Path

import pandas as pd


def load_dataframe(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(uploaded_file)

        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(uploaded_file)

        raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    except Exception as exc:
        raise ValueError(f"Failed to read file: {exc}") from exc


def dataframe_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
