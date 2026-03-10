import pandas as pd
from pathlib import Path

def load_dataframe(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported format")

def dataframe_overview(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }
