from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd

from src.analyzer import detect_outliers


def fig_to_bytes(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def build_missing_chart(df: pd.DataFrame):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind="bar", ax=ax)
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing Count")
    return fig_to_bytes(fig)


def build_dtype_chart(df: pd.DataFrame):
    dtype_counts = df.dtypes.astype(str).value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    dtype_counts.plot(kind="bar", ax=ax)
    ax.set_title("Data Types Distribution")
    ax.set_xlabel("Data Type")
    ax.set_ylabel("Count")
    return fig_to_bytes(fig)


def build_outlier_chart(df: pd.DataFrame):
    outliers = detect_outliers(df)

    if not outliers:
        return None

    outlier_series = pd.Series(outliers).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    outlier_series.plot(kind="bar", ax=ax)
    ax.set_title("Outliers by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Outlier Count")
    return fig_to_bytes(fig)
