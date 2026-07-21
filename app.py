import os

import pandas as pd
import streamlit as st
import pandas as pd

from src.ai_assistant import ask_gpt
from src.analyzer import assess_readiness, detect_outliers
from src.io_utils import dataframe_overview, load_dataframe
from src.pdf_report import build_pdf_report


st.set_page_config(
    page_title="AI Data Quality Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("AI Data Quality Analyst")
st.caption("Upload your file and assess its overall data quality.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Please configure it before running the app.")
    st.stop()


def build_dataset_context(df: pd.DataFrame, structured: dict) -> dict:
    missing_by_column = df.isna().sum().sort_values(ascending=False)
    missing_by_column = {k: int(v) for k, v in missing_by_column.items() if int(v) > 0}

    column_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    column_names = list(df.columns)

    sample_rows = df.head(5).fillna("").astype(str).to_dict(orient="records")

    numeric_df = df.select_dtypes(include="number")
    numeric_summary = {}
    if not numeric_df.empty:
        desc = numeric_df.describe().T.fillna(0)
        for col in desc.index:
            numeric_summary[col] = {
                "count": float(desc.loc[col, "count"]) if "count" in desc.columns else 0,
                "mean": float(desc.loc[col, "mean"]) if "mean" in desc.columns else 0,
                "std": float(desc.loc[col, "std"]) if "std" in desc.columns else 0,
                "min": float(desc.loc[col, "min"]) if "min" in desc.columns else 0,
                "25%": float(desc.loc[col, "25%"]) if "25%" in desc.columns else 0,
                "50%": float(desc.loc[col, "50%"]) if "50%" in desc.columns else 0,
                "75%": float(desc.loc[col, "75%"]) if "75%" in desc.columns else 0,
                "max": float(desc.loc[col, "max"]) if "max" in desc.columns else 0,
            }

    categorical_summary = {}
    object_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in object_cols:
        top_values = df[col].astype(str).replace("nan", pd.NA).dropna().value_counts().head(5)
        categorical_summary[col] = top_values.to_dict()

    outlier_map = detect_outliers(df)

    dataset_context = {
        "overview": structured,
        "column_names": column_names,
        "column_types": column_types,
        "missing_by_column": missing_by_column,
        "numeric_summary": numeric_summary,
        "categorical_top_values": categorical_summary,
        "outliers_by_column": outlier_map,
        "sample_rows": sample_rows,
    }

    return dataset_context


def render_dashboard(df, structured):
    st.subheader("Data Quality Dashboard")

    score = structured["score"]
    st.progress(score, text=f"Data Quality Score: {score}/100")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", structured["rows"])
    c2.metric("Columns", structured["columns"])
    c3.metric("Missing Cells", structured["missing_cells"])
    c4.metric("Duplicate Rows", structured["duplicate_rows"])

    tab1, tab2, tab3 = st.tabs(["Missing Values", "Data Types", "Outliers"])

    with tab1:
        st.write("Missing Values by Column")
        missing = df.isna().sum()
        missing_df = pd.DataFrame({
            "column": missing.index,
            "missing": missing.values,
        })
        missing_df = missing_df[missing_df["missing"] > 0]

        if not missing_df.empty:
            st.bar_chart(missing_df.set_index("column"))
        else:
            st.info("No missing values found.")

    with tab2:
        st.write("Data Types Distribution")
        dtype_counts = df.dtypes.astype(str).value_counts()
        dtype_df = pd.DataFrame({
            "dtype": dtype_counts.index,
            "count": dtype_counts.values,
        }).set_index("dtype")
        st.bar_chart(dtype_df)

    with tab3:
        st.write("Outliers by Column")
        outliers = detect_outliers(df)

        if outliers:
            outlier_df = pd.DataFrame({
                "column": list(outliers.keys()),
                "outliers": list(outliers.values()),
            }).set_index("column")
            st.bar_chart(outlier_df)
        else:
            st.info("No numeric outliers detected.")


if "analysis_text" not in st.session_state:
    st.session_state.analysis_text = ""

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False


uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

if uploaded_file is None:
    st.info("Please upload a dataset first.")
    st.stop()

try:
    df = load_dataframe(uploaded_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if df.empty:
    st.warning("The uploaded dataset is empty.")
    st.stop()


overview = dataframe_overview(df)
score, status, extra = assess_readiness(df)

structured = {
    **overview,
    **extra,
    "status": status,
}

dataset_context = build_dataset_context(df, structured)

st.markdown("### Data Quality Assessment")


with st.form("question_form"):
    question = st.text_area(
        "Ask your question",
        placeholder="Example: What columns have the highest number of missing values?",
    )
    _, button_col = st.columns([6, 1])
    with button_col:
        submitted = st.form_submit_button("Send", use_container_width=True)


if submitted:
    with st.spinner("Analyzing dataset..."):
        st.session_state.analysis_text = ""
        st.session_state.pdf_bytes = None
        st.session_state.question_submitted = False

        try:
            final_question = question or "Assess the overall quality of this dataset."

            analysis_text = ask_gpt(
                question=final_question,
                dataset_context=dataset_context,
            )

            pdf_bytes = build_pdf_report(
                filename=uploaded_file.name,
                question=final_question,
                structured=structured,
                analysis_text=analysis_text,
                df=df,
            )

            st.session_state.analysis_text = analysis_text
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.question_submitted = True

            st.success("Analysis completed and PDF report generated successfully ✅")

        except Exception as exc:
            st.error(f"Analysis or PDF generation failed: {exc}")


st.subheader("Data Quality Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Score", f"{structured['score']}/100")
c2.metric("Rows", structured["rows"])
c3.metric("Columns", structured["columns"])
c4.metric("Missing Cells", structured["missing_cells"])


if st.session_state.analysis_text:
    st.subheader("Primary Assessment")
    st.write(st.session_state.analysis_text)


if st.session_state.question_submitted:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    render_dashboard(df, structured)

    st.subheader("Missing Values by Column")
    missing_display = df.isna().sum().sort_values(ascending=False)
    missing_display = missing_display[missing_display > 0]

    if not missing_display.empty:
        st.dataframe(
            pd.DataFrame({
                "column": missing_display.index,
                "missing_count": missing_display.values
            }),
            use_container_width=True,
        )
    else:
        st.info("No missing values found in the dataset.")


if st.session_state.pdf_bytes is not None:
    st.subheader("Download Report")
    st.download_button(
        label="Download PDF Report",
        data=st.session_state.pdf_bytes,
        file_name="data_quality_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
