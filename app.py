import os
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="AI Data Quality Analyst",
    page_icon="📊",
    layout="wide"
)

st.title("AI Data Quality Analyst")


# -----------------------
# Load Data
# -----------------------

def load_dataframe(uploaded_file):

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(uploaded_file)

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file format")


# -----------------------
# Dataset Overview
# -----------------------

def dataframe_overview(df):

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# -----------------------
# Quality Score
# -----------------------

def quality_score(df):

    rows, cols = df.shape
    total_cells = max(rows * cols, 1)

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_penalty = min(40, (missing_cells / total_cells) * 100)
    duplicate_penalty = min(20, (duplicate_rows / rows) * 100 if rows else 0)

    score = int(max(0, 100 - missing_penalty - duplicate_penalty))

    if score > 85:
        readiness = "Ready for AI"
    elif score > 65:
        readiness = "Needs Cleaning"
    else:
        readiness = "High Risk Dataset"

    return score, readiness


# -----------------------
# Outlier Detection
# -----------------------

def detect_outliers(df):

    numeric = df.select_dtypes(include="number")

    results = {}

    for col in numeric.columns:

        series = numeric[col].dropna()

        if len(series) < 4:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = series[(series < lower) | (series > upper)]

        results[col] = len(outliers)

    return results


# -----------------------
# Dashboard
# -----------------------

def render_dashboard(df, structured):

    st.subheader("Dashboard")

    score = structured["score"]
    readiness = structured["readiness"]

    st.progress(score, text=f"Data Quality Score: {score}/100")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", structured["rows"])
    col2.metric("Columns", structured["columns"])
    col3.metric("Missing Cells", structured["missing_cells"])
    col4.metric("Duplicate Rows", structured["duplicate_rows"])

    if readiness == "Ready for AI":
        st.success(readiness)
    elif readiness == "Needs Cleaning":
        st.warning(readiness)
    else:
        st.error(readiness)

    tab1, tab2, tab3 = st.tabs(
        ["Missing Values", "Data Types", "Outliers"]
    )

    with tab1:

        st.write("Missing Values by Column")

        missing = df.isna().sum()

        missing_df = pd.DataFrame({
            "column": missing.index,
            "missing": missing.values
        })

        missing_df = missing_df[missing_df["missing"] > 0]

        if not missing_df.empty:
            st.bar_chart(missing_df.set_index("column"))
        else:
            st.info("No missing values")

    with tab2:

        st.write("Data Types Distribution")

        dtype_counts = df.dtypes.astype(str).value_counts()

        dtype_df = pd.DataFrame({
            "dtype": dtype_counts.index,
            "count": dtype_counts.values
        }).set_index("dtype")

        st.bar_chart(dtype_df)

    with tab3:

        st.write("Outliers by Column")

        outliers = detect_outliers(df)

        if outliers:

            outlier_df = pd.DataFrame({
                "column": outliers.keys(),
                "outliers": outliers.values()
            }).set_index("column")

            st.bar_chart(outlier_df)

        else:

            st.info("No numeric outliers detected")


# -----------------------
# PDF Report
# -----------------------

def build_pdf_report(filename, question, structured, analyses):

    buffer = BytesIO()

    pdf = canvas.Canvas(buffer, pagesize=A4)

    y = 800

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "AI Data Quality Report")

    y -= 40

    pdf.setFont("Helvetica", 11)

    pdf.drawString(50, y, f"File: {filename}")
    y -= 20

    pdf.drawString(50, y, f"Question: {question}")
    y -= 30

    pdf.drawString(50, y, "Dataset Overview")
    y -= 20

    pdf.drawString(50, y, f"Rows: {structured['rows']}")
    y -= 20
    pdf.drawString(50, y, f"Columns: {structured['columns']}")
    y -= 20
    pdf.drawString(50, y, f"Missing Cells: {structured['missing_cells']}")
    y -= 20
    pdf.drawString(50, y, f"Duplicate Rows: {structured['duplicate_rows']}")
    y -= 20
    pdf.drawString(50, y, f"Quality Score: {structured['score']}/100")

    y -= 40

    for analysis in analyses:

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, analysis["title"])
        y -= 20

        pdf.setFont("Helvetica", 10)

        for line in analysis["content"].split("\n"):

            if y < 100:
                pdf.showPage()
                y = 800

            pdf.drawString(50, y, line[:100])
            y -= 15

        y -= 20

    pdf.save()

    buffer.seek(0)

    return buffer.getvalue()


# -----------------------
# GPT
# -----------------------

def ask_gpt(client, question, structured, mode):

    prompt = f"""

You are a senior data quality analyst.

User Question:
{question}

Analysis Mode:
{mode}

Dataset Information:
{structured}

Provide:

- Executive summary
- Key findings
- Data quality score /100
- Readiness for AI/ML
- Recommended fixes

Avoid repeating previous analyses.

"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    if hasattr(response, "output_text"):
        return response.output_text

    return str(response)


# -----------------------
# Session
# -----------------------

if "analyses" not in st.session_state:
    st.session_state.analyses = []

if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None


# -----------------------
# Upload
# -----------------------

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload dataset first.")
    st.stop()

df = load_dataframe(uploaded_file)

overview = dataframe_overview(df)

score, readiness = quality_score(df)

structured = {
    **overview,
    "score": score,
    "readiness": readiness
}


# -----------------------
# Question
# -----------------------

with st.form("question_form"):

    question = st.text_area(
        "Ask your question",
        placeholder="Example: Is this dataset ready for training?"
    )

    col1, col2 = st.columns([8,1])

    with col2:
        submitted = st.form_submit_button("Send")


if submitted:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    answer = ask_gpt(client, question, structured, "Auto")

    st.session_state.analyses = [{
        "title": "Primary GPT-5 Analysis",
        "mode": "Auto",
        "content": answer
    }]

    st.session_state.question_submitted = True

    st.session_state.pdf_bytes = build_pdf_report(
        uploaded_file.name,
        question,
        structured,
        st.session_state.analyses
    )


# -----------------------
# Analyses
# -----------------------

if st.session_state.analyses:

    for analysis in st.session_state.analyses:

        st.subheader(analysis["title"])
        st.write(analysis["content"])


# -----------------------
# Preview + Dashboard
# -----------------------

if st.session_state.question_submitted:

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    render_dashboard(df, structured)


# -----------------------
# Sidebar Analysis Mode
# -----------------------

st.sidebar.header("Analysis Mode")

mode = st.sidebar.selectbox(
    "Choose analysis",
    [
        "Select analysis",
        "Missing Values",
        "Duplicate Rows",
        "Data Types Check",
        "Outlier Detection",
    ],
    disabled=not st.session_state.question_submitted
)

apply_mode = st.sidebar.button(
    "Add Analysis",
    disabled=not st.session_state.question_submitted or mode == "Select analysis"
)

if apply_mode:

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    answer = ask_gpt(client, question, structured, mode)

    st.session_state.analyses.append({
        "title": f"Additional Analysis — {mode}",
        "mode": mode,
        "content": answer
    })

    st.session_state.pdf_bytes = build_pdf_report(
        uploaded_file.name,
        question,
        structured,
        st.session_state.analyses
    )


# -----------------------
# Download
# -----------------------

if st.session_state.pdf_bytes:

    st.download_button(
        "Download PDF Report",
        st.session_state.pdf_bytes,
        "data_quality_report.pdf"
    )
