import os
from io import BytesIO
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
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
# Goal-Based Readiness
# -----------------------
def assess_readiness(df, goal):
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)

    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    missing_ratio = missing_cells / total_cells
    duplicate_ratio = (duplicate_rows / rows) if rows else 0

    # القيم الأساسية
    missing_penalty = min(40, missing_ratio * 100)
    duplicate_penalty = min(20, duplicate_ratio * 100)
    base_score = int(max(0, 100 - missing_penalty - duplicate_penalty))

    # تشديد أو تخفيف حسب الهدف
    if goal == "تقرير":
        adjusted_score = min(100, base_score + 8)
        if adjusted_score >= 75:
            status = "جاهزة للتقرير"
        elif adjusted_score >= 55:
            status = "تحتاج تنظيف بسيط قبل التقرير"
        else:
            status = "غير مناسبة للتقرير حالياً"

    elif goal == "تحليل":
        adjusted_score = base_score
        if adjusted_score >= 80:
            status = "جاهزة للتحليل"
        elif adjusted_score >= 60:
            status = "تحتاج تنظيف قبل التحليل"
        else:
            status = "جودة البيانات ضعيفة للتحليل"

    elif goal == "مودل":
        adjusted_score = max(0, base_score - 8)
        if adjusted_score >= 85:
            status = "جاهزة لبناء مودل"
        elif adjusted_score >= 65:
            status = "تحتاج معالجة قبل بناء المودل"
        else:
            status = "غير مناسبة لبناء مودل حالياً"
    else:
        adjusted_score = base_score
        status = "لم يتم تحديد الهدف"

    return adjusted_score, status


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
# Chart Builders
# -----------------------
def fig_to_bytes(fig):
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def build_missing_chart(df):
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


def build_dtype_chart(df):
    dtype_counts = df.dtypes.astype(str).value_counts()

    fig, ax = plt.subplots(figsize=(7, 4))
    dtype_counts.plot(kind="bar", ax=ax)
    ax.set_title("Data Types Distribution")
    ax.set_xlabel("Data Type")
    ax.set_ylabel("Count")
    return fig_to_bytes(fig)


def build_outlier_chart(df):
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


# -----------------------
# Dashboard
# -----------------------
def render_dashboard(df, structured):
    st.subheader("Dashboard")

    score = structured["score"]
    status = structured["status"]
    selected_goal = structured["goal"]

    st.progress(score, text=f"درجة الجاهزية: {score}/100")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", structured["rows"])
    col2.metric("Columns", structured["columns"])
    col3.metric("Missing Cells", structured["missing_cells"])
    col4.metric("Duplicate Rows", structured["duplicate_rows"])

    st.info(f"الهدف المختار: {selected_goal}")

    if "جاهزة" in status:
        st.success(status)
    elif "تحتاج" in status:
        st.warning(status)
    else:
        st.error(status)

    tab1, tab2, tab3 = st.tabs(["Missing Values", "Data Types", "Outliers"])

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
                "column": list(outliers.keys()),
                "outliers": list(outliers.values())
            }).set_index("column")

            st.bar_chart(outlier_df)
        else:
            st.info("No numeric outliers detected")


# -----------------------
# PDF Helpers
# -----------------------
def draw_wrapped_text(pdf, text, x, y, width_chars=95, line_height=14, font_name="Helvetica", font_size=10):
    pdf.setFont(font_name, font_size)

    for paragraph in text.split("\n"):
        wrapped_lines = wrap(paragraph, width=width_chars) if paragraph.strip() else [""]
        for line in wrapped_lines:
            if y < 70:
                pdf.showPage()
                y = 800
                pdf.setFont(font_name, font_size)
            pdf.drawString(x, y, line)
            y -= line_height

    return y


def draw_image_on_pdf(pdf, image_buffer, title, y):
    if image_buffer is None:
        return y

    if y < 320:
        pdf.showPage()
        y = 800

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, title)
    y -= 20

    img = ImageReader(image_buffer)
    pdf.drawImage(img, 50, y - 220, width=500, height=220, preserveAspectRatio=True, mask='auto')
    y -= 250

    return y


# -----------------------
# PDF Report
# -----------------------
def build_pdf_report(filename, question, structured, analyses, df):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    y = 800

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "AI Data Quality Report")
    y -= 35

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"File: {filename}")
    y -= 20
    pdf.drawString(50, y, f"Goal: {structured['goal']}")
    y -= 20

    pdf.drawString(50, y, "Question:")
    y -= 15
    y = draw_wrapped_text(pdf, question, 50, y, width_chars=90, line_height=14, font_size=10)
    y -= 10

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Dataset Overview")
    y -= 20

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, f"Rows: {structured['rows']}")
    y -= 16
    pdf.drawString(50, y, f"Columns: {structured['columns']}")
    y -= 16
    pdf.drawString(50, y, f"Missing Cells: {structured['missing_cells']}")
    y -= 16
    pdf.drawString(50, y, f"Duplicate Rows: {structured['duplicate_rows']}")
    y -= 16
    pdf.drawString(50, y, f"Readiness Score: {structured['score']}/100")
    y -= 16
    pdf.drawString(50, y, f"Status: {structured['status']}")
    y -= 30

    for analysis in analyses:
        if y < 140:
            pdf.showPage()
            y = 800

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, analysis["title"])
        y -= 20

        y = draw_wrapped_text(
            pdf,
            analysis["content"],
            50,
            y,
            width_chars=95,
            line_height=14,
            font_name="Helvetica",
            font_size=10
        )
        y -= 20

    # الرسومات البيانية
    missing_chart = build_missing_chart(df)
    dtype_chart = build_dtype_chart(df)
    outlier_chart = build_outlier_chart(df)

    y = draw_image_on_pdf(pdf, missing_chart, "Chart: Missing Values by Column", y)
    y = draw_image_on_pdf(pdf, dtype_chart, "Chart: Data Types Distribution", y)
    y = draw_image_on_pdf(pdf, outlier_chart, "Chart: Outliers by Column", y)

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# -----------------------
# GPT
# -----------------------
def ask_gpt(client, question, structured, mode):
    prompt = f"""
You are a senior data quality analyst.

User Goal:
{structured['goal']}

User Question:
{question}

Analysis Mode:
{mode}

Dataset Information:
{structured}

Instructions:
- Explain whether the dataset is suitable for the selected goal
- Do not use the phrase "Ready for AI"
- Use the selected goal instead:
  - if the goal is report, evaluate report readiness
  - if the goal is analysis, evaluate analysis readiness
  - if the goal is model, evaluate model readiness

Provide:
- Executive summary
- Key findings
- Readiness score /100
- Is it ready for the selected goal or not
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

if "submitted_message" not in st.session_state:
    st.session_state.submitted_message = ""


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

# -----------------------
# Goal Selection
# -----------------------
user_goal = st.radio(
    "وش هدفك من البيانات؟",
    ["تقرير", "تحليل", "مودل"],
    horizontal=True
)

overview = dataframe_overview(df)
score, status = assess_readiness(df, user_goal)

structured = {
    **overview,
    "score": score,
    "status": status,
    "goal": user_goal
}


# -----------------------
# Question
# -----------------------
with st.form("question_form"):
    question = st.text_area(
        "Ask your question",
        placeholder="مثال: هل هذه البيانات مناسبة لهدفي؟"
    )

    col1, col2 = st.columns([8, 1])

    with col2:
        submitted = st.form_submit_button("Send")

if submitted:
    with st.spinner("جاري الإرسال والتحليل..."):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        answer = ask_gpt(client, question, structured, "Auto")

        st.session_state.analyses = [{
            "title": "Primary GPT-5 Analysis",
            "mode": "Auto",
            "content": answer
        }]

        st.session_state.question_submitted = True
        st.session_state.submitted_message = "تم الإرسال بنجاح ✅"

        st.session_state.pdf_bytes = build_pdf_report(
            uploaded_file.name,
            question,
            structured,
            st.session_state.analyses,
            df
        )

if st.session_state.submitted_message:
    st.success(st.session_state.submitted_message)


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
    with st.spinner("جاري إضافة التحليل..."):
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
            st.session_state.analyses,
            df
        )

    st.success("تمت إضافة التحليل بنجاح ✅")


# -----------------------
# Download
# -----------------------
if st.session_state.pdf_bytes:
    st.download_button(
        "Download PDF Report",
        st.session_state.pdf_bytes,
        "data_quality_report.pdf",
        mime="application/pdf"
    )
