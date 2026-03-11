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

st.title("📊 AI Data Quality Analyst")
st.caption("Upload your file, choose your goal, and evaluate whether the dataset is ready for that purpose.")


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
        results[col] = int(len(outliers))

    return results


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

    numeric_cols = df.select_dtypes(include="number").shape[1]
    object_cols = df.select_dtypes(include="object").shape[1]
    outliers = detect_outliers(df)
    total_outliers = sum(outliers.values())

    missing_penalty = min(40, missing_ratio * 100)
    duplicate_penalty = min(20, duplicate_ratio * 100)
    outlier_penalty = min(15, (total_outliers / max(rows, 1)) * 100)

    base_score = 100 - missing_penalty - duplicate_penalty - outlier_penalty

    if goal == "Build Report":
        score = int(max(0, min(100, base_score + 8)))
        if score >= 75:
            status = "Ready for report building"
        elif score >= 55:
            status = "Needs minor cleaning before report building"
        else:
            status = "Not suitable for report building yet"

    elif goal == "Run Analysis":
        score = int(max(0, min(100, base_score)))
        if score >= 80:
            status = "Ready for analysis"
        elif score >= 60:
            status = "Needs cleaning before analysis"
        else:
            status = "Low data quality for analysis"

    elif goal == "Build Model":
        model_penalty = 0

        if numeric_cols == 0:
            model_penalty += 20

        if missing_ratio > 0.10:
            model_penalty += 10

        if duplicate_ratio > 0.05:
            model_penalty += 8

        score = int(max(0, min(100, base_score - 8 - model_penalty)))

        if score >= 85:
            status = "Ready for model building"
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
        "numeric_columns": int(numeric_cols),
        "text_columns": int(object_cols),
        "outlier_count": int(total_outliers),
    }

    return score, status, summary


# -----------------------
# Recommendations by Goal
# -----------------------
def goal_recommendations(goal, structured):
    recs = []

    if structured["missing_cells"] > 0:
        recs.append("Handle missing values before proceeding.")
    if structured["duplicate_rows"] > 0:
        recs.append("Remove duplicate rows to improve consistency.")
    if structured["outlier_count"] > 0:
        recs.append("Review numeric outliers and validate extreme values.")

    if goal == "Build Report":
        recs.append("Ensure key business columns are complete for accurate reporting.")
        recs.append("Standardize labels and categories for cleaner report visuals.")

    elif goal == "Run Analysis":
        recs.append("Validate data types before running analytical workflows.")
        recs.append("Check column consistency and business logic across fields.")

    elif goal == "Build Model":
        recs.append("Encode categorical columns before model training.")
        recs.append("Split features and target clearly before building the model.")
        recs.append("Consider scaling numeric features if required by the algorithm.")
        recs.append("Review class balance if this dataset will be used for classification.")

    return recs


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
    goal = structured["goal"]

    st.progress(score, text=f"Readiness Score: {score}/100")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", structured["rows"])
    c2.metric("Columns", structured["columns"])
    c3.metric("Missing Cells", structured["missing_cells"])
    c4.metric("Duplicate Rows", structured["duplicate_rows"])

    st.info(f"Selected Goal: {goal}")

    if "Ready" in status:
        st.success(status)
    elif "Needs" in status:
        st.warning(status)
    else:
        st.error(status)

    tab1, tab2, tab3 = st.tabs(["Missing Values", "Data Types", "Outliers"])

    with tab1:
        st.write("Missing Values by Column")
        missing = df.isna().sum()
        missing_df = pd.DataFrame({"column": missing.index, "missing": missing.values})
        missing_df = missing_df[missing_df["missing"] > 0]

        if not missing_df.empty:
            st.bar_chart(missing_df.set_index("column"))
        else:
            st.info("No missing values found.")

    with tab2:
        st.write("Data Types Distribution")
        dtype_counts = df.dtypes.astype(str).value_counts()
        dtype_df = pd.DataFrame({"dtype": dtype_counts.index, "count": dtype_counts.values}).set_index("dtype")
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
            st.info("No numeric outliers detected.")


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
    pdf.drawImage(img, 50, y - 220, width=500, height=220, preserveAspectRatio=True, mask="auto")
    y -= 250

    return y


# -----------------------
# PDF Report
# -----------------------
def build_pdf_report(filename, question, structured, analyses, df, recommendations):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    y = 800

    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(50, y, "AI Data Quality Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"File: {filename}")
    y -= 18
    pdf.drawString(50, y, f"Goal: {structured['goal']}")
    y -= 18
    pdf.drawString(50, y, f"Readiness Score: {structured['score']}/100")
    y -= 18
    pdf.drawString(50, y, f"Status: {structured['status']}")
    y -= 28

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Question")
    y -= 16
    y = draw_wrapped_text(pdf, question or "No question provided.", 50, y, width_chars=92)
    y -= 16

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Dataset Overview")
    y -= 16

    overview_lines = [
        f"Rows: {structured['rows']}",
        f"Columns: {structured['columns']}",
        f"Missing Cells: {structured['missing_cells']}",
        f"Duplicate Rows: {structured['duplicate_rows']}",
        f"Numeric Columns: {structured['numeric_columns']}",
        f"Text Columns: {structured['text_columns']}",
        f"Detected Outliers: {structured['outlier_count']}",
    ]

    y = draw_wrapped_text(pdf, "\n".join(overview_lines), 50, y, width_chars=90)
    y -= 10

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Recommendations")
    y -= 16
    rec_text = "\n".join([f"- {r}" for r in recommendations])
    y = draw_wrapped_text(pdf, rec_text, 50, y, width_chars=92)
    y -= 16

    for analysis in analyses:
        if y < 140:
            pdf.showPage()
            y = 800

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, analysis["title"])
        y -= 18

        y = draw_wrapped_text(pdf, analysis["content"], 50, y, width_chars=95)
        y -= 16

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

Question:
{question}

Analysis Mode:
{mode}

Dataset Information:
{structured}

Rules:
- Do not use the phrase "Ready for AI"
- Evaluate readiness strictly against the selected goal
- Be practical and specific

Provide:
1. Executive summary
2. Key findings
3. Readiness score out of 100
4. Whether the dataset is ready for the selected goal
5. Specific recommended fixes
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    if hasattr(response, "output_text"):
        return response.output_text

    return str(response)


# -----------------------
# Session State
# -----------------------
if "analyses" not in st.session_state:
    st.session_state.analyses = []

if "question_submitted" not in st.session_state:
    st.session_state.question_submitted = False

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

if "submitted_message" not in st.session_state:
    st.session_state.submitted_message = ""

if "last_goal" not in st.session_state:
    st.session_state.last_goal = None


# -----------------------
# Upload
# -----------------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Please upload a dataset first.")
    st.stop()

df = load_dataframe(uploaded_file)


# -----------------------
# Goal Selection
# -----------------------
goal = st.radio(
    "Select your goal",
    ["Build Report", "Data Analysis", "Build Model"],
    horizontal=True
)

overview = dataframe_overview(df)
score, status, extra = assess_readiness(df, goal)

structured = {
    **overview,
    **extra,
    "score": score,
    "status": status,
    "goal": goal
}

recommendations = goal_recommendations(goal, structured)

if st.session_state.last_goal != goal:
    st.session_state.last_goal = goal
    st.session_state.submitted_message = ""


# -----------------------
# Main Action Selector
# -----------------------
action_label = {
    "Build Report": "Build Report Readiness Assessment",
    "Data Analysis": "Run Analysis Readiness Assessment",
    "Build Model": "Build Model Readiness Assessment"
}[goal]

st.markdown(f"### {action_label}")

with st.form("question_form"):
    question = st.text_area(
        "Ask your question",
        placeholder="Example: Is this dataset suitable for the selected goal?"
    )

    submitted = st.form_submit_button("Send")


# -----------------------
# Submit Action
# -----------------------
if submitted:
    with st.spinner("Sending request and analyzing dataset..."):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        answer = ask_gpt(client, question or "Assess the dataset for the selected goal.", structured, "Auto")

        st.session_state.analyses = [{
            "title": f"Primary Assessment — {goal}",
            "mode": "Auto",
            "content": answer
        }]

        st.session_state.question_submitted = True
        st.session_state.submitted_message = "Request sent successfully ✅"

        st.session_state.pdf_bytes = build_pdf_report(
            uploaded_file.name,
            question or "Assess the dataset for the selected goal.",
            structured,
            st.session_state.analyses,
            df,
            recommendations
        )

if st.session_state.submitted_message:
    st.success(st.session_state.submitted_message)


# -----------------------
# Quick Readiness Summary
# -----------------------
st.subheader("Readiness Summary")
c1, c2 = st.columns([1, 2])

with c1:
    st.metric("Score", f"{structured['score']}/100")

with c2:
    if "Ready" in structured["status"]:
        st.success(structured["status"])
    elif "Needs" in structured["status"]:
        st.warning(structured["status"])
    else:
        st.error(structured["status"])


# -----------------------
# Recommendations UI
# -----------------------
with st.expander("Recommended Actions", expanded=True):
    for rec in recommendations:
        st.write(f"- {rec}")


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
# Sidebar Extra Analyses
# -----------------------
st.sidebar.header("Additional Analyses")

mode = st.sidebar.selectbox(
    "Choose an additional analysis",
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
    with st.spinner("Adding analysis..."):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        answer = ask_gpt(
            client,
            question or "Assess the dataset for the selected goal.",
            structured,
            mode
        )

        st.session_state.analyses.append({
            "title": f"Additional Analysis — {mode}",
            "mode": mode,
            "content": answer
        })

        st.session_state.pdf_bytes = build_pdf_report(
            uploaded_file.name,
            question or "Assess the dataset for the selected goal.",
            structured,
            st.session_state.analyses,
            df,
            recommendations
        )

    st.success("Analysis added successfully ✅")


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
