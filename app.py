import os
import sys
from pathlib import Path

import streamlit as st
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from analyzer import analyze_dataset
from agent import (
    build_analysis_context,
    generate_final_report,
    infer_task_focus,
)
from io_utils import load_dataframe, dataframe_overview

st.set_page_config(page_title="AI Data Quality Analyst", layout="wide")
st.title("AI Data Quality Analyst (GPT-5)")

st.sidebar.header("Configuration")

analysis_mode = st.sidebar.selectbox(
    "Analysis mode",
    [
        "Full Quality Audit",
        "Missing Values",
        "Duplicate Rows",
        "Data Types Check",
        "Outlier Detection",
        "Column Consistency",
    ],
)

agent_task = st.text_area(
    "Agent task",
    placeholder="Example: check if dataset has null values, duplicates, or outliers"
)

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"]
)

run_analysis = st.button("Run Analysis")
run_gpt = st.button("Analyze with GPT-5")

if uploaded_file is not None:
    try:
        df = load_dataframe(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    overview = dataframe_overview(df)

    # interpret free-text task
    task_focus = infer_task_focus(agent_task)

    # run selected mode, but keep task focus for report + GPT prompt
    analysis = analyze_dataset(df, analysis_mode=analysis_mode)

    context = build_analysis_context(
        user_goal=agent_task if agent_task.strip() else "Assess data quality for AI readiness",
        filename=uploaded_file.name,
        analysis_mode=analysis_mode,
        analysis=analysis,
        task_focus=task_focus,
    )

    final_report = generate_final_report(context)

    if run_analysis or run_gpt:
        st.subheader("Overview")
        st.json(overview)

        st.subheader("Detected Agent Intent")
        st.write(task_focus)

        st.subheader("Analysis Results")

        if analysis_mode == "Missing Values":
            st.json(analysis.get("missing_by_column", []))
            st.metric("Total Missing Cells", analysis.get("total_missing_values", 0))

        elif analysis_mode == "Duplicate Rows":
            st.metric("Duplicate Rows", analysis.get("duplicate_rows", 0))

        elif analysis_mode == "Data Types Check":
            st.json(analysis.get("mixed_type_columns", []))
            st.json(analysis.get("dtypes", {}))

        elif analysis_mode == "Outlier Detection":
            st.json(analysis.get("outlier_summary", []))

        elif analysis_mode == "Column Consistency":
            st.json(analysis.get("consistency_issues", []))

        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Quality Score", analysis.get("quality_score", 0))
            with col2:
                st.metric("Readiness", analysis.get("readiness", "Unknown"))
            st.json(analysis)

        st.subheader("Final Report")
        st.json(final_report)

    if run_gpt:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.error("OPENAI_API_KEY is missing. Set it in your terminal before running the app.")
            st.stop()

        try:
            client = OpenAI(api_key=api_key)

            prompt = f"""
You are a senior data quality analyst.

User task:
{agent_task if agent_task.strip() else "No custom task provided"}

Detected task focus:
{task_focus}

Selected analysis mode:
{analysis_mode}

Filename:
{uploaded_file.name}

Dataset overview:
{overview}

Rule-based analysis output:
{analysis}

Final local report:
{final_report}

Instructions:
- Answer based on the user's task and the selected analysis mode.
- If the task asks about null values, focus heavily on missing values.
- If the task asks about duplicates, focus heavily on duplicate rows.
- If the task asks about outliers, focus heavily on outlier detection.
- If the task asks about types, focus heavily on data type issues.
- If the task asks about consistency, focus heavily on text/categorical consistency.
- If the mode is Full Quality Audit, provide a broad assessment.

Return:
1. Executive summary
2. Main findings
3. Business risk
4. Recommended fixes
5. Priority next steps
"""

            with st.spinner("Analyzing with GPT-5..."):
                response = client.responses.create(
                    model="gpt-5",
                    input=prompt
                )

            st.subheader("GPT-5 Analysis")
            st.write(response.output_text)

        except Exception as e:
            st.error(f"OpenAI API error: {e}")
else:
    st.info("Upload a file first, then click Run Analysis or Analyze with GPT-5.")
