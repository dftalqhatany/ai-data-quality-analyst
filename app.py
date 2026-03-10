
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.agent import build_analysis_context, generate_final_report
from src.analyzer import analyze_dataset
from src.io_utils import load_dataframe, dataframe_overview

st.set_page_config(page_title="AI Data Quality Analyst", page_icon="📊", layout="wide")

st.title("📊 AI Data Quality Analyst")
st.caption("Upload a dataset and generate a data quality report for AI readiness.")

with st.sidebar:
    st.header("Configuration")
    analysis_mode = st.selectbox(
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

col1, col2 = st.columns([2,1])

with col1:
    uploaded_file = st.file_uploader("Upload dataset", type=["csv","xlsx","xls"])

with col2:
    user_goal = st.text_area(
        "Agent task",
        placeholder="Example: Check if this dataset is ready for AI training"
    )

run_clicked = st.button("Analyze Data")

if run_clicked:
    if uploaded_file is None:
        st.error("Upload dataset first")
        st.stop()

    df = load_dataframe(uploaded_file)

    analysis = analyze_dataset(df, analysis_mode)
    context = build_analysis_context(
        user_goal=user_goal,
        filename=uploaded_file.name,
        analysis_mode=analysis_mode,
        analysis=analysis
    )

    report = generate_final_report(context)

    st.success("Analysis completed")

    overview = dataframe_overview(df)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows", overview["rows"])
    c2.metric("Columns", overview["columns"])
    c3.metric("Missing", overview["missing_cells"])
    c4.metric("Duplicates", overview["duplicate_rows"])

    st.metric("Quality Score", f"{analysis['quality_score']}/100")
    st.info(f"Dataset Readiness: {analysis['readiness']}")

    st.subheader("Executive Summary")
    st.write(report["executive_summary"])

    st.subheader("Recommendations")
    for r in report["recommendations"]:
        st.markdown(f"- {r}")

    st.subheader("Issues Table")
    st.dataframe(pd.DataFrame(analysis["issue_summary"]))

    st.subheader("Column Risks")
    st.dataframe(pd.DataFrame(analysis["column_risks"]))

    st.subheader("Preview")
    st.dataframe(df.head())

    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "analysis": analysis,
        "report": report
    }

    st.download_button(
        "Download JSON report",
        data=json.dumps(payload,indent=2),
        file_name="data_quality_report.json"
    )
