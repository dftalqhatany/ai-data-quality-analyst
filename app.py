import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

# Add src folder to path
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from analyzer import analyze_dataset
from agent import build_analysis_context, generate_final_report
from io_utils import load_dataframe, dataframe_overview

st.set_page_config(page_title="AI Data Quality Analyst", layout="wide")
st.title("AI Data Quality Analyst (GPT-5)")

api_key = os.getenv("OPENAI_API_KEY")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
analysis_mode = st.selectbox(
    "Analysis Mode",
    ["Full Quality Audit", "Quick Scan"]
)
user_goal = st.text_area(
    "What do you want from the analysis?",
    placeholder="Example: Tell me whether this dataset is ready for AI model training."
)

if uploaded_file is not None:
    try:
        df = load_dataframe(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

    st.subheader("Preview")
    st.dataframe(df.head())

    overview = dataframe_overview(df)
    analysis = analyze_dataset(df, analysis_mode=analysis_mode)

    context = build_analysis_context(
        user_goal=user_goal if user_goal else "Assess data quality for AI readiness",
        filename=uploaded_file.name,
        analysis_mode=analysis_mode,
        analysis=analysis
    )

    final_report = generate_final_report(context)

    st.subheader("Local Analysis")
    st.write("**Overview**")
    st.json(overview)

    st.write("**Analysis Output**")
    st.json(analysis)

    st.write("**Final Report**")
    st.json(final_report)

    if st.button("Analyze with GPT-5"):
        if not api_key:
            st.error("OPENAI_API_KEY is missing. Set it in your environment before running the app.")
            st.stop()

        try:
            client = OpenAI(api_key=api_key)

            prompt = f"""
You are a senior data quality analyst.

The user goal is:
{context["user_goal"]}

Filename:
{context["filename"]}

Analysis mode:
{context["analysis_mode"]}

Dataset overview:
{overview}

Rule-based analysis:
{analysis}

Generated local report:
{final_report}

Please provide:
1. Executive summary
2. Key data quality risks
3. AI readiness assessment
4. Recommended cleaning steps
5. Priority actions before model training
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
