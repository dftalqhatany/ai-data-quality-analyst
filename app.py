import os

import pandas as pd
import streamlit as st

from src.ai_assistant import ask_gpt
from src.analyzer import assess_readiness, detect_outliers, goal_recommendations
from src.io_utils import dataframe_overview, load_dataframe
from src.pdf_report import build_pdf_report


st.set_page_config(
    page_title="AI Data Quality Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("AI Data Quality Analyst")
st.caption("Upload your file, choose your goal, and evaluate whether the dataset is ready for that purpose.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Please configure it before running the app.")
    st.stop()


def render_dashboard(df, structured):
    st.subheader("Dashboard")

    score = structured["score"]
    st.progress(score, text=f"Readiness Score: {score}/100")

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


goal_options = {
    "report": "Build Report",
    "analysis": "Data Analysis",
    "model": "Build Model",
}

goal_key = st.radio(
    "Select your goal",
    options=list(goal_options.keys()),
    format_func=lambda x: goal_options[x],
    horizontal=True,
)

goal_label = goal_options[goal_key]

overview = dataframe_overview(df)
score, status, extra = assess_readiness(df, goal_key)

structured = {
    **overview,
    **extra,
    "goal": goal_label,
}

recommendations = goal_recommendations(goal_key, structured)

action_label_map = {
    "report": "Build Report Readiness Assessment",
    "analysis": "Data Analysis Readiness Assessment",
    "model": "Build Model Readiness Assessment",
}

st.markdown(f"### {action_label_map.get(goal_key, 'Dataset Readiness Assessment')}")


with st.form("question_form"):
    question = st.text_area(
        "Ask your question",
        placeholder="Example: Is this dataset suitable for the selected goal?",
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
            final_question = question or "Assess the dataset for the selected goal."

            analysis_text = ask_gpt(
                question=final_question,
                structured=structured,
                goal_label=goal_label,
                recommendations=recommendations,
            )

            pdf_bytes = build_pdf_report(
                filename=uploaded_file.name,
                question=final_question,
                structured=structured,
                analysis_text=analysis_text,
                df=df,
                recommendations=recommendations,
                goal_label=goal_label,
            )

            st.session_state.analysis_text = analysis_text
            st.session_state.pdf_bytes = pdf_bytes
            st.session_state.question_submitted = True

            st.success("Analysis completed and PDF report generated successfully ✅")

        except Exception as exc:
            st.error(f"Analysis or PDF generation failed: {exc}")


st.subheader("Readiness Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Score", f"{structured['score']}/100")
c2.metric("Rows", structured["rows"])
c3.metric("Columns", structured["columns"])
c4.metric("Missing Cells", structured["missing_cells"])


if st.session_state.analysis_text:
    st.subheader(f"Primary Assessment — {goal_label}")
    st.write(st.session_state.analysis_text)


if st.session_state.question_submitted:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    render_dashboard(df, structured)


if st.session_state.question_submitted:
    st.subheader("Recommended Actions")
    for rec in recommendations:
        st.write(f"- {rec}")


if st.session_state.pdf_bytes is not None:
    st.subheader("Download Report")
    st.download_button(
        label="Download PDF Report",
        data=st.session_state.pdf_bytes,
        file_name="data_quality_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
