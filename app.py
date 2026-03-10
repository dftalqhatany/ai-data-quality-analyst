import streamlit as st
import pandas as pd
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("AI Data Quality Analyst (GPT-5)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview")
    st.dataframe(df.head())

    report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }

    if st.button("Analyze with GPT-5"):

        response = client.responses.create(
            model="gpt-5",
            input=f"""
            Analyze this dataset quality report and give insights:

            {report}
            """
        )

        st.write(response.output_text)
