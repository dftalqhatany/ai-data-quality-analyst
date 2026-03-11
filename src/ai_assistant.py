import os
from openai import OpenAI


def contains_arabic(text: str) -> bool:
    if not text:
        return False
    return any("\u0600" <= ch <= "\u06FF" for ch in str(text))


def ask_gpt(question: str, dataset_context: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    response_language = "Arabic" if contains_arabic(question) else "English"

    prompt = f"""
You are a senior data quality analyst.

User Question:
{question}

Dataset Context:
{dataset_context}

Language:
{response_language}

Rules:
- Base your answer strictly on the dataset context.
- If the user asks about missing values by column, use missing_by_column.
- Focus only on data quality assessment.
- Be clear and practical.

Provide:
1. Executive summary
2. Direct answer to the question
3. Key data quality findings
4. Data quality score out of 100
5. Recommended fixes
"""

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
    )

    if hasattr(response, "output_text"):
        return response.output_text

    return str(response)
