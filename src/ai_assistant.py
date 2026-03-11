import os

from openai import OpenAI


def contains_arabic(text: str) -> bool:
    if not text:
        return False
    return any("\u0600" <= ch <= "\u06FF" for ch in str(text))


def ask_gpt(question: str, structured: dict, goal_label: str, recommendations: list[str]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Please configure it before running the app.")

    response_language = "Arabic" if contains_arabic(question) else "English"

    prompt = f"""
You are a senior data quality analyst.

Response language:
{response_language}

User Goal:
{goal_label}

Question:
{question}

Dataset Information:
{structured}

Recommendations Draft:
{recommendations}

Rules:
- Do not use the phrase "Ready for AI"
- Evaluate readiness strictly against the selected goal
- Be practical and specific
- Keep the answer structured and concise
- Respond fully in {response_language}
- If the response language is Arabic, write clear professional Arabic

Provide:
1. Executive summary
2. Key findings
3. Readiness score out of 100
4. Whether the dataset is suitable for the selected goal
5. Specific recommended fixes
"""

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
    )

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    return str(response)
