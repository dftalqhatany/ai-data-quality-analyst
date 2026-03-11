from io import BytesIO
from pathlib import Path
from textwrap import wrap
import urllib.request

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from src.charts import build_dtype_chart, build_missing_chart, build_outlier_chart


PAGE_WIDTH, PAGE_HEIGHT = A4
RIGHT_MARGIN_X = 545

ARABIC_FONT_NAME = "Cairo"
ARABIC_FONT_PATH = Path("Cairo.ttf")
FONT_URL = "https://raw.githubusercontent.com/google/fonts/main/ofl/cairo/Cairo%5Bslnt%2Cwght%5D.ttf"


def register_arabic_font():
    if not ARABIC_FONT_PATH.exists():
        urllib.request.urlretrieve(FONT_URL, ARABIC_FONT_PATH)

    registered_fonts = pdfmetrics.getRegisteredFontNames()
    if ARABIC_FONT_NAME not in registered_fonts:
        pdfmetrics.registerFont(TTFont(ARABIC_FONT_NAME, str(ARABIC_FONT_PATH)))


def contains_arabic(text: str) -> bool:
    if not text:
        return False
    return any("\u0600" <= ch <= "\u06FF" for ch in str(text))


def prepare_arabic_text(text: str) -> str:
    reshaped = arabic_reshaper.reshape(str(text))
    return get_display(reshaped)


def normalize_pdf_text(text: str) -> str:
    text = str(text)
    if contains_arabic(text):
        return prepare_arabic_text(text)
    return text


def draw_section_title(pdf, title: str, y: int):
    if contains_arabic(title):
        pdf.setFont(ARABIC_FONT_NAME, 12)
        pdf.drawRightString(RIGHT_MARGIN_X, y, normalize_pdf_text(title))
    else:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, title)
    return y - 16


def draw_wrapped_text(pdf, text, x, y, width_chars=95, line_height=14, font_name="Helvetica", font_size=10, rtl=False):
    pdf.setFont(font_name, font_size)

    for paragraph in str(text).split("\n"):
        content = normalize_pdf_text(paragraph) if rtl else paragraph
        wrapped_lines = wrap(content, width=width_chars) if paragraph.strip() else [""]

        for line in wrapped_lines:
            if y < 70:
                pdf.showPage()
                y = 800
                pdf.setFont(font_name, font_size)

            if rtl:
                pdf.drawRightString(RIGHT_MARGIN_X, y, line)
            else:
                pdf.drawString(x, y, line)

            y -= line_height

    return y


def draw_key_value_line(pdf, label: str, value: str, y: int):
    full_text = f"{label}: {value}"

    if contains_arabic(full_text):
        pdf.setFont(ARABIC_FONT_NAME, 11)
        pdf.drawRightString(RIGHT_MARGIN_X, y, normalize_pdf_text(full_text))
    else:
        pdf.setFont("Helvetica", 11)
        pdf.drawString(50, y, full_text)

    return y - 18


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

    pdf.drawImage(
        img,
        50,
        y - 220,
        width=500,
        height=220,
        preserveAspectRatio=True,
        mask="auto",
    )

    y -= 250
    return y


def build_pdf_report(filename, question, structured, analysis_text, df):
    register_arabic_font()

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    y = 800

    report_title = "تقرير جودة البيانات" if contains_arabic(question) else "Data Quality Report"

    if contains_arabic(report_title):
        pdf.setFont(ARABIC_FONT_NAME, 15)
        pdf.drawRightString(RIGHT_MARGIN_X, y, normalize_pdf_text(report_title))
    else:
        pdf.setFont("Helvetica-Bold", 15)
        pdf.drawString(50, y, report_title)

    y -= 30

    y = draw_key_value_line(pdf, "File", filename, y)
    y = draw_key_value_line(pdf, "Data Quality Score", f"{structured['score']}/100", y)
    y = draw_key_value_line(pdf, "Status", structured["status"], y)

    y -= 10

    question_title = "السؤال" if contains_arabic(question) else "Question"
    y = draw_section_title(pdf, question_title, y)

    question_is_arabic = contains_arabic(question)

    y = draw_wrapped_text(
        pdf,
        question or "Assess the quality of this dataset.",
        50,
        y,
        width_chars=92,
        font_name=ARABIC_FONT_NAME if question_is_arabic else "Helvetica",
        font_size=10,
        rtl=question_is_arabic,
    )

    y -= 16

    overview_title = "نظرة عامة على البيانات" if contains_arabic(question) else "Dataset Overview"
    y = draw_section_title(pdf, overview_title, y)

    overview_lines = [
        f"Rows: {structured['rows']}",
        f"Columns: {structured['columns']}",
        f"Missing Cells: {structured['missing_cells']}",
        f"Duplicate Rows: {structured['duplicate_rows']}",
        f"Numeric Columns: {structured['numeric_columns']}",
        f"Text Columns: {structured['text_columns']}",
        f"Detected Outliers: {structured['outlier_count']}",
    ]

    y = draw_wrapped_text(
        pdf,
        "\n".join(overview_lines),
        50,
        y,
        width_chars=90,
        font_name="Helvetica",
        font_size=10,
        rtl=False,
    )

    y -= 10

    analysis_title = "التقييم" if contains_arabic(analysis_text) else "Assessment"
    y = draw_section_title(pdf, analysis_title, y)

    analysis_is_arabic = contains_arabic(analysis_text)

    y = draw_wrapped_text(
        pdf,
        analysis_text,
        50,
        y,
        width_chars=95,
        font_name=ARABIC_FONT_NAME if analysis_is_arabic else "Helvetica",
        font_size=10,
        rtl=analysis_is_arabic,
    )

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
