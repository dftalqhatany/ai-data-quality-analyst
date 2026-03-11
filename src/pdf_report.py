from io import BytesIO
from textwrap import wrap

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from src.charts import build_dtype_chart, build_missing_chart, build_outlier_chart


def draw_wrapped_text(
    pdf,
    text,
    x,
    y,
    width_chars=95,
    line_height=14,
    font_name="Helvetica",
    font_size=10,
):
    pdf.setFont(font_name, font_size)

    for paragraph in str(text).split("\n"):
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


def build_pdf_report(filename, question, structured, analysis_text, df, recommendations, goal_label):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    y = 800

    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(50, y, "AI Data Quality Report")
    y -= 30

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"File: {filename}")
    y -= 18
    pdf.drawString(50, y, f"Goal: {goal_label}")
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
    pdf.drawString(50, y, "Assessment")
    y -= 18
    y = draw_wrapped_text(pdf, analysis_text, 50, y, width_chars=95)
    y -= 16

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Recommended Actions")
    y -= 16
    rec_text = "\n".join([f"- {r}" for r in recommendations]) if recommendations else "- No actions recommended."
    y = draw_wrapped_text(pdf, rec_text, 50, y, width_chars=92)
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
