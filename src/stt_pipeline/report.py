from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .models import CallAnalysis


def write_sentiment_report_pdf(report_path: Path, calls: list[CallAnalysis], input_dir: Path, model_name: str) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    document = SimpleDocTemplate(
        str(report_path),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    body_style = styles["BodyText"]
    body_style.leading = 14
    small_style = ParagraphStyle(
        name="SmallBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11,
        alignment=TA_LEFT,
    )

    story: list[object] = []
    story.append(Paragraph("Call Sentiment Report", title_style))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph(f"Input folder: {input_dir}", body_style))
    story.append(Paragraph(f"Whisper model: {model_name}", body_style))
    story.append(Paragraph(f"Calls processed: {len(calls)}", body_style))
    story.append(Spacer(1, 0.2 * inch))

    table_data = [["Candidate", "Sentiment", "Compound", "Positive", "Neutral", "Negative", "Score", "Decision"]]
    for call in calls:
        table_data.append(
            [
                call.transcript.source_path.stem,
                call.sentiment.label,
                f"{call.sentiment.compound:.2f}",
                f"{call.sentiment.positive:.2f}",
                f"{call.sentiment.neutral:.2f}",
                f"{call.sentiment.negative:.2f}",
                str(call.score.score),
                call.batch_decision,
            ]
        )

    table = Table(table_data, repeatRows=1, colWidths=[1.15 * inch, 0.8 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.55 * inch, 1.05 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3c88")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d2fe")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#eef2ff")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))

    for call in calls:
        story.append(Paragraph(f"{call.transcript.source_path.stem}", styles["Heading2"]))
        story.append(Paragraph(f"Sentiment label: {call.sentiment.label}", body_style))
        story.append(Paragraph(f"Compound score: {call.sentiment.compound:.2f}", body_style))
        story.append(Paragraph(f"Distribution: positive {call.sentiment.positive:.2f}, neutral {call.sentiment.neutral:.2f}, negative {call.sentiment.negative:.2f}", body_style))
        story.append(Paragraph(escape(call.sentiment.summary), body_style))
        story.append(Paragraph(f"Shortlist score: {call.score.score} ({call.batch_decision})", body_style))
        if call.batch_reason:
            story.append(Paragraph(escape(call.batch_reason), small_style))
        story.append(Paragraph("Transcript excerpt:", body_style))
        excerpt = call.transcript.text.strip().replace("\n", " ") or "No transcript text available."
        story.append(Paragraph(escape(excerpt[:400]), small_style))
        story.append(Spacer(1, 0.15 * inch))

    document.build(story)
    return report_path