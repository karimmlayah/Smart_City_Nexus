"""Exports CSV et rapport PDF multi-pages pour missions UAV."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from django.conf import settings
from django.http import HttpResponse
from django.utils import timezone

from monitoring.models import UavStructuralAnalysis


def _display_label(label: str) -> str:
    mapping = {
        "Endommagé": "Damaged",
        "Effondré": "Collapsed",
        "Critique": "Critical",
        "Modéré": "Moderate",
        "Faible": "Low",
    }
    return mapping.get(label or "", label or "—")


def history_csv_response() -> HttpResponse:
    qs = UavStructuralAnalysis.objects.order_by("-created_at")[:500]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "id",
            "created_at",
            "primary_filename",
            "comparison_filename",
            "zone",
            "surface_m2",
            "prediction",
            "confidence",
            "risk_score",
            "operational_level",
            "co2_kg",
            "comparison_prediction",
        ]
    )
    for r in qs:
        w.writerow(
            [
                r.pk,
                r.created_at.isoformat(),
                r.primary_filename,
                r.comparison_filename or "",
                r.zone,
                r.surface_m2,
                r.prediction_label,
                f"{r.confidence:.4f}",
                f"{r.risk_score:.2f}",
                r.operational_level,
                f"{r.co2_estimate_kg:.2f}",
                r.comparison_label or "",
            ]
        )
    resp = HttpResponse(buf.getvalue(), content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = 'attachment; filename="uav_mission_history.csv"'
    return resp


def build_multipage_pdf() -> bytes:
    """Rapport synthétique + détail des dernières analyses."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Image as RLImage
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return b""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Smart City UAV — Report",
    )
    styles = getSampleStyleSheet()
    story: list[Any] = []

    story.append(Paragraph("<b>Smart City — UAV Mission Dashboard</b>", styles["Title"]))
    story.append(
        Paragraph(
            f"Generated on {timezone.now().strftime('%Y-%m-%d %H:%M')} · Platform analysis engine.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.4 * cm))

    qs = list(UavStructuralAnalysis.objects.order_by("-created_at")[:15])
    total = UavStructuralAnalysis.objects.count()
    story.append(Paragraph(f"<b>Analyses on record:</b> {total}", styles["Heading3"]))
    story.append(Spacer(1, 0.2 * cm))

    if qs:
        data = [["ID", "Date", "Prediction", "Conf.", "Risk /10", "Level", "CO₂ (t)", "Zone"]]
        for r in qs[:10]:
            data.append(
                [
                    str(r.pk),
                    r.created_at.strftime("%Y-%m-%d %H:%M"),
                    _display_label(r.prediction_label),
                    f"{r.confidence * 100:.1f}%",
                    f"{r.risk_score:.1f}",
                    _display_label(r.operational_level),
                    f"{r.co2_estimate_kg / 1000:.3f}",
                    r.zone,
                ]
            )
        t = Table(data, repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F80ED")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F5F5"), colors.white]),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 0.6 * cm))

        latest = qs[0]
        story.append(Paragraph("<b>Latest analysis (visual preview)</b>", styles["Heading3"]))
        story.append(
            Paragraph(
                f"{_display_label(latest.prediction_label)} · confidence {latest.confidence * 100:.1f}% · risk {latest.risk_score:.1f}/10 · {_display_label(latest.operational_level)}",
                styles["Normal"],
            )
        )
        media_root = Path(settings.MEDIA_ROOT)

        def try_img(url_field: str, caption: str):
            rel = (url_field or "").replace(settings.MEDIA_URL.rstrip("/") + "/", "").lstrip("/")
            if not rel:
                return
            p = media_root / rel
            if not p.is_file():
                return
            try:
                img = RLImage(str(p), width=14 * cm, height=5 * cm)
                story.append(Spacer(1, 0.15 * cm))
                story.append(Paragraph(f"<i>{caption}</i>", styles["Normal"]))
                story.append(img)
            except Exception:
                pass

        try_img(latest.primary_original_url, "Drone image (reference)")
        try_img(latest.primary_gradcam_url, "AI attention map")
        try_img(latest.primary_saliency_url, "Detailed signal focus")
    else:
        story.append(Paragraph("No analyses recorded yet.", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


def history_pdf_response() -> HttpResponse:
    pdf_bytes = build_multipage_pdf()
    if not pdf_bytes:
        return HttpResponse(
            "PDF unavailable (reportlab not installed).",
            status=500,
            content_type="text/plain; charset=utf-8",
        )
    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename="uav_report.pdf"'
    return resp
