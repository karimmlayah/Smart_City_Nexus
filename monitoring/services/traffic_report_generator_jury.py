from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from monitoring.services.traffic_nexus_pdf import (
    draw_card,
    draw_kpi_card,
    draw_section_title,
    draw_wrapped_text,
    fit_text,
)


def _render_jury_pdf(data: dict, debug_layout: bool = False) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    from monitoring.services.traffic_nexus_pdf import build_traffic_nexus_pdf

    # Reuse premium renderer and keep optional debug switch for layout boxes.
    payload = dict(data or {})
    payload["debug_layout"] = bool(debug_layout)
    return build_traffic_nexus_pdf(payload)


def generate_traffic_report(data: dict, output_path: str, debug_layout: bool = False) -> str:
    """
    Mandatory stable API used by the app.
    Writes the report to output_path and returns output_path.
    """
    out = _render_jury_pdf(data, debug_layout=debug_layout)
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(out)
    return str(p)


def build_report_bytes_with_fallback(data: dict, debug_layout: bool = False) -> bytes:
    """
    Fallback chain: jury -> premium -> pro -> legacy
    """
    # jury
    with NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        try:
            generate_traffic_report(data, tmp.name, debug_layout=debug_layout)
            return Path(tmp.name).read_bytes()
        except Exception:
            pass

    # premium (existing module)
    try:
        from monitoring.services.traffic_nexus_pdf import build_traffic_nexus_pdf

        return build_traffic_nexus_pdf(data or {})
    except Exception:
        pass

    # pro / legacy lightweight fallback
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    with NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        c = canvas.Canvas(tmp.name, pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 16)
        c.drawString(36, h - 56, "Traffic Nexus Report (Fallback)")
        c.setFont("Helvetica", 10)
        api_ok = bool((data or {}).get("api_key_configured"))
        msg = "API key not configured. Using local fallback." if not api_ok else "External API key configured."
        c.drawString(36, h - 78, msg)
        c.drawString(36, h - 98, "Report generation fallback path used to avoid production regression.")
        c.save()
        return Path(tmp.name).read_bytes()
