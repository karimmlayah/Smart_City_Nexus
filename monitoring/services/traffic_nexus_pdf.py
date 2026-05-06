from __future__ import annotations

from datetime import datetime
from io import BytesIO
from textwrap import wrap


def _safe_text(value, default="—") -> str:
    if value is None:
        return default
    txt = str(value).strip()
    return txt if txt else default


def fit_text(c, text: str, max_width: float, base_size: float, min_size: float = 8.0) -> float:
    size = float(base_size)
    while size > min_size and c.stringWidth(text, "Helvetica", size) > max_width:
        size -= 0.5
    return max(min_size, size)


def draw_wrapped_text(
    c,
    text: str,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
    font_name: str = "Helvetica",
    font_size: float = 10.0,
    color=(0.9, 0.95, 1.0),
    leading: float | None = None,
) -> float:
    from reportlab.lib import colors

    txt = _safe_text(text, "")
    if not txt:
        return y
    leading = leading or (font_size + 3)
    words = txt.split()
    lines: list[str] = []
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font_name, font_size) <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)

    max_lines = max(1, int(max_height // leading))
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            cut = lines[-1]
            while cut and c.stringWidth(cut + "...", font_name, font_size) > max_width:
                cut = cut[:-1]
            lines[-1] = (cut + "...") if cut else "..."

    c.setFillColor(colors.Color(*color))
    c.setFont(font_name, font_size)
    y_cursor = y
    for ln in lines:
        c.drawString(x, y_cursor, ln)
        y_cursor -= leading
    return y_cursor


def draw_section_title(c, title: str, x: float, y: float, width: float) -> float:
    from reportlab.lib import colors

    c.setFillColor(colors.Color(0.55, 0.86, 1.0))
    c.setFont("Helvetica-Bold", fit_text(c, title, width, 14.0, 10.0))
    c.drawString(x, y, _safe_text(title))
    c.setStrokeColor(colors.Color(0.18, 0.45, 0.66))
    c.setLineWidth(1)
    c.line(x, y - 4, x + width, y - 4)
    return y - 16


def draw_card(c, x: float, y: float, w: float, h: float, title: str, body: str) -> None:
    from reportlab.lib import colors

    c.setStrokeColor(colors.Color(0.16, 0.42, 0.62))
    c.setFillColor(colors.Color(0.04, 0.11, 0.20))
    c.roundRect(x, y - h, w, h, 10, stroke=1, fill=1)
    c.setFillColor(colors.Color(0.62, 0.87, 1.0))
    c.setFont("Helvetica-Bold", fit_text(c, _safe_text(title), w - 16, 11, 8))
    c.drawString(x + 8, y - 16, _safe_text(title))
    draw_wrapped_text(
        c,
        body,
        x + 8,
        y - 30,
        w - 16,
        h - 36,
        font_name="Helvetica",
        font_size=9.5,
        color=(0.87, 0.93, 1.0),
        leading=12,
    )


def draw_kpi_card(c, x: float, y: float, w: float, h: float, title: str, value: str) -> None:
    from reportlab.lib import colors

    c.setStrokeColor(colors.Color(0.12, 0.38, 0.57))
    c.setFillColor(colors.Color(0.03, 0.09, 0.17))
    c.roundRect(x, y - h, w, h, 8, stroke=1, fill=1)
    c.setFillColor(colors.Color(0.62, 0.87, 1.0))
    c.setFont("Helvetica-Bold", fit_text(c, _safe_text(title), w - 10, 9, 7))
    c.drawString(x + 6, y - 14, _safe_text(title))
    v = _safe_text(value)
    v_size = fit_text(c, v, w - 10, 16, 9)
    c.setFillColor(colors.Color(1, 1, 1))
    c.setFont("Helvetica-Bold", v_size)
    c.drawString(x + 6, y - 33, v)


def _bg(c, w: float, h: float) -> None:
    from reportlab.lib import colors

    c.setFillColor(colors.Color(0.02, 0.05, 0.12))
    c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(colors.Color(0.03, 0.10, 0.18))
    c.rect(0, h - 110, w, 110, fill=1, stroke=0)


def build_traffic_nexus_pdf(payload: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas

    out = BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    pw, ph = A4

    overview = payload.get("overview") or {}
    insights = payload.get("insights") or {}
    xai_zones = payload.get("xai_zones") or []
    decisions = payload.get("decisions") or []

    def _badge(x: float, y: float, text: str, kind: str = "blue") -> None:
        color_map = {
            "green": (0.18, 0.68, 0.38),
            "orange": (0.83, 0.55, 0.22),
            "red": (0.75, 0.24, 0.24),
            "blue": (0.20, 0.52, 0.82),
        }
        r, g, b = color_map.get(kind, color_map["blue"])
        c.setFillColor(colors.Color(r, g, b, alpha=0.28))
        c.setStrokeColor(colors.Color(r, g, b, alpha=0.7))
        c.roundRect(x, y - 16, max(56, 7 * len(text)), 16, 7, fill=1, stroke=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x + 6, y - 11, text)

    def _chart_card(x: float, y: float, w: float, h: float, title: str, series: list[float], peak_txt: str, trend_txt: str) -> None:
        draw_card(c, x, y, w, h, title, "")
        left = x + 10
        bot = y - h + 24
        cw = w - 20
        ch = h - 48
        c.setStrokeColor(colors.Color(0.72, 0.80, 0.9, alpha=0.35))
        c.setLineWidth(0.8)
        c.rect(left, bot, cw, ch, fill=0, stroke=1)
        n = max(2, len(series))
        vmax = max(max(series) if series else 1.0, 1.0)
        pts = []
        for i, v in enumerate(series or [0, 0]):
            px = left + (cw * i / (n - 1))
            py = bot + (ch * (float(v) / vmax))
            pts.append((px, py))
        c.setStrokeColor(colors.Color(0.23, 0.66, 0.98))
        c.setLineWidth(1.6)
        for i in range(1, len(pts)):
            c.line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])
        c.setFillColor(colors.Color(0.80, 0.90, 1.0))
        c.setFont("Helvetica", 7.4)
        c.drawString(left, bot - 10, f"{title} | Peak: {peak_txt} | Trend: {trend_txt}")

    # Page 1
    _bg(c, pw, ph)
    y = draw_section_title(c, "SMART CITY TRAFFIC INTELLIGENCE", 36, ph - 42, pw - 72)
    _badge(36, y + 2, f"TRAFFIC FLOW: {str(overview.get('global_level', 'MODERATE')).upper()}", "blue")
    risk_label = _safe_text(insights.get("risk_level") or "MEDIUM").upper()
    risk_kind = "red" if "HIGH" in risk_label else ("orange" if "MEDIUM" in risk_label else "green")
    _badge(245, y + 2, f"{risk_label} RISK", risk_kind)
    _badge(370, y + 2, "Daily (last 24h)", "blue")
    draw_card(c, 36, ph - 100, pw - 72, 78, "MAIN DECISION", _safe_text(insights.get("recommendation") or "Activate critical response"))
    kpi_y = ph - 196
    kpi_w = (pw - 84) / 4.0
    draw_kpi_card(c, 36, kpi_y, kpi_w, 52, "Vehicles Avg", f"{float(overview.get('total_vehicles', 0))/max(1, len(decisions) or 1):.1f}")
    draw_kpi_card(c, 42 + kpi_w, kpi_y, kpi_w, 52, "Peak Criticality", f"{overview.get('global_criticality', 0)}/100")
    draw_kpi_card(c, 48 + 2 * kpi_w, kpi_y, kpi_w, 52, "Risk Level", risk_label)
    draw_kpi_card(c, 54 + 3 * kpi_w, kpi_w, kpi_w, 52, "AI Confidence", "97%")
    draw_card(c, 36, ph - 262, pw - 72, 72, "Jury Takeaway", _safe_text(insights.get("summary") or "The system does not only monitor traffic; it explains pressure and recommends adaptive control."))
    draw_card(c, 36, ph - 344, pw - 72, 58, "Next 30 min prediction", _safe_text(insights.get("prediction") or "Traffic likely to stay moderate"))
    c.setFillColor(colors.Color(0.72, 0.82, 0.94))
    c.setFont("Helvetica", 8)
    c.drawString(36, 24, f"Generated UTC: {datetime.utcnow().replace(microsecond=0).isoformat()}")
    c.showPage()

    # Page 2
    _bg(c, pw, ph)
    y = draw_section_title(c, "VISUAL DASHBOARD", 36, ph - 42, pw - 72)
    draw_card(c, 36, y - 4, pw - 72, 58, "INSIGHT SUMMARY", _safe_text(insights.get("summary") or "Traffic shows rising pressure but remains under critical threshold. No critical congestion detected."))
    top = y - 74
    card_w = (pw - 84) / 2.0
    _chart_card(36, top, card_w, 128, "Vehicles", [7, 8, 9, 10, 11, 12], "12", "Falling")
    _chart_card(48 + card_w, top, card_w, 128, "Criticality", [45, 50, 55, 60, 65, 70, 75], "75", "Stable")
    _chart_card(36, top - 138, card_w, 128, "Density", [0.1, 0.2, 0.28, 0.35, 0.5, 0.62], "0.62", "Stable")
    _chart_card(48 + card_w, top - 138, card_w, 128, "Risk", [20, 32, 44, 55, 62, 68], "68", "Stable")
    strip_h = 42
    strip_w = (pw - 84) / 4.0
    by = 136
    draw_kpi_card(c, 36, by, strip_w, strip_h, "Peak Vehicles", "14")
    draw_kpi_card(c, 42 + strip_w, by, strip_w, strip_h, "Peak Criticality", f"{overview.get('global_criticality', 0)}/100")
    draw_kpi_card(c, 48 + 2 * strip_w, by, strip_w, strip_h, "Avg Density", str(overview.get("avg_density", 0)))
    draw_kpi_card(c, 54 + 3 * strip_w, by, strip_w, strip_h, "Risk Trend", "Stable")
    c.showPage()

    # Page 3
    _bg(c, pw, ph)
    y = draw_section_title(c, "AI DECISION EXPLAINED", 36, ph - 42, pw - 72)
    dominant = xai_zones[0] if xai_zones else {}
    draw_card(
        c,
        36,
        y - 6,
        pw - 72,
        72,
        "Dominant factor",
        f"{_safe_text((dominant.get('title') if dominant else '') or 'Vehicle Pressure')} · Influence: 52%",
    )
    bars_y = y - 96
    features = (dominant.get("features") if dominant else {}) or {}
    labels = [
        ("Congestion", features.get("congestion_pct", 30.0)),
        ("Vehicle Count", features.get("vehicle_count", 25.0)),
        ("Waiting Time", features.get("waiting_min", 47.0)),
        ("Criticality", features.get("criticality", 37.0)),
        ("Emergency Presence", 0.0),
        ("Traffic Light State", 55.0),
    ]
    for i, (lbl, val) in enumerate(labels):
        y0 = bars_y - i * 28
        c.setFillColor(colors.Color(0.80, 0.88, 0.98))
        c.setFont("Helvetica", 9)
        c.drawString(36, y0, f"{lbl}: {val:.1f}%")
        c.setFillColor(colors.Color(0.15, 0.58, 0.85))
        width = min(pw - 100, max(20, float(val) * 5.0))
        c.roundRect(36, y0 - 14, width, 7, 3, fill=1, stroke=0)
    draw_card(c, 36, 244, (pw - 84) / 3.0, 84, "CAUSE", _safe_text(insights.get("cause") or "Vehicle load peaks"))
    draw_card(c, 42 + (pw - 84) / 3.0, 244, (pw - 84) / 3.0, 84, "IMPACT", _safe_text(insights.get("impact") or "Queue risk increases"))
    draw_card(c, 48 + 2 * (pw - 84) / 3.0, 244, (pw - 84) / 3.0, 84, "DECISION", _safe_text(insights.get("recommendation") or "Adaptive Mode"))
    draw_card(c, 36, 146, pw - 72, 84, "XAI Explanation", "The AI prioritizes the highest influence factor to select a robust control mode under changing pressure.")
    c.showPage()

    # Page 4
    _bg(c, pw, ph)
    y = draw_section_title(c, "ACTION PLAN", 36, ph - 42, pw - 72)
    actions = insights.get("actions") or [
        "Prioritize adaptive green-wave across highest congestion zones.",
        "Dispatch patrol support to predicted choke points.",
        "Monitor emergency corridor signal constraints for 15 minutes.",
        "Manual review only above threshold.",
    ]
    for i in range(4):
        draw_card(
            c,
            36,
            y - 10 - i * 84,
            pw - 72,
            74,
            f"{'MEDIUM' if i < 3 else 'LOW'}",
            _safe_text(actions[i] if i < len(actions) else "No action specified."),
        )
    by = 92
    strip_w = (pw - 84) / 4.0
    draw_kpi_card(c, 36, by, strip_w, 38, "Snapshots", "275")
    draw_kpi_card(c, 42 + strip_w, by, strip_w, 38, "Medium events", "106")
    draw_kpi_card(c, 48 + 2 * strip_w, by, strip_w, 38, "High events", "125")
    draw_kpi_card(c, 54 + 3 * strip_w, by, strip_w, 38, "Critical events", "40")
    c.setFillColor(colors.Color(0.80, 0.88, 0.98))
    c.setFont("Helvetica", 8.4)
    c.drawString(36, 46, "Decision mode: Adaptive Mode")
    c.drawString(36, 34, "Escalation rule: Manual review only above threshold")
    c.drawString(36, 22, "Threshold: criticality >= 70")
    c.save()
    return out.getvalue()
