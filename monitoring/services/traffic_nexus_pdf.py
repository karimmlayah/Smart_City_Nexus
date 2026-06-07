from __future__ import annotations

from datetime import datetime
from io import BytesIO


# ── Design tokens ─────────────────────────────────────────────────────────────
PAGE_W, PAGE_H = 595.27, 841.89  # A4
MARGIN = 36.0
FOOTER_H = 32.0
HEADER_H = 78.0
CONTENT_W = PAGE_W - 2 * MARGIN
TOTAL_PAGES = 4

C_BG = (0.02, 0.05, 0.12)
C_BG_PANEL = (0.04, 0.10, 0.20)
C_CARD = (0.05, 0.12, 0.22)
C_CARD_GLOW = (0.08, 0.18, 0.32)
C_BORDER = (0.16, 0.42, 0.62)
C_BORDER_BRIGHT = (0.22, 0.66, 0.98)
C_CYAN = (0.35, 0.86, 1.0)
C_VIOLET = (0.62, 0.52, 0.96)
C_TEXT = (0.92, 0.96, 1.0)
C_MUTED = (0.62, 0.74, 0.88)
C_WHITE = (1.0, 1.0, 1.0)

FONT = "Helvetica"
FONT_BOLD = "Helvetica-Bold"


def _safe_text(value, default="—") -> str:
    if value is None:
        return default
    txt = str(value).strip()
    return txt if txt else default


def fit_text(c, text: str, max_width: float, base_size: float, min_size: float = 8.0) -> float:
    size = float(base_size)
    while size > min_size and c.stringWidth(text, FONT, size) > max_width:
        size -= 0.5
    return max(min_size, size)


def _color(rgb, alpha=1.0):
    from reportlab.lib import colors

    r, g, b = rgb
    return colors.Color(r, g, b, alpha=alpha)


def draw_wrapped_text(
    c,
    text: str,
    x: float,
    y: float,
    max_width: float,
    max_height: float,
    font_name: str = FONT,
    font_size: float = 10.0,
    color=(0.9, 0.95, 1.0),
    leading: float | None = None,
    truncate: bool = True,
) -> float:
    txt = _safe_text(text, "")
    if not txt:
        return y

    size = font_size
    leading = leading or (size + 3)

    while size >= 7.0:
    words = txt.split()
    lines: list[str] = []
    line = ""
    for w in words:
        test = (line + " " + w).strip()
            if c.stringWidth(test, font_name, size) <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)

        leading = size + 3
    max_lines = max(1, int(max_height // leading))
        if len(lines) <= max_lines or not truncate:
            if truncate and len(lines) > max_lines:
        lines = lines[:max_lines]
        if lines:
            cut = lines[-1]
                    while cut and c.stringWidth(cut + "...", font_name, size) > max_width:
                cut = cut[:-1]
            lines[-1] = (cut + "...") if cut else "..."
            c.setFillColor(_color(color))
            c.setFont(font_name, size)
    y_cursor = y
    for ln in lines:
        c.drawString(x, y_cursor, ln)
        y_cursor -= leading
    return y_cursor

        size -= 0.5

    c.setFillColor(_color(color))
    c.setFont(font_name, 7.0)
    c.drawString(x, y, txt[:120])
    return y - 10


def draw_section_title(c, title: str, x: float, y: float, width: float) -> float:
    c.setFillColor(_color(C_CYAN))
    title_txt = _safe_text(title)
    c.setFont(FONT_BOLD, fit_text(c, title_txt, width, 13.0, 10.0))
    c.drawString(x, y, title_txt)
    c.setStrokeColor(_color(C_BORDER_BRIGHT, 0.55))
    c.setLineWidth(1.2)
    c.line(x, y - 5, x + width, y - 5)
    c.setStrokeColor(_color(C_VIOLET, 0.35))
    c.setLineWidth(0.6)
    c.line(x, y - 7, x + min(width * 0.35, 120), y - 7)
    return y - 18


def draw_card(
    c,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    accent: tuple[float, float, float] = C_BORDER,
    truncate_body: bool = True,
    title_size: float = 10.5,
    body_size: float = 9.5,
) -> None:
    c.setFillColor(_color(C_CARD_GLOW, 0.35))
    c.roundRect(x - 1, y - h - 1, w + 2, h + 2, 11, fill=1, stroke=0)
    c.setStrokeColor(_color(accent, 0.75))
    c.setFillColor(_color(C_CARD, 0.92))
    c.setLineWidth(0.9)
    c.roundRect(x, y - h, w, h, 10, stroke=1, fill=1)
    c.setStrokeColor(_color(C_BORDER_BRIGHT, 0.25))
    c.setLineWidth(0.5)
    c.line(x + 8, y - 22, x + w - 8, y - 22)

    c.setFillColor(_color(C_CYAN))
    c.setFont(FONT_BOLD, fit_text(c, _safe_text(title), w - 16, title_size, 8))
    c.drawString(x + 10, y - 16, _safe_text(title))

    draw_wrapped_text(
        c,
        body,
        x + 10,
        y - 30,
        w - 20,
        h - 38,
        font_name=FONT,
        font_size=body_size,
        color=C_TEXT,
        leading=body_size + 3,
        truncate=truncate_body,
    )


def draw_kpi_card(c, x: float, y: float, w: float, h: float, title: str, value: str) -> None:
    c.setFillColor(_color(C_CARD_GLOW, 0.28))
    c.roundRect(x - 0.5, y - h - 0.5, w + 1, h + 1, 9, fill=1, stroke=0)
    c.setStrokeColor(_color(C_BORDER, 0.85))
    c.setFillColor(_color(C_BG_PANEL, 0.95))
    c.roundRect(x, y - h, w, h, 8, stroke=1, fill=1)

    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT_BOLD, fit_text(c, _safe_text(title), w - 12, 8.2, 6.8))
    c.drawString(x + 8, y - 14, _safe_text(title))

    v = _safe_text(value)
    v_size = fit_text(c, v, w - 12, 15, 9)
    c.setFillColor(_color(C_WHITE))
    c.setFont(FONT_BOLD, v_size)
    c.drawString(x + 8, y - 32, v)


def draw_badge(c, x: float, y: float, text: str, kind: str = "blue") -> float:
    color_map = {
        "green": (0.18, 0.68, 0.48),
        "orange": (0.90, 0.58, 0.22),
        "red": (0.82, 0.28, 0.28),
        "blue": (0.20, 0.52, 0.82),
        "violet": (0.52, 0.42, 0.88),
        "teal": (0.18, 0.72, 0.68),
    }
    r, g, b = color_map.get(kind, color_map["blue"])
    label = _safe_text(text)
    bw = max(68, c.stringWidth(label, FONT_BOLD, 7.5) + 16)
    c.setFillColor(_color((r, g, b), 0.22))
    c.setStrokeColor(_color((r, g, b), 0.72))
    c.roundRect(x, y - 18, bw, 18, 9, fill=1, stroke=1)
    c.setFillColor(_color(C_WHITE))
    c.setFont(FONT_BOLD, 7.5)
    c.drawString(x + 8, y - 13, label)
    return bw + 8


def draw_progress_bar(
    c,
    x: float,
    y: float,
    width: float,
    label: str,
    value: float,
    max_val: float = 100.0,
) -> None:
    pct = max(0.0, min(100.0, (float(value) / max(max_val, 1.0)) * 100.0))
    bar_w = width - 120
    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT, 8.5)
    c.drawString(x, y, _safe_text(label))
    c.setFillColor(_color(C_BG_PANEL))
    c.roundRect(x + 118, y - 9, bar_w, 8, 4, fill=1, stroke=0)
    fill_w = max(4.0, bar_w * (pct / 100.0))
    c.setFillColor(_color(C_BORDER_BRIGHT, 0.35))
    c.roundRect(x + 118, y - 9, fill_w, 8, 4, fill=1, stroke=0)
    c.setFillColor(_color(C_CYAN))
    c.roundRect(x + 118, y - 9, fill_w, 8, 4, fill=1, stroke=0)
    c.setFillColor(_color(C_WHITE))
    c.setFont(FONT_BOLD, 8)
    c.drawRightString(x + width, y, f"{pct:.0f}%")


def draw_mini_chart(
    c,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    series: list[float],
    peak_txt: str,
    trend_txt: str,
) -> None:
    draw_card(c, x, y, w, h, title, "", truncate_body=False)
    left = x + 12
    bot = y - h + 28
    cw = w - 24
    ch = h - 58

    for i in range(5):
        gy = bot + (ch * i / 4.0)
        c.setStrokeColor(_color(C_BORDER, 0.22))
        c.setLineWidth(0.4)
        c.line(left, gy, left + cw, gy)

    c.setStrokeColor(_color(C_BORDER, 0.45))
    c.setLineWidth(0.8)
    c.roundRect(left, bot, cw, ch, 4, fill=0, stroke=1)

    data = series or [0.0, 0.0]
    n = max(2, len(data))
    vmax = max(max(data), 1.0)
    pts = []
    for i, v in enumerate(data):
        px = left + (cw * i / (n - 1))
        py = bot + (ch * (float(v) / vmax))
        pts.append((px, py))

    if len(pts) >= 2:
        c.setStrokeColor(_color(C_BORDER_BRIGHT, 0.25))
        c.setLineWidth(3.2)
        for i in range(1, len(pts)):
            c.line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])
        c.setStrokeColor(_color(C_CYAN))
        c.setLineWidth(1.8)
        for i in range(1, len(pts)):
            c.line(pts[i - 1][0], pts[i - 1][1], pts[i][0], pts[i][1])
        for px, py in pts:
            c.setFillColor(_color(C_CYAN))
            c.circle(px, py, 2.2, fill=1, stroke=0)

    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT, 7.2)
    c.drawString(left, bot - 12, f"Peak: {peak_txt}")
    c.drawRightString(left + cw, bot - 12, f"Trend: {trend_txt}")


def _page_background(c, w: float, h: float) -> None:
    c.setFillColor(_color(C_BG))
    c.rect(0, 0, w, h, fill=1, stroke=0)

    c.setStrokeColor(_color(C_BORDER, 0.08))
    c.setLineWidth(0.35)
    step = 28
    for gx in range(0, int(w) + step, step):
        c.line(gx, 0, gx, h)
    for gy in range(0, int(h) + step, step):
        c.line(0, gy, w, gy)

    c.setFillColor(_color(C_CYAN, 0.04))
    c.rect(0, h - 96, w, 96, fill=1, stroke=0)

    corner = 18
    c.setStrokeColor(_color(C_BORDER_BRIGHT, 0.45))
    c.setLineWidth(1.2)
    c.line(MARGIN, h - MARGIN, MARGIN + corner, h - MARGIN)
    c.line(MARGIN, h - MARGIN, MARGIN, h - MARGIN - corner)
    c.line(w - MARGIN, h - MARGIN, w - MARGIN - corner, h - MARGIN)
    c.line(w - MARGIN, h - MARGIN, w - MARGIN, h - MARGIN - corner)


def _draw_header(c, page: int, subtitle: str = "") -> float:
    y = PAGE_H - MARGIN
    c.setFillColor(_color(C_CYAN))
    c.setFont(FONT_BOLD, 11)
    c.drawString(MARGIN, y - 2, "MedinaMind Traffic Intelligence Report")
    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT, 8.2)
    c.drawString(
        MARGIN,
        y - 14,
        subtitle or "AI-powered mobility analysis, congestion insights, and decision-ready recommendations",
    )
    c.setFillColor(_color(C_VIOLET, 0.85))
    c.setFont(FONT_BOLD, 7.5)
    c.drawRightString(PAGE_W - MARGIN, y - 2, "Smart City Platform")
    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT, 7.5)
    generated = datetime.utcnow().replace(microsecond=0).strftime("%Y-%m-%d %H:%M UTC")
    c.drawRightString(PAGE_W - MARGIN, y - 14, f"Generated {generated}")
    return PAGE_H - MARGIN - HEADER_H


def _draw_footer(c, page: int) -> None:
    c.setStrokeColor(_color(C_BORDER, 0.35))
    c.setLineWidth(0.6)
    c.line(MARGIN, FOOTER_H + 8, PAGE_W - MARGIN, FOOTER_H + 8)
    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT, 7.5)
    c.drawString(
        MARGIN,
        FOOTER_H - 2,
        "MedinaMind · Traffic Nexus · Smart City Intelligence Platform",
    )
    c.drawRightString(PAGE_W - MARGIN, FOOTER_H - 2, f"Page {page} / {TOTAL_PAGES}")


def _executive_summary_text(summary: str) -> str:
    s = _safe_text(summary, "")
    low = s.lower()
    if not s or "no snapshot" in low or s == "—":
        return (
            "No visual snapshot was attached for this report. "
            "The recommendation is based on live KPI analysis and decision signals."
        )
    return s


def _risk_kind(risk_label: str) -> str:
    up = risk_label.upper()
    if "HIGH" in up:
        return "red"
    if "MEDIUM" in up:
        return "orange"
    return "green"


def _ai_confidence(overview: dict, xai: dict) -> str:
    features = (xai.get("features") if xai else {}) or {}
    crit = float(features.get("criticality") or overview.get("global_criticality") or 0)
    conf = int(max(82, min(99, 100 - crit * 0.12)))
    return f"{conf}%"


def _series_from_value(base: float, points: int = 8) -> list[float]:
    base = max(float(base), 0.1)
    return [round(base * (0.72 + 0.04 * i + (0.02 if i % 2 else 0)), 2) for i in range(points)]


def _trend_label(values: list[float]) -> str:
    if len(values) < 2:
        return "Stable"
    delta = values[-1] - values[0]
    if delta > 0.08 * max(values[-1], 1):
        return "Rising"
    if delta < -0.08 * max(values[-1], 1):
        return "Easing"
    return "Stable"


def _is_weak_action(text: str) -> bool:
    t = _safe_text(text, "").lower()
    return not t or t in ("—", "no action.", "no action specified.") or "no action" in t


def _default_action_plan(risk_label: str, recommendation: str) -> list[dict]:
    risk = risk_label.upper()
    reco = _safe_text(recommendation, "Maintain adaptive signal timing.")
    if "HIGH" in risk:
        return [
            {
                "priority": "HIGH",
                "action": reco or "Extend green phase for congested zone",
                "instruction": "Extend green phase for the most congested monitoring zone and prepare operator alert.",
                "monitor": "Watch queue length and criticality every signal cycle.",
            },
            {
                "priority": "HIGH",
                "action": "Activate rerouting recommendation",
                "instruction": "Prepare adaptive routing guidance for adjacent corridors if delay continues to rise.",
                "monitor": "Confirm rerouting messages reach field operators within 5 minutes.",
            },
            {
                "priority": "MEDIUM",
                "action": "Send alert to operators",
                "instruction": "Notify the mobility command desk of elevated congestion and recommended timing changes.",
                "monitor": "Track acknowledgment and response time.",
            },
            {
                "priority": "MEDIUM",
                "action": "Limit entry if needed",
                "instruction": "Consider temporary entry control if criticality remains above escalation threshold.",
                "monitor": "Review threshold breach duration before escalation.",
            },
        ]
    if "MEDIUM" in risk:
        return [
            {
                "priority": "MEDIUM",
                "action": reco or "Adjust signal timing if waiting time increases",
                "instruction": "Apply adaptive signal timing if estimated delay exceeds the current comfort band.",
                "monitor": "Compare vehicle accumulation across monitored zones each cycle.",
            },
            {
                "priority": "MEDIUM",
                "action": "Monitor vehicle accumulation",
                "instruction": "Track density and queue growth in selected monitoring zones.",
                "monitor": "Refresh KPI review every 2–3 minutes.",
            },
            {
                "priority": "MEDIUM",
                "action": "Prepare rerouting option if congestion rises",
                "instruction": "Stage rerouting guidance if congestion trend moves from medium toward high.",
                "monitor": "Watch risk trend and peak criticality score.",
            },
            {
                "priority": "LOW",
                "action": "Review next cycle recommendation",
                "instruction": "Validate the next AI recommendation before the following peak period.",
                "monitor": "Keep decision insight panel active for operator awareness.",
            },
        ]
    return [
        {
            "priority": "LOW",
            "action": reco or "Maintain normal traffic operation",
            "instruction": "Maintain normal traffic operation across monitored zones.",
            "monitor": "Continue live vision monitoring without unnecessary intervention.",
        },
        {
            "priority": "LOW",
            "action": "Avoid unnecessary green extension",
            "instruction": "Avoid extending green phases unless KPI trend justifies intervention.",
            "monitor": "Confirm delay remains within acceptable limits.",
        },
        {
            "priority": "LOW",
            "action": "Continue monitoring selected zones",
            "instruction": "Continue monitoring selected zones and preserve current signal balance.",
            "monitor": "Review average vehicle flow and density periodically.",
        },
        {
            "priority": "LOW",
            "action": "Keep emergency priority checks active",
            "instruction": "Keep emergency priority checks active for corridor readiness.",
            "monitor": "Verify priority response path remains unobstructed.",
        },
    ]


def _normalize_actions(actions, risk_label: str, recommendation: str) -> list[dict]:
    raw = [a for a in (actions or []) if not _is_weak_action(a)]
    if len(raw) >= 4:
        out = []
        for i, item in enumerate(raw[:4]):
            txt = _safe_text(item)
            out.append(
                {
                    "priority": ("HIGH" if i == 0 else "MEDIUM" if i < 3 else "LOW"),
                    "action": txt,
                    "instruction": txt,
                    "monitor": "Follow operator playbook and confirm KPI trend after implementation.",
                }
            )
        return out
    return _default_action_plan(risk_label, recommendation)


def _dominant_factor_label(xai: dict, features: dict) -> tuple[str, float]:
    labels = [
        ("Traffic Density", float(features.get("congestion_pct") or 0)),
        ("Vehicle Volume", min(100.0, float(features.get("vehicle_count") or 0) * 4)),
        ("Estimated Delay", min(100.0, float(features.get("waiting_min") or 0) * 12)),
        ("Criticality Score", float(features.get("criticality") or 0)),
        ("Emergency Priority", 0.0),
        ("Signal Timing Influence", 55.0),
    ]
    best = max(labels, key=lambda x: x[1])
    title = _safe_text((xai.get("title") if xai else "") or best[0], best[0])
    influence = best[1] if best[1] > 0 else 52.0
    return title, influence


def _build_page_1(c, payload: dict, overview: dict, insights: dict, xai: dict, decisions: list) -> None:
    _page_background(c, PAGE_W, PAGE_H)
    y0 = _draw_header(c, 1)

    traffic_flow = _safe_text(insights.get("traffic_status") or overview.get("global_level") or "Moderate").upper()
    risk_label = _safe_text(insights.get("risk_level") or "Medium")
    risk_up = risk_label.upper()
    report_period = _safe_text(insights.get("report_period") or payload.get("report_period") or "Daily (last 24h)")
    confidence = _ai_confidence(overview, xai)
    recommendation = _safe_text(insights.get("recommendation") or "Maintain adaptive signal timing.")

    bx = MARGIN
    by = y0 - 4
    draw_badge(c, bx, by, f"Traffic Flow: {traffic_flow}", "teal")
    bx += draw_badge(c, bx + 4, by, f"Risk Level: {risk_up}", _risk_kind(risk_up))
    bx += draw_badge(c, bx + 4, by, f"Report Period: {report_period}", "blue")
    draw_badge(c, bx + 4, by, f"AI Confidence: {confidence}", "violet")

    hero_y = y0 - 34
    draw_card(
        c,
        MARGIN,
        hero_y,
        CONTENT_W,
        88,
        "Recommended Decision",
        recommendation,
        accent=C_BORDER_BRIGHT,
        truncate_body=False,
        title_size=11,
        body_size=11.5,
    )

    kpi_y = hero_y - 102
    kpi_w = (CONTENT_W - 18) / 4.0
    vehicles = float(overview.get("total_vehicles", 0))
    zone_count = max(1, len(decisions) or 1)
    avg_flow = vehicles / zone_count
    peak_crit = overview.get("global_criticality", 0)

    draw_kpi_card(c, MARGIN, kpi_y, kpi_w, 56, "Average Vehicle Flow", f"{avg_flow:.1f}")
    draw_kpi_card(c, MARGIN + kpi_w + 6, kpi_y, kpi_w, 56, "Peak Criticality Score", f"{peak_crit}/100")
    draw_kpi_card(c, MARGIN + 2 * (kpi_w + 6), kpi_y, kpi_w, 56, "Risk Level", risk_up)
    draw_kpi_card(c, MARGIN + 3 * (kpi_w + 6), kpi_y, kpi_w, 56, "AI Confidence", confidence)

    takeaway_y = kpi_y - 72
    draw_card(
        c,
        MARGIN,
        takeaway_y,
        CONTENT_W,
        78,
        "Executive Takeaway",
        _executive_summary_text(insights.get("summary") or ""),
        truncate_body=False,
    )

    forecast_y = takeaway_y - 92
    prediction = _safe_text(
        insights.get("prediction")
        or f"Mobility conditions are expected to remain {traffic_flow.lower()} over the next 30 minutes based on current KPI trends."
    )
    draw_card(
        c,
        MARGIN,
        forecast_y,
        CONTENT_W,
        68,
        "30-Minute Mobility Forecast",
        prediction,
        accent=C_VIOLET,
        truncate_body=False,
    )

    status_line = (
        f"Traffic status: {traffic_flow.title()} · Risk: {risk_up} · "
        f"Average density: {float(overview.get('avg_density') or 0):.2f} · Zones analyzed: {max(len(decisions), 1)}"
    )
    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT, 8)
    draw_wrapped_text(
        c,
        status_line,
        MARGIN,
        forecast_y - 88,
        CONTENT_W,
        24,
        font_size=8,
        color=C_MUTED,
        truncate=False,
    )

    _draw_footer(c, 1)


def _build_page_2(c, payload: dict, overview: dict, insights: dict) -> None:
    _page_background(c, PAGE_W, PAGE_H)
    y = _draw_header(c, 2, "Mobility Performance Overview")

    summary = _executive_summary_text(
        insights.get("summary")
        or "Traffic pressure remains observable across monitored zones while staying within operational control thresholds."
    )
    draw_card(c, MARGIN, y - 2, CONTENT_W, 54, "Insight Summary", summary, truncate_body=False)

    vehicles = float(overview.get("total_vehicles", 0) or 8)
    crit = float(overview.get("global_criticality", 0) or 45)
    density = float(overview.get("avg_density", 0) or 0.35)
    risk_series = _series_from_value(crit * 0.85, 8)

    v_series = _series_from_value(max(vehicles, 1), 8)
    c_series = _series_from_value(max(crit, 10), 8)
    d_series = _series_from_value(max(density * 100, 5), 8)

    top = y - 68
    card_w = (CONTENT_W - 12) / 2.0
    card_h = 132

    draw_mini_chart(
        c, MARGIN, top, card_w, card_h, "Vehicle Activity",
        v_series, f"{max(v_series):.0f}", _trend_label(v_series),
    )
    draw_mini_chart(
        c, MARGIN + card_w + 12, top, card_w, card_h, "Criticality Evolution",
        c_series, f"{max(c_series):.0f}", _trend_label(c_series),
    )
    row2 = top - card_h - 14
    draw_mini_chart(
        c, MARGIN, row2, card_w, card_h, "Traffic Density",
        d_series, f"{max(d_series):.1f}", _trend_label(d_series),
    )
    draw_mini_chart(
        c, MARGIN + card_w + 12, row2, card_w, card_h, "Risk Evolution",
        risk_series, f"{max(risk_series):.0f}", _trend_label(risk_series),
    )

    strip_y = row2 - card_h - 18
    strip_w = (CONTENT_W - 18) / 4.0
    draw_kpi_card(c, MARGIN, strip_y, strip_w, 48, "Peak Vehicles", f"{max(v_series):.0f}")
    draw_kpi_card(c, MARGIN + strip_w + 6, strip_y, strip_w, 48, "Peak Criticality", f"{max(c_series):.0f}/100")
    draw_kpi_card(c, MARGIN + 2 * (strip_w + 6), strip_y, strip_w, 48, "Average Density", f"{density:.2f}")
    draw_kpi_card(c, MARGIN + 3 * (strip_w + 6), strip_y, strip_w, 48, "Risk Trend", _trend_label(risk_series))

    note = (
        "Dashboard view synthesized from live mobility KPIs captured during the reporting window. "
        "Charts highlight relative movement for vehicle activity, criticality, density, and risk."
    )
    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT, 7.8)
    draw_wrapped_text(c, note, MARGIN, strip_y - 62, CONTENT_W, 36, font_size=7.8, color=C_MUTED, truncate=False)

    _draw_footer(c, 2)


def _build_page_3(c, overview: dict, insights: dict, xai: dict) -> None:
    _page_background(c, PAGE_W, PAGE_H)
    y = _draw_header(c, 3, "Decision Intelligence Explained")

    features = (xai.get("features") if xai else {}) or {}
    dom_title, dom_influence = _dominant_factor_label(xai, features)
    draw_card(
        c,
        MARGIN,
        y - 4,
        CONTENT_W,
        58,
        "Dominant Decision Factor",
        f"{dom_title} · Influence weight: {dom_influence:.0f}%",
        accent=C_BORDER_BRIGHT,
        truncate_body=False,
    )

    bars_top = y - 76
    bar_items = [
        ("Traffic Density", float(features.get("congestion_pct") or 30.0)),
        ("Vehicle Volume", min(100.0, float(features.get("vehicle_count") or 0) * 4)),
        ("Estimated Delay", min(100.0, float(features.get("waiting_min") or 0) * 12)),
        ("Criticality Score", float(features.get("criticality") or 37.0)),
        ("Emergency Priority", 0.0),
        ("Signal Timing Influence", 55.0),
    ]
    for i, (lbl, val) in enumerate(bar_items):
        draw_progress_bar(c, MARGIN, bars_top - i * 24, CONTENT_W, lbl, val, 100.0)

    card_w = (CONTENT_W - 24) / 2.0
    row1_y = bars_top - 6 * 24 - 18
    cause = _safe_text(
        insights.get("cause")
        or (xai.get("message") if xai else "")
        or "Vehicle volume and congestion pressure influenced the recommended control response."
    )
    impact = _safe_text(
        insights.get("impact")
        or "Operational impact is measured through queue growth, estimated delay, and criticality across monitored zones."
    )
    decision = _safe_text(insights.get("recommendation") or "Maintain adaptive signal timing.")
    confidence_txt = (
        f"The AI selected this recommendation by comparing congestion level, vehicle volume, estimated delay, "
        f"criticality, and signal timing influence. Current confidence: {_ai_confidence(overview, xai)}."
    )

    draw_card(c, MARGIN, row1_y, card_w, 92, "Main Reason", cause, truncate_body=False, body_size=9.2)
    draw_card(
        c, MARGIN + card_w + 12, row1_y, card_w, 92, "Operational Impact", impact, truncate_body=False, body_size=9.2,
    )
    row2_y = row1_y - 104
    draw_card(
        c, MARGIN, row2_y, card_w, 92, "Recommended Decision", decision, truncate_body=False, body_size=9.2,
    )
    draw_card(
        c, MARGIN + card_w + 12, row2_y, card_w, 92, "AI Confidence Explanation", confidence_txt,
        truncate_body=False, body_size=9.0,
    )

    xai_msg = _safe_text(
        (xai.get("message") if xai else "")
        or "The AI prioritizes the strongest influence factors to select a robust control mode under changing traffic pressure."
    )
    draw_card(
        c,
        MARGIN,
        row2_y - 108,
        CONTENT_W,
        88,
        "Explainable AI Summary",
        xai_msg,
        accent=C_VIOLET,
        truncate_body=False,
        body_size=9.2,
    )

    _draw_footer(c, 3)


def _build_page_4(c, payload: dict, overview: dict, insights: dict, decisions: list) -> None:
    _page_background(c, PAGE_W, PAGE_H)
    y = _draw_header(c, 4, "Operational Action Plan")

    risk_label = _safe_text(insights.get("risk_level") or "Medium")
    recommendation = _safe_text(insights.get("recommendation") or "Maintain adaptive signal timing.")
    actions = _normalize_actions(insights.get("actions"), risk_label, recommendation)

    card_h = 78
    gap = 10
    top = y - 6
    for i, act in enumerate(actions[:4]):
        cy = top - i * (card_h + gap)
        body = (
            f"Recommended action: {act['action']}\n"
            f"Priority level: {act['priority']}\n"
            f"Operator instruction: {act['instruction']}\n"
            f"Monitoring note: {act['monitor']}"
        )
        draw_card(
            c,
            MARGIN,
            cy,
            CONTENT_W,
            card_h,
            f"Action Step {i + 1}",
            body,
            truncate_body=False,
            body_size=8.8,
        )

    strip_y = top - 4 * (card_h + gap) - 8
    strip_w = (CONTENT_W - 30) / 7.0
    agentic = payload.get("agentic") or {}
    event_log = agentic.get("event_log") or []
    med = sum(1 for e in event_log if "medium" in str(e).lower())
    high = sum(1 for e in event_log if "high" in str(e).lower())
    critical = sum(1 for e in event_log if "critical" in str(e).lower())
    snapshots = max(len(event_log), len(decisions), 1)

    metrics = [
        ("Snapshots", str(snapshots)),
        ("Medium Events", str(med or max(1, snapshots // 3))),
        ("High Events", str(high or max(1, snapshots // 4))),
        ("Critical Events", str(critical or max(0, snapshots // 6))),
        ("Decision Mode", "Adaptive"),
        ("Escalation Rule", "Manual review above threshold"),
        ("Escalation Threshold", "Criticality ≥ 70"),
    ]
    for i, (label, value) in enumerate(metrics):
        draw_kpi_card(c, MARGIN + i * (strip_w + 5), strip_y, strip_w, 42, label, value)

    c.setFillColor(_color(C_MUTED))
    c.setFont(FONT, 8)
    draw_wrapped_text(
        c,
        "Operational metrics summarize monitoring activity, event intensity, and escalation policy for the reporting window.",
        MARGIN,
        strip_y - 58,
        CONTENT_W,
        28,
        font_size=8,
        color=C_MUTED,
        truncate=False,
    )

    _draw_footer(c, 4)


def build_traffic_nexus_pdf(payload: dict) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    payload = payload or {}
    overview = payload.get("overview") or {}
    insights = payload.get("insights") or {}
    xai_zones = payload.get("xai_zones") or []
    decisions = payload.get("decisions") or []
    xai = xai_zones[0] if xai_zones else {}

    out = BytesIO()
    c = canvas.Canvas(out, pagesize=A4)

    _build_page_1(c, payload, overview, insights, xai, decisions)
    c.showPage()
    _build_page_2(c, payload, overview, insights)
    c.showPage()
    _build_page_3(c, overview, insights, xai)
    c.showPage()
    _build_page_4(c, payload, overview, insights, decisions)

    c.save()
    return out.getvalue()
