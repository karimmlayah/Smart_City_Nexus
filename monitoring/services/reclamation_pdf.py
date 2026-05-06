"""
Rapport PDF décisionnel pour signalements citoyens — Smart City AI (ReportLab + charts optionnels).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4
from xml.sax.saxutils import escape

from django.conf import settings

from monitoring.ai_agents import AI_W_CONFIDENCE, AI_W_LOCATION, AI_W_SEVERITY
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# --- Public API versions (footer / compatibility) ---------------------------------
DEFAULT_REPORT_VERSION = "1.0.0"


def media_url_to_fs(url: str | None) -> str:
    """Convertit une URL MEDIA en chemin absolu si le fichier existe."""
    if not url:
        return ""
    try:
        mu = settings.MEDIA_URL
        rel = url.replace(mu, "", 1).lstrip("/").replace("\\", "/")
        if url.startswith("http"):
            from urllib.parse import urlparse

            rel = urlparse(url).path.replace(mu, "", 1).lstrip("/")
        p = (Path(settings.MEDIA_ROOT) / rel).resolve()
        return str(p) if p.is_file() else ""
    except Exception:
        return ""


def _tone_palette(priority_tone: str) -> dict[str, Any]:
    """Red / Orange / Blue / Green per spec."""
    t = (priority_tone or "medium").lower()
    if t == "critical":
        return {
            "main": colors.HexColor("#c62828"),
            "light": colors.HexColor("#ffebee"),
            "label": "Critical",
            "hex_main": "#c62828",
        }
    if t == "high":
        return {
            "main": colors.HexColor("#ef6c00"),
            "light": colors.HexColor("#fff3e0"),
            "label": "High",
            "hex_main": "#ef6c00",
        }
    if t == "medium":
        return {
            "main": colors.HexColor("#1565c0"),
            "light": colors.HexColor("#e3f2fd"),
            "label": "Medium",
            "hex_main": "#1565c0",
        }
    return {
        "main": colors.HexColor("#2e7d32"),
        "light": colors.HexColor("#e8f5e9"),
        "label": "Low",
        "hex_main": "#2e7d32",
    }


def _safe_para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(str(text or "")), style)


def build_executive_summary(
    *,
    category_label: str,
    detected_issue: str,
    final_priority: str,
    ai_score: float,
    risk_level: str,
    urgency: str,
    groq_summary: str,
    palette: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Section A — narrative + KPI strip."""
    # One short narrative sentence for jury / municipal readers
    issue_short = (detected_issue or category_label or "an incident")[:80]
    hx = palette.get("hex_main") or "#1565c0"
    summary_line = (
        f"A citizen reported <b>{escape(category_label)}</b>. AI analysis points to "
        f"<b>{escape(issue_short)}</b> with <font color=\"{hx}\"><b>{escape(final_priority)}</b></font> priority. "
        f"{escape((groq_summary or '')[:280])}"
    )
    if len(summary_line) > 600:
        summary_line = summary_line[:600] + "…"

    intro_style = ParagraphStyle(
        name="ExecIntro",
        parent=styles["body"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=10,
    )
    rows = [
        Paragraph(summary_line, intro_style),
        Spacer(1, 6),
    ]

    # KPI mini-table
    kpi_data = [
        [
            _safe_para("⚠ Priority", styles["kpi_label"]),
            _safe_para(str(final_priority), styles["kpi_value"]),
            _safe_para("📊 AI score", styles["kpi_label"]),
            _safe_para(f"{ai_score:.1f} / 100", styles["kpi_value"]),
        ],
        [
            _safe_para("🧠 Risk level", styles["kpi_label"]),
            _safe_para(str(risk_level), styles["kpi_value"]),
            _safe_para("⏱ Urgency", styles["kpi_label"]),
            _safe_para(str(urgency), styles["kpi_value"]),
        ],
    ]
    kt = Table(kpi_data, colWidths=[3.2 * cm, 4.8 * cm, 3.2 * cm, 4.8 * cm])
    kt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), palette["light"]),
                ("BOX", (0, 0), (-1, -1), 0.8, palette["main"]),
                ("LINEABOVE", (0, 1), (-1, 1), 0.4, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    rows.append(kt)
    return rows


def build_charts(
    *,
    pipeline: dict[str, Any],
    category_distribution: dict[str, int] | None,
    out_dir: Path,
) -> tuple[str | None, str | None]:
    """
    Returns paths to PNG files for score breakdown + optional category distribution.
    Falls back gracefully if matplotlib unavailable or errors.
    """
    chart_a: str | None = None
    chart_b: str | None = None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)

        sev = float(pipeline.get("severity_score") or 0)
        loc = float(pipeline.get("location_score") or 0)
        conf = float(pipeline.get("confidence_score") or 0)

        # Weighted contribution to AI score (same formula as pipeline)
        w_sev = AI_W_SEVERITY * sev
        w_loc = AI_W_LOCATION * loc
        w_conf = AI_W_CONFIDENCE * conf

        fig1, ax1 = plt.subplots(figsize=(6.2, 2.8))
        labels = [
            f"Severity ×{AI_W_SEVERITY}",
            f"Location ×{AI_W_LOCATION}",
            f"Confidence ×{AI_W_CONFIDENCE}",
        ]
        vals = [w_sev, w_loc, w_conf]
        colors_bars = ["#c62828", "#1565c0", "#2e7d32"]
        ax1.barh(labels, vals, color=colors_bars, height=0.55)
        ax1.set_xlabel("Contribution to aggregate AI score")
        ax1.set_title("AI score breakdown (weighted components)")
        xmax = max(vals) if vals else 1.0
        ax1.set_xlim(0, max(100.0, xmax * 1.15))
        fig1.tight_layout()
        p1 = out_dir / f"chart_score_{uuid4().hex[:10]}.png"
        fig1.savefig(str(p1), dpi=120)
        plt.close(fig1)
        chart_a = str(p1)

        # Category distribution (other reports in system)
        if category_distribution and len(category_distribution) > 1:
            fig2, ax2 = plt.subplots(figsize=(6.2, 3.2))
            keys = list(category_distribution.keys())[:8]
            vals2 = [category_distribution[k] for k in keys]
            ax2.bar(range(len(keys)), vals2, color="#1565c0")
            ax2.set_xticks(range(len(keys)))
            ax2.set_xticklabels(keys, rotation=35, ha="right", fontsize=8)
            ax2.set_ylabel("Report count")
            ax2.set_title("Category distribution (all citizen reports in DB)")
            fig2.tight_layout()
            p2 = out_dir / f"chart_cat_{uuid4().hex[:10]}.png"
            fig2.savefig(str(p2), dpi=120)
            plt.close(fig2)
            chart_b = str(p2)
    except Exception:
        return None, None
    return chart_a, chart_b


def build_visual_section(
    *,
    original_fs: str,
    annotated_fs: str,
    heatmap_fs: str,
    media_type: str,
    styles: dict[str, ParagraphStyle],
    max_w_cm: float = 14.0,
) -> list[Any]:
    """Section F — images with captions."""
    flow: list[Any] = []
    cap_style = ParagraphStyle(
        name="ImgCap",
        parent=styles["caption"],
        fontSize=8,
        textColor=colors.HexColor("#424242"),
        alignment=TA_CENTER,
    )

    def add_img(path: str, title: str, caption: str) -> None:
        if not path or not Path(path).is_file():
            return
        try:
            img = RLImage(path, width=max_w_cm * cm, height=4.5 * cm, kind="proportional")
            flow.append(Spacer(1, 6))
            flow.append(_safe_para(f"📷 {title}", styles["section"]))
            flow.append(img)
            flow.append(_safe_para(caption, cap_style))
        except Exception:
            pass

    add_img(
        original_fs,
        "Original media",
        "Citizen-submitted evidence (unaltered)." if media_type == "image" else "Video report — key frame context.",
    )
    add_img(
        annotated_fs,
        "YOLO detection overlay",
        "Model bounding boxes on detected objects / damage regions.",
    )
    add_img(
        heatmap_fs,
        "XAI attention heatmap",
        "Model attention focuses on damaged road surface and salient texture changes.",
    )
    return flow


def build_action_plan(
    *,
    groq_json: dict[str, Any],
    final_priority: str,
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Structured recommendations + heuristic TND / time estimates."""
    actions = list(groq_json.get("recommended_actions") or [])
    maint = str(groq_json.get("maintenance_plan") or "")
    urgency = str(groq_json.get("estimated_urgency") or "")

    # Split crude tiers
    immediate = actions[:3] if actions else ["Dispatch field verification.", "Secure perimeter if hazard."]
    short_term = actions[3:6] if len(actions) > 3 else ["Schedule crew window per municipal SLA."]
    long_term = [maint] if maint else ["Infrastructure audit & preventive maintenance cycle."]

    # Heuristic cost / time from priority
    pr = (final_priority or "Medium").lower()
    if pr == "critical":
        tnd_min, tnd_max = 12000, 22000
        time_hint = "Immediate dispatch · intervention window 12–24h"
    elif pr == "high":
        tnd_min, tnd_max = 5500, 12000
        time_hint = "24–48h mobilisation"
    elif pr == "medium":
        tnd_min, tnd_max = 2000, 6000
        time_hint = "48–120h planning"
    else:
        tnd_min, tnd_max = 800, 3500
        time_hint = "Planned within 1–2 weeks"

    blocks: list[Any] = []
    blocks.append(_safe_para("<b>Immediate actions</b>", styles["sub"]))
    for a in immediate:
        blocks.append(_safe_para(f"• {a}", styles["body"]))
    blocks.append(Spacer(1, 4))
    blocks.append(_safe_para("<b>Short-term plan</b>", styles["sub"]))
    for a in short_term:
        blocks.append(_safe_para(f"• {a}", styles["body"]))
    blocks.append(Spacer(1, 4))
    blocks.append(_safe_para("<b>Long-term fix</b>", styles["sub"]))
    for a in long_term:
        blocks.append(_safe_para(f"• {a}", styles["body"]))
    blocks.append(Spacer(1, 8))
    est = Table(
        [
            [
                _safe_para("Estimated response time", styles["table_head"]),
                _safe_para(escape(time_hint) + f" (Groq: {escape(urgency)})", styles["body"]),
            ],
            [
                _safe_para("Indicative budget (TND)", styles["table_head"]),
                _safe_para(
                    f"{tnd_min:,} – {tnd_max:,} TND (order of magnitude — validate with procurement)",
                    styles["body"],
                ),
            ],
        ],
        colWidths=[4.5 * cm, 12.5 * cm],
    )
    est.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bdbdbd")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eceff1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    blocks.append(est)
    return blocks


def _risk_impact_text(
    *,
    category_label: str,
    risk_level: str,
    final_priority: str,
) -> str:
    """Section I — synthetic narrative."""
    cat = (category_label or "").lower()
    impact_traffic = (
        "Potential lane obstruction or reduced ride comfort — adjust traffic flow if near intersections."
        if "pothole" in cat or "route" in cat
        else "May affect pedestrian or vehicle flow depending on exact location."
    )
    safety = (
        "Elevated injury risk for two-wheelers and night visibility if untreated."
        if risk_level in ("Critical", "High")
        else "Moderate safety concern — monitor weather and traffic peaks."
    )
    infra = (
        "Accelerated pavement degradation and water ingress if left unresolved."
        if final_priority in ("Critical", "High")
        else "Progressive wear — schedule inspection before seasonal peaks."
    )
    return f"Traffic: {impact_traffic} Safety: {safety} Infrastructure: {infra}"


def _zone_label(nearby_count: int, final_priority: str) -> str:
    if nearby_count >= 5 and final_priority in ("Critical", "High"):
        return "Critical zone — clustered citizen signals"
    if nearby_count >= 3:
        return "Elevated activity zone — recurring complaints nearby"
    return "Standard zone"


def build_copilot_pdf_flowables(
    copilot_plan: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    """Sections PDF pour Municipality AI Copilot."""
    cp = {k: v for k, v in copilot_plan.items() if k != "_meta"}
    flow: list[Any] = []
    flow.append(_safe_para("Municipality AI Copilot — plan d’intervention", styles["section"]))
    flow.append(
        _safe_para(
            f"<b>Risque</b> : {escape(str(cp.get('risk_level')))} · "
            f"<b>Urgence</b> : {escape(str(cp.get('urgency')))} · "
            f"<b>Source</b> : {escape(str(cp.get('source')))}",
            styles["body"],
        )
    )
    flow.append(_safe_para(f"<b>Résumé</b> — {escape(str(cp.get('executive_summary') or ''))}", styles["body"]))
    flow.append(_safe_para(f"<b>Pourquoi cette priorité</b> — {escape(str(cp.get('priority_reason') or ''))}", styles["body"]))

    da = cp.get("damage_assessment") or {}
    flow.append(_safe_para("<b>Évaluation dégâts</b>", styles["sub"]))
    flow.append(
        _safe_para(
            escape(
                f"Type estimé: {da.get('estimated_damage_type')} · Taille: {da.get('estimated_size')} · "
                f"Confiance: {da.get('confidence')} — {da.get('visual_evidence', '')[:700]}"
            ),
            styles["body"],
        )
    )

    flow.append(
        _safe_para(
            f"<b>Coût total estimatif (TND)</b> : {cp.get('total_estimated_cost_tnd')}",
            styles["sub"],
        )
    )

    for ph in cp.get("intervention_plan") or []:
        acts = " ".join(f"• {escape(str(a))}" for a in (ph.get("actions") or [])[:12])
        flow.append(
            _safe_para(
                f"<b>{escape(str(ph.get('phase')))}</b> — échéance {escape(str(ph.get('deadline')))} · "
                f"{escape(str(ph.get('responsible_team')))} · ~{ph.get('estimated_cost_tnd')} TND",
                styles["body"],
            )
        )
        flow.append(_safe_para(acts, styles["body"]))
        mats = ", ".join(str(m) for m in (ph.get("materials") or [])[:15])
        if mats:
            flow.append(_safe_para(f"Matériels : {escape(mats)}", styles["caption"]))

    rp = cp.get("resource_plan") or {}
    flow.append(_safe_para("<b>Ressources</b>", styles["sub"]))
    flow.append(
        _safe_para(
            escape(
                f"Équipes : {', '.join(rp.get('teams_needed') or [])} · "
                f"Effectifs ~{rp.get('workers_estimate')} · "
                f"Équipement : {', '.join(rp.get('equipment') or [])[:400]}"
            ),
            styles["body"],
        )
    )

    flow.append(_safe_para(f"<b>Message citoyens</b> — {escape(str(cp.get('citizen_message') or ''))}", styles["body"]))
    flow.append(_safe_para(f"<b>Message municipalité</b> — {escape(str(cp.get('municipality_message') or ''))}", styles["body"]))
    fu = cp.get("follow_up") or {}
    flow.append(
        _safe_para(
            escape(
                f"<b>Suivi</b> : inspection complémentaire = {fu.get('needs_second_inspection')} · "
                f"{fu.get('suggested_status')} · contrôle J+{fu.get('verification_after_days')}"
            ),
            styles["body"],
        )
    )
    return flow


def export_reclamation_ai_pdf(
    *,
    report_pk: int,
    created_label: str,
    category_label: str,
    description: str,
    latitude: float | None,
    longitude: float | None,
    pipeline: dict[str, Any],
    vision: dict[str, Any],
    xai_result: dict[str, Any],
    groq_json: dict[str, Any],
    selected_model_key: str,
    conf_threshold: float,
    media_type: str = "image",
    original_media_url: str | None = None,
    category_distribution: dict[str, int] | None = None,
    total_system_reports: int | None = None,
    copilot_plan: dict[str, Any] | None = None,
) -> str:
    """
    Writes a professional PDF under MEDIA_ROOT/reclamation_reports/.
    Same entry signature as before + optional enrichment kwargs (backward compatible).
    """
    out_dir = Path(settings.MEDIA_ROOT) / "reclamation_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reclamation_{report_pk}_{uuid4().hex[:10]}.pdf"

    tone = pipeline.get("priority_tone") or "medium"
    palette = _tone_palette(str(tone))

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleMain",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0d47a1"),
            spaceAfter=4,
            fontName="Helvetica-Bold",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#546e7a"),
            spaceAfter=12,
            fontName="Helvetica",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1565c0"),
            spaceBefore=14,
            spaceAfter=8,
            fontName="Helvetica-Bold",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Sub",
            fontSize=10,
            leading=13,
            spaceBefore=6,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="kpi_label",
            fontSize=9,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#37474f"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="kpi_value",
            fontSize=10,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0d47a1"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="table_head",
            parent=styles["Body"],
            fontName="Helvetica-Bold",
        )
    )
    style_map = {
        "title": styles["TitleMain"],
        "subtitle": styles["Subtitle"],
        "section": styles["Section"],
        "sub": styles["Sub"],
        "body": styles["Body"],
        "caption": styles["Caption"],
        "kpi_label": styles["kpi_label"],
        "kpi_value": styles["kpi_value"],
        "table_head": styles["table_head"],
    }

    story: list[Any] = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_version = getattr(settings, "SMARTCITY_REPORT_VERSION", DEFAULT_REPORT_VERSION)

    # ----- Header band (table mimicking logo + titles + meta) -----
    header_table = Table(
        [
            [
                _safe_para(
                    "<b><font size=11>[ LOGO ]</font></b><br/><font size=7>Smart City</font>",
                    styles["Normal"],
                ),
                Paragraph(
                    "<para align=center><b><font size=16 color='#0d47a1'>Smart City AI Incident Report</font></b><br/>"
                    "<font size=10 color='#546e7a'>Citizen Signal → AI Analysis → Decision Support</font></para>",
                    styles["Normal"],
                ),
                Paragraph(
                    f"<para align=right><font size=9><b>Report ID</b> #{report_pk}<br/>"
                    f"<b>Date</b> {escape(created_label)}<br/>"
                    f"<b>Generated</b> {escape(now_str)}</font></para>",
                    styles["Normal"],
                ),
            ]
        ],
        colWidths=[3.2 * cm, 11.0 * cm, 5.8 * cm],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#eceff1")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#90a4ae")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 12))

    # Priority ribbon
    ribbon = Table(
        [
            [
                _safe_para(f"Priority band: {pipeline.get('final_priority')}", style_map["kpi_value"]),
            ]
        ],
        colWidths=[17 * cm],
    )
    ribbon.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), palette["light"]),
                ("LINEBELOW", (0, 0), (-1, -1), 3, palette["main"]),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(ribbon)
    story.append(Spacer(1, 10))

    # A. Executive summary
    story.append(_safe_para("A. Executive summary", style_map["section"]))
    risk_ll = str(groq_json.get("risk_level") or "—")
    urg_ll = str(groq_json.get("estimated_urgency") or "—")
    if copilot_plan:
        risk_ll = str(copilot_plan.get("risk_level") or risk_ll)
        urg_ll = str(copilot_plan.get("urgency") or urg_ll)
    story.extend(
        build_executive_summary(
            category_label=category_label,
            detected_issue=str(pipeline.get("detected_issue") or ""),
            final_priority=str(pipeline.get("final_priority") or ""),
            ai_score=float(pipeline.get("ai_score") or 0),
            risk_level=risk_ll,
            urgency=urg_ll,
            groq_summary=str(groq_json.get("summary") or ""),
            palette=palette,
            styles=style_map,
        )
    )

    # B. Incident details
    story.append(_safe_para("B. Incident details", style_map["section"]))
    gps = (
        f"{latitude:.6f}, {longitude:.6f}"
        if latitude is not None and longitude is not None
        else "N/A"
    )
    det_rows = [
        ["📍 Category", category_label],
        ["Description", (description or "—")[:2000]],
        ["GPS", gps],
        ["Date / time", created_label],
        ["Media type", media_type or "—"],
    ]
    t_det = Table(
        [[_safe_para(f"<b>{escape(a)}</b>", style_map["body"]), _safe_para(escape(str(b)), style_map["body"])] for a, b in det_rows],
        colWidths=[4.0 * cm, 13.0 * cm],
    )
    t_det.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cfd8dc")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f5f5f5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_det)

    # C. AI analysis
    story.append(_safe_para("C. AI analysis (vision)", style_map["section"]))
    story.append(
        _safe_para(
            f"<b>🤖 Detected issue</b> — {escape(str(pipeline.get('detected_issue') or '—'))}",
            style_map["body"],
        )
    )
    lbls = ", ".join(vision.get("labels") or []) or "—"
    story.append(
        _safe_para(
            f"<b>YOLO</b> — boxes: <b>{vision.get('detection_box_count', 0)}</b>, "
            f"max confidence: <b>{vision.get('max_confidence')}</b>, labels: {escape(lbls)}.",
            style_map["body"],
        )
    )
    story.append(
        _safe_para(
            f"<b>Model</b> — key <code>{escape(selected_model_key)}</code>, file "
            f"{escape(str(vision.get('model_path_used') or '')[:90])}",
            style_map["body"],
        )
    )
    story.append(
        _safe_para(
            f"<b>Confidence settings</b> — inference threshold (Ultralytics <i>conf</i>) = "
            f"<b>{conf_threshold:.4f}</b>. Detections below this score are discarded before display.",
            style_map["body"],
        )
    )

    # D. Multi-agent reasoning
    story.append(_safe_para("D. Multi-agent reasoning", style_map["section"]))
    sev_r = " ".join(pipeline.get("severity_reasons") or [])
    loc_r = " ".join(pipeline.get("location_reasons") or [])
    if pipeline.get("yolo_used"):
        conf_expl = (
            f"YOLO-driven confidence (max detection ×100): {pipeline.get('confidence_score')}."
        )
    else:
        conf_expl = (
            f"Rule-based confidence cap 60% (model not used): {pipeline.get('confidence_score')}."
        )
    story.append(_safe_para(f"<b>Severity</b> → {escape(sev_r)}", style_map["body"]))
    story.append(_safe_para(f"<b>Location</b> → {escape(loc_r)}", style_map["body"]))
    story.append(_safe_para(f"<b>Confidence</b> → {escape(conf_expl)}", style_map["body"]))
    formula = (
        f"ai_score = {AI_W_SEVERITY}×severity + {AI_W_LOCATION}×location + "
        f"{AI_W_CONFIDENCE}×confidence → <b>{pipeline.get('ai_score')}</b>"
    )
    story.append(_safe_para(formula, style_map["sub"]))
    why = pipeline.get("why_score") or []
    for line in why:
        story.append(_safe_para(f"• {escape(str(line))}", style_map["body"]))

    # Charts (optional)
    chart_dir = out_dir / "charts"
    c1, c2 = build_charts(
        pipeline=pipeline,
        category_distribution=category_distribution,
        out_dir=chart_dir,
    )
    story.append(_safe_para("📊 Analytics charts", style_map["section"]))
    if c1 and Path(c1).is_file():
        try:
            story.append(RLImage(c1, width=15 * cm, height=6.5 * cm))
            story.append(Spacer(1, 6))
        except Exception:
            story.append(_safe_para("(Score chart unavailable.)", style_map["caption"]))
    else:
        story.append(_safe_para("(Score chart skipped — matplotlib unavailable or error.)", style_map["caption"]))
    if c2 and Path(c2).is_file():
        try:
            story.append(RLImage(c2, width=15 * cm, height=7 * cm))
        except Exception:
            pass

    # E. Visual evidence (starts new page if charts made previous page long)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cfd8dc")))
    story.append(Spacer(1, 10))
    story.append(_safe_para("E. Visual evidence", style_map["section"]))
    orig_fs = media_url_to_fs(original_media_url)
    ann_fs = media_url_to_fs(vision.get("annotated_image_url"))
    heat_fs = media_url_to_fs(xai_result.get("heatmap_url"))
    story.extend(
        build_visual_section(
            original_fs=orig_fs,
            annotated_fs=ann_fs,
            heatmap_fs=heat_fs,
            media_type=media_type or "image",
            styles=style_map,
        )
    )

    # F. Recommendations — Copilot ou ancien format
    story.append(_safe_para("F. Municipality AI Copilot", style_map["section"]))
    if copilot_plan:
        story.extend(build_copilot_pdf_flowables(copilot_plan, style_map))
    else:
        story.extend(
            build_action_plan(
                groq_json=groq_json,
                final_priority=str(pipeline.get("final_priority") or ""),
                styles=style_map,
            )
        )

    # G. Municipality action plan
    story.append(_safe_para("G. Municipality action plan", style_map["section"]))
    nearby = int(pipeline.get("nearby_count") or 0)
    copilot_team = ""
    if copilot_plan:
        rp = copilot_plan.get("resource_plan") or {}
        copilot_team = ", ".join(rp.get("teams_needed") or [])[:200]
    map_tbl = Table(
        [
            [
                _safe_para("Priority ranking", style_map["table_head"]),
                _safe_para(str(pipeline.get("final_priority")), style_map["body"]),
            ],
            [
                _safe_para("Responsible team", style_map["table_head"]),
                _safe_para(
                    str(copilot_team or groq_json.get("responsible_team") or "—"),
                    style_map["body"],
                ),
            ],
            [
                _safe_para("Suggested owner (agents)", style_map["table_head"]),
                _safe_para(str(pipeline.get("suggested_owner") or "—"), style_map["body"]),
            ],
            [
                _safe_para("Resources (heuristic)", style_map["table_head"]),
                _safe_para(
                    "1 inspection team · 1 road crew · core equipment: compactor, hot-mix, signage, PPE "
                    "(adjust to field assessment).",
                    style_map["body"],
                ),
            ],
            [
                _safe_para("Timeline", style_map["table_head"]),
                _safe_para(str(groq_json.get("estimated_urgency") or "—"), style_map["body"]),
            ],
        ],
        colWidths=[4.8 * cm, 12.2 * cm],
    )
    map_tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#90a4ae")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eceff1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(map_tbl)

    # H. Citizen safety (highlighted)
    cit = str(groq_json.get("citizen_safety_message") or "")
    if cit:
        box = Table(
            [[_safe_para(f"<b>⚠ Citizen safety</b><br/>{escape(cit)}", style_map["body"])]],
            colWidths=[17 * cm],
        )
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffebee")),
                    ("BOX", (0, 0), (-1, -1), 1.5, colors.HexColor("#c62828")),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(Spacer(1, 10))
        story.append(_safe_para("H. Citizen safety message", style_map["section"]))
        story.append(box)

    # I. Risk & impact
    story.append(_safe_para("I. Risk & impact", style_map["section"]))
    story.append(
        _safe_para(
            escape(
                _risk_impact_text(
                    category_label=category_label,
                    risk_level=str(groq_json.get("risk_level") or ""),
                    final_priority=str(pipeline.get("final_priority") or ""),
                )
            ),
            style_map["body"],
        )
    )

    # J. Analytics & context
    story.append(_safe_para("J. Analytics & context", style_map["section"]))
    cluster = nearby + 1
    zone = _zone_label(nearby, str(pipeline.get("final_priority") or ""))
    story.append(
        _safe_para(
            f"Similar reports in zone (~1.3 km): <b>{nearby}</b> other report(s). "
            f"<b>Cluster size</b> (incl. this signal): {cluster}. "
            f"<b>Zone classification:</b> {escape(zone)}.",
            style_map["body"],
        )
    )
    if total_system_reports is not None:
        story.append(
            _safe_para(
                f"Total citizen reports in system: <b>{total_system_reports}</b>.",
                style_map["body"],
            )
        )
    story.append(
        _safe_para(
            f"<b>Recommendation source:</b> {escape(str(groq_json.get('source') or '?'))} — "
            f"{escape(str(groq_json.get('summary') or '')[:400])}",
            style_map["body"],
        )
    )

    # Groq full narrative block (retain compatibility / storytelling)
    story.append(_safe_para("AI recommendation (full text)", style_map["sub"]))
    story.append(_safe_para(escape(str(groq_json.get("maintenance_plan") or "")), style_map["body"]))

    # K. Footer on canvas via doc.build ... use onLaterPages
    def _draw_footer(canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setStrokeColor(colors.HexColor("#90a4ae"))
        canvas_obj.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
        canvas_obj.setFont("Helvetica", 8)
        canvas_obj.setFillColor(colors.HexColor("#546e7a"))
        canvas_obj.drawString(
            18 * mm,
            10 * mm,
            f"Generated by Smart City AI System · {now_str} · v{report_version}",
        )
        canvas_obj.restoreState()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title=f"SmartCity-Report-{report_pk}",
    )
    doc.build(story, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
    return str(out_path)
