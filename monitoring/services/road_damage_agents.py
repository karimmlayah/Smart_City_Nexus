from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import requests
from django.conf import settings
from django.core.mail import send_mail
from django.utils import timezone
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def assess_image_quality(image_path: str | Path) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {"ok": False, "reason": "image_unreadable"}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    too_blurry = blur_score < float(getattr(settings, "ROAD_DAMAGE_QUALITY_MIN_BLUR", 70.0))
    too_dark = brightness < float(getattr(settings, "ROAD_DAMAGE_QUALITY_MIN_BRIGHTNESS", 50.0))
    too_bright = brightness > float(getattr(settings, "ROAD_DAMAGE_QUALITY_MAX_BRIGHTNESS", 220.0))
    ok = not (too_blurry or too_dark or too_bright)
    message = "Qualite acceptable."
    if not ok:
        message = "Image de mauvaise qualite. Refaire capture (plus nette/lumineuse)."
    return {
        "ok": ok,
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "too_blurry": too_blurry,
        "too_dark": too_dark,
        "too_bright": too_bright,
        "message": message,
    }


def anti_false_positive_adjust(result: dict) -> dict:
    max_conf = float(result.get("max_conf") or 0.0)
    damage_ratio = float(result.get("damage_ratio") or 0.0)
    damage_frames = int(result.get("damage_frames") or 0)
    verdict = str(result.get("verdict") or "")
    corrected = False
    reason = ""
    if verdict == "damage" and (max_conf < 0.42 or (damage_ratio < 0.1 and damage_frames <= 1)):
        result["verdict"] = "no_damage"
        result["verdict_label"] = "Doute eleve: verification manuelle recommandee"
        corrected = True
        reason = "Signal faible (confiance/ratio) - possible faux positif."
    return {"result": result, "corrected": corrected, "reason": reason}


def compute_priority_and_plan(*, risk_score: float, confidence: float, damage_ratio: float) -> dict:
    s = max(0.0, min(1.0, risk_score))
    if s >= 0.8:
        priority = "critique"
        budget = (7000, 18000)
        deadline_days = 1
    elif s >= 0.55:
        priority = "elevee"
        budget = (2500, 9000)
        deadline_days = 7
    else:
        priority = "moyenne"
        budget = (800, 3000)
        deadline_days = 30
    plan = [
        {"horizon": "immediat", "task": "Securiser la zone et signaliser le danger."},
        {"horizon": "7 jours", "task": "Inspection detaillee + estimation technique."},
        {"horizon": "30 jours", "task": "Refection durable et controle post-travaux."},
    ]
    if priority == "moyenne":
        plan[0]["task"] = "Surveillance terrain et signalisation legere."
    return {
        "priority": priority,
        "budget_min": budget[0],
        "budget_max": budget[1],
        "deadline_days": deadline_days,
        "plan": plan,
        "kpi": {
            "risk_pct": int(round(s * 100)),
            "confidence_pct": int(round(confidence * 100)),
            "frequency_pct": int(round(damage_ratio * 100)),
        },
    }


def build_ticket(priority: str, route_name: str) -> dict:
    code = f"RD-{timezone.now():%Y%m%d}-{uuid4().hex[:8].upper()}"
    deadline_map = {"critique": 1, "elevee": 7, "moyenne": 30}
    d = deadline_map.get(priority, 30)
    due = (timezone.now() + timedelta(days=d)).date().isoformat()
    return {"ticket_id": code, "deadline_days": d, "deadline_date": due, "route_name": route_name or "N/A"}


def send_risk_alerts(*, risk_score: float, title: str, message: str) -> dict:
    threshold = float(getattr(settings, "ROAD_DAMAGE_ALERT_NOTIFY_MIN_SCORE", 0.75))
    if risk_score < threshold:
        return {"sent": False, "reason": "below_threshold"}
    sent_channels: list[str] = []
    errors: list[str] = []

    to_emails = [x.strip() for x in str(getattr(settings, "ROAD_DAMAGE_ALERT_EMAILS", "")).split(",") if x.strip()]
    if to_emails:
        try:
            send_mail(
                subject=title,
                message=message,
                from_email=getattr(settings, "DEFAULT_FROM_EMAIL", "smartcity@localhost"),
                recipient_list=to_emails,
                fail_silently=False,
            )
            sent_channels.append("email")
        except Exception as exc:
            errors.append(f"email:{exc}")

    tg_webhook = str(getattr(settings, "ROAD_DAMAGE_TELEGRAM_WEBHOOK", "")).strip()
    if tg_webhook:
        try:
            requests.post(tg_webhook, json={"text": f"{title}\n{message}"}, timeout=12)
            sent_channels.append("telegram")
        except Exception as exc:
            errors.append(f"telegram:{exc}")

    wa_webhook = str(getattr(settings, "ROAD_DAMAGE_WHATSAPP_WEBHOOK", "")).strip()
    if wa_webhook:
        try:
            requests.post(wa_webhook, json={"text": f"{title}\n{message}"}, timeout=12)
            sent_channels.append("whatsapp")
        except Exception as exc:
            errors.append(f"whatsapp:{exc}")

    return {"sent": bool(sent_channels), "channels": sent_channels, "errors": errors}


def geo_link(latitude: float | None, longitude: float | None) -> str:
    if latitude is None or longitude is None:
        return ""
    return f"https://www.google.com/maps?q={latitude},{longitude}"


def export_pdf_report(
    *,
    route_name: str,
    risk_label: str,
    risk_pct: int,
    priority: str,
    ticket_id: str,
    report_text: str,
    annotated_image_path: str = "",
    heatmap_image_path: str = "",
) -> str:
    out_dir = Path(settings.MEDIA_ROOT) / "road_damage_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_{uuid4().hex}.pdf"

    c = canvas.Canvas(str(out_path), pagesize=A4)
    w, h = A4
    y = h - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "Road Damage - Rapport Professionnel")
    y -= 24
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Route: {route_name or 'N/A'}")
    y -= 14
    c.drawString(40, y, f"Niveau de risque: {risk_label} ({risk_pct}%)")
    y -= 14
    c.drawString(40, y, f"Priorite: {priority} | Ticket: {ticket_id}")
    y -= 18
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Rapport LLM")
    y -= 14
    c.setFont("Helvetica", 9)
    for line in (report_text or "").splitlines():
        if y < 60:
            c.showPage()
            y = h - 40
            c.setFont("Helvetica", 9)
        c.drawString(40, y, line[:120])
        y -= 12

    def draw_img(path: str, title: str, y_pos: float) -> float:
        p = Path(path) if path else None
        if not p or not p.is_file():
            return y_pos
        if y_pos < 260:
            c.showPage()
            y_pos = h - 40
        c.setFont("Helvetica-Bold", 11)
        c.drawString(40, y_pos, title)
        y_pos -= 10
        c.drawImage(str(p), 40, y_pos - 170, width=240, height=160, preserveAspectRatio=True, mask="auto")
        return y_pos - 190

    y = draw_img(annotated_image_path, "Image annotee", y)
    y = draw_img(heatmap_image_path, "Heatmap XAI", y)
    c.save()
    return str(out_path)
