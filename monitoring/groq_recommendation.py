"""
Recommandations JSON via Groq (si clé) ou règles déterministes.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from django.conf import settings


def _rule_based(payload: dict[str, Any]) -> dict[str, Any]:
    cat = (payload.get("category_code") or payload.get("category") or "").lower()
    desc = (payload.get("description") or "").lower()
    ai_score = float(payload.get("ai_score") or 50)
    prio = payload.get("final_priority") or "Medium"

    if prio in ("Critical", "High") or ai_score >= 75:
        risk = "Critical" if prio == "Critical" else "High"
        urgency = "Immediate"
    elif ai_score >= 55:
        risk = "Medium"
        urgency = "24h"
    else:
        risk = "Low"
        urgency = "1 week"

    team = "Municipality"
    if cat == "pothole" or "pothole" in desc or "nid" in desc:
        team = "Road maintenance team"
        actions = [
            "Secure perimeter and temporary signage.",
            "Schedule asphalt / pothole repair crew.",
            "Verify drainage after intervention.",
        ]
        maint = "Cut degraded pavement, clean, tack coat, hot-mix patch, compact."
    elif cat == "garbage":
        team = "Environmental service"
        actions = ["Dispatch cleaning unit.", "Assess bin placement.", "Sort recyclables if applicable."]
        maint = "Street sweep + selective collection within 48h."
    elif cat in ("smoke", "flood"):
        team = "Emergency service"
        actions = ["Verify source and coordinate fire/environment.", "Protect citizens and divert traffic."]
        maint = "Incident-led response and infrastructure inspection."
    else:
        actions = ["Field verification.", "Update citizen ticket."]
        maint = "Standard municipal review."

    labels = payload.get("yolo_labels") or []
    summary = (
        f"Citizen report ({cat or 'general'}). AI aggregate score {ai_score:.0f}. "
        f"Detections: {', '.join(labels) if labels else 'none (stub)'}. Priority {prio}."
    )

    return {
        "summary": summary,
        "risk_level": risk,
        "recommended_actions": actions,
        "responsible_team": team,
        "citizen_safety_message": "Stay clear of the hazard zone and follow official diversions.",
        "maintenance_plan": maint,
        "estimated_urgency": urgency,
        "source": "rules",
    }


def _groq_chat(messages: list[dict], model: str | None = None) -> str | None:
    key = (getattr(settings, "GROQ_API_KEY", None) or "").strip()
    if not key:
        return None
    model = model or getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
    url = "https://api.groq.com/openai/v1/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1200,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        data = json.loads(raw)
        return (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except (json.JSONDecodeError, IndexError, TypeError):
        return None


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def build_prompt_payload(
    *,
    category_code: str,
    category_label: str,
    description: str,
    latitude: float | None,
    longitude: float | None,
    yolo_labels: list[str],
    max_yolo_conf: float,
    ai_score: float,
    severity_score: float,
    location_score: float,
    confidence_score: float,
    xai_explanation: str | None,
    final_priority: str,
    yolo_conf_threshold: float | None = None,
    detection_box_count: int | None = None,
    vision_model_key: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "category_code": category_code,
        "category_label": category_label,
        "category": category_label,
        "description": description,
        "gps": f"{latitude},{longitude}" if latitude is not None and longitude is not None else "",
        "yolo_labels": yolo_labels,
        "max_yolo_confidence": max_yolo_conf,
        "ai_score": ai_score,
        "severity_score": severity_score,
        "location_score": location_score,
        "confidence_score": confidence_score,
        "xai_explanation": xai_explanation or "",
        "final_priority": final_priority,
    }
    if yolo_conf_threshold is not None:
        out["yolo_conf_threshold"] = round(float(yolo_conf_threshold), 4)
    if detection_box_count is not None:
        out["detection_box_count"] = int(detection_box_count)
    if vision_model_key:
        out["vision_model_key"] = vision_model_key
    return out


def get_recommendation_json(payload: dict[str, Any]) -> dict[str, Any]:
    """Retourne toujours un dict avec les clés attendues."""
    base_rules = _rule_based(payload)

    sys_prompt = (
        "You are a Smart City operations assistant. Reply ONLY with valid JSON matching keys: "
        "summary (string), risk_level (Critical|High|Medium|Low), recommended_actions (array of strings), "
        "responsible_team (string), citizen_safety_message (string), maintenance_plan (string), "
        "estimated_urgency (Immediate|24h|48h|1 week). "
        "Use French for narrative strings where natural."
    )
    user_content = json.dumps(payload, ensure_ascii=False)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
    ]
    raw = _groq_chat(messages)
    parsed = _parse_json_obj(raw) if raw else None

    if isinstance(parsed, dict) and parsed.get("summary"):
        parsed["source"] = "groq"
        for k in (
            "recommended_actions",
            "responsible_team",
            "citizen_safety_message",
            "maintenance_plan",
            "estimated_urgency",
            "risk_level",
        ):
            parsed.setdefault(k, base_rules.get(k))
        return parsed

    return base_rules
