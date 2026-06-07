"""MedinaMind AI Orb — Groq assistant API with platform intent routing."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests
from django.conf import settings
from django.http import JsonResponse
from django.urls import reverse
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_POST

logger = logging.getLogger(__name__)

_ORB_SYSTEM = (
    "You are MedinaMind Orb, the voice AI assistant for a Smart City command platform. "
    "Always reply in the same language the user speaks (English, French, Arabic, Spanish, German, etc.). "
    "If the language is unclear, use English. "
    "Keep replies concise for spoken delivery (2–4 sentences unless asked for detail). "
    "You help operators with city monitoring, traffic, road damage, fire/smoke, waste, drones, and alerts. "
    "You do not have live sensor access — if asked for real-time data, explain that and suggest which module to open. "
    "Be professional, futuristic, and helpful for a municipal operations center."
)

_INTENT_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"\b(open|go to|show|launch|navigate to|ouvre|ouvrir|affiche|afficher)\b.*\b(traffic control|traffic nexus|trafic)\b",
            re.I,
        ),
        "traffic_control",
        "Opening Traffic Control.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(smart traffic|traffic light|signal control)\b", re.I),
        "traffic_signal",
        "Opening Smart Traffic Lights.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(red light|violation|traffic violation)\b", re.I),
        "traffic_violation",
        "Opening Red Light Detection.",
    ),
    (
        re.compile(
            r"\b(open|go to|show|launch|ouvre|ouvrir|affiche)\b.*\b(road analysis|road damage|pothole|route)\b",
            re.I,
        ),
        "road_analysis",
        "Opening Road Analysis.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(city monitoring|surveillance|fusion)\b", re.I),
        "city_monitoring",
        "Opening City Monitoring.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(fire|smoke|fire camera)\b", re.I),
        "fire_smoke",
        "Opening Fire and Smoke detection.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(waste|street waste|garbage)\b", re.I),
        "street_waste",
        "Opening Street Waste detection.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(drone|uav)\b", re.I),
        "drone_monitoring",
        "Opening Drone Monitoring.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(camera settings|smart camera|esp32)\b", re.I),
        "camera_settings",
        "Opening Camera Settings.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(dashboard|ai dashboard|command center)\b", re.I),
        "ai_dashboard",
        "Opening the AI Dashboard.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(settings|platform settings)\b", re.I),
        "platform_settings",
        "Opening Platform Settings.",
    ),
    (
        re.compile(r"\b(open|go to|show|launch)\b.*\b(home|homepage|landing)\b", re.I),
        "home",
        "Opening the MedinaMind homepage.",
    ),
    (
        re.compile(r"\b(generate|create)\b.*\breport\b", re.I),
        "generate_report",
        "I can help you generate a report from the relevant module. Opening the AI Dashboard reports view.",
    ),
    (
        re.compile(r"\b(summarize|summary of)\b.*\b(incident|alert|issue)\b", re.I),
        "summarize_incident",
        "",
    ),
]


def _module_urls() -> dict[str, str]:
    return {
        "home": reverse("monitoring:home"),
        "ai_dashboard": reverse("monitoring:ai_dashboard"),
        "city_monitoring": reverse("monitoring:surveillance_dashboard"),
        "road_analysis": reverse("monitoring:road_damage_test"),
        "fire_smoke": reverse("monitoring:fire_camera_detect"),
        "street_waste": reverse("monitoring:waste_street_detection"),
        "drone_monitoring": reverse("monitoring:uav_dashboard"),
        "traffic_control": reverse("monitoring:traffic_nexus_dashboard"),
        "traffic_signal": reverse("monitoring:traffic_signal_control"),
        "traffic_violation": reverse("monitoring:traffic_violation"),
        "camera_settings": reverse("monitoring:camera_settings"),
        "platform_settings": reverse("monitoring:platform_settings"),
        "generate_report": reverse("monitoring:ai_dashboard"),
    }


def match_intent(message: str) -> dict[str, Any] | None:
    text = (message or "").strip()
    if not text:
        return None
    for pattern, intent_id, spoken in _INTENT_RULES:
        if pattern.search(text):
            urls = _module_urls()
            url = urls.get(intent_id)
            if not url:
                continue
            action = "navigate"
            if intent_id == "generate_report":
                url = urls["ai_dashboard"] + "?focus=reports"
            return {
                "intent": intent_id,
                "action": action,
                "url": url,
                "spoken": spoken or None,
            }
    return None


def _groq_chat(messages: list[dict[str, str]]) -> tuple[str | None, str | None]:
    api_key = (getattr(settings, "GROQ_API_KEY", None) or "").strip()
    if not api_key:
        return None, "Groq API key is missing. Add GROQ_API_KEY to your .env file."

    model = (getattr(settings, "GROQ_MODEL", None) or "llama-3.3-70b-versatile").strip()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.55,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
    except requests.RequestException as exc:
        logger.warning("Groq orb request failed: %s", exc)
        return None, "Unable to reach Groq. Please try again shortly."

    try:
        data = resp.json()
    except ValueError:
        data = {}

    if resp.status_code >= 400:
        err = data.get("error") or {}
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return None, msg or f"Groq API error ({resp.status_code})."

    choices = data.get("choices") or []
    if not choices:
        return None, "Empty response from Groq."

    content = (choices[0].get("message") or {}).get("content") or ""
    return content.strip(), None


@never_cache
@require_POST
def assistant_groq(request):
    """POST /api/assistant/groq/ — voice orb assistant (Groq-backed)."""
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    message = (body.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "The message field is required."}, status=400)
    if len(message) > 4000:
        message = message[:4000]

    intent_hit = match_intent(message)
    if intent_hit and intent_hit.get("spoken"):
        return JsonResponse(
            {
                "reply": intent_hit["spoken"],
                "intent": intent_hit["intent"],
                "action": intent_hit["action"],
                "url": intent_hit["url"],
                "source": "intent",
            }
        )

    history: list[dict[str, str]] = []
    raw_history = body.get("history")
    if isinstance(raw_history, list):
        for item in raw_history[-8:]:
            if not isinstance(item, dict):
                continue
            role = (item.get("role") or "").strip()
            content = (item.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                history.append({"role": role, "content": content[:2000]})

    api_messages = [{"role": "system", "content": _ORB_SYSTEM}] + history
    api_messages.append({"role": "user", "content": message})

    reply, err = _groq_chat(api_messages)
    if err:
        return JsonResponse({"error": err}, status=503 if "missing" in err.lower() else 502)

    if intent_hit and not intent_hit.get("spoken"):
        return JsonResponse(
            {
                "reply": reply,
                "intent": intent_hit["intent"],
                "action": intent_hit["action"],
                "url": intent_hit["url"],
                "source": "groq+intent",
            }
        )

    post_intent = match_intent(reply or "")
    payload: dict[str, Any] = {"reply": reply, "source": "groq"}
    if post_intent:
        payload.update(
            {
                "intent": post_intent["intent"],
                "action": post_intent["action"],
                "url": post_intent["url"],
            }
        )
    return JsonResponse(payload)
