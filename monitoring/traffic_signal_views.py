"""Standalone traffic light upload page with ESP32 handoff."""
from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path

import requests
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from monitoring.runtime import ml_deps_available, ml_unavailable_http

logger = logging.getLogger(__name__)


def _upload_dir() -> Path:
    d = Path(settings.MEDIA_ROOT) / "traffic_signal" / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _safe_name(name: str) -> str:
    base = Path(name or "upload").name
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    return base or "upload"


def _build_esp32_payload(result: dict) -> dict:
    signal = result.get("state") or "unknown"
    return {
        "source": "smartcity_traffic_signal",
        "signal": signal,
        "label": "Feu rouge" if signal == "red" else "Feu vert" if signal == "green" else "Non détecté",
        "confidence": result.get("confidence") or 0,
        "red_count": 1 if signal == "red" else 0,
        "green_count": 1 if signal == "green" else 0,
        "media_type": result.get("media_type") or "",
    }


def _send_to_esp32(url: str, payload: dict) -> dict:
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=5,
        )
        text = response.text[:500]
        logger.info("ESP32 status=%s response=%s", response.status_code, text)
        response.raise_for_status()
        return {
            "ok": True,
            "status_code": response.status_code,
            "message": text or "ok",
            "response_text": text,
        }
    except requests.exceptions.RequestException as exc:
        logger.warning("ESP32 request failed: %s", exc)
        return {
            "ok": False,
            "status_code": None,
            "message": str(exc),
            "response_text": "",
        }


@require_http_methods(["GET", "POST"])
def traffic_signal_control_page(request):
    context = {
        "default_esp32_url": getattr(settings, "TRAFFIC_SIGNAL_ESP32_URL", ""),
        "result": None,
        "esp32_result": None,
        "error": "",
    }

    if request.method == "GET":
        return render(request, "monitoring/traffic_signal_control.html", context)

    context["default_esp32_url"] = (request.POST.get("esp32_url") or "").strip()
    media = request.FILES.get("media")
    if not media:
        context["error"] = "Choisissez une image ou une vidéo à analyser."
        return render(request, "monitoring/traffic_signal_control.html", context)

    upload_path = _upload_dir() / f"{uuid.uuid4().hex[:10]}_{_safe_name(media.name)}"
    with open(upload_path, "wb") as out:
        for chunk in media.chunks():
            out.write(chunk)

    if not ml_deps_available():
        context["error"] = "Traffic signal ML analysis is not available on this host."
        return render(request, "monitoring/traffic_signal_control.html", context)

    from monitoring.services.traffic_signal_detection import analyze_traffic_signal_file

    result = analyze_traffic_signal_file(upload_path)
    context["result"] = result
    if not result.get("success"):
        context["error"] = result.get("error_detail") or result.get("error") or "Analyse impossible."
        return render(request, "monitoring/traffic_signal_control.html", context)

    esp32_url = context["default_esp32_url"] or getattr(settings, "TRAFFIC_SIGNAL_ESP32_URL", "")
    payload = _build_esp32_payload(result)
    payload_json = json.dumps(payload, ensure_ascii=False)
    if esp32_url:
        send_result = _send_to_esp32(esp32_url, payload)
        context["esp32_result"] = {
            **send_result,
            "payload": payload,
            "payload_json": payload_json,
            "url": esp32_url,
        }
    else:
        context["esp32_result"] = {
            "ok": False,
            "status_code": None,
            "message": "URL ESP32 manquante.",
            "response_text": "",
            "payload": payload,
            "payload_json": payload_json,
            "url": "",
        }

    return render(request, "monitoring/traffic_signal_control.html", context)
