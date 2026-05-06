"""
API dédiée « Réglages caméra » ESP32-CAM — chemins /camera/...

Mapping champ Django ↔ paramètre ESP : noms identiques (voir esp_commands.INT_KEYS / BOOL_KEYS).
Stockage : FireCameraConfig.ui_prefs (SQLite) — une ligne, pas de doublon de modèle.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any
from urllib.parse import quote

import requests
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from monitoring.services.fire_cam import esp_commands
from monitoring.services.fire_cam.esp_net import base_url, clean_host, resolve_esp_host
from monitoring.services.fire_cam.store import get_fire_config

logger = logging.getLogger(__name__)

# Image 1×1 PNG gris (aperçu si ESP hors ligne — évite img cassée).
_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _host_from_request_data(data: dict[str, Any], cfg) -> str:
    raw = data.get("esp_ip")
    if isinstance(raw, str) and raw.strip():
        h = clean_host(raw.strip())
        if h:
            return h
    return resolve_esp_host(cfg)


def _host_for_capture_proxy_get(request) -> str:
    raw = request.GET.get("esp_ip") or request.GET.get("host")
    if isinstance(raw, str) and raw.strip():
        h = clean_host(raw.strip())
        if h:
            return h
    cfg = get_fire_config()
    return resolve_esp_host(cfg)


def camera_capture_proxy_response(request) -> HttpResponse:
    """
    GET /camera/capture-proxy/ — JPEG depuis l’ESP ou PNG placeholder (200, pas de crash).
    """
    host = _host_for_capture_proxy_get(request)
    if not host:
        resp = HttpResponse(_PLACEHOLDER_PNG, content_type="image/png")
        resp["X-ESP-Reachable"] = "0"
        resp["X-Camera-Error"] = "no_host"
        return resp
    bu = base_url(host)
    if not bu:
        r = HttpResponse(_PLACEHOLDER_PNG, content_type="image/png")
        r["X-ESP-Reachable"] = "0"
        return r
    try:
        up = requests.get(f"{bu}/capture", timeout=(2, 5))
        if not up.ok:
            logger.warning("capture proxy HTTP %s pour %s", up.status_code, host)
            resp = HttpResponse(_PLACEHOLDER_PNG, content_type="image/png")
            resp["X-ESP-Reachable"] = "0"
            resp["X-Camera-Error"] = f"http_{up.status_code}"
            return resp
        ct = up.headers.get("Content-Type") or "image/jpeg"
        resp = HttpResponse(up.content, content_type=ct)
        resp["X-ESP-Reachable"] = "1"
        return resp
    except requests.RequestException as exc:
        logger.warning("capture proxy échec %s : %s", host, exc)
        resp = HttpResponse(_PLACEHOLDER_PNG, content_type="image/png")
        resp["X-ESP-Reachable"] = "0"
        resp["X-Camera-Error"] = type(exc).__name__
        return resp


@require_GET
def camera_capture_proxy(request):
    return camera_capture_proxy_response(request)


@require_POST
def camera_settings_update(request):
    """
    POST /camera/settings/update/
    Body JSON: { "field": "brightness", "value": 2, "esp_ip": "192.168.0.148" (optionnel) }
    """
    try:
        data = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return JsonResponse(
            {"db_ok": False, "esp_ok": False, "message": "JSON invalide.", "capture_url": ""},
            status=400,
        )

    field = (data.get("field") or data.get("var") or "").strip()
    if field not in esp_commands.BOOL_KEYS | esp_commands.INT_KEYS:
        return JsonResponse(
            {
                "db_ok": False,
                "esp_ok": False,
                "message": f"Champ inconnu : {field!r}",
                "capture_url": "",
            },
            status=400,
        )

    val = data.get("value", data.get("val"))
    cfg = get_fire_config()
    merged = esp_commands.merge_prefs(cfg.ui_prefs)
    merged[field] = esp_commands.normalize_pref(field, val)

    cfg.ui_prefs = merged
    update_fields: list[str] = ["ui_prefs", "updated_at"]
    raw_ip = data.get("esp_ip")
    if isinstance(raw_ip, str) and raw_ip.strip():
        h_ip = clean_host(raw_ip.strip())
        if h_ip:
            cfg.esp_ip = h_ip
            update_fields.append("esp_ip")

    try:
        cfg.save(update_fields=update_fields)
    except Exception as exc:
        logger.exception("camera_settings_update save DB")
        return JsonResponse(
            {
                "db_ok": False,
                "esp_ok": False,
                "message": f"Erreur base : {exc}",
                "capture_url": "",
            },
            status=500,
        )

    host = _host_from_request_data(data, cfg)
    esp_ok = False
    if host:
        esp_ok = esp_commands.send_cmd(host, field, merged[field])
    else:
        logger.info("camera_settings_update sans host ESP — DB OK seulement (%s)", field)

    ts = int(time.time() * 1000)
    cap_path = reverse("monitoring:camera_capture_proxy") + f"?t={ts}"
    if isinstance(raw_ip, str) and raw_ip.strip() and clean_host(raw_ip.strip()):
        cap_path += f"&esp_ip={quote(raw_ip.strip(), safe='')}"

    msg: str
    if not host:
        msg = "Base OK — pas d’IP ESP : enregistrement local uniquement."
    elif esp_ok:
        msg = "Base OK + ESP OK (/control ou /cmd)."
    else:
        msg = "Base OK — ESP refuse /control et /cmd (404 ?). Vérifie firmware / IP."

    logger.info(
        "camera_settings_update field=%s db_ok=True esp_ok=%s host=%s",
        field,
        esp_ok,
        host or "-",
    )

    return JsonResponse(
        {
            "db_ok": True,
            "esp_ok": esp_ok,
            "message": msg,
            "capture_url": cap_path,
            "field": field,
            "prefs": merged,
            "esp_ip_db": (cfg.esp_ip or "").strip(),
        }
    )


@require_POST
def camera_settings_sync_full(request):
    """
    POST /camera/settings/sync-full/
    Body JSON optionnel : { "prefs": { ... }, "esp_ip": "..." }
    Sinon ré-applique les prefs déjà en base vers l’ESP (ordre firmware).
    """
    try:
        data = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "message": "JSON invalide"}, status=400)

    cfg = get_fire_config()
    if isinstance(data.get("prefs"), dict):
        merged = esp_commands.merge_prefs(data["prefs"])
        cfg.ui_prefs = merged
    else:
        merged = esp_commands.merge_prefs(cfg.ui_prefs)

    if isinstance(data.get("esp_ip"), str) and data["esp_ip"].strip():
        h = clean_host(data["esp_ip"].strip())
        if h:
            cfg.esp_ip = h

    cfg.save(update_fields=["esp_ip", "ui_prefs", "updated_at"])

    host = resolve_esp_host(cfg)
    if host:
        esp_commands.apply_prefs_to_esp(host, merged)

    ts = int(time.time() * 1000)
    cap = reverse("monitoring:camera_capture_proxy") + f"?t={ts}"

    return JsonResponse(
        {
            "ok": True,
            "db_ok": True,
            "esp_applied": bool(host),
            "message": "Synchronisation complète envoyée." if host else "Base sauvegardée — IP ESP manquante.",
            "capture_url": cap,
            "prefs": merged,
            "esp_ip_db": (cfg.esp_ip or "").strip(),
        }
    )
