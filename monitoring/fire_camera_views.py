"""Vues Django — caméra ESP32 feu/fumée (sans Streamlit)."""

from __future__ import annotations

import base64
import logging
import json
import os
import time
from pathlib import Path
from urllib.parse import urlencode
from typing import Any

import cv2
from django.conf import settings
from django.contrib import messages
from django.core.cache import cache
import requests
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from monitoring.services.fire_cam.capture import fetch_jpeg_capture
from monitoring.services.fire_cam.esp_net import (
    base_url,
    clean_host,
    get_default_ip,
    resolve_esp_host,
    save_ip,
    stream_url_candidates,
    stream_url_exact,
    test_port_80,
    test_port_81_stream,
)

logger = logging.getLogger(__name__)
from monitoring.services.fire_cam.fire_twilio_tracker import (
    draw_fire_alert_overlay,
    is_fire_detection,
    reset_fire_tracker_state,
    step_fire_twilio,
)
from monitoring.services.fire_cam import esp_commands
from monitoring.services.fire_cam.store import get_fire_config
from monitoring.services.fire_cam.twilio_voice_alert import (
    place_twilio_alert_call,
    twilio_env_configured,
    twilio_env_report,
)

_yolo_inst: Any = None
_yolo_path: str = ""


def _session_cache_key(request) -> str:
    if not request.session.session_key:
        request.session.create()
    return f"fire_cam_twilio:{request.session.session_key}"


def _xai_payload(
    yolo: Any,
    dets: list[tuple[int, float, tuple[int, int, int, int]]],
) -> dict[str, Any]:
    """Résumé XAI (liste des boîtes + texte) pour l’UI « Analyse ONNX »."""
    rows: list[dict[str, Any]] = []
    fire_n = 0
    names = getattr(yolo, "class_names", []) or []
    for cls_id, score, box in dets:
        x1, y1, x2, y2 = box
        label = names[cls_id] if 0 <= cls_id < len(names) else f"class_{cls_id}"
        is_f = is_fire_detection(cls_id, list(names))
        if is_f:
            fire_n += 1
        rows.append(
            {
                "class_name": label,
                "confidence": round(float(score), 4),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "is_fire_related": is_f,
            }
        )
    if not rows:
        summary = (
            "Aucune détection au-dessus du seuil de confiance — inférence YOLO ONNX "
            "(entrée letterbox 640×640, sortie NMS)."
        )
    else:
        summary = f"{len(rows)} objet(s) détecté(s). "
        if fire_n > 0:
            summary += (
                f"{fire_n} correspond à la règle feu/fumée (noms de classes ou TWILIO_FIRE_CLASS_IDS). "
                "Les cadres verts sur l’image reprennent ces scores."
            )
        else:
            summary += (
                "Aucune classe interprétée comme feu/fumée sur cette frame — seuil ou scène sans risque."
            )
    return {
        "detections": rows,
        "summary_fr": summary,
        "letterbox": 640,
    }


def _get_yolo() -> tuple[Any, str | None]:
    global _yolo_inst, _yolo_path
    try:
        from monitoring.services.fire_cam.onnx_yolo import YoloOnnx
    except ModuleNotFoundError as exc:
        return None, f"Installez onnxruntime (pip install onnxruntime). Détail: {exc}"

    path = Path(str(settings.FIRE_ONNX_MODEL_PATH))
    if not path.is_file():
        return None, str(path)
    sp = str(path.resolve())
    if _yolo_inst is None or _yolo_path != sp:
        _yolo_inst = YoloOnnx(path)
        _yolo_path = sp
    return _yolo_inst, None


@ensure_csrf_cookie
def fire_camera_settings(request):
    cfg = get_fire_config()
    prefs = esp_commands.merge_prefs(cfg.ui_prefs)

    if request.method == "POST":
        act = (request.POST.get("action") or "").strip()
        esp_raw = (request.POST.get("esp_ip") or "").strip()
        if act == "save_road_live":
            cfg = get_fire_config()
            reg_models = getattr(settings, "ROAD_VISION_MODELS", {}) or {}
            rmk = (request.POST.get("road_live_model_key") or "").strip()
            if not rmk:
                cfg.road_live_model_key = ""
            elif rmk in reg_models:
                cfg.road_live_model_key = rmk
            else:
                messages.warning(
                    request,
                    f"Modèle road live « {rmk} » inconnu dans ROAD_VISION_MODELS.",
                )
            rcr = (request.POST.get("road_live_conf") or "").strip()
            if not rcr:
                cfg.road_live_conf = None
            else:
                try:
                    cfg.road_live_conf = max(
                        0.01, min(1.0, float(rcr.replace(",", ".")))
                    )
                except (TypeError, ValueError):
                    messages.warning(request, "Seuil confidence live invalide (laisser vide pour défaut projet).")
            cfg.save(
                update_fields=["road_live_model_key", "road_live_conf", "updated_at"]
            )
            messages.success(
                request,
                "Réglages YOLO drone / analyse live enregistrés (utilisés sur /reclamations/…/ai/?live=1 lorsque l’URL n’impose pas model/conf).",
            )
            return redirect("monitoring:camera_settings")
        if act == "test80":
            probe = esp_raw or resolve_esp_host(cfg)
            ok, msg = test_port_80(probe)
            (messages.success if ok else messages.error)(request, msg[:4000])
            if ok and esp_raw:
                save_ip(esp_raw)
        elif act == "test81":
            probe = esp_raw or resolve_esp_host(cfg)
            ok, msg = test_port_81_stream(probe)
            (messages.success if ok else messages.error)(request, msg[:4000])
            if ok and esp_raw:
                save_ip(esp_raw)
        elif act == "save_all":
            prefs_new = esp_commands.prefs_from_post(request.POST)
            cfg.esp_ip = clean_host(esp_raw)
            cfg.ui_prefs = prefs_new
            cfg.save(update_fields=["esp_ip", "ui_prefs", "updated_at"])
            host_apply = clean_host(cfg.esp_ip)
            if host_apply:
                esp_commands.apply_prefs_to_esp(host_apply, prefs_new)
                messages.success(
                    request,
                    "Réglages image enregistrés dans la base Django et appliqués sur la caméra ESP.",
                )
            else:
                messages.warning(
                    request,
                    "Préférences enregistrées en base. Indique une IP valide pour envoyer les commandes à l’ESP.",
                )
            return redirect("monitoring:camera_settings")
        return redirect("monitoring:camera_settings")

    host = resolve_esp_host(cfg)

    try:
        from monitoring import vision_analysis

        rv_models = vision_analysis.list_models_for_ui()
    except Exception:
        rv_models = []

    road_conf_fallback = float(
        getattr(
            settings,
            "RECLAMATION_VISION_CONF_DEFAULT",
            getattr(settings, "ROAD_DAMAGE_YOLO_CONF", 0.25),
        )
    )

    ctx = {
        "esp_ip": cfg.esp_ip or get_default_ip(),
        "prefs": prefs,
        "stream_url": stream_url_exact(host) if host else "",
        "control_url": f"{base_url(host)}/" if host else "",
        "capture_url": f"{base_url(host)}/capture" if host else "",
        "has_esp_host": bool(host),
        "fs_options": esp_commands.FS_OPTIONS,
        "wb_options": esp_commands.WB_OPTIONS,
        "fx_options": esp_commands.FX_OPTIONS,
        "gain_labels": esp_commands.GAIN_LABELS,
        "patch_url": reverse("monitoring:fire_camera_settings_patch"),
        "camera_update_url": reverse("monitoring:camera_settings_update"),
        "camera_sync_full_url": reverse("monitoring:camera_settings_sync_full"),
        "esp_capture_proxy_url": reverse("monitoring:esp_capture_proxy"),
        "camera_capture_proxy_url": reverse("monitoring:camera_capture_proxy"),
        "esp_stream_proxy_url": reverse("monitoring:esp_stream_proxy"),
        "fire_camera_state_url": reverse("monitoring:fire_camera_settings_state"),
        "road_vision_models": rv_models,
        "road_live_model_key": getattr(cfg, "road_live_model_key", "") or "",
        "road_live_conf": getattr(cfg, "road_live_conf", None),
        "road_conf_fallback": road_conf_fallback,
    }
    return render(request, "monitoring/fire_camera_settings.html", ctx)


@require_POST
def fire_camera_settings_patch(request):
    """Mise à jour incrémentale (live) : une variable + sync ui_prefs + /cmd ESP."""
    try:
        data = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "json"}, status=400)
    var = (data.get("var") or "").strip()
    if var not in esp_commands.BOOL_KEYS | esp_commands.INT_KEYS:
        return JsonResponse({"ok": False, "error": "unknown_var"}, status=400)
    val = data.get("val")
    cfg = get_fire_config()
    merged = esp_commands.merge_prefs(cfg.ui_prefs)
    merged[var] = esp_commands.normalize_pref(var, val)
    cfg.ui_prefs = merged
    update_fields = ["ui_prefs", "updated_at"]
    # IP du champ : enregistrée en base à chaque réglage (pas besoin du bouton « Enregistrer tout »).
    raw_override = data.get("esp_ip")
    esp_ip_saved = False
    if isinstance(raw_override, str) and raw_override.strip():
        h_ip = clean_host(raw_override.strip())
        if h_ip:
            cfg.esp_ip = h_ip
            update_fields.append("esp_ip")
            esp_ip_saved = True
    cfg.save(update_fields=update_fields)
    # IP formulaire prioritaire pour /cmd juste après sauvegarde.
    override = ""
    if isinstance(raw_override, str) and raw_override.strip():
        override = clean_host(raw_override.strip())
    host = override or resolve_esp_host(cfg)
    cmd_ok = False
    if host:
        cmd_ok = esp_commands.send_cmd(host, var, merged[var])
    return JsonResponse(
        {
            "ok": True,
            "var": var,
            "esp_host": host,
            "cmd_ok": cmd_ok,
            "saved_prefs": True,
            "esp_ip_saved": esp_ip_saved,
            "reason": ""
            if host
            else "no_host",
            # Source de vérité SQLite pour réconcilier la UI (pas de sliders obsolètes).
            "prefs": merged,
            "esp_ip_db": (cfg.esp_ip or "").strip(),
        }
    )


def _host_for_esp_proxy(request) -> str:
    """IP depuis ?esp_ip= (formulaire) ou base FireCameraConfig."""
    raw = request.GET.get("esp_ip") or request.GET.get("host")
    if isinstance(raw, str) and raw.strip():
        h = clean_host(raw.strip())
        if h:
            return h
    cfg = get_fire_config()
    return resolve_esp_host(cfg)


@require_GET
def fire_camera_settings_state(request):
    """Lecture seule : IP + prefs fusionnées (SQLite)."""
    cfg = get_fire_config()
    merged = esp_commands.merge_prefs(cfg.ui_prefs)
    return JsonResponse(
        {
            "esp_ip": (cfg.esp_ip or "").strip(),
            "prefs": merged,
        }
    )


@require_GET
def esp_capture_proxy(request):
    """Proxy same-origin vers http://ESP/capture (évite CORS / mixed content côté navigateur)."""
    host = _host_for_esp_proxy(request)
    if not host:
        return HttpResponse(
            "Host ESP inconnu (saisir l’IP ou enregistrer en base).",
            status=404,
            content_type="text/plain; charset=utf-8",
        )
    bu = base_url(host)
    if not bu:
        return HttpResponse(status=404)
    try:
        r = requests.get(f"{bu}/capture", timeout=(2, 3))
        if not r.ok:
            return HttpResponse(status=502)
        ct = r.headers.get("Content-Type") or "image/jpeg"
        return HttpResponse(r.content, content_type=ct)
    except requests.RequestException:
        return HttpResponse(status=502)


@require_GET
def esp_stream_proxy(request):
    """Proxy MJPEG — essaie :81/stream puis /stream sur port 80 (firmwares Arduino)."""
    host = _host_for_esp_proxy(request)
    if not host:
        return HttpResponse(status=404)
    urls = stream_url_candidates(host)
    if not urls:
        return HttpResponse(status=404)
    last_exc: BaseException | None = None
    for url in urls:
        try:
            upstream = requests.get(url, stream=True, timeout=(10, 180))
            upstream.raise_for_status()
            ct = upstream.headers.get("Content-Type") or "multipart/x-mixed-replace; boundary=frame"

            def chunk_iter():
                try:
                    for chunk in upstream.iter_content(chunk_size=8192):
                        if chunk:
                            yield chunk
                finally:
                    upstream.close()

            resp = StreamingHttpResponse(chunk_iter(), content_type=ct)
            resp["Cache-Control"] = "no-store"
            resp["X-Accel-Buffering"] = "no"
            resp["X-ESP-Stream-URL"] = url
            return resp
        except requests.RequestException as exc:
            logger.warning("esp_stream_proxy %s → %s", url, exc)
            last_exc = exc
            continue
    logger.warning(
        "esp_stream_proxy tous les flux ont échoué pour %s (dernière erreur : %s)",
        host,
        last_exc,
    )
    return HttpResponse(status=502)


@ensure_csrf_cookie
def fire_camera_detect(request):
    cfg = get_fire_config()
    if request.method == "POST":
        esp_raw = (request.POST.get("esp_ip") or "").strip()
        new_ip = clean_host(esp_raw)
        if not new_ip:
            messages.error(
                request,
                "IP ESP32-CAM vide/invalide. Entrez une IP (ex: 192.168.0.148).",
            )
            return redirect("monitoring:fire_camera_detect")
        save_ip(new_ip)
        messages.success(request, f"IP mise à jour : {new_ip}")
        return redirect("monitoring:fire_camera_detect")
    host = resolve_esp_host(cfg)
    mp = Path(str(settings.FIRE_ONNX_MODEL_PATH))
    try:
        import onnxruntime  # noqa: F401

        onnx_ok = True
    except ImportError:
        onnx_ok = False
    model_ok = mp.is_file()
    analysis_ready = bool(host) and model_ok and onnx_ok
    stream_proxy_path = reverse("monitoring:esp_stream_proxy")
    stream_src = stream_proxy_path
    if host:
        stream_src = f"{stream_proxy_path}?{urlencode({'esp_ip': host, '_': str(int(time.time() * 1000))})}"
    ctx = {
        "stream_url": stream_url_exact(host) if host else "",
        "esp_stream_proxy_url": stream_proxy_path,
        "esp_stream_src": stream_src,
        "esp_host": host,
        "esp_ip": host,
        "has_ip": bool(host),
        "model_ok": model_ok,
        "onnx_ok": onnx_ok,
        "analysis_ready": analysis_ready,
        "model_path": str(mp),
        "analyze_url": reverse("monitoring:fire_camera_analyze"),
        "twilio_test_url": reverse("monitoring:fire_camera_twilio_test"),
        "twilio_ready": twilio_env_configured(),
        "twilio_rows": twilio_env_report(),
        "drone_host_set": bool((os.environ.get("DRONE_HOST") or "").strip()),
    }
    return render(request, "monitoring/fire_camera_detect.html", ctx)


@require_POST
def fire_camera_analyze(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        body = {}

    key = _session_cache_key(request)
    if body.get("reset"):
        st: dict = {}
        reset_fire_tracker_state(st)
        cache.set(key, st, 86400)
        return JsonResponse({"ok": True, "reset": True})

    state = cache.get(key)
    if not isinstance(state, dict):
        state = {}

    cfg = get_fire_config()
    host = resolve_esp_host(cfg)
    bu = base_url(host)
    if not bu:
        return JsonResponse(
            {
                "ok": False,
                "error": "no_ip",
                "message": "Aucune IP ESP : enregistrez-la dans Réglages caméra feu, ou définissez DRONE_HOST dans le fichier .env à la racine du projet.",
            },
            status=400,
        )

    yolo, err = _get_yolo()
    if yolo is None:
        detail = err or ""
        if "onnxruntime" in detail.lower():
            msg = "Paquet Python manquant : exécutez « pip install onnxruntime » dans votre environnement."
        else:
            msg = (
                "Fichier modèle ONNX introuvable. Copiez votre export « best.onnx » vers le chemin indiqué dans « path », "
                "ou définissez la variable FIRE_ONNX_MODEL dans .env avec le chemin absolu vers le fichier."
            )
        return JsonResponse(
            {"ok": False, "error": "no_model", "path": detail, "message": msg},
            status=500,
        )

    frame = fetch_jpeg_capture(bu)
    if frame is None:
        return JsonResponse({"ok": False, "error": "no_frame"})

    conf = float(body.get("conf", 0.65))
    iou = float(body.get("iou", 0.45))
    yolo.conf_thres = conf
    yolo.iou_thres = iou
    dets = yolo.predict_boxes(frame)
    out = yolo.draw_boxes(frame, dets)

    twilio_on = bool(body.get("twilio", False))
    alert_after = float(body.get("alert_after", 2.0))
    grace = float(body.get("grace", 0.8))

    info = step_fire_twilio(
        state,
        dets,
        yolo.class_names,
        twilio_enabled=twilio_on,
        alert_after_sec=alert_after,
        grace_sec=grace,
    )
    if info.get("show_alert"):
        draw_fire_alert_overlay(out)

    cache.set(key, state, 86400)

    ok_j, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok_j:
        return JsonResponse({"ok": False, "error": "encode"}, status=500)
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    xai = _xai_payload(yolo, dets)

    return JsonResponse(
        {
            "ok": True,
            "jpeg_b64": b64,
            "xai": xai,
            "status": info["status"],
            "show_alert": info["show_alert"],
            "has_fire": info["has_fire"],
            "risk_seconds": round(float(info.get("risk_seconds") or 0), 2),
            "twilio_called": info["twilio_called"],
            "twilio_outcome": info.get("twilio_outcome"),
            "twilio_detail": (info.get("twilio_outcome_detail") or "")[:500],
            "twilio_error": (info.get("twilio_error") or "")[:500],
            "twilio_popup": bool(info.get("twilio_popup")),
            "twilio_popup_l1": (info.get("twilio_popup_l1") or "")[:300],
            "twilio_popup_l2": (info.get("twilio_popup_l2") or "")[:500],
            "twilio_modal_kind": (info.get("twilio_modal_kind") or "")[:32],
        }
    )


@require_POST
def fire_camera_twilio_test(request):
    ok, msg = place_twilio_alert_call()
    msg_lower = (msg or "").lower()
    if ok:
        variant = "success"
        title = "Test Twilio réussi"
    elif not twilio_env_configured() or "non configuré" in msg_lower or "manquant" in msg_lower:
        variant = "warning"
        title = "Twilio non configuré"
    else:
        variant = "danger"
        title = "Test Twilio échoué"
    return JsonResponse(
        {
            "ok": ok,
            "message": msg,
            "modal_title": title,
            "modal_body": msg or "",
            "modal_variant": variant,
        }
    )
