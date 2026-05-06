"""API REST Traffic Nexus — cadre vidéo + analyse ROI (logique tdss)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import cv2
from django.conf import settings
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from monitoring.traffic_nexus_helpers import (
    bgr_to_jpeg_base64,
    extract_polygon_points,
    resize_frame_max_side,
    scale_polygon,
)
from monitoring.traffic_nexus_inference import (
    analyze_frame,
    analyze_from_candidates,
    decode_base64_image,
    decode_upload_first_frame,
    load_frame_from_youtube,
)
from monitoring.traffic_nexus_live import (
    close_session,
    create_session,
    get_precomputed_candidates,
    get_session,
    read_next_frame,
)
from monitoring.traffic_nexus_paths import resolve_trusted_media_file
from monitoring.services.traffic_report_generator_jury import build_report_bytes_with_fallback
from monitoring.traffic_tdss.esp32_bridge import esp32_decision_signature, post_json
from monitoring.traffic_tdss.video_processing import read_first_frame

logger = logging.getLogger(__name__)

# Doit couvrir les modes gérés par monitoring.traffic_tdss.vehicle_detection._render_and_count_candidates
_TRAFFIC_DISPLAY_MODES = frozenset(
    {"normal_detection", "speed_status", "minimal", "congestion_heatmap"}
)


def _json_err(message: str, status: int = 400, **extra):
    return JsonResponse({"success": False, "message": message, **extra}, status=status)


def _clamp_conf(v: float) -> float:
    return max(0.05, min(0.95, float(v)))


def _clamp_imgsz(v: int) -> int:
    return max(160, min(960, int(v)))


def _clamp_max_det(v: int) -> int:
    return max(5, min(200, int(v)))


def _rectangles_to_polygons(rects: list) -> list[list[tuple[int, int]]]:
    out: list[list[tuple[int, int]]] = []
    for r in rects:
        if not isinstance(r, (list, tuple)) or len(r) < 4:
            continue
        x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        out.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    return out


def _merge_roi(body: dict, frame_hw: tuple[int, int]) -> list[list[tuple[int, int]]]:
    fh, fw = frame_hw
    polygons_in: list = body.get("polygons") or []
    roi: list[list[tuple[int, int]]] = []
    for poly in polygons_in:
        if isinstance(poly, list) and len(poly) >= 3:
            roi.append([(int(p[0]), int(p[1])) for p in poly if isinstance(p, (list, tuple)) and len(p) >= 2])

    rects = body.get("rectangles") or []
    roi.extend(_rectangles_to_polygons(rects))

    canvas_json = body.get("canvas_json")
    cw = int(body.get("canvas_width") or 0)
    ch = int(body.get("canvas_height") or 0)
    if canvas_json and cw > 0 and ch > 0:
        extracted = extract_polygon_points(canvas_json if isinstance(canvas_json, dict) else {})
        for poly in extracted:
            roi.append(scale_polygon(poly, cw, ch, fw, fh))
    return roi


def _apply_esp32_bridge_http(
    result: dict,
    *,
    esp32_enabled: bool,
    esp32_url: str,
    esp32_timeout: float,
    bridge_in: dict | None,
) -> None:
    """
    POST JSON uniquement si décision confirmée change (signature), comme Streamlit.
    Réinitialise la signature si l’URL ESP32 change.
    """
    bi = bridge_in if isinstance(bridge_in, dict) else {}
    esp_block = result.get("esp32") or {}
    logs = list(esp_block.get("confirm_logs") or [])
    payload = esp_block.get("payload")
    confirm_state_out = esp_block.get("confirm_state")

    bridge_out: dict = {
        "confirm_state": confirm_state_out,
        "last_post_signature": bi.get("last_post_signature"),
        "stored_url": bi.get("stored_url"),
    }
    url = (esp32_url or "").strip()
    if bridge_out.get("stored_url") != url:
        bridge_out["last_post_signature"] = None
        bridge_out["stored_url"] = url if url else None

    if not esp32_enabled:
        logs.append("ESP32 skipped: disabled")
        result["esp32_post"] = {"ok": False, "skipped": True, "reason": "disabled"}
    elif not url:
        logs.append("ESP32 skipped: no URL")
        result["esp32_post"] = {"ok": False, "skipped": True, "reason": "no_url"}
    elif not payload:
        logs.append("ESP32 skipped: empty payload")
        result["esp32_post"] = {"ok": False, "skipped": True, "reason": "empty_payload"}
    else:
        sig = esp32_decision_signature(payload)
        last_sig = bridge_out.get("last_post_signature")
        if sig == last_sig:
            logs.append("ESP32 skipped: confirmed decision unchanged")
            result["esp32_post"] = {"ok": True, "skipped": True, "reason": "unchanged"}
        else:
            ok, detail = post_json(url, payload, timeout=min(30.0, max(0.5, esp32_timeout)))
            if ok:
                bridge_out["last_post_signature"] = sig
                logs.append("ESP32 sent: confirmed decision changed")
                result["esp32_post"] = {"ok": True, "skipped": False, "detail": (detail or "")[:500]}
            else:
                logs.append(f"ESP32 POST failed: {(detail or '')[:160]}")
                result["esp32_post"] = {"ok": False, "skipped": False, "detail": (detail or "")[:500]}

    result["esp32_logs"] = logs
    result["esp32_bridge_state"] = bridge_out


def _load_local_frame(path: str):
    img = cv2.imread(path)
    if img is not None:
        return img
    return read_first_frame(path)


@csrf_exempt
@require_http_methods(["POST"])
def traffic_first_frame(request):
    """YouTube URL (JSON), fichier upload, ou chemin local → image JPEG redimensionnée (base64)."""
    max_side = int(getattr(settings, "TRAFFIC_NEXUS_FRAME_MAX_SIDE", 1280) or 1280)
    frame = None

    if request.FILES.get("file"):
        up = request.FILES["file"]
        raw = up.read()
        orig_name = getattr(up, "name", "") or ""
        frame = decode_upload_first_frame(raw, original_filename=orig_name)
    else:
        try:
            body = json.loads(request.body.decode() or "{}")
        except json.JSONDecodeError:
            return _json_err("JSON invalide")
        url = (body.get("youtube_url") or body.get("url") or "").strip()
        local_path = (body.get("local_file_path") or body.get("local_path") or "").strip()
        if local_path:
            trusted = resolve_trusted_media_file(local_path)
            if not trusted:
                return _json_err("Chemin local refusé ou introuvable.", status=422)
            frame = _load_local_frame(trusted)
        elif url:
            frame = load_frame_from_youtube(url)

    if frame is None:
        return _json_err("Impossible de lire une image : fichier, youtube_url ou local_file_path.", status=422)

    frame = resize_frame_max_side(frame, max_side=max_side)
    h, w = frame.shape[:2]
    b64 = bgr_to_jpeg_base64(frame)
    if not b64:
        return _json_err("Encodage image échoué", status=500)

    return JsonResponse(
        {
            "success": True,
            "image_base64": b64,
            "width": w,
            "height": h,
            "max_side": max_side,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def traffic_analyze(request):
    """Image base64 + polygones / rectangles + options → détection, décisions, XAI, agents."""
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return _json_err("JSON invalide")

    b64 = (body.get("image_base64") or "").strip()
    frame = decode_base64_image(b64) if b64 else None
    if frame is None:
        return _json_err("image_base64 manquant ou invalide.", status=422)

    fh, fw = frame.shape[:2]
    roi = _merge_roi(body, (fh, fw))

    mode = (body.get("mode") or "vehicle").strip()
    model_key = (body.get("model") or "yolov8n").strip()
    model_mode = (body.get("model_mode") or "single_model").strip().lower()
    dual = model_mode == "dual_model_fusion" or bool(body.get("dual_models"))
    display_mode = (body.get("display_mode") or "normal_detection").strip().lower()
    if display_mode not in _TRAFFIC_DISPLAY_MODES:
        display_mode = "normal_detection"
    try:
        conf = _clamp_conf(float(body.get("conf", 0.25)))
    except (TypeError, ValueError):
        conf = 0.25
    try:
        imgsz = _clamp_imgsz(int(body.get("imgsz", 640)))
    except (TypeError, ValueError):
        imgsz = 640
    try:
        max_det = _clamp_max_det(int(body.get("max_det", 80)))
    except (TypeError, ValueError):
        max_det = 80
    mock_emergency = bool(body.get("mock_emergency"))
    draw_confidence = True
    show_speed_overlays = bool(body.get("show_speed_overlays", True))
    if display_mode != "speed_status":
        show_speed_overlays = False
    try:
        overlay_min_conf = float(body.get("overlay_min_conf", 0.35))
    except (TypeError, ValueError):
        overlay_min_conf = 0.35
    try:
        overlay_min_area = int(body.get("overlay_min_area", 180))
    except (TypeError, ValueError):
        overlay_min_area = 180
    try:
        esp32_sim_zone = int(body.get("esp32_sim_zone") or body.get("priority_zone_sim") or 0)
    except (TypeError, ValueError):
        esp32_sim_zone = 0
    vehicle_labels = (body.get("vehicle_labels") or "").strip()
    custom_model_path = (body.get("custom_model_path") or "").strip()

    bridge_in = body.get("esp32_bridge_state")
    if not isinstance(bridge_in, dict):
        bridge_in = {}
    esp32_confirm_in = bridge_in.get("confirm_state")
    esp32_enabled = bool(body.get("esp32_enabled"))
    display_state_in = body.get("display_state")
    if not isinstance(display_state_in, dict):
        display_state_in = {}

    paths = getattr(settings, "TRAFFIC_NEXUS_MODEL_PATHS", {})
    if model_key not in paths and model_key != "yolov8n":
        model_key = "yolov8n"

    vehicle_mode = mode.lower() in ("vehicle", "vehicles", "counting")
    if vehicle_mode and dual:
        pass
    elif not custom_model_path and not paths.get(model_key):
        return _json_err(
            f"Modèle « {model_key} » introuvable. Vérifiez traffic_nexus/traffic_nexus/ ou settings.",
            status=422,
            extra={"error": "model_missing"},
        )

    try:
        result = analyze_frame(
            frame,
            roi,
            mode=mode,
            model_key=model_key,
            dual_models=dual,
            conf=conf,
            imgsz=imgsz,
            max_det=max_det,
            draw_confidence=draw_confidence,
            min_confidence=max(0.0, min(1.0, overlay_min_conf)),
            min_box_area=max(1, overlay_min_area),
            display_mode=display_mode,
            show_speed_overlays=show_speed_overlays,
            display_state=display_state_in,
            mock_emergency=mock_emergency,
            vehicle_labels=vehicle_labels or None,
            custom_model_path=custom_model_path or None,
            esp32_sim_zone=esp32_sim_zone,
            esp32_confirm_state=esp32_confirm_in if isinstance(esp32_confirm_in, dict) else None,
            now_s=time.time(),
        )
    except Exception as exc:
        logger.exception("traffic analyze failed")
        return _json_err(str(exc), status=500, extra={"error": "inference"})

    if not result.get("success"):
        return JsonResponse(result, status=422)

    esp32_url = (body.get("esp32_url") or "").strip()
    try:
        esp32_timeout = float(body.get("esp32_timeout", 2.0))
    except (TypeError, ValueError):
        esp32_timeout = 2.0
    _apply_esp32_bridge_http(
        result,
        esp32_enabled=esp32_enabled,
        esp32_url=esp32_url,
        esp32_timeout=esp32_timeout,
        bridge_in=bridge_in,
    )
    result["api_key_configured"] = bool((os.getenv("GROQ_API_KEY") or "").strip())
    result["display_mode"] = display_mode
    result["model_mode"] = "dual_model_fusion" if dual else "single_model"

    return JsonResponse(result)


def _parse_live_meta(request) -> dict:
    if request.FILES.get("file"):
        raw = request.POST.get("meta") or request.POST.get("payload") or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    try:
        return json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return {}


@csrf_exempt
@require_http_methods(["POST"])
def traffic_live_init(request):
    """Démarre une session vidéo (YouTube, chemin local, ou fichier upload)."""
    meta = _parse_live_meta(request)
    temp_upload = None
    if request.FILES.get("file"):
        import tempfile

        f = request.FILES["file"]
        suf = Path(f.name).suffix or ".mp4"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
        for chunk in f.chunks():
            tmp.write(chunk)
        tmp.close()
        temp_upload = tmp.name

    youtube_url = (meta.get("youtube_url") or meta.get("url") or "").strip()
    local_file_path = (meta.get("local_file_path") or meta.get("local_path") or "").strip()
    processing_mode = (meta.get("processing_mode") or "live").strip()
    mode = (meta.get("mode") or "vehicle").strip()
    model_key = (meta.get("model") or "yolov8n").strip()
    dual = bool(meta.get("dual_models"))
    try:
        conf = _clamp_conf(float(meta.get("conf", 0.25)))
    except (TypeError, ValueError):
        conf = 0.25
    try:
        imgsz = _clamp_imgsz(int(meta.get("imgsz", 640)))
    except (TypeError, ValueError):
        imgsz = 640
    try:
        max_det = _clamp_max_det(int(meta.get("max_det", 80)))
    except (TypeError, ValueError):
        max_det = 80
    try:
        frame_skip = max(1, min(5, int(meta.get("frame_skip", 1))))
    except (TypeError, ValueError):
        frame_skip = 1
    vehicle_labels = (meta.get("vehicle_labels") or "").strip()
    custom_model_path = (meta.get("custom_model_path") or "").strip()
    turbo_local = bool(meta.get("turbo_local"))
    if turbo_local:
        imgsz = min(imgsz, 416)
        max_det = min(max_det, 48)

    sid, err, session_meta, first_frame = create_session(
        youtube_url=youtube_url,
        local_file_path=local_file_path,
        temp_upload_path=temp_upload,
        processing_mode=processing_mode,
        mode=mode,
        model_key=model_key,
        dual_models=dual,
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        frame_skip=frame_skip,
        vehicle_labels=vehicle_labels,
        custom_model_path=custom_model_path,
        turbo_local=turbo_local,
    )
    if err or not sid or not session_meta or first_frame is None:
        return _json_err(err or "Session live impossible.", status=422)

    max_side = int(getattr(settings, "TRAFFIC_NEXUS_FRAME_MAX_SIDE", 1280) or 1280)
    frame = first_frame
    frame = resize_frame_max_side(frame, max_side=max_side)
    h, w = frame.shape[:2]
    b64 = bgr_to_jpeg_base64(frame)
    if not b64:
        close_session(sid)
        return _json_err("Encodage image échoué", status=500)

    return JsonResponse(
        {
            "success": True,
            "session_id": sid,
            "image_base64": b64,
            "width": w,
            "height": h,
            "max_side": max_side,
            **session_meta,
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def traffic_live_tick(request):
    """Lit la frame suivante et exécute l’analyse (YOLO ou cache optimisé).

    Client peut envoyer ``turbo_local`` pour lecture fichier/upload plus légère (imgsz/max_det/overlays).
    Le throttling « Dashboard refresh (frames) » est uniquement côté navigateur (pas ce endpoint).
    """
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return _json_err("JSON invalide")

    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        return _json_err("session_id requis.", status=422)

    try:
        frame_skip = max(1, min(5, int(body.get("frame_skip", 1))))
    except (TypeError, ValueError):
        frame_skip = 1

    turbo_local = bool(body.get("turbo_local"))

    frame, eof, frame_idx = read_next_frame(session_id, frame_skip=frame_skip)
    if eof or frame is None:
        close_session(session_id)
        return JsonResponse({"success": False, "ended": True, "message": "Fin de vidéo ou session invalide."})

    max_side = int(getattr(settings, "TRAFFIC_NEXUS_FRAME_MAX_SIDE", 1280) or 1280)
    frame = resize_frame_max_side(frame, max_side=max_side)
    fh, fw = frame.shape[:2]
    roi = _merge_roi(body, (fh, fw))

    mode = (body.get("mode") or "vehicle").strip()
    model_key = (body.get("model") or "yolov8n").strip()
    model_mode = (body.get("model_mode") or "single_model").strip().lower()
    dual = model_mode == "dual_model_fusion" or bool(body.get("dual_models"))
    display_mode = (body.get("display_mode") or "normal_detection").strip().lower()
    if display_mode not in _TRAFFIC_DISPLAY_MODES:
        display_mode = "normal_detection"
    try:
        conf = _clamp_conf(float(body.get("conf", 0.25)))
    except (TypeError, ValueError):
        conf = 0.25
    try:
        imgsz = _clamp_imgsz(int(body.get("imgsz", 640)))
    except (TypeError, ValueError):
        imgsz = 640
    try:
        max_det = _clamp_max_det(int(body.get("max_det", 80)))
    except (TypeError, ValueError):
        max_det = 80
    # Turbo local playback: lower imgsz / max_det ceiling per tick (UI sends overrides; this guards the API).
    if turbo_local:
        imgsz = min(imgsz, 416)
        max_det = min(max_det, 48)
    mock_emergency = bool(body.get("mock_emergency"))
    draw_confidence = True
    show_speed_overlays = bool(body.get("show_speed_overlays", True))
    if display_mode != "speed_status":
        show_speed_overlays = False
    try:
        overlay_min_conf = float(body.get("overlay_min_conf", 0.35))
    except (TypeError, ValueError):
        overlay_min_conf = 0.35
    try:
        overlay_min_area = int(body.get("overlay_min_area", 180))
    except (TypeError, ValueError):
        overlay_min_area = 180
    try:
        esp32_sim_zone = int(body.get("esp32_sim_zone") or 0)
    except (TypeError, ValueError):
        esp32_sim_zone = 0
    vehicle_labels = (body.get("vehicle_labels") or "").strip()
    custom_model_path = (body.get("custom_model_path") or "").strip()

    bridge_in = body.get("esp32_bridge_state")
    if not isinstance(bridge_in, dict):
        bridge_in = {}
    esp32_confirm_in = bridge_in.get("confirm_state")
    esp32_enabled = bool(body.get("esp32_enabled"))
    display_state_in = body.get("display_state")
    if not isinstance(display_state_in, dict):
        display_state_in = {}
    _confirm_kw = {
        "esp32_confirm_state": esp32_confirm_in if isinstance(esp32_confirm_in, dict) else None,
        "now_s": time.time(),
    }

    sess = get_session(session_id)
    use_cache = bool(sess and sess.optimized and sess.precomputed)

    try:
        if use_cache:
            cands = get_precomputed_candidates(session_id, frame_idx)
            if cands is not None:
                result = analyze_from_candidates(
                    frame,
                    cands,
                    roi,
                    mock_emergency=mock_emergency,
                    esp32_sim_zone=esp32_sim_zone,
                    draw_confidence=draw_confidence,
                    min_confidence=max(0.0, min(1.0, overlay_min_conf)),
                    min_box_area=max(1, overlay_min_area),
                    display_mode=display_mode,
                    show_speed_overlays=show_speed_overlays,
                    display_state=display_state_in,
                    **_confirm_kw,
                )
            else:
                result = analyze_frame(
                    frame,
                    roi,
                    mode=mode,
                    model_key=model_key,
                    dual_models=dual,
                    conf=conf,
                    imgsz=imgsz,
                    max_det=max_det,
                    draw_confidence=draw_confidence,
                    min_confidence=max(0.0, min(1.0, overlay_min_conf)),
                    min_box_area=max(1, overlay_min_area),
                    display_mode=display_mode,
                    show_speed_overlays=show_speed_overlays,
                    display_state=display_state_in,
                    mock_emergency=mock_emergency,
                    vehicle_labels=vehicle_labels or None,
                    custom_model_path=custom_model_path or None,
                    esp32_sim_zone=esp32_sim_zone,
                    **_confirm_kw,
                )
        else:
            result = analyze_frame(
                frame,
                roi,
                mode=mode,
                model_key=model_key,
                dual_models=dual,
                conf=conf,
                imgsz=imgsz,
                max_det=max_det,
                mock_emergency=mock_emergency,
                vehicle_labels=vehicle_labels or None,
                custom_model_path=custom_model_path or None,
                esp32_sim_zone=esp32_sim_zone,
                draw_confidence=draw_confidence,
                min_confidence=max(0.0, min(1.0, overlay_min_conf)),
                min_box_area=max(1, overlay_min_area),
                display_mode=display_mode,
                show_speed_overlays=show_speed_overlays,
                display_state=display_state_in,
                **_confirm_kw,
            )
    except Exception as exc:
        logger.exception("traffic live tick failed")
        return _json_err(str(exc), status=500, extra={"error": "inference"})

    if not result.get("success"):
        return JsonResponse({**result, "frame_index": frame_idx}, status=422)

    esp32_url = (body.get("esp32_url") or "").strip()
    try:
        esp32_timeout = float(body.get("esp32_timeout", 2.0))
    except (TypeError, ValueError):
        esp32_timeout = 2.0
    _apply_esp32_bridge_http(
        result,
        esp32_enabled=esp32_enabled,
        esp32_url=esp32_url,
        esp32_timeout=esp32_timeout,
        bridge_in=bridge_in,
    )
    result["api_key_configured"] = bool((os.getenv("GROQ_API_KEY") or "").strip())
    result["display_mode"] = display_mode
    result["model_mode"] = "dual_model_fusion" if dual else "single_model"

    out = {**result, "frame_index": frame_idx, "ended": False}
    return JsonResponse(out)


@csrf_exempt
@require_http_methods(["POST"])
def traffic_live_close(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return _json_err("JSON invalide")
    sid = (body.get("session_id") or "").strip()
    if sid:
        close_session(sid)
    return JsonResponse({"success": True})


@require_http_methods(["GET"])
def traffic_models(request):
    paths = getattr(settings, "TRAFFIC_NEXUS_MODEL_PATHS", {})
    avail = {k: bool(v) for k, v in paths.items()}
    return JsonResponse({"success": True, "models": avail})


@csrf_exempt
@require_http_methods(["POST"])
def traffic_report_pdf(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return _json_err("JSON invalide")

    try:
        data = body if isinstance(body, dict) else {}
        data["api_key_configured"] = bool((os.getenv("GROQ_API_KEY") or "").strip())
        pdf_bytes = build_report_bytes_with_fallback(data, debug_layout=bool(data.get("debug_layout")))
    except ModuleNotFoundError:
        return _json_err("PDF indisponible (reportlab manquant).", status=501)
    except Exception as exc:
        logger.exception("traffic report pdf failed")
        return _json_err(f"Génération PDF échouée: {exc}", status=500)

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = 'attachment; filename="traffic_nexus_report.pdf"'
    return resp
