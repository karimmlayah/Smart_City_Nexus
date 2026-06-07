"""API Waste Street Detection — /api/waste/*."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path

import json
import logging
import os
import uuid
from pathlib import Path

import requests
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from monitoring.models import WasteReport
from monitoring.runtime import ML_UNAVAILABLE_MESSAGE, ml_deps_available
from monitoring.services.waste_email import send_waste_alert_email
from monitoring.services.waste_report_service import (
    compute_statistics,
    format_location_line,
    list_reports_for_api,
)
from monitoring.services.waste_severity import compute_severity_and_action
from monitoring.services.waste_sms import send_waste_alert_sms

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024


def _json_err(message: str, status: int = 400, **extra):
    return JsonResponse({"success": False, "message": message, **extra}, status=status)


def _location_hint(city: str = "", address: str = "") -> str:
    return " ".join(x.strip() for x in [city or "", address or ""] if x and x.strip())


def _parse_detect_payload(request):
    ct = (request.META.get("CONTENT_TYPE") or "").split(";")[0].strip().lower()
    city = ""
    address = ""
    if ct == "application/json":
        try:
            body = json.loads(request.body.decode() or "{}")
        except json.JSONDecodeError:
            return None, "invalid_json"
        url = (body.get("image_url") or "").strip()
        city = (body.get("city") or "").strip()
        address = (body.get("address") or "").strip()
        if not url:
            return None, "missing_image_url"
        return {"image_url": url, "city": city, "address": address}, None
    f = request.FILES.get("image")
    city = (request.POST.get("city") or "").strip()
    address = (request.POST.get("address") or "").strip()
    if f:
        return {"file": f, "city": city, "address": address}, None
    url = (request.POST.get("image_url") or "").strip()
    if url:
        return {"image_url": url, "city": city, "address": address}, None
    return None, "missing_image"


def _bgr_from_upload(fobj):
    import cv2
    import numpy as np

    raw = fobj.read()
    if not raw:
        return None, "empty_upload"
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return (img, None) if img is not None else (None, "decode_failed")


def _bgr_from_url(url: str):
    import cv2
    import numpy as np
    if not url.startswith(("http://", "https://")):
        return None, "invalid_url_scheme"
    try:
        r = requests.get(url, timeout=25, stream=True)
        r.raise_for_status()
        total = 0
        parts: list[bytes] = []
        for chunk in r.iter_content(65536):
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                return None, "image_too_large"
            parts.append(chunk)
        data = b"".join(parts)
        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return (img, None) if img is not None else (None, "decode_failed")
    except Exception as exc:
        logger.warning("waste image url fetch failed: %s", exc)
        return None, f"download_failed:{exc}"


def _save_upload_copy(frame_bgr: np.ndarray) -> str | None:
    try:
        d = Path(settings.MEDIA_ROOT) / "waste_uploads"
        d.mkdir(parents=True, exist_ok=True)
        fp = d / f"{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(fp), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        rel = fp.relative_to(Path(settings.MEDIA_ROOT))
        return f"{settings.MEDIA_URL.rstrip('/')}/{rel.as_posix().replace(chr(92), '/')}"
    except Exception as exc:
        logger.warning("waste upload copy failed: %s", exc)
        return None


@csrf_exempt
@require_http_methods(["POST"])
def waste_detect(request):
    if not ml_deps_available():
        return _json_err(ML_UNAVAILABLE_MESSAGE, status=503)
    from monitoring.services.waste_detection import detect_waste_bgr

    spec, err = _parse_detect_payload(request)
    if spec is None:
        return _json_err(err or "bad_request", extra={"error": err})

    hint = _location_hint(spec.get("city") or "", spec.get("address") or "")
    frame = None
    source_image_url = None
    err_code = None

    if "file" in spec:
        frame, err_code = _bgr_from_upload(spec["file"])
        if frame is not None:
            source_image_url = _save_upload_copy(frame)
    else:
        url = spec["image_url"]
        frame, err_code = _bgr_from_url(url)
        source_image_url = url

    if frame is None:
        return _json_err(
            "Error while processing image",
            status=422,
            extra={"error": err_code or "load_failed"},
        )

    result = detect_waste_bgr(frame, location_hint=hint)
    out = {
        "success": bool(result.get("success")),
        "detections": result.get("detections") or [],
        "count": int(result.get("count") or 0),
        "annotated_image_url": result.get("annotated_image_url"),
        "max_confidence": float(result.get("max_confidence") or 0),
        "severity": result.get("severity") or "Low",
        "recommended_action": result.get("recommended_action") or "",
        "sensitive_area_boost": bool(result.get("sensitive_area_boost")),
        "source_image_url": source_image_url,
    }
    if result.get("error"):
        out["error"] = result["error"]
        if result.get("error_detail"):
            out["error_detail"] = result["error_detail"]
        if result.get("model_path"):
            out["model_path"] = result["model_path"]

    return JsonResponse(out)


def _confidence_pct(raw: float) -> float:
    return raw * 100 if raw <= 1 else raw


def _map_severity(s: str | None) -> str | None:
    s_clean = (s or "").strip()
    if not s_clean:
        return None
    for val in WasteReport.Severity.values:
        if val.lower() == s_clean.lower():
            return val
    return None


@csrf_exempt
@require_http_methods(["POST"])
def waste_send_sms(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return _json_err("Invalid JSON", status=400)

    location = (body.get("location") or "").strip() or "Unknown location"
    count = int(body.get("count") or 0)
    conf_raw = float(body.get("confidence") or 0)
    pct = _confidence_pct(conf_raw)
    notes = (body.get("notes") or "").strip()
    severity = (body.get("severity") or "").strip()
    recommended_action = (body.get("recommended_action") or "").strip()
    report_id = body.get("report_id")

    sms_out = send_waste_alert_sms(
        location=location,
        count=count,
        confidence_pct=pct,
        severity=severity,
        recommended_action=recommended_action,
        notes=notes,
    )

    body_txt = sms_out.get("payload_preview") or ""

    if sms_out["success"]:
        sms_status = (
            WasteReport.SmsStatus.SENT
            if sms_out.get("mode") == "twilio"
            else WasteReport.SmsStatus.MOCK_SENT
        )
    else:
        sms_status = WasteReport.SmsStatus.FAILED

    if report_id is not None:
        try:
            wr = WasteReport.objects.filter(pk=int(report_id)).first()
            if wr:
                wr.sms_status = sms_status
                wr.sms_body = body_txt[:2000]
                wr.sms_twilio_sid = (sms_out.get("sid") or "")[:64]
                wr.sms_detail = str(sms_out.get("message") or "")[:2000]
                wr.save(
                    update_fields=["sms_status", "sms_body", "sms_twilio_sid", "sms_detail"]
                )
        except (TypeError, ValueError):
            pass

    return JsonResponse(
        {
            "success": bool(sms_out.get("success")),
            "message": sms_out.get("message") or "",
            "mode": sms_out.get("mode"),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def waste_send_email(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return _json_err("Invalid JSON", status=400)

    to_email = (body.get("to") or "").strip() or (
        getattr(settings, "WASTE_DEFAULT_MUNICIPALITY_EMAIL", "")
        or os.environ.get("EMAIL_USER", "")
        or getattr(settings, "EMAIL_HOST_USER", "")
        or ""
    )
    subject = (body.get("subject") or "Smart City Waste Alert").strip()
    message = (body.get("message") or "").strip()
    report_id = body.get("report_id")
    report_blob = body.get("report")

    if not message and isinstance(report_blob, dict):
        lines = [
            f"Report ID: {report_blob.get('id', '—')}",
            f"Time: {report_blob.get('created_at', '—')}",
            f"Location: {report_blob.get('location', '—')}",
            f"Objects: {report_blob.get('count', '—')}",
            f"Max confidence: {report_blob.get('max_confidence', '—')}",
            f"Severity: {report_blob.get('severity', '—')}",
            f"Action: {report_blob.get('recommended_action', '—')}",
        ]
        message = "\n".join(lines)

    mail_out = send_waste_alert_email(to_email=to_email, subject=subject, body=message)

    if mail_out["success"]:
        email_status = (
            WasteReport.EmailStatus.SENT
            if mail_out.get("mode") == "smtp"
            else WasteReport.EmailStatus.MOCK_SENT
        )
    else:
        email_status = WasteReport.EmailStatus.FAILED

    if report_id is not None:
        try:
            wr = WasteReport.objects.filter(pk=int(report_id)).first()
            if wr:
                wr.email_status = email_status
                wr.email_detail = str(mail_out.get("message") or "")[:2000]
                wr.save(update_fields=["email_status", "email_detail"])
        except (TypeError, ValueError):
            pass

    return JsonResponse(
        {
            "success": bool(mail_out.get("success")),
            "message": mail_out.get("message") or "",
            "mode": mail_out.get("mode"),
        }
    )


@csrf_exempt
@require_http_methods(["POST"])
def waste_reports_create(request):
    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        return _json_err("Invalid JSON", status=400)

    city = (body.get("city") or "").strip()
    address = (body.get("address") or "").strip()
    loc_line = format_location_line(city, address, body.get("location") or "")
    lat = body.get("latitude")
    lng = body.get("longitude")
    try:
        latitude = float(lat) if lat is not None and str(lat).strip() != "" else None
    except (TypeError, ValueError):
        latitude = None
    try:
        longitude = float(lng) if lng is not None and str(lng).strip() != "" else None
    except (TypeError, ValueError):
        longitude = None

    notes = (body.get("notes") or "").strip()
    detections = body.get("detections") or []
    count = int(body.get("count") or 0)
    max_conf = float(body.get("max_confidence") or 0)
    severity_in = (body.get("severity") or "").strip()
    action_in = (body.get("recommended_action") or "").strip()
    annot_url = (body.get("annotated_image_url") or "").strip()
    src_url = (body.get("source_image_url") or "").strip()
    sens_flag = bool(body.get("sensitive_area_boost"))

    hint = _location_hint(city, address)
    sev, act, sens_calc = compute_severity_and_action(
        count=count,
        max_confidence=max_conf,
        location_hint=hint,
    )
    severity_final = _map_severity(severity_in) or sev
    action_final = action_in or act
    sensitive_final = sens_flag or sens_calc

    wr = WasteReport.objects.create(
        location=loc_line[:400],
        city=city[:160],
        address=address[:400],
        latitude=latitude,
        longitude=longitude,
        notes=notes[:8000],
        detected_objects=detections if isinstance(detections, list) else [],
        object_count=count,
        max_confidence=min(1.0, max(0.0, max_conf if max_conf <= 1 else max_conf / 100)),
        severity=severity_final,
        recommended_action=action_final[:4000],
        sensitive_area_boost=sensitive_final,
        annotated_image_url=annot_url[:512],
        source_image_url=src_url[:1024],
    )

    built_report = {
        "id": wr.pk,
        "created_at": wr.created_at.isoformat(),
        "location": wr.location,
        "city": wr.city,
        "address": wr.address,
        "latitude": wr.latitude,
        "longitude": wr.longitude,
        "image_source": wr.source_image_url or "(upload)",
        "detected_objects": wr.detected_objects,
        "count": wr.object_count,
        "max_confidence": wr.max_confidence,
        "severity": wr.severity,
        "recommended_action": wr.recommended_action,
        "notes": wr.notes,
        "sms_status": wr.sms_status,
        "email_status": wr.email_status,
        "annotated_image_url": wr.annotated_image_url,
        "source_image_url": wr.source_image_url,
    }

    return JsonResponse({"success": True, "report": built_report})


@require_http_methods(["GET"])
def waste_reports_list(request):
    return JsonResponse({"success": True, "reports": list_reports_for_api()})


@require_http_methods(["GET"])
def waste_statistics(request):
    return JsonResponse({"success": True, **compute_statistics()})


@csrf_exempt
@require_http_methods(["GET", "POST", "OPTIONS"])
def waste_reports_dispatch(request):
    if request.method == "POST":
        return waste_reports_create(request)
    if request.method == "GET":
        return waste_reports_list(request)
    return _json_err("Method not allowed", status=405)
