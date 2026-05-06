"""
Django views wrapping the Smart City traffic-violation pipeline (web_app.py).
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

from django.conf import settings
from django.http import FileResponse, Http404, HttpResponse, JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .smart_city_web import get_smart_city_web

try:
    from werkzeug.utils import secure_filename as _werkzeug_secure_filename
except ImportError:

    def _werkzeug_secure_filename(name: str) -> str:
        p = Path(name)
        return p.name if p.name else "upload"


def _vehicle_registry():
    """SQLite registry from hazemproj Smart city/database.py (vehicles + violations)."""
    import sys

    root = str(settings.SMART_CITY_VIOLATION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import database as vehicle_db_module

    return vehicle_db_module.db


def _safe_output_path(output_dir: Path, filename: str) -> Path:
    if not filename or filename.startswith("/"):
        raise Http404("Invalid file")
    rel = Path(filename)
    if any(p == ".." for p in rel.parts):
        raise Http404("Invalid path")
    full = (output_dir / rel).resolve()
    base = output_dir.resolve()
    if full != base and base not in full.parents:
        raise Http404("Outside output directory")
    if not full.is_file():
        raise Http404("Not found")
    return full


def _mimetype_for_path(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
    }.get(ext, "application/octet-stream")


def _registry_context():
    """SQLite vehicles + violations from hazemproj/Smart city/vehicles.db."""
    try:
        registry = _vehicle_registry()
        return {
            "vehicles": registry.get_all_vehicles(),
            "violations": registry.get_violations(limit=100),
            "registry_available": True,
            "registry_error": None,
        }
    except Exception as exc:
        return {
            "vehicles": [],
            "violations": [],
            "registry_available": False,
            "registry_error": str(exc),
        }


def _render_traffic_violation(request, **extra):
    ctx = _registry_context()
    ctx.update(extra)
    return render(request, "monitoring/traffic_violation.html", ctx)


@require_GET
def traffic_violation_page(request):
    return _render_traffic_violation(request)


@require_POST
def traffic_violation_analyze(request):
    try:
        sc = get_smart_city_web()
    except Exception as exc:
        return _render_traffic_violation(
            request,
            error=f"Impossible de charger le pipeline Smart City (dépendances ou web_app). Détail : {exc}",
        )
    UPLOAD_DIR = sc.UPLOAD_DIR
    OUTPUT_DIR = sc.OUTPUT_DIR
    ALLOWED_EXTENSIONS = sc.ALLOWED_EXTENSIONS
    STREET_SIGN_IMAGE_EXTS = sc.STREET_SIGN_IMAGE_EXTS

    import uuid

    token = uuid.uuid4().hex[:8]
    is_image = False
    is_video = False
    upload_path = None
    stop_line = None

    stop_line_json = (request.POST.get("stop_line_data") or "").strip()
    if stop_line_json:
        try:
            stop_line_data = json.loads(stop_line_json)
            stop_line = [(int(p["x"]), int(p["y"])) for p in stop_line_data]
        except (ValueError, TypeError, KeyError):
            pass

    media = request.FILES.get("media")
    if media and getattr(media, "name", None):
        safe_name = _werkzeug_secure_filename(media.name)
        ext = Path(safe_name).suffix.lower()
        upload_path = UPLOAD_DIR / f"{token}_{safe_name}"
        with open(upload_path, "wb") as out:
            for chunk in media.chunks():
                out.write(chunk)
        is_image = ext in STREET_SIGN_IMAGE_EXTS
        is_video = ext in ALLOWED_EXTENSIONS
        if not (is_image or is_video):
            return _render_traffic_violation(
                request,
                error="Type de fichier non pris en charge. Utilisez une image (jpg, png…) ou une vidéo (mp4…).",
            )
    else:
        data_url = (request.POST.get("camera_image_data") or "").strip()
        if not data_url.startswith("data:image/"):
            return _render_traffic_violation(
                request,
                error="Choisissez un fichier ou capturez une image depuis la caméra.",
            )
        try:
            _, b64_data = data_url.split(",", 1)
            import base64

            image_bytes = base64.b64decode(b64_data)
            upload_path = UPLOAD_DIR / f"{token}_camera.jpg"
            upload_path.write_bytes(image_bytes)
            is_image = True
        except Exception:
            return _render_traffic_violation(
                request,
                error="Impossible de lire l’image capturée par la caméra.",
            )

    try:
        if is_image:
            output_name = f"{token}_street_sign.png"
            output_path = OUTPUT_DIR / output_name
            result = sc.analyze_street_sign_image(upload_path, output_path)
        else:
            output_name = f"{token}_street_sign.mp4"
            output_path = OUTPUT_DIR / output_name
            if stop_line:
                result = sc.analyze_street_sign_video_with_violations(
                    upload_path, output_path, token, stop_line
                )
            else:
                result = sc.analyze_street_sign_video_tracked(upload_path, output_path)
    except Exception as exc:
        return _render_traffic_violation(
            request,
            error=f"Analyse impossible : {exc}",
        )

    try:
        if upload_path and upload_path.exists():
            upload_path.unlink()
    except OSError:
        pass

    try:
        for old_video in OUTPUT_DIR.glob("*_street_sign.mp4"):
            if old_video.name != output_name:
                old_video.unlink(missing_ok=True)
        for old_video in OUTPUT_DIR.glob("*_animal.mp4"):
            old_video.unlink(missing_ok=True)
    except OSError:
        pass

    if result.get("stop_line_used") and result.get("violation_count", 0) > 0:
        result_path = OUTPUT_DIR / f"{token}_violation_results.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        if getattr(sc, "AI_AGENTS_AVAILABLE", False) and getattr(sc, "OPENAI_API_KEY", ""):
            result["dashboard_token"] = token

    output_url = reverse("monitoring:traffic_violation_output", kwargs={"filename": output_name})
    return _render_traffic_violation(
        request,
        result=result,
        output_name=output_name,
        output_url=output_url,
    )


@csrf_exempt
@require_POST
def traffic_violation_suggest_stop_line(request):
    sc = get_smart_city_web()
    import base64

    import cv2
    import numpy as np

    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)
    frame_data = body.get("frame_data", "")
    if not str(frame_data).startswith("data:image/"):
        return JsonResponse({"ok": False, "error": "Invalid frame_data"}, status=400)
    try:
        _, b64_data = frame_data.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return JsonResponse({"ok": False, "error": "Could not decode image"}, status=400)
        suggestion = sc.suggest_stop_line_with_openai(frame)
        if "error" in suggestion:
            return JsonResponse({"ok": False, "error": suggestion["error"]}, status=500)
        return JsonResponse(
            {
                "ok": True,
                "stop_line_points": suggestion.get("stop_line_points", []),
                "confidence": suggestion.get("confidence", 0.0),
                "explanation": suggestion.get("explanation", ""),
                "traffic_light_type": suggestion.get("traffic_light_type", ""),
                "lane_type": suggestion.get("lane_type", ""),
            }
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def traffic_violation_check_violations(request):
    sc = get_smart_city_web()
    UPLOAD_DIR = sc.UPLOAD_DIR
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)
    tracking_token = str(body.get("tracking_token", "")).strip()
    raw_line = body.get("stop_line", [])
    if not tracking_token:
        return JsonResponse({"ok": False, "error": "tracking_token is required"}, status=400)
    if len(raw_line) != 2:
        return JsonResponse({"ok": False, "error": "stop_line must have exactly 2 points"}, status=400)
    try:
        stop_line = [(int(p["x"]), int(p["y"])) for p in raw_line]
        result = sc.check_violations_from_tracking(tracking_token, stop_line)
        tracking_path = UPLOAD_DIR / f"{tracking_token}_tracking.json"
        tracking_data = None
        if tracking_path.exists():
            tracking_data = json.loads(tracking_path.read_text(encoding="utf-8"))
        result["tracking_data"] = tracking_data
        return JsonResponse(result)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)


@csrf_exempt
@require_POST
def traffic_violation_realtime_frame(request):
    sc = get_smart_city_web()
    if not hasattr(sc, "_street_rt_roboflow_counter"):
        sc._street_rt_roboflow_counter = 0

    import base64

    import cv2
    import numpy as np

    data_url = (request.POST.get("frame_data") or "").strip()
    if not data_url.startswith("data:image/"):
        return JsonResponse({"ok": False, "error": "Missing frame_data"}, status=400)
    try:
        _, b64_data = data_url.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JsonResponse({"ok": False, "error": "Could not decode frame"}, status=400)
        sc._street_rt_roboflow_counter += 1
        dets, road_zone, plate_backend = sc._detect_combined_street_objects(
            frame,
            sign_conf=0.5,
            traffic_conf=0.35,
            stream_frame_index=sc._street_rt_roboflow_counter,
        )
        real_xai = (request.POST.get("real_xai") or "").strip() in {"1", "true", "on"}
        xai_cells = []
        xai_meta = {"faithfulness": 0.0, "baseline_conf": 0.0}
        if real_xai and dets:
            top_det = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
            xai_payload = sc._compute_real_xai_cells(frame, top_det, grid=4)
            xai_cells = xai_payload.get("cells", [])
            xai_meta = {
                "faithfulness": float(xai_payload.get("faithfulness", 0.0)),
                "baseline_conf": float(xai_payload.get("baseline_conf", 0.0)),
                "target_label": str(top_det.get("label", "")),
            }
        return JsonResponse(
            {
                "ok": True,
                "detections": dets,
                "xai_cells": xai_cells,
                "xai_meta": xai_meta,
                "road_zone": road_zone,
                "plate_detector_backend": plate_backend,
            }
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)}, status=500)


def traffic_violation_output(request, filename):
    sc = get_smart_city_web()
    path = _safe_output_path(sc.OUTPUT_DIR, filename)
    resp = FileResponse(open(path, "rb"), content_type=_mimetype_for_path(path))
    resp["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    if path.suffix.lower() == ".mp4":
        resp["Accept-Ranges"] = "bytes"
    return resp


def violation_dashboard(request, token: str):
    sc = get_smart_city_web()
    OUTPUT_DIR = sc.OUTPUT_DIR

    if not getattr(sc, "AI_AGENTS_AVAILABLE", False):
        return HttpResponse(
            "AI agents require reportlab. Install: pip install reportlab",
            status=500,
            content_type="text/plain",
        )
    if not getattr(sc, "OPENAI_API_KEY", ""):
        return HttpResponse("OpenAI API key not configured (.env)", status=500, content_type="text/plain")

    result_path = OUTPUT_DIR / f"{token}_violation_results.json"
    if not result_path.exists():
        raise Http404("Violation results not found")

    results = json.loads(result_path.read_text(encoding="utf-8"))

    analysis_agent = sc.ViolationAnalysisAgent(sc.OPENAI_API_KEY)
    summary_agent = sc.ViolationSummaryAgent(sc.OPENAI_API_KEY)
    recommendation_agent = sc.RecommendationAgent(sc.OPENAI_API_KEY)
    pdf_gen = sc.ViolationPDFGenerator(OUTPUT_DIR / f"{token}_reports")

    violations_out = []
    ocr_debug_list = []

    for car_id in results.get("violated_ids", []):
        violation_data = {
            "car_id": car_id,
            "traffic_light": "red",
            "crossed_stop_line": True,
            "plate_images_count": 0,
            "confidence": 0.85,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "frame_number": 0,
        }

        ai_result = analysis_agent.analyze_violation(violation_data)

        plate_images = []
        plates_dir = OUTPUT_DIR / f"{token}_plates"
        if plates_dir.exists():
            for plate_file in plates_dir.glob(f"plate_car{car_id}_*.jpg"):
                conf_str = plate_file.stem.split("_conf")[-1]
                try:
                    conf = int(conf_str) / 100.0
                    plate_images.append((plate_file, conf))
                except ValueError:
                    pass

        violation_data["plate_images_count"] = len(plate_images)

        plate_number = "NO_PLATE"
        plate_verified = False
        ocr_confidence = 0.0
        ocr_notes = ""
        ocr_debug_entry = {
            "car_id": car_id,
            "ocr_raw": None,
            "auto_formatted": None,
            "openai_result": None,
            "final_plate": None,
            "error": None,
        }

        if getattr(sc, "OCR_AVAILABLE", False) and getattr(sc, "OCR_API_KEY", "") and getattr(
            sc, "OPENAI_API_KEY", ""
        ) and plate_images:
            try:
                ocr_result = sc.process_violation_plates(
                    car_id=car_id,
                    plate_images=plate_images,
                    ocr_api_key=sc.OCR_API_KEY,
                    openai_api_key=sc.OPENAI_API_KEY,
                )

                plate_number = ocr_result.get("plate_number", "OCR_FAILED")
                plate_verified = ocr_result.get("is_tunisian_plate", False)
                ocr_confidence = ocr_result.get("confidence", 0.0)
                ocr_notes = ocr_result.get("notes", "")
                ocr_raw = ocr_result.get("ocr_raw", "")

                ocr_debug_entry["ocr_raw"] = ocr_raw
                ocr_debug_entry["auto_formatted"] = (
                    ocr_raw.replace("-", " تونس ").replace("_", " تونس ") if ocr_raw else None
                )
                ocr_debug_entry["openai_result"] = ocr_result.get("notes", "")
                ocr_debug_entry["final_plate"] = plate_number

                violation_data["plate_number"] = plate_number
                violation_data["plate_verified"] = plate_verified
                violation_data["ocr_confidence"] = ocr_confidence
                violation_data["ocr_raw"] = ocr_raw

            except Exception as ocr_error:
                plate_number = "OCR_ERROR"
                ocr_debug_entry["error"] = str(ocr_error)
        else:
            reasons = []
            if not getattr(sc, "OCR_AVAILABLE", False):
                reasons.append("OCR module not available")
            if not getattr(sc, "OCR_API_KEY", ""):
                reasons.append("OCR_API_KEY not set")
            if not getattr(sc, "OPENAI_API_KEY", ""):
                reasons.append("OPENAI_API_KEY not set")
            if not plate_images:
                reasons.append("No plate images")
            ocr_debug_entry["error"] = "Skipped: " + ", ".join(reasons)

        ocr_debug_list.append(ocr_debug_entry)

        vehicle_info = None
        email_sent = False
        if getattr(sc, "DB_EMAIL_AVAILABLE", False) and plate_number not in ("NO_PLATE", "OCR_FAILED", "OCR_ERROR"):
            try:
                vehicle_info = sc.db.search_vehicle(plate_number)

                if vehicle_info:
                    violation_id = sc.db.record_violation(
                        plate_number=plate_number,
                        vehicle_id=vehicle_info["id"],
                        video_token=token,
                        car_tracking_id=car_id,
                        fine_amount=85.0,
                    )

                    if getattr(sc, "EMAIL_USER", "") and getattr(sc, "EMAIL_PASSWORD", ""):
                        notifier = sc.ViolationEmailNotifier(
                            smtp_host=getattr(sc, "EMAIL_HOST", "smtp.gmail.com"),
                            smtp_port=getattr(sc, "EMAIL_PORT", 587),
                            email_user=sc.EMAIL_USER,
                            email_password=sc.EMAIL_PASSWORD,
                        )

                        best_plate_img = (
                            sorted(plate_images, key=lambda x: x[1], reverse=True)[0][0] if plate_images else None
                        )

                        email_sent = notifier.send_violation_email(
                            to_email=vehicle_info["email"],
                            owner_name=vehicle_info["owner_name"],
                            plate_number=plate_number,
                            violation_date=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                            fine_amount=85.0,
                            plate_image_path=best_plate_img,
                        )

                        if email_sent:
                            sc.db.mark_email_sent(violation_id)
                else:
                    sc.db.record_violation(
                        plate_number=plate_number,
                        vehicle_id=None,
                        video_token=token,
                        car_tracking_id=car_id,
                        fine_amount=85.0,
                    )
            except Exception:
                pass

        pdf_path = pdf_gen.generate_violation_report(
            car_id=car_id,
            violation_data=violation_data,
            plate_images=plate_images,
            ai_analysis=ai_result.get("analysis", "Analysis not available"),
        )

        violations_out.append(
            {
                "car_id": car_id,
                "timestamp": violation_data["timestamp"],
                "confidence": violation_data["confidence"],
                "plate_count": len(plate_images),
                "plate_number": plate_number,
                "plate_verified": plate_verified,
                "ocr_confidence": int(ocr_confidence * 100),
                "ocr_notes": ocr_notes,
                "ocr_raw": violation_data.get("ocr_raw", ""),
                "owner_name": vehicle_info["owner_name"] if vehicle_info else "Unknown",
                "owner_email": vehicle_info["email"] if vehicle_info else "N/A",
                "email_sent": email_sent,
                "pdf_filename": f"{token}_reports/{pdf_path.name}",
                "ai_analysis": ai_result.get("analysis", ""),
            }
        )

    video_metadata = {
        "duration": f"{results.get('frames_processed', 0)} frames",
        "frames_processed": results.get("frames_processed", 0),
        "date": datetime.now().strftime("%Y-%m-%d"),
    }

    summary_result = summary_agent.generate_summary(violations_out, video_metadata)

    violation_stats = {
        "total_violations": len(violations_out),
        "violation_rate": len(violations_out) / max(results.get("frames_processed", 1), 1),
        "peak_time": "N/A",
        "common_vehicle": "car",
        "avg_confidence": sum(v["confidence"] for v in violations_out) / max(len(violations_out), 1),
    }

    recommendations_result = recommendation_agent.get_recommendations(violation_stats)

    pdf_gen.generate_summary_report(
        violations=violations_out,
        video_metadata=video_metadata,
        ai_summary=summary_result.get("summary", ""),
        ai_recommendations=recommendations_result.get("recommendations", ""),
    )

    stats = {
        "total_violations": len(violations_out),
        "unique_cars": len(violations_out),
        "plates_detected": len(results.get("plate_images", {}) or {}),
        "avg_confidence": int(violation_stats["avg_confidence"] * 100),
    }

    show_ocr_comparison = any(
        v.get("ocr_raw") and v.get("ocr_raw") != v.get("plate_number") for v in violations_out
    )

    return render(
        request,
        "monitoring/violation_dashboard.html",
        {
            "token": token,
            "violations": violations_out,
            "stats": stats,
            "plate_images": results.get("plate_images", {}),
            "ai_summary": summary_result.get("summary", "").replace("\n", "<br>"),
            "ai_recommendations": recommendations_result.get("recommendations", "").replace("\n", "<br>"),
            "ocr_debug": ocr_debug_list,
            "show_ocr_comparison": show_ocr_comparison,
        },
    )


def violation_download_reports(request, token: str):
    sc = get_smart_city_web()
    reports_dir = sc.OUTPUT_DIR / f"{token}_reports"
    if not reports_dir.exists():
        raise Http404("Reports not found")

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for pdf_file in reports_dir.glob("*.pdf"):
            zf.write(pdf_file, pdf_file.name)
    buf.seek(0)
    resp = FileResponse(buf, as_attachment=True, filename=f"violation_reports_{token}.zip")
    resp["Content-Type"] = "application/zip"
    return resp
