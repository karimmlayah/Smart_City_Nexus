"""API Dashboard UAV — analyse CNN, historique, exports."""

from __future__ import annotations

import logging

import cv2
import numpy as np
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from monitoring.models import UavStructuralAnalysis
from monitoring.services.uav_cnn_inference import predict_structural_state
from monitoring.services.uav_export import history_csv_response, history_pdf_response
from monitoring.services.uav_risk_engine import (
    build_recommendations,
    compute_risk_score,
    estimate_co2_kg,
    operational_level_from_score,
)

logger = logging.getLogger(__name__)


def _json_err(message: str, status: int = 400, **extra):
    return JsonResponse({"success": False, "message": message, **extra}, status=status)


def _bgr_from_upload(fobj) -> tuple[np.ndarray | None, str | None]:
    raw = fobj.read()
    if not raw:
        return None, "empty_upload"
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return (img, None) if img is not None else (None, "decode_failed")


@csrf_exempt
@require_http_methods(["POST"])
def uav_analyze(request):
    primary_file = request.FILES.get("image_primary") or request.FILES.get("image")
    if not primary_file:
        return _json_err("Fichier image UAV requis (image_primary).")

    frame_p, err = _bgr_from_upload(primary_file)
    if frame_p is None:
        return _json_err(err or "decode_failed", status=422)

    zone = (request.POST.get("zone") or "B").strip().upper()[:1]
    if zone not in ("A", "B", "C"):
        zone = "B"
    try:
        surface_m2 = float((request.POST.get("surface_m2") or "0").replace(",", "."))
    except ValueError:
        surface_m2 = 0.0

    out_primary = predict_structural_state(frame_p)
    if not out_primary.get("success"):
        return JsonResponse({**out_primary}, status=422)

    pred_idx = int(out_primary["predicted_index"])
    conf = float(out_primary["confidence"])
    probs = list(out_primary["probabilities"])

    risk = compute_risk_score(pred_idx, conf, probs, zone)
    op_label, op_key = operational_level_from_score(risk)
    co2 = estimate_co2_kg(pred_idx, surface_m2, conf)
    reco = build_recommendations(pred_idx, conf, zone, risk, op_label)

    comparison_block = None
    comp_file = request.FILES.get("image_compare") or request.FILES.get("comparison")
    comp_probs = []
    comp_idx = None
    comp_label = ""
    comp_conf = None
    comp_grad_url = ""
    comp_sal_url = ""
    comp_orig_url = ""

    if comp_file:
        frame_c, cerr = _bgr_from_upload(comp_file)
        if frame_c is not None:
            out_c = predict_structural_state(frame_c)
            if out_c.get("success"):
                comparison_block = {
                    "probabilities_dict": out_c["probabilities_dict"],
                    "probabilities": out_c["probabilities"],
                    "predicted_label": out_c["predicted_label"],
                    "badge_key": out_c["badge_key"],
                    "confidence": out_c["confidence"],
                    "gradcam_image_url": out_c["gradcam_image_url"],
                    "saliency_image_url": out_c["saliency_image_url"],
                    "original_image_url": out_c["original_image_url"],
                    "labels_order": out_c["labels_order"],
                }
                comp_probs = out_c["probabilities"]
                comp_idx = out_c["predicted_index"]
                comp_label = out_c["predicted_label"]
                comp_conf = out_c["confidence"]
                comp_grad_url = out_c["gradcam_image_url"]
                comp_sal_url = out_c["saliency_image_url"]
                comp_orig_url = out_c["original_image_url"]

    rec = UavStructuralAnalysis.objects.create(
        primary_filename=getattr(primary_file, "name", "")[:260] or "upload.jpg",
        comparison_filename=getattr(comp_file, "name", "")[:260] if comp_file else "",
        zone=zone,
        surface_m2=surface_m2,
        prediction_index=pred_idx,
        prediction_label=out_primary["predicted_label"],
        probabilities=probs,
        confidence=conf,
        comparison_index=comp_idx,
        comparison_label=comp_label or "",
        comparison_probabilities=comp_probs,
        comparison_confidence=comp_conf,
        primary_original_url=out_primary["original_image_url"],
        primary_gradcam_url=out_primary["gradcam_image_url"],
        primary_saliency_url=out_primary["saliency_image_url"],
        comparison_original_url=comp_orig_url,
        comparison_gradcam_url=comp_grad_url,
        comparison_saliency_url=comp_sal_url,
        risk_score=risk,
        operational_level=op_label,
        operational_level_key=op_key,
        co2_estimate_kg=co2,
        recommendations=reco,
    )

    history_slice = list(
        UavStructuralAnalysis.objects.order_by("-created_at").values(
            "id",
            "created_at",
            "primary_filename",
            "prediction_label",
            "confidence",
            "risk_score",
            "co2_estimate_kg",
        )[:10]
    )

    payload = {
        "success": True,
        "analysis_id": rec.pk,
        "zone": zone,
        "surface_m2": surface_m2,
        "perception_status": "Perception · analyse terminée",
        "primary": {
            "probabilities": probs,
            "probabilities_dict": out_primary["probabilities_dict"],
            "predicted_label": out_primary["predicted_label"],
            "badge_key": out_primary["badge_key"],
            "confidence": conf,
            "original_image_url": out_primary["original_image_url"],
            "gradcam_image_url": out_primary["gradcam_image_url"],
            "saliency_image_url": out_primary["saliency_image_url"],
            "labels_order": out_primary["labels_order"],
        },
        "comparison": comparison_block,
        "risk_score": round(risk, 2),
        "operational_level": op_label,
        "operational_level_key": op_key,
        "co2_estimate_kg": co2,
        "recommendations": reco,
        "history_preview": [
            {
                "id": row["id"],
                "created_at": row["created_at"].isoformat(),
                "filename": row["primary_filename"],
                "prediction": row["prediction_label"],
                "confidence": row["confidence"],
                "risk_score": row["risk_score"],
                "co2_kg": row["co2_estimate_kg"],
            }
            for row in history_slice
        ],
        "model_accuracy_display": getattr(settings, "UAV_DISPLAY_MODEL_ACCURACY", "96.06%"),
    }
    return JsonResponse(payload)


@require_http_methods(["GET"])
def uav_history(request):
    rows = UavStructuralAnalysis.objects.order_by("-created_at")[:10]
    data = []
    for r in rows:
        data.append(
            {
                "id": r.pk,
                "created_at": r.created_at.isoformat(),
                "filename": r.primary_filename,
                "prediction": r.prediction_label,
                "confidence": r.confidence,
                "risk_score": r.risk_score,
                "co2_kg": r.co2_estimate_kg,
                "comparison_prediction": r.comparison_label or "",
            }
        )
    return JsonResponse({"success": True, "rows": data})


@require_http_methods(["GET"])
def uav_export_csv(request):
    return history_csv_response()


@require_http_methods(["GET"])
def uav_export_pdf(request):
    return history_pdf_response()


@require_http_methods(["GET"])
def uav_sim_buildings(request):
    """Pastilles démo pour carte 2D / jumeau (positions fixes)."""
    # Quartier fictif — même centre pour Leaflet + Three.js
    base_lat, base_lng = 36.8065, 10.1815
    import random

    random.seed(42)
    buildings = []
    for i in range(24):
        buildings.append(
            {
                "id": i + 1,
                "lat": base_lat + (random.random() - 0.5) * 0.022,
                "lng": base_lng + (random.random() - 0.5) * 0.022,
                "state": random.choice(["intact", "damaged", "collapsed"]),
                "label": f"Bât. {i + 1}",
            }
        )
    latest = (
        UavStructuralAnalysis.objects.order_by("-created_at")
        .values("prediction_label", "prediction_index")
        .first()
    )
    return JsonResponse({"success": True, "buildings": buildings, "latest_hint": latest})
