"""
Inférence détection déchets (YOLO Ultralytics) pour Waste Street Detection.

Configurer dans Django settings :
  WASTE_MODEL_PATH — chemin absolu ou relatif au fichier .pt entraîné déchets/rue.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.conf import settings

from monitoring.services.waste_severity import compute_severity_and_action

logger = logging.getLogger(__name__)


def _analysis_dir() -> Path:
    d = Path(settings.MEDIA_ROOT) / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _waste_model_path() -> Path | None:
    raw = (getattr(settings, "WASTE_MODEL_PATH", "") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = Path(settings.BASE_DIR) / p
    return p.resolve() if p.is_file() else None


_model_cache: dict[str, Any] = {}


def _get_model():
    path = _waste_model_path()
    if path is None:
        return None, None
    key = str(path)
    if key not in _model_cache:
        try:
            from ultralytics import YOLO

            _model_cache[key] = YOLO(str(path))
        except Exception as exc:
            logger.exception("waste model load failed: %s", exc)
            return None, None
    return _model_cache[key], path


def detect_waste_bgr(frame_bgr: np.ndarray, location_hint: str = "") -> dict[str, Any]:
    """
    Retourne un dict aligné sur l'API /api/waste/detect/ :
    success, detections[], count, annotated_image_url, max_confidence,
    severity, recommended_action, sensitive_area_boost, error?
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "success": False,
            "detections": [],
            "count": 0,
            "annotated_image_url": None,
            "max_confidence": 0.0,
            "severity": "Low",
            "recommended_action": "",
            "sensitive_area_boost": False,
            "error": "empty_image",
        }

    model, path_used = _get_model()
    if model is None:
        return {
            "success": False,
            "detections": [],
            "count": 0,
            "annotated_image_url": None,
            "max_confidence": 0.0,
            "severity": "Low",
            "recommended_action": "",
            "sensitive_area_boost": False,
            "error": "model_missing",
            "error_detail": (
                "Définissez WASTE_MODEL_PATH dans settings.py "
                "(chemin vers un fichier .pt valide)."
            ),
        }

    conf = float(getattr(settings, "WASTE_YOLO_CONF", 0.25))
    iou = float(getattr(settings, "WASTE_YOLO_IOU", 0.45))
    imgsz = int(getattr(settings, "WASTE_YOLO_IMGSZ", 640))

    try:
        results = model.predict(
            source=frame_bgr,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
        r0 = results[0]
        names = getattr(model, "names", {}) or {}
        if isinstance(names, dict):
            id_to_name = {int(k): str(v) for k, v in names.items()}
        else:
            id_to_name = {i: str(n) for i, n in enumerate(names)}

        detections: list[dict[str, Any]] = []
        max_cf = 0.0
        boxes = getattr(r0, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.numpy()
            for i in range(len(xyxy)):
                c = float(confs[i])
                ci = int(clss[i])
                max_cf = max(max_cf, c)
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                label = id_to_name.get(ci, str(ci))
                detections.append(
                    {
                        "label": label,
                        "confidence": round(c, 4),
                        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    }
                )

        annotated_url = None
        try:
            plot_rgb = r0.plot()
            if plot_rgb is not None:
                plot_bgr = plot_rgb[:, :, ::-1]
                out = _analysis_dir() / f"waste_anno_{uuid.uuid4().hex}.jpg"
                cv2.imwrite(str(out), plot_bgr)
                rel = out.relative_to(Path(settings.MEDIA_ROOT))
                annotated_url = (
                    f"{settings.MEDIA_URL.rstrip('/')}/"
                    f"{rel.as_posix().replace(chr(92), '/')}"
                )
        except Exception as exc:
            logger.warning("waste annotate save failed: %s", exc)

        severity, action, sens = compute_severity_and_action(
            count=len(detections),
            max_confidence=max_cf,
            location_hint=location_hint or "",
        )

        return {
            "success": True,
            "detections": detections,
            "count": len(detections),
            "annotated_image_url": annotated_url,
            "max_confidence": round(max_cf, 4),
            "severity": severity,
            "recommended_action": action,
            "sensitive_area_boost": sens,
            "model_path": str(path_used),
        }
    except Exception as exc:
        logger.exception("waste inference failed")
        return {
            "success": False,
            "detections": [],
            "count": 0,
            "annotated_image_url": None,
            "max_confidence": 0.0,
            "severity": "Low",
            "recommended_action": "",
            "sensitive_area_boost": False,
            "error": "inference_failed",
            "error_detail": str(exc),
        }
