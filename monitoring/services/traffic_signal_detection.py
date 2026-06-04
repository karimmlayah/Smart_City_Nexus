"""Traffic light red/green detection with a YOLO .pt model."""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

_model_cache: dict[str, Any] = {}


def _analysis_dir() -> Path:
    d = Path(settings.MEDIA_ROOT) / "traffic_signal"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _media_url(path: Path) -> str:
    rel = path.relative_to(Path(settings.MEDIA_ROOT))
    return f"{settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"


def _configured_model_path() -> Path | None:
    raw = (getattr(settings, "TRAFFIC_SIGNAL_MODEL_PATH", "") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = Path(settings.BASE_DIR) / p
    return p.resolve() if p.is_file() else None


def _get_model():
    path = _configured_model_path()
    if path is None:
        return None, None
    key = str(path)
    if key not in _model_cache:
        try:
            from ultralytics import YOLO

            _model_cache[key] = YOLO(str(path))
        except Exception as exc:
            logger.exception("traffic signal model load failed: %s", exc)
            return None, None
    return _model_cache[key], path


def _id_to_name(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def _signal_for_label(label: str) -> str | None:
    low = label.strip().lower()
    red_keys = getattr(settings, "TRAFFIC_SIGNAL_RED_LABELS", ("red light", "red", "rouge"))
    green_keys = getattr(settings, "TRAFFIC_SIGNAL_GREEN_LABELS", ("green light", "green", "vert"))
    if any(str(k).lower() in low for k in red_keys):
        return "red"
    if any(str(k).lower() in low for k in green_keys):
        return "green"
    return None


def _run_frame(model: Any, frame_bgr: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    conf = float(getattr(settings, "TRAFFIC_SIGNAL_CONF", 0.25))
    iou = float(getattr(settings, "TRAFFIC_SIGNAL_IOU", 0.45))
    imgsz = int(getattr(settings, "TRAFFIC_SIGNAL_IMGSZ", 640))
    results = model.predict(source=frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r0 = results[0]
    names = _id_to_name(model)
    detections: list[dict[str, Any]] = []

    boxes = getattr(r0, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.numpy()
        for i in range(len(xyxy)):
            label = names.get(int(clss[i]), str(int(clss[i])))
            signal = _signal_for_label(label)
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            detections.append(
                {
                    "label": label,
                    "signal": signal or "other",
                    "confidence": round(float(confs[i]), 4),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )

    annotated = None
    try:
        annotated = r0.plot()
    except Exception as exc:
        logger.warning("traffic signal annotation failed: %s", exc)
    return detections, annotated


def _summarize_signal(detections: list[dict[str, Any]]) -> dict[str, Any]:
    red = [d for d in detections if d.get("signal") == "red"]
    green = [d for d in detections if d.get("signal") == "green"]
    red_conf = max((float(d.get("confidence") or 0) for d in red), default=0.0)
    green_conf = max((float(d.get("confidence") or 0) for d in green), default=0.0)
    if red_conf <= 0 and green_conf <= 0:
        state = "unknown"
        confidence = 0.0
    elif red_conf >= green_conf:
        state = "red"
        confidence = red_conf
    else:
        state = "green"
        confidence = green_conf
    return {
        "state": state,
        "label": {"red": "Feu rouge", "green": "Feu vert"}.get(state, "Non détecté"),
        "confidence": round(confidence, 4),
        "red_count": len(red),
        "green_count": len(green),
        "red_confidence": round(red_conf, 4),
        "green_confidence": round(green_conf, 4),
    }


def analyze_traffic_signal_file(path: Path) -> dict[str, Any]:
    """Analyze an image or video and return a UI/ESP32-ready result."""
    model, model_path = _get_model()
    if model is None:
        return {
            "success": False,
            "error": "model_missing",
            "error_detail": "Modèle introuvable. Vérifiez TRAFFIC_SIGNAL_MODEL_PATH dans settings.py.",
        }

    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return _analyze_image(path, model, model_path)
    if ext in VIDEO_EXTS:
        return _analyze_video(path, model, model_path)
    return {
        "success": False,
        "error": "unsupported_media",
        "error_detail": "Format non pris en charge. Utilisez une image ou une vidéo.",
    }


def _analyze_image(path: Path, model: Any, model_path: Path) -> dict[str, Any]:
    frame = cv2.imread(str(path))
    if frame is None or frame.size == 0:
        return {"success": False, "error": "image_decode_failed", "error_detail": "Image illisible."}

    detections, annotated = _run_frame(model, frame)
    summary = _summarize_signal(detections)
    out = _analysis_dir() / f"traffic_signal_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(out), annotated if annotated is not None else frame)
    return {
        "success": True,
        "media_type": "image",
        "detections": detections,
        "detection_count": len(detections),
        "annotated_url": _media_url(out),
        "model_path": str(model_path),
        **summary,
    }


def _analyze_video(path: Path, model: Any, model_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"success": False, "error": "video_decode_failed", "error_detail": "Vidéo illisible."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return {"success": False, "error": "video_decode_failed", "error_detail": "Dimensions vidéo invalides."}

    out = _analysis_dir() / f"traffic_signal_{uuid.uuid4().hex}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_step = max(1, int(getattr(settings, "TRAFFIC_SIGNAL_VIDEO_FRAME_STEP", 3)))
    max_frames = int(getattr(settings, "TRAFFIC_SIGNAL_VIDEO_MAX_FRAMES", 900))

    all_detections: list[dict[str, Any]] = []
    frame_idx = 0
    last_annotated: np.ndarray | None = None

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            detections, annotated = _run_frame(model, frame)
            for det in detections:
                det = dict(det)
                det["frame"] = frame_idx
                all_detections.append(det)
            last_annotated = annotated if annotated is not None else frame
            writer.write(last_annotated)
        else:
            writer.write(last_annotated if last_annotated is not None else frame)
        frame_idx += 1

    cap.release()
    writer.release()

    summary = _summarize_signal(all_detections)
    return {
        "success": True,
        "media_type": "video",
        "detections": all_detections[:80],
        "detection_count": len(all_detections),
        "frames_processed": frame_idx,
        "annotated_url": _media_url(out),
        "model_path": str(model_path),
        **summary,
    }
