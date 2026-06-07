"""
Analyse vision YOLO (Ultralytics) pour CitizenReclamation — road damage / détection locale.
Repli gracieux si le modèle ou ultralytics manque.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from functools import lru_cache

from django.conf import settings


def _classification_registry() -> dict[str, str]:
    spec = getattr(settings, "ROAD_VISION_CLASSIFICATION_MODELS", None)
    if isinstance(spec, dict) and spec:
        return spec
    return getattr(settings, "ROAD_VISION_MODELS", {})


def pick_model_path(model_key: str | None) -> tuple[str | None, str, str]:
    """
    Retourne (chemin absolu, clé demandée, clé réellement utilisée après repli si fichier absent).
    """
    registry = _classification_registry()
    default_key = getattr(settings, "ROAD_VISION_MODEL_DEFAULT_KEY", "yolov8n_100ep")
    requested = (model_key or default_key).strip()
    if requested not in registry:
        requested = default_key

    def _first_existing(order: list[str]) -> tuple[str | None, str]:
        for k in order:
            if k not in registry:
                continue
            p = Path(registry[k])
            if p.is_file():
                return str(p.resolve()), k
        return None, requested

    if requested in registry:
        p = Path(registry[requested])
        if p.is_file():
            return str(p.resolve()), requested, requested

    fb_extra = getattr(settings, "ROAD_VISION_CLASSIFICATION_FALLBACK_ORDER", ()) or ()
    seen: set[str] = set()
    fallback_order: list[str] = []
    for k in (requested, default_key, *fb_extra, *registry.keys()):
        if k in registry and k not in seen:
            seen.add(k)
            fallback_order.append(k)
    path, used_key = _first_existing(fallback_order)
    return path, requested, used_key


def resolve_model_path(model_key: str | None = None) -> str | None:
    """Chemin du fichier modèle pour une clé de registre (avec repli si manquant)."""
    path, _, _ = pick_model_path(model_key)
    return path


def _model_backend(path: str | None) -> str:
    if not path:
        return "yolo"
    return "fasterrcnn" if Path(path).suffix.lower() == ".pth" else "yolo"


@lru_cache(maxsize=16)
def _yolo_cached(path: str):
    from ultralytics import YOLO

    return YOLO(path)


def list_models_for_ui() -> list[dict[str, Any]]:
    """Liste pour menus classification (clé, libellé, chemin, existe sur disque)."""
    registry = _classification_registry()
    labels = getattr(settings, "ROAD_VISION_CLASSIFICATION_LABELS", None) or getattr(
        settings, "ROAD_VISION_MODEL_LABELS", {}
    )
    dk = getattr(settings, "ROAD_VISION_MODEL_DEFAULT_KEY", "yolov8n_100ep")
    ui_order = getattr(settings, "ROAD_VISION_CLASSIFICATION_UI_ORDER", None) or tuple(
        sorted(registry.keys())
    )
    ordered_keys = [k for k in ui_order if k in registry]
    for k in registry:
        if k not in ordered_keys:
            ordered_keys.append(k)
    rows = []
    for key in ordered_keys:
        p = Path(registry[key])
        rows.append(
            {
                "key": key,
                "label": labels.get(key, key),
                "path": str(p),
                "exists": p.is_file(),
                "is_default": key == dk,
            }
        )
    return rows


def _analysis_out_dir() -> Path:
    d = Path(settings.MEDIA_ROOT) / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stub_result(reason: str, media_type: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "ok": False,
        "stub": True,
        "error": reason,
        "media_type": media_type,
        "labels": [],
        "boxes": [],
        "annotated_image_url": None,
        "annotated_segmentation_image_url": None,
        "frame_results": [],
        "max_confidence": 0.0,
        "model_path_used": None,
        "model_key_requested": None,
        "model_key_resolved": None,
        "model_fallback": False,
    }
    if meta:
        out.update(meta)
    return out


@lru_cache(maxsize=16)
def _fasterrcnn_cached(path: str):
    from monitoring.services.fasterrcnn_inference import load_fasterrcnn

    return load_fasterrcnn(path)


def _load_vision_model(model_key: str | None = None):
    """Charge YOLO (.pt) ou Faster R-CNN (.pth). Retourne (model, backend, req_k, used_k, err)."""
    path, req_k, used_k = pick_model_path(model_key)
    if not path:
        return None, "yolo", req_k, used_k, "model_file_missing"

    backend = _model_backend(path)
    if backend == "fasterrcnn":
        try:
            return _fasterrcnn_cached(path), backend, req_k, used_k, None
        except Exception as exc:
            return None, backend, req_k, used_k, f"model_load_error:{exc}"

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return None, backend, req_k, used_k, "ultralytics_not_installed"
    try:
        return _yolo_cached(path), backend, req_k, used_k, None
    except Exception as exc:
        return None, backend, req_k, used_k, f"model_load_error:{exc}"


def _load_yolo(model_key: str | None = None):
    """Compatibilité — délègue à _load_vision_model (YOLO uniquement si backend yolo)."""
    model, backend, req_k, used_k, err = _load_vision_model(model_key)
    if backend != "yolo":
        return model, req_k, used_k, err
    return model, req_k, used_k, err


def _names_map(model) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def _segmentation_model_path() -> str | None:
    """Chemin absolu du modèle segmentation route (aligné sur ROAD_DAMAGE_MODEL_PATH / laboratoire)."""
    p = (getattr(settings, "ROAD_DAMAGE_MODEL_PATH", "") or "").strip()
    if not p:
        info = getattr(settings, "ROAD_VISION_SEGMENTATION_INFO", None) or {}
        if isinstance(info, dict):
            p = (info.get("path") or "").strip()
    if not p:
        return None
    rp = Path(p).resolve()
    return str(rp) if rp.is_file() else None


def _result_plot_bgr(raw_result) -> Any:
    """RGB plot Ultralytics → ndarray BGR ou None."""
    import numpy as np

    try:
        plot_rgb = raw_result.plot()
        if plot_rgb is None:
            return None
        return plot_rgb[:, :, ::-1] if isinstance(plot_rgb, np.ndarray) else None
    except Exception:
        return None


def _save_plot_bgr_to_analysis(plot_bgr) -> str | None:
    """Écrit une image BGR annotée sous MEDIA/analysis/, retourne URL relative MEDIA."""
    import cv2

    if plot_bgr is None:
        return None
    try:
        out = _analysis_out_dir() / f"anno_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(out), plot_bgr)
        rel = out.relative_to(Path(settings.MEDIA_ROOT))
        return f"{settings.MEDIA_URL.rstrip('/')}/{rel.as_posix().replace(chr(92), '/')}"
    except Exception:
        return None


def predict_image_bgr(
    model,
    frame_bgr,
    conf: float | None = None,
    iou: float | None = None,
    backend: str = "yolo",
) -> dict[str, Any]:
    if backend == "fasterrcnn":
        from monitoring.services.fasterrcnn_inference import predict_fasterrcnn_bgr

        return predict_fasterrcnn_bgr(model, frame_bgr, conf=conf)

    conf = conf if conf is not None else float(getattr(settings, "ROAD_DAMAGE_YOLO_CONF", 0.25))
    iou = iou if iou is not None else float(getattr(settings, "ROAD_DAMAGE_YOLO_IOU", 0.45))
    imgsz = int(getattr(settings, "ROAD_DAMAGE_YOLO_IMGSZ", 640))
    results = model.predict(source=frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    r0 = results[0]
    names = _names_map(model)
    boxes_out: list[dict[str, Any]] = []
    max_conf = 0.0
    boxes = getattr(r0, "boxes", None)
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy.numpy()
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf.numpy()
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls.numpy()
        for i in range(len(xyxy)):
            c = float(confs[i])
            ci = int(clss[i])
            max_conf = max(max_conf, c)
            x1, y1, x2, y2 = [float(x) for x in xyxy[i]]
            label = names.get(ci, str(ci))
            boxes_out.append(
                {
                    "label": label,
                    "class_id": ci,
                    "confidence": round(c, 4),
                    "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )
    labels_unique = sorted({b["label"] for b in boxes_out})
    return {
        "boxes": boxes_out,
        "labels": labels_unique,
        "max_confidence": round(max_conf, 4),
        "raw_result": r0,
    }


def analyze_image_file(
    abs_path: str | Path,
    model_key: str | None = None,
    conf: float | None = None,
) -> dict[str, Any]:
    """Inférence image + image annotée dans media/analysis/."""
    import cv2

    abs_path = Path(abs_path)
    media_type = "image"
    conf_effective = (
        float(conf)
        if conf is not None
        else float(getattr(settings, "ROAD_DAMAGE_YOLO_CONF", 0.25))
    )
    if not abs_path.is_file():
        return _stub_result("file_not_found", media_type, {"yolo_conf_threshold": conf_effective})

    model, backend, req_k, used_k, err = _load_vision_model(model_key)
    path_used = pick_model_path(model_key)[0]
    meta = {
        "model_key_requested": req_k,
        "model_key_resolved": used_k,
        "model_fallback": req_k != used_k,
        "model_path_used": path_used,
        "model_backend": backend,
        "yolo_conf_threshold": conf_effective,
    }
    if model is None:
        return _stub_result(err or "no_model", media_type, meta)

    frame = cv2.imread(str(abs_path))
    if frame is None:
        return _stub_result("cv2_imread_failed", media_type, meta)

    pred = predict_image_bgr(model, frame, conf=conf, backend=backend)
    seg_path = _segmentation_model_path()
    meta["segmentation_model_path_used"] = seg_path
    meta["segmentation_masks_drawn"] = False
    annotated_url = None
    annotated_seg_url = None
    try:
        if backend == "fasterrcnn":
            annotated_url = _save_plot_bgr_to_analysis(pred.get("annotated_bgr"))
        else:
            annotated_url = _save_plot_bgr_to_analysis(_result_plot_bgr(pred["raw_result"]))
        if seg_path and backend == "yolo":
            try:
                imgsz = int(getattr(settings, "ROAD_DAMAGE_YOLO_IMGSZ", 640))
                iou_effective = float(getattr(settings, "ROAD_DAMAGE_YOLO_IOU", 0.45))
                seg_model = _yolo_cached(seg_path)
                sr0 = seg_model.predict(
                    source=frame,
                    conf=conf_effective,
                    iou=iou_effective,
                    imgsz=imgsz,
                    verbose=False,
                )[0]
                annotated_seg_url = _save_plot_bgr_to_analysis(_result_plot_bgr(sr0))
                meta["segmentation_masks_drawn"] = annotated_seg_url is not None
            except Exception:
                pass
    except Exception:
        pass

    pred.pop("raw_result", None)
    pred.pop("annotated_bgr", None)
    return {
        "ok": True,
        "stub": False,
        "media_type": "image",
        "annotated_image_url": annotated_url,
        "annotated_segmentation_image_url": annotated_seg_url,
        "frame_results": [],
        **pred,
        **meta,
    }


def analyze_bgr_frame(
    frame_bgr,
    model_key: str | None = None,
    conf: float | None = None,
) -> dict[str, Any]:
    """Inférence YOLO sur une image OpenCV BGR (ex. cliché ESP32 `/capture`) — même sortie que ``analyze_image_file``."""
    import cv2
    import numpy as np

    media_type = "image"
    conf_effective = (
        float(conf)
        if conf is not None
        else float(getattr(settings, "ROAD_DAMAGE_YOLO_CONF", 0.25))
    )
    if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
        return _stub_result("empty_frame", media_type, {"yolo_conf_threshold": conf_effective})

    model, backend, req_k, used_k, err = _load_vision_model(model_key)
    path_used = pick_model_path(model_key)[0]
    meta = {
        "model_key_requested": req_k,
        "model_key_resolved": used_k,
        "model_fallback": req_k != used_k,
        "model_path_used": path_used,
        "model_backend": backend,
        "yolo_conf_threshold": conf_effective,
    }
    if model is None:
        return _stub_result(err or "no_model", media_type, meta)

    pred = predict_image_bgr(model, frame_bgr, conf=conf, backend=backend)
    seg_path = _segmentation_model_path()
    meta["segmentation_model_path_used"] = seg_path
    meta["segmentation_masks_drawn"] = False
    annotated_url = None
    annotated_seg_url = None
    try:
        if backend == "fasterrcnn":
            annotated_url = _save_plot_bgr_to_analysis(pred.get("annotated_bgr"))
        else:
            annotated_url = _save_plot_bgr_to_analysis(_result_plot_bgr(pred["raw_result"]))
        if seg_path and backend == "yolo":
            try:
                imgsz = int(getattr(settings, "ROAD_DAMAGE_YOLO_IMGSZ", 640))
                iou_effective = float(getattr(settings, "ROAD_DAMAGE_YOLO_IOU", 0.45))
                seg_model = _yolo_cached(seg_path)
                sr0 = seg_model.predict(
                    source=frame_bgr,
                    conf=conf_effective,
                    iou=iou_effective,
                    imgsz=imgsz,
                    verbose=False,
                )[0]
                annotated_seg_url = _save_plot_bgr_to_analysis(_result_plot_bgr(sr0))
                meta["segmentation_masks_drawn"] = annotated_seg_url is not None
            except Exception:
                pass
    except Exception:
        pass

    pred.pop("raw_result", None)
    pred.pop("annotated_bgr", None)
    return {
        "ok": True,
        "stub": False,
        "media_type": "image",
        "annotated_image_url": annotated_url,
        "annotated_segmentation_image_url": annotated_seg_url,
        "frame_results": [],
        "source": "esp_live_frame",
        **pred,
        **meta,
    }


def analyze_video_file(
    abs_path: str | Path,
    max_frames: int = 8,
    interval_sec: float = 1.0,
    model_key: str | None = None,
    conf: float | None = None,
) -> dict[str, Any]:
    """Échantillonne ~1 frame/s, max max_frames, YOLO sur chaque frame + aperçus annotés."""
    import cv2

    abs_path = Path(abs_path)
    media_type = "video"
    conf_effective = (
        float(conf)
        if conf is not None
        else float(getattr(settings, "ROAD_DAMAGE_YOLO_CONF", 0.25))
    )
    if not abs_path.is_file():
        return _stub_result("file_not_found", media_type, {"yolo_conf_threshold": conf_effective})

    model, backend, req_k, used_k, err = _load_vision_model(model_key)
    path_used = pick_model_path(model_key)[0]
    meta = {
        "model_key_requested": req_k,
        "model_key_resolved": used_k,
        "model_fallback": req_k != used_k,
        "model_path_used": path_used,
        "model_backend": backend,
        "yolo_conf_threshold": conf_effective,
    }
    if model is None:
        return _stub_result(err or "no_model", media_type, meta)

    cap = cv2.VideoCapture(str(abs_path))
    if not cap.isOpened():
        return _stub_result("video_open_failed", media_type, meta)

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps < 1:
        fps = 25.0
    frame_step = max(1, int(round(fps * interval_sec)))
    frame_results: list[dict[str, Any]] = []
    idx = 0
    sampled = 0
    all_labels: set[str] = set()
    max_conf_global = 0.0

    while sampled < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            pr = predict_image_bgr(model, frame, conf=conf, backend=backend)
            max_conf_global = max(max_conf_global, pr["max_confidence"])
            all_labels.update(pr["labels"])
            thumb_url = None
            try:
                if backend == "fasterrcnn":
                    plot_bgr = pr.get("annotated_bgr")
                else:
                    plot_rgb = pr["raw_result"].plot()
                    plot_bgr = plot_rgb[:, :, ::-1] if plot_rgb is not None else None
                if plot_bgr is not None:
                    out = _analysis_out_dir() / f"vid_frame_{uuid.uuid4().hex}.jpg"
                    cv2.imwrite(str(out), plot_bgr)
                    rel = out.relative_to(Path(settings.MEDIA_ROOT))
                    thumb_url = f"{settings.MEDIA_URL.rstrip('/')}/{rel.as_posix().replace(chr(92), '/')}"
            except Exception:
                pass
            pr.pop("raw_result", None)
            pr.pop("annotated_bgr", None)
            frame_results.append(
                {
                    "frame_index": idx,
                    "time_sec": round(idx / fps, 2),
                    "labels": pr["labels"],
                    "boxes": pr["boxes"],
                    "max_confidence": pr["max_confidence"],
                    "annotated_thumb_url": thumb_url,
                }
            )
            sampled += 1
        idx += 1

    cap.release()
    return {
        "ok": True,
        "stub": False,
        "media_type": "video",
        "annotated_image_url": frame_results[0]["annotated_thumb_url"] if frame_results else None,
        "frame_results": frame_results,
        "labels": sorted(all_labels),
        "boxes": [],
        "max_confidence": round(max_conf_global, 4),
        **meta,
    }


def analyze_reclamation_media(
    reclamation,
    model_key: str | None = None,
    conf: float | None = None,
) -> dict[str, Any]:
    """Point d'entrée depuis la vue : utilise reclamation.media.path."""
    try:
        path = reclamation.media.path
    except Exception as exc:
        return _stub_result(f"media_path:{exc}", getattr(reclamation, "media_type", "image"))

    mt = getattr(reclamation, "media_type", "") or "image"
    if mt == "video":
        return analyze_video_file(path, model_key=model_key, conf=conf)
    return analyze_image_file(path, model_key=model_key, conf=conf)


def count_detection_boxes(pred_dict: dict[str, Any]) -> int:
    """Nombre total de boîtes après seuil (image ou vidéo agrégée par frames)."""
    frs = pred_dict.get("frame_results") or []
    if frs:
        return sum(len(fr.get("boxes") or []) for fr in frs)
    return len(pred_dict.get("boxes") or [])


def max_confidence_from_boxes(pred_dict: dict[str, Any]) -> float:
    """Score max pour XAI (baseline occlusion)."""
    if pred_dict.get("frame_results"):
        return float(
            max((fr.get("max_confidence") or 0) for fr in pred_dict["frame_results"])
            or pred_dict.get("max_confidence")
            or 0
        )
    return float(pred_dict.get("max_confidence") or 0)


def baseline_predict_confidence(
    frame_bgr, model, conf: float | None = None, backend: str = "yolo"
) -> float:
    pr = predict_image_bgr(model, frame_bgr, conf=conf, backend=backend)
    return float(pr["max_confidence"])
