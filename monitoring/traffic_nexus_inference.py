"""Inférence Traffic Nexus (YOLO + tdss) pour l’intégration Django."""
from __future__ import annotations

import base64
import logging
import threading
import time
from collections import Counter
from dataclasses import asdict
from typing import Any, Optional, Set

import cv2
import numpy as np
from django.conf import settings
from ultralytics import YOLO

from monitoring.traffic_nexus_helpers import (
    VEHICLE_LABELS,
    build_scene_state,
    counter_list_serializable,
    polygon_area,
)
from monitoring.traffic_nexus_paths import resolve_trusted_model_weights
from monitoring.traffic_tdss.agentic_system import AgenticOrchestrator
from monitoring.traffic_tdss.congestion_analysis import ZoneTrafficState, compute_zone_states
from monitoring.traffic_tdss.decision_engine import DecisionEngine, ZoneDecision
from monitoring.traffic_tdss.esp32_bridge import (
    build_esp32_payload,
    infer_emergency_zone,
    new_emergency_confirmation_state,
    step_emergency_confirmation,
)
from monitoring.traffic_tdss.vehicle_detection import (
    ZoneDetection,
    detect_from_candidates,
    detect_in_zones,
    detect_in_zones_multi,
)
from monitoring.traffic_tdss.video_processing import read_first_frame, resolve_video_source
from monitoring.traffic_tdss.xai import explain_decision

logger = logging.getLogger(__name__)

_models_lock = threading.Lock()
_models_cache: dict[str, YOLO] = {}
_last_model_config_diag: tuple[str, str, bool, str] | None = None


def get_cached_yolo(path: str) -> YOLO:
    return _get_model(path)


def _get_model(path: str) -> YOLO:
    with _models_lock:
        if path not in _models_cache:
            _models_cache[path] = YOLO(path)
            logger.info("Traffic Nexus YOLO loaded: %s", path)
        return _models_cache[path]


def parse_allowed_labels(vehicle_labels_csv: str | None, vehicle_mode: bool) -> Optional[Set[str]]:
    if not vehicle_mode:
        return None
    s = (vehicle_labels_csv or "").strip()
    if not s:
        return set(VEHICLE_LABELS)
    tokens = {x.strip().lower() for x in s.split(",") if x.strip()}
    if "*" in tokens or "all" in tokens:
        return None
    return tokens


def load_frame_from_youtube(url: str) -> np.ndarray | None:
    try:
        resolved = resolve_video_source(url.strip())
        return read_first_frame(resolved)
    except Exception as exc:
        logger.warning("traffic first frame youtube failed: %s", exc)
        return None


def decode_upload_bytes(raw: bytes) -> np.ndarray | None:
    if not raw:
        return None
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def decode_upload_first_frame(raw: bytes, *, original_filename: str = "") -> np.ndarray | None:
    """Decode a raster image from bytes, or the first frame of an uploaded video (mp4, etc.)."""
    from pathlib import Path
    import tempfile

    img = decode_upload_bytes(raw)
    if img is not None:
        return img
    if not raw:
        return None
    suf = Path(original_filename).suffix.lower()
    video_exts = {
        ".mp4",
        ".webm",
        ".avi",
        ".mov",
        ".mkv",
        ".m4v",
        ".mpeg",
        ".mpg",
        ".wmv",
        ".ogv",
    }
    if suf not in video_exts:
        suf = ".mp4"
    tmp_path: str | None = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
        tmp.write(raw)
        tmp.close()
        tmp_path = tmp.name
        from monitoring.traffic_tdss.video_processing import read_first_frame

        return read_first_frame(tmp_path)
    except Exception as exc:
        logger.warning("decode_upload_first_frame failed: %s", exc)
        return None
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def decode_base64_image(b64: str) -> np.ndarray | None:
    try:
        raw = base64.standard_b64decode(b64)
        return decode_upload_bytes(raw)
    except Exception:
        return None


def zone_decision_to_dict(d: ZoneDecision) -> dict[str, Any]:
    return asdict(d)


def coerce_emergency_confirm_state(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Réhydrate l’état mutable pour step_emergency_confirmation (aligné Streamlit)."""
    base = new_emergency_confirmation_state()
    if not raw or not isinstance(raw, dict):
        return base
    for k in base:
        if k in raw:
            base[k] = raw[k]
    return base


def resolve_model_config(
    mode: str,
    custom_model_path: str | None,
    defaults: dict,
    use_dual_models: bool,
) -> dict[str, Any]:
    """
    Résolution centralisée et stricte du modèle/pipeline actif.
    """
    mode_norm = (mode or "vehicle").strip().lower()
    is_congestion = mode_norm in ("congestion", "congestion_detection")
    custom_resolved = resolve_trusted_model_weights(custom_model_path) if custom_model_path else None

    if is_congestion:
        # Uniquement yolo_congestion.pt — pas de fusion avec models_traffic/best.pt.
        congestion_path = (
            defaults.get("yolo_congestion")
            or defaults.get("congestion")
            or defaults.get("yolov8n")
            or ""
        )
        return {
            "active_model_path": congestion_path,
            "secondary_model_path": "",
            "use_dual_models": False,
            "allowed_labels": None,
            "pipeline": "congestion",
        }

    active = custom_resolved or (defaults.get("best") or defaults.get("yolov8n") or "")
    secondary = defaults.get("yolov8n") or ""
    dual_ok = bool(use_dual_models and active and secondary)
    return {
        "active_model_path": active,
        "secondary_model_path": secondary,
        "use_dual_models": dual_ok,
        "allowed_labels": set(VEHICLE_LABELS),
        "pipeline": "vehicle",
    }


def run_yolo_detection(
    frame_bgr: np.ndarray,
    roi_polygons: list[list[tuple[int, int]]],
    *,
    mode: str,
    model_key: str,
    dual_models: bool,
    conf: float,
    imgsz: int,
    max_det: int,
    draw_confidence: bool = True,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
    display_mode: str = "normal_detection",
    show_speed_overlays: bool = True,
    track_state: dict[str, Any] | None = None,
    vehicle_labels_csv: str | None = None,
    custom_model_path: str | None = None,
    overlay_on_client: bool = False,
) -> tuple[Optional[ZoneDetection], Optional[dict[str, Any]]]:
    global _last_model_config_diag

    draw_boxes = not overlay_on_client or (display_mode or "").strip().lower() in (
        "congestion_heatmap",
    )

    defaults = getattr(settings, "TRAFFIC_NEXUS_MODEL_PATHS", {}) or {}
    mode_norm = (mode or "vehicle").strip().lower()
    is_congestion_mode = mode_norm in ("congestion", "congestion_detection")
    # UI model_key remappe « best » pour le pipeline véhicule seul (choix du modèle unique).
    # En double modèle véhicule, « best » doit rester models_traffic/best.pt (ambulances), pas yolov8n.
    if model_key in defaults and not is_congestion_mode:
        defaults = dict(defaults)
        if not (dual_models and mode_norm == "vehicle"):
            defaults["best"] = defaults.get(model_key) or defaults.get("best")
    cfg = resolve_model_config(mode, custom_model_path, defaults, dual_models)
    allowed = parse_allowed_labels(vehicle_labels_csv, cfg["pipeline"] == "vehicle")
    if cfg["pipeline"] == "congestion":
        allowed = None

    diag = (
        str(mode or "vehicle"),
        str(cfg.get("active_model_path") or ""),
        bool(cfg.get("use_dual_models")),
        str(cfg.get("pipeline") or ""),
    )
    if _last_model_config_diag != diag:
        _last_model_config_diag = diag
        logger.info(
            "[ModelConfig]\nmode=%s\nactive_model=%s\ndual_models=%s\npipeline=%s",
            diag[0],
            diag[1],
            diag[2],
            diag[3],
        )

    if cfg["pipeline"] == "vehicle" and cfg["use_dual_models"]:
        p_best = str(cfg.get("active_model_path") or "")
        p_yolo = str(cfg.get("secondary_model_path") or "")
        if not (p_best and p_yolo):
            return None, {
                "success": False,
                "error": "dual_models_missing",
                "message": "best.pt ou yolov8n.pt introuvable (ou chemin personnalisé invalide).",
            }
        mb = _get_model(p_best)
        mn = _get_model(p_yolo)
        mb.overrides["conf"] = conf
        mn.overrides["conf"] = conf
        det = detect_in_zones_multi(
            models=[mb, mn],
            frame=frame_bgr,
            roi_polygons=roi_polygons,
            allowed_labels=allowed,
            imgsz=imgsz,
            max_det=max_det,
            draw_confidence=draw_confidence,
            min_confidence=min_confidence,
            min_box_area=min_box_area,
            display_mode=display_mode,
            show_speed_overlays=show_speed_overlays,
            track_state=track_state,
            draw_boxes=draw_boxes,
        )
        det.model_config = {
            "selected_mode": str(mode or "vehicle"),
            "active_model_path": p_best,
            "dual_models_enabled": True,
            "inference_pipeline": "vehicle",
            "secondary_model_path": p_yolo,
        }
        return det, None

    path = str(cfg.get("active_model_path") or "")
    if not path:
        return None, {
            "success": False,
            "error": "model_missing",
            "message": "Aucun modèle YOLO configuré ou chemin personnalisé invalide.",
        }
    model = _get_model(path)
    model.overrides["conf"] = conf
    det = detect_in_zones(
        model=model,
        frame=frame_bgr,
        roi_polygons=roi_polygons,
        allowed_labels=allowed,
        imgsz=imgsz,
        max_det=max_det,
        draw_confidence=draw_confidence,
        min_confidence=min_confidence,
        min_box_area=min_box_area,
        display_mode=display_mode,
        show_speed_overlays=show_speed_overlays,
        track_state=track_state,
        draw_boxes=draw_boxes,
    )
    det.model_config = {
        "selected_mode": str(mode or "vehicle"),
        "active_model_path": path,
        "dual_models_enabled": False,
        "inference_pipeline": str(cfg.get("pipeline") or "vehicle"),
    }
    return det, None


def finalize_analysis(
    det: ZoneDetection,
    frame_bgr: np.ndarray,
    roi_polygons: list[list[tuple[int, int]]],
    *,
    mock_emergency: bool,
    esp32_sim_zone: int = 0,
    esp32_confirm_state: dict[str, Any] | None = None,
    now_s: float | None = None,
    overlay_on_client: bool = False,
) -> dict[str, Any]:
    polygons = roi_polygons if roi_polygons else []
    zone_areas = [polygon_area(p) for p in polygons] if polygons else [float(frame_bgr.shape[0] * frame_bgr.shape[1])]

    zone_states = compute_zone_states(det.per_zone_counts, zone_areas)
    engine = DecisionEngine()
    decisions = engine.decide_many(zone_states)
    green_wave = DecisionEngine.green_wave(decisions)

    t_now = float(now_s) if now_s is not None else time.time()
    raw_emergency, raw_priority_zone = infer_emergency_zone(
        det.per_zone_counts,
        mock_emergency,
        int(esp32_sim_zone),
    )
    n_z = len(decisions)
    rz = raw_priority_zone
    if rz is not None and n_z > 0:
        rz = max(0, min(n_z - 1, int(rz)))
    elif rz is not None and n_z == 0:
        rz = None

    confirm_state = coerce_emergency_confirm_state(esp32_confirm_state)
    confirmed_emergency, confirmed_priority_zone, confirm_logs = step_emergency_confirmation(
        confirm_state,
        raw_emergency,
        rz,
        t_now,
    )
    cpz = confirmed_priority_zone
    if cpz is not None and n_z > 0:
        cpz = max(0, min(n_z - 1, int(cpz)))

    esp32_payload = build_esp32_payload(decisions, confirmed_emergency, cpz)

    xai_zones = []
    for i, (zone_st, dec) in enumerate(zip(zone_states, decisions)):
        exp = explain_decision(zone_st, dec)
        xai_zones.append(
            {
                "zone": i + 1,
                "title": exp.title,
                "message": exp.message,
                "features": {
                    "congestion_pct": exp.feature_congestion,
                    "vehicle_count": exp.feature_vehicle_count,
                    "waiting_min": exp.feature_waiting_time,
                    "criticality": exp.feature_criticality,
                },
            }
        )

    scene = build_scene_state(zone_states, decisions, det.per_zone_counts, mock_emergency)
    orch = AgenticOrchestrator()
    agentic_bundle = orch.run_agentic_system(scene)

    total_vehicles = sum(s.vehicle_count for s in zone_states)
    avg_density = float(np.mean([s.density for s in zone_states])) if zone_states else 0.0
    global_level = max((d.congestion_level for d in decisions), default="low")
    global_criticality = int(np.mean([d.traffic_criticality_score for d in decisions])) if decisions else 0

    _, buf = cv2.imencode(
        ".jpg",
        det.processed_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 76 if overlay_on_client else 88],
    )
    out_b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")

    def _state_dict(zs: ZoneTrafficState) -> dict[str, Any]:
        return {
            "vehicle_count": zs.vehicle_count,
            "density": round(zs.density, 4),
            "weighted_score": round(zs.weighted_score, 2),
            "level": zs.level,
        }

    return {
        "success": True,
        "annotated_image_base64": out_b64,
        "detections_total": det.detections_total,
        "per_zone_counts": counter_list_serializable(det.per_zone_counts),
        "zone_states": [_state_dict(z) for z in zone_states],
        "decisions": [zone_decision_to_dict(d) for d in decisions],
        "green_wave": green_wave,
        "overview": {
            "total_vehicles": total_vehicles,
            "avg_density": round(avg_density, 4),
            "global_level": global_level,
            "global_criticality": global_criticality,
        },
        "xai_zones": xai_zones,
        "agentic": agentic_bundle,
        "scene_state": scene,
        "esp32": {
            "payload": esp32_payload,
            "raw_emergency": raw_emergency,
            "raw_priority_zone": rz,
            "confirmed_emergency": confirmed_emergency,
            "confirmed_priority_zone": cpz,
            "confirm_logs": confirm_logs,
            "confirm_state": confirm_state,
        },
        "display_state": {
            "track_state": det.track_state or {},
            "vehicle_detections": det.vehicle_detections or [],
        },
        "vehicle_detections": det.vehicle_detections or [],
        "model_config": det.model_config or {},
    }


def analyze_frame(
    frame_bgr: np.ndarray,
    roi_polygons: list[list[tuple[int, int]]],
    *,
    mode: str,
    model_key: str,
    dual_models: bool,
    conf: float,
    imgsz: int,
    max_det: int,
    draw_confidence: bool = True,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
    display_mode: str = "normal_detection",
    show_speed_overlays: bool = True,
    display_state: dict[str, Any] | None = None,
    mock_emergency: bool = False,
    vehicle_labels: str | None = None,
    custom_model_path: str | None = None,
    esp32_sim_zone: int = 0,
    esp32_confirm_state: dict[str, Any] | None = None,
    now_s: float | None = None,
    overlay_on_client: bool = False,
) -> dict[str, Any]:
    """Une passe détection + décision + XAI + agents (sans boucle vidéo)."""
    det, err = run_yolo_detection(
        frame_bgr,
        roi_polygons,
        mode=mode,
        model_key=model_key,
        dual_models=dual_models,
        conf=conf,
        imgsz=imgsz,
        max_det=max_det,
        draw_confidence=True,
        min_confidence=min_confidence,
        min_box_area=min_box_area,
        display_mode=display_mode,
        show_speed_overlays=show_speed_overlays,
        track_state=(display_state or {}).get("track_state") if isinstance(display_state, dict) else None,
        vehicle_labels_csv=vehicle_labels,
        custom_model_path=custom_model_path,
        overlay_on_client=overlay_on_client,
    )
    if err:
        return err
    assert det is not None
    return finalize_analysis(
        det,
        frame_bgr,
        roi_polygons,
        mock_emergency=mock_emergency,
        esp32_sim_zone=esp32_sim_zone,
        esp32_confirm_state=esp32_confirm_state,
        now_s=now_s,
        overlay_on_client=overlay_on_client,
    )


def analyze_from_candidates(
    frame_bgr: np.ndarray,
    candidates: list[dict[str, Any]],
    roi_polygons: list[list[tuple[int, int]]],
    *,
    mock_emergency: bool = False,
    esp32_sim_zone: int = 0,
    draw_confidence: bool = True,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
    display_mode: str = "normal_detection",
    show_speed_overlays: bool = True,
    display_state: dict[str, Any] | None = None,
    esp32_confirm_state: dict[str, Any] | None = None,
    now_s: float | None = None,
    overlay_on_client: bool = False,
) -> dict[str, Any]:
    """Analyse à partir de détections précalculées (mode cache optimisé)."""
    draw_boxes = not overlay_on_client or (display_mode or "").strip().lower() in ("congestion_heatmap",)
    det = detect_from_candidates(
        frame_bgr,
        candidates,
        roi_polygons,
        draw_confidence=True,
        min_confidence=min_confidence,
        min_box_area=min_box_area,
        display_mode=display_mode,
        show_speed_overlays=show_speed_overlays,
        track_state=(display_state or {}).get("track_state") if isinstance(display_state, dict) else None,
        draw_boxes=draw_boxes,
    )
    return finalize_analysis(
        det,
        frame_bgr,
        roi_polygons,
        mock_emergency=mock_emergency,
        esp32_sim_zone=esp32_sim_zone,
        esp32_confirm_state=esp32_confirm_state,
        now_s=now_s,
        overlay_on_client=overlay_on_client,
    )
