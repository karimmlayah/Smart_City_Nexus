"""Sessions vidéo Traffic Nexus (live / cache optimisé) pour l’API Django."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
from django.conf import settings

from monitoring.traffic_nexus_inference import get_cached_yolo, parse_allowed_labels
from monitoring.traffic_nexus_paths import resolve_trusted_media_file
from monitoring.traffic_tdss.vehicle_detection import infer_candidates_multi
from monitoring.traffic_tdss.video_processing import resolve_video_source

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_sessions: Dict[str, "LiveSession"] = {}

SESSION_TTL_S = 3600
MAX_PRECOMPUTE_FRAMES = 250


@dataclass
class LiveSession:
    cap: Optional[cv2.VideoCapture]
    source_label: str
    temp_path: Optional[str]
    frame_idx: int
    total_frames: int
    fps: float
    optimized: bool
    precomputed: Dict[int, List[Dict[str, Any]]]
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self.temp_path and Path(self.temp_path).is_file():
            try:
                Path(self.temp_path).unlink(missing_ok=True)
            except Exception:
                pass
            self.temp_path = None


def _cleanup_stale() -> None:
    now = time.time()
    dead = [sid for sid, s in _sessions.items() if now - s.last_access > SESSION_TTL_S]
    for sid in dead:
        _sessions.pop(sid, None).release()


def resolve_capture_source(
    *,
    youtube_url: str,
    local_file_path: str,
    temp_upload_path: Optional[str],
) -> tuple[Optional[cv2.VideoCapture], Optional[str], str]:
    """Retourne (cap, erreur_message, label_debug)."""
    if temp_upload_path:
        cap = cv2.VideoCapture(temp_upload_path)
        if cap.isOpened():
            return cap, None, Path(temp_upload_path).name
        cap.release()
        return None, "Impossible d’ouvrir le fichier téléversé.", ""

    lp = (local_file_path or "").strip()
    if lp:
        trusted = resolve_trusted_media_file(lp)
        if not trusted:
            return None, "Chemin local refusé ou fichier introuvable (restez sous le dossier projet).", ""
        cap = cv2.VideoCapture(trusted)
        if cap.isOpened():
            return cap, None, trusted
        cap.release()
        return None, "Impossible d’ouvrir la vidéo locale.", ""

    yt = (youtube_url or "").strip()
    if yt:
        try:
            resolved = resolve_video_source(yt)
        except Exception as exc:
            return None, str(exc), ""
        cap = cv2.VideoCapture(resolved)
        if cap.isOpened():
            return cap, None, yt
        cap.release()
        return None, "Impossible d’ouvrir le flux YouTube résolu.", ""

    return None, "Indiquez une URL YouTube, un chemin local ou téléversez un fichier.", ""


def _precompute_candidates(
    cap: cv2.VideoCapture,
    *,
    models: List,
    allowed_labels: Optional[Set[str]],
    imgsz: int,
    max_det: int,
    frame_step: int,
) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    frame_idx = 0
    processed = 0
    while processed < MAX_PRECOMPUTE_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % max(1, frame_step) != 0:
            continue
        out[frame_idx] = infer_candidates_multi(
            models=models,
            frame=frame,
            allowed_labels=allowed_labels,
            imgsz=imgsz,
            max_det=max_det,
        )
        processed += 1
    return out


def create_session(
    *,
    youtube_url: str = "",
    local_file_path: str = "",
    temp_upload_path: Optional[str] = None,
    processing_mode: str = "live",
    mode: str = "vehicle",
    model_key: str = "yolov8n",
    dual_models: bool = False,
    conf: float = 0.25,
    imgsz: int = 640,
    max_det: int = 80,
    frame_skip: int = 1,
    vehicle_labels: str = "",
    custom_model_path: str = "",
    turbo_local: bool = False,
) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """
    Ouvre la vidéo, optionnellement précalcule les candidats (optimisé).
    Retourne (session_id, erreur, meta).
    """
    _cleanup_stale()
    cap, err, label = resolve_capture_source(
        youtube_url=youtube_url,
        local_file_path=local_file_path,
        temp_upload_path=temp_upload_path,
    )
    if err or cap is None:
        return None, err or "Source invalide", None, None

    # Turbo local playback: smaller decode buffer can reduce latency on disk/upload captures (backend-dependent).
    lp = (local_file_path or "").strip()
    is_file_capture = bool(lp or temp_upload_path)
    if turbo_local and is_file_capture and cap is not None:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    paths = getattr(settings, "TRAFFIC_NEXUS_MODEL_PATHS", {})
    vehicle_mode = (mode or "vehicle").lower() in ("vehicle", "vehicles", "counting")
    allowed = parse_allowed_labels(vehicle_labels, vehicle_mode)

    from monitoring.traffic_nexus_paths import resolve_trusted_model_weights

    custom_resolved = resolve_trusted_model_weights(custom_model_path) if custom_model_path else None

    p_cong = (paths.get("yolo_congestion") or paths.get("congestion") or "").strip()

    models: List = []
    try:
        if vehicle_mode and dual_models:
            p_best_dual = custom_resolved or (paths.get("best") or "")
            p_yolo = paths.get("yolov8n") or ""
            if not (p_best_dual and p_yolo):
                cap.release()
                return None, "Double modèle : best.pt et yolov8n.pt requis.", None, None
            models = [get_cached_yolo(p_best_dual), get_cached_yolo(p_yolo)]
            for m in models:
                m.overrides["conf"] = conf
        elif not vehicle_mode:
            path_cong = custom_resolved or p_cong
            if not path_cong:
                cap.release()
                return None, "Modèle congestion (yolo_congestion.pt) introuvable.", None, None
            m = get_cached_yolo(path_cong)
            m.overrides["conf"] = conf
            models = [m]
        else:
            path = custom_resolved or (paths.get(model_key) or paths.get("yolov8n") or "")
            if not path:
                cap.release()
                return None, "Modèle YOLO introuvable.", None, None
            m = get_cached_yolo(path)
            m.overrides["conf"] = conf
            models = [m]
    except Exception as exc:
        cap.release()
        return None, str(exc), None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    wants_opt = (processing_mode or "live").lower() in ("optimized", "cache", "auto-cache")
    # Flux sans durée connue (YouTube) : pas de précalcul fiable (pas de seek).
    can_precompute = wants_opt and total_frames > 0

    precomputed: Dict[int, List[Dict[str, Any]]] = {}
    if can_precompute and len(models) >= 1:
        precomputed = _precompute_candidates(
            cap,
            models=models,
            allowed_labels=allowed,
            imgsz=imgsz,
            max_det=max_det,
            frame_step=max(1, frame_skip),
        )
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ok0, fr0 = cap.read()
    if not ok0 or fr0 is None:
        cap.release()
        return None, "Impossible de lire la première image de la vidéo.", None, None

    sid = uuid.uuid4().hex
    sess = LiveSession(
        cap=cap,
        source_label=label,
        temp_path=temp_upload_path,
        frame_idx=1,
        total_frames=total_frames,
        fps=fps,
        optimized=can_precompute and len(precomputed) > 0,
        precomputed=precomputed,
    )
    with _lock:
        _sessions[sid] = sess

    meta = {
        "session_id": sid,
        "source": label,
        "optimized": sess.optimized,
        "optimized_requested": wants_opt,
        "optimized_skipped_stream": wants_opt and total_frames <= 0,
        "precomputed_frames": len(precomputed),
        "total_frames": total_frames,
        "fps": fps,
        "turbo_local": bool(turbo_local and is_file_capture),
    }
    return sid, None, meta, fr0


def read_next_frame(session_id: str, frame_skip: int) -> tuple[Optional[np.ndarray], bool, int]:
    """Lit la frame ; avance de frame_skip lectures. Retourne (frame, eof, frame_index)."""
    with _lock:
        sess = _sessions.get(session_id)
        if not sess or sess.cap is None:
            return None, True, 0
        sess.touch()
        cap = sess.cap
        frame = None
        last_idx = sess.frame_idx
        for _ in range(max(1, frame_skip)):
            ok, fr = cap.read()
            if not ok:
                return None, True, last_idx
            frame = fr
            sess.frame_idx += 1
            last_idx = sess.frame_idx
    return frame, False, last_idx


def get_session(session_id: str) -> Optional[LiveSession]:
    with _lock:
        s = _sessions.get(session_id)
        if s:
            s.touch()
        return s


def pop_session(session_id: str) -> None:
    with _lock:
        s = _sessions.pop(session_id, None)
    if s:
        s.release()


def get_precomputed_candidates(session_id: str, frame_idx: int) -> Optional[List[Dict[str, Any]]]:
    sess = get_session(session_id)
    if not sess or not sess.precomputed:
        return None
    return sess.precomputed.get(frame_idx)


def close_session(session_id: str) -> None:
    pop_session(session_id)
