from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import threading
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
from django.conf import settings

from .fight_classifier import _yt_dlp_download_capped, is_youtube_url

_road_model = None
_road_model_lock = threading.Lock()


def get_road_damage_model():
    global _road_model
    with _road_model_lock:
        if _road_model is None:
            from ultralytics import YOLO

            path = (getattr(settings, "ROAD_DAMAGE_MODEL_PATH", "") or "").strip()
            if not path:
                path = str(Path(settings.BASE_DIR) / "best (2).pt")
            if path.endswith(".pt") and ("/" in path or "\\" in path):
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Modele introuvable: {path}")
            _road_model = YOLO(path)
        return _road_model


def _predict_damages_in_frame(det_model, frame_bgr) -> tuple[object, list[dict], float]:
    conf = float(getattr(settings, "ROAD_DAMAGE_YOLO_CONF", 0.25))
    iou = float(getattr(settings, "ROAD_DAMAGE_YOLO_IOU", 0.45))
    imgsz = int(getattr(settings, "ROAD_DAMAGE_YOLO_IMGSZ", 640))
    r = det_model.predict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )[0]
    detections: list[dict] = []
    max_conf = 0.0
    if r.boxes is None or len(r.boxes) == 0:
        return r, detections, max_conf
    cls_t = r.boxes.cls
    conf_t = r.boxes.conf
    xyxy_t = r.boxes.xyxy
    cls_arr = cls_t.cpu().numpy() if hasattr(cls_t, "cpu") else np.asarray(cls_t)
    conf_arr = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
    xyxy_arr = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)

    target_labels = [
        str(x).strip().lower()
        for x in getattr(settings, "ROAD_DAMAGE_TARGET_LABELS", [])
        if str(x).strip()
    ]
    wanted = set(target_labels)

    for i in range(len(r.boxes)):
        cls_id = int(cls_arr[i])
        cf = float(conf_arr[i])
        name = str(det_model.names.get(cls_id, cls_id))
        name_l = name.strip().lower()
        if wanted and name_l not in wanted:
            continue
        max_conf = max(max_conf, cf)
        x1, y1, x2, y2 = [float(v) for v in xyxy_arr[i]]
        detections.append(
            {
                "label": name,
                "conf": round(cf, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return r, detections, max_conf


def _draw_damage_masks(result_obj, frame_bgr, min_conf: float) -> np.ndarray:
    # Keep notebook-like rendering: Ultralytics plot() overlays segmentation masks.
    boxes = getattr(result_obj, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return frame_bgr
    conf_t = boxes.conf
    conf_arr = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
    keep_idx = [i for i, c in enumerate(conf_arr) if float(c) >= min_conf]
    if not keep_idx:
        return frame_bgr
    return result_obj.plot()


def _transcode_for_browser(mp4_path: Path) -> Path:
    out_path = mp4_path.with_name(f"{mp4_path.stem}_web.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        pass
    return mp4_path


def scan_video_for_road_damage(
    cap,
    video_label: str = "Video",
    annotated_output_path: Path | None = None,
) -> dict:
    model = get_road_damage_model()
    stride = max(1, int(getattr(settings, "ROAD_DAMAGE_SCAN_STRIDE", 2)))
    max_frames = int(getattr(settings, "ROAD_DAMAGE_MAX_FRAMES", 20000))
    timeline_max = int(getattr(settings, "ROAD_DAMAGE_TIMELINE_MAX_POINTS", 800))
    alert_thr = float(getattr(settings, "ROAD_DAMAGE_ALERT_MIN_CONF", 0.5))
    draw_thr = float(getattr(settings, "ROAD_DAMAGE_DRAW_MIN_CONF", alert_thr))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    writer = None
    if annotated_output_path is not None:
        annotated_output_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if w > 0 and h > 0:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_output_path), fourcc, max(1.0, fps), (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    frames_analyzed = 0
    damage_frames = 0
    max_conf_global = 0.0
    details_tail: list[dict] = []
    timeline: list[dict] = []

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % stride != 0:
            if writer is not None:
                writer.write(frame)
            frame_idx += 1
            continue

        frame_idx += 1
        frames_analyzed += 1
        t = (frame_idx - 1) / fps if fps > 0 else 0.0
        pred, dets, frame_max = _predict_damages_in_frame(model, frame)
        max_conf_global = max(max_conf_global, frame_max)
        has_damage = bool(dets and frame_max >= alert_thr)
        if has_damage:
            damage_frames += 1
        if writer is not None:
            writer.write(_draw_damage_masks(pred, frame, min_conf=draw_thr))

        compact = ", ".join(f"{x['label']} {x['conf'] * 100:.0f}%" for x in dets[:4]) if dets else ""
        row = {
            "t": round(float(t), 2),
            "has_damage": has_damage,
            "damage_max": round(frame_max, 4),
            "damage_compact": compact,
        }
        timeline.append(row)
        details_tail.append(row)
        if len(details_tail) > 64:
            details_tail.pop(0)

    if frames_analyzed == 0:
        if writer is not None:
            writer.release()
            try:
                annotated_output_path.unlink(missing_ok=True)
            except OSError:
                pass
        return {"error": "Impossible d'analyser la video (flux vide ou non lisible)."}

    if len(timeline) > timeline_max:
        idx = np.linspace(0, len(timeline) - 1, timeline_max, dtype=int)
        timeline = [timeline[i] for i in idx]

    verdict = "damage" if damage_frames > 0 else "no_damage"
    if writer is not None:
        writer.release()

    annotated_video_path = None
    if annotated_output_path is not None and annotated_output_path.is_file():
        if annotated_output_path.stat().st_size > 0:
            playable = _transcode_for_browser(annotated_output_path)
            annotated_video_path = str(playable)

    return {
        "video_label": video_label,
        "frames_analyzed": frames_analyzed,
        "damage_frames": damage_frames,
        "damage_ratio": round(damage_frames / max(1, frames_analyzed), 4),
        "max_conf": round(max_conf_global, 4),
        "verdict": verdict,
        "verdict_label": "Road damage detecte" if verdict == "damage" else "Aucun road damage detecte",
        "details": details_tail,
        "timeline": timeline,
        "annotated_video_path": annotated_video_path,
    }


def analyze_local_video_path(path: str | Path, annotated_output_path: str | Path | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"error": "Fichier video introuvable."}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "Impossible d'ouvrir la video (format non supporte ou fichier corrompu)."}
    try:
        ann = Path(annotated_output_path) if annotated_output_path else None
        return scan_video_for_road_damage(
            cap,
            video_label=path.name,
            annotated_output_path=ann,
        )
    finally:
        cap.release()


def _download_video_from_url(url: str) -> Path:
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
    out_dir = Path(settings.MEDIA_ROOT) / "road_damage_url_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"download_{threading.get_ident()}_{Path(parsed.path).stem or 'video'}.mp4"
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(out_file, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    return out_file


def analyze_video_url(url: str, annotated_output_path: str | Path | None = None) -> dict:
    if is_youtube_url(url):
        return analyze_youtube_url(url, annotated_output_path=annotated_output_path)
    local_path = _download_video_from_url(url)
    try:
        return analyze_local_video_path(local_path, annotated_output_path=annotated_output_path)
    finally:
        try:
            local_path.unlink(missing_ok=True)
        except OSError:
            pass


def analyze_youtube_url(url: str, annotated_output_path: str | Path | None = None) -> dict:
    path = _yt_dlp_download_capped(url)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        try:
            parent = path.parent
            path.unlink(missing_ok=True)
            shutil.rmtree(parent, ignore_errors=True)
        except OSError:
            pass
        return {"error": "Ouverture video impossible"}
    try:
        ann = Path(annotated_output_path) if annotated_output_path else None
        return scan_video_for_road_damage(
            cap,
            video_label="URL Video",
            annotated_output_path=ann,
        )
    finally:
        cap.release()
        try:
            parent = path.parent
            path.unlink(missing_ok=True)
            shutil.rmtree(parent, ignore_errors=True)
        except OSError:
            pass


def analyze_image_path(path: str | Path, annotated_output_path: str | Path | None = None) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"error": "Image introuvable."}
    frame = cv2.imread(str(path))
    if frame is None:
        return {"error": "Impossible de lire l'image."}

    model = get_road_damage_model()
    alert_thr = float(getattr(settings, "ROAD_DAMAGE_ALERT_MIN_CONF", 0.5))
    draw_thr = float(getattr(settings, "ROAD_DAMAGE_DRAW_MIN_CONF", alert_thr))
    pred, dets, frame_max = _predict_damages_in_frame(model, frame)
    has_damage = bool(dets and frame_max >= alert_thr)

    annotated_image_path = None
    if annotated_output_path is not None:
        ann = Path(annotated_output_path)
        ann.parent.mkdir(parents=True, exist_ok=True)
        drawn = _draw_damage_masks(pred, frame, min_conf=draw_thr)
        if cv2.imwrite(str(ann), drawn):
            annotated_image_path = str(ann)

    compact = ", ".join(f"{x['label']} {x['conf'] * 100:.0f}%" for x in dets[:8]) if dets else ""
    return {
        "video_label": path.name,
        "is_image": True,
        "frames_analyzed": 1,
        "damage_frames": 1 if has_damage else 0,
        "damage_ratio": 1.0 if has_damage else 0.0,
        "max_conf": round(frame_max, 4),
        "verdict": "damage" if has_damage else "no_damage",
        "verdict_label": "Road damage detecte" if has_damage else "Aucun road damage detecte",
        "details": [
            {
                "t": 0.0,
                "has_damage": has_damage,
                "damage_max": round(frame_max, 4),
                "damage_compact": compact,
            }
        ],
        "timeline": [],
        "annotated_image_path": annotated_image_path,
    }


def analyze_image_url(url: str, annotated_output_path: str | Path | None = None) -> dict:
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
    out_dir = Path(settings.MEDIA_ROOT) / "road_damage_url_images"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in (".jpg", ".jpeg", ".png", ".webp"):
        suffix = ".jpg"
    out_file = out_dir / f"download_{threading.get_ident()}{suffix}"
    with requests.get(url, stream=True, timeout=40) as resp:
        resp.raise_for_status()
        with open(out_file, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    try:
        return analyze_image_path(out_file, annotated_output_path=annotated_output_path)
    finally:
        try:
            out_file.unlink(missing_ok=True)
        except OSError:
            pass
