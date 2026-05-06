from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import threading

import cv2
import numpy as np
from django.conf import settings

from .fight_classifier import _yt_dlp_download_capped

_weapon_model = None
_weapon_model_lock = threading.Lock()
_weapon_models_by_path: dict[str, object] = {}


def get_weapon_test_model(model_path: str | None = None):
    global _weapon_model
    with _weapon_model_lock:
        path = (model_path or getattr(settings, "WEAPON_TEST_MODEL_PATH", "") or "").strip()
        if not path:
            # Direct simple model: YOLOv8 pretrained detector.
            path = "yolov8n.pt"
        if path.endswith(".pt") and ("/" in path or "\\" in path):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Modele introuvable: {path}")
        if model_path is None:
            if _weapon_model is None:
                from ultralytics import YOLO

                _weapon_model = YOLO(path)
            return _weapon_model
        existing = _weapon_models_by_path.get(path)
        if existing is not None:
            return existing
        from ultralytics import YOLO

        loaded = YOLO(path)
        _weapon_models_by_path[path] = loaded
        return loaded


def _detect_weapons_in_frame(
    det_model,
    frame_bgr,
    target_labels: list[str] | tuple[str, ...] | None = None,
) -> tuple[list[dict], float]:
    conf = float(getattr(settings, "WEAPON_YOLO_CONF", 0.35))
    iou = float(getattr(settings, "WEAPON_YOLO_IOU", 0.45))
    imgsz = int(getattr(settings, "WEAPON_YOLO_IMGSZ", 640))
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
        return detections, max_conf
    cls_t = r.boxes.cls
    conf_t = r.boxes.conf
    xyxy_t = r.boxes.xyxy
    cls_arr = cls_t.cpu().numpy() if hasattr(cls_t, "cpu") else np.asarray(cls_t)
    conf_arr = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
    xyxy_arr = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)
    configured_labels = (
        target_labels
        if target_labels is not None
        else getattr(settings, "WEAPON_TEST_TARGET_LABELS", ["knife"])
    )
    wanted = {str(x).strip().lower() for x in configured_labels if str(x).strip()}
    use_filter = bool(wanted)
    for i in range(len(r.boxes)):
        cls_id = int(cls_arr[i])
        cf = float(conf_arr[i])
        name = str(det_model.names.get(cls_id, cls_id))
        name_l = name.strip().lower()
        if use_filter and name_l not in wanted:
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
    return detections, max_conf


def _draw_weapon_markers(frame_bgr, detections: list[dict], min_conf: float) -> np.ndarray:
    out = frame_bgr.copy()
    for d in detections:
        conf = float(d.get("conf", 0.0))
        if conf < min_conf:
            continue
        b = d.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        color = (0, 255, 255)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{d.get('label', 'weapon')} {conf*100:.0f}%"
        cv2.putText(
            out,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _crop_detection_region(
    frame_bgr: np.ndarray,
    bbox: list[float],
    pad: float,
) -> np.ndarray | None:
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad
    py = bh * pad
    ix1 = int(max(0, x1 - px))
    iy1 = int(max(0, y1 - py))
    ix2 = int(min(w, x2 + px))
    iy2 = int(min(h, y2 + py))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return frame_bgr[iy1:iy2, ix1:ix2].copy()


def _maybe_save_knife_crops(
    frame_bgr: np.ndarray,
    detections: list[dict],
    min_conf: float,
    crops_dir: Path | None,
    frame_idx: int,
    t_sec: float,
    out_list: list[dict],
    max_crops: int,
) -> None:
    if crops_dir is None or len(out_list) >= max_crops:
        return
    crops_dir.mkdir(parents=True, exist_ok=True)
    pad = float(getattr(settings, "WEAPON_TEST_CROP_PADDING", 0.18))
    for j, d in enumerate(detections):
        if len(out_list) >= max_crops:
            break
        cf = float(d.get("conf", 0.0))
        if cf < min_conf:
            continue
        b = d.get("bbox")
        if not b:
            continue
        crop = _crop_detection_region(frame_bgr, b, pad)
        if crop is None or crop.size == 0:
            continue
        fname = f"f{frame_idx:06d}_{j:02d}_{cf:.2f}.jpg"
        full = crops_dir / fname
        if cv2.imwrite(str(full), crop):
            rel = full.relative_to(Path(settings.MEDIA_ROOT))
            out_list.append(
                {
                    "rel_path": rel.as_posix(),
                    "t": round(float(t_sec), 2),
                    "conf": round(cf, 4),
                    "label": str(d.get("label", "knife")),
                }
            )


def _transcode_for_browser(mp4_path: Path) -> Path:
    """
    Force H.264 + yuv420p + faststart for reliable HTML5 playback.
    Keeps the original file if ffmpeg is unavailable or transcoding fails.
    """
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


def scan_video_for_weapons(
    cap,
    video_label: str = "Video",
    annotated_output_path: Path | None = None,
    crops_output_dir: Path | None = None,
    model_path: str | None = None,
    target_labels: list[str] | tuple[str, ...] | None = None,
    scan_stride: int | None = None,
    max_frames: int | None = None,
    timeline_max_points: int | None = None,
    alert_min_conf: float | None = None,
    draw_min_conf: float | None = None,
    transcode_output: bool = True,
) -> dict:
    model = get_weapon_test_model(model_path=model_path)
    stride = max(
        1,
        int(
            scan_stride
            if scan_stride is not None
            else getattr(settings, "WEAPON_TEST_SCAN_STRIDE", 2)
        ),
    )
    max_frames = int(
        max_frames
        if max_frames is not None
        else getattr(settings, "WEAPON_TEST_MAX_FRAMES", 20000)
    )
    timeline_max = int(
        timeline_max_points
        if timeline_max_points is not None
        else getattr(settings, "WEAPON_TEST_TIMELINE_MAX_POINTS", 800)
    )
    alert_thr = float(
        alert_min_conf
        if alert_min_conf is not None
        else getattr(settings, "WEAPON_ALERT_MIN_CONF", 0.82)
    )
    draw_thr = float(
        draw_min_conf
        if draw_min_conf is not None
        else getattr(settings, "WEAPON_DRAW_MIN_CONF", alert_thr)
    )
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
    weapon_frames = 0
    max_conf_global = 0.0
    details_tail: list[dict] = []
    timeline: list[dict] = []
    max_crops = int(getattr(settings, "WEAPON_TEST_MAX_CROP_SNAPSHOTS", 24))
    knife_crops: list[dict] = []

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
        dets, frame_max = _detect_weapons_in_frame(model, frame, target_labels=target_labels)
        max_conf_global = max(max_conf_global, frame_max)
        has_weapon = bool(dets and frame_max >= alert_thr)
        if has_weapon:
            weapon_frames += 1
            _maybe_save_knife_crops(
                frame,
                dets,
                draw_thr,
                crops_output_dir,
                frame_idx,
                t,
                knife_crops,
                max_crops,
            )
        if writer is not None:
            writer.write(_draw_weapon_markers(frame, dets, min_conf=draw_thr))
        compact = ", ".join(f"{x['label']} {x['conf'] * 100:.0f}%" for x in dets[:4]) if dets else ""
        row = {
            "t": round(float(t), 2),
            "has_weapon": has_weapon,
            "weapon_max": round(frame_max, 4),
            "weapon_compact": compact,
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

    verdict = "knife" if weapon_frames > 0 else "no_knife"
    if writer is not None:
        writer.release()

    annotated_video_path = None
    if annotated_output_path is not None and annotated_output_path.is_file():
        if annotated_output_path.stat().st_size > 0:
            playable = (
                _transcode_for_browser(annotated_output_path)
                if transcode_output
                else annotated_output_path
            )
            annotated_video_path = str(playable)

    return {
        "video_label": video_label,
        "frames_analyzed": frames_analyzed,
        "weapon_frames": weapon_frames,
        "weapon_ratio": round(weapon_frames / max(1, frames_analyzed), 4),
        "max_conf": round(max_conf_global, 4),
        "verdict": verdict,
        "verdict_label": "Knife detecte" if verdict == "knife" else "Aucun knife detecte",
        "details": details_tail,
        "timeline": timeline,
        "annotated_video_path": annotated_video_path,
        "knife_crops": knife_crops,
    }


def analyze_local_video_path(
    path: str | Path,
    annotated_output_path: str | Path | None = None,
    crops_output_dir: str | Path | None = None,
    model_path: str | None = None,
    target_labels: list[str] | tuple[str, ...] | None = None,
    scan_stride: int | None = None,
    max_frames: int | None = None,
    timeline_max_points: int | None = None,
    alert_min_conf: float | None = None,
    draw_min_conf: float | None = None,
    transcode_output: bool = True,
) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"error": "Fichier video introuvable."}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "Impossible d'ouvrir la video (format non supporte ou fichier corrompu)."}
    try:
        ann = Path(annotated_output_path) if annotated_output_path else None
        cdir = Path(crops_output_dir) if crops_output_dir else None
        return scan_video_for_weapons(
            cap,
            video_label=path.name,
            annotated_output_path=ann,
            crops_output_dir=cdir,
            model_path=model_path,
            target_labels=target_labels,
            scan_stride=scan_stride,
            max_frames=max_frames,
            timeline_max_points=timeline_max_points,
            alert_min_conf=alert_min_conf,
            draw_min_conf=draw_min_conf,
            transcode_output=transcode_output,
        )
    finally:
        cap.release()


def analyze_youtube_url(
    url: str,
    annotated_output_path: str | Path | None = None,
    crops_output_dir: str | Path | None = None,
    model_path: str | None = None,
    target_labels: list[str] | tuple[str, ...] | None = None,
    scan_stride: int | None = None,
    max_frames: int | None = None,
    timeline_max_points: int | None = None,
    alert_min_conf: float | None = None,
    draw_min_conf: float | None = None,
    transcode_output: bool = True,
) -> dict:
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
        cdir = Path(crops_output_dir) if crops_output_dir else None
        return scan_video_for_weapons(
            cap,
            video_label="YouTube",
            annotated_output_path=ann,
            crops_output_dir=cdir,
            model_path=model_path,
            target_labels=target_labels,
            scan_stride=scan_stride,
            max_frames=max_frames,
            timeline_max_points=timeline_max_points,
            alert_min_conf=alert_min_conf,
            draw_min_conf=draw_min_conf,
            transcode_output=transcode_output,
        )
    finally:
        cap.release()
        try:
            parent = path.parent
            path.unlink(missing_ok=True)
            shutil.rmtree(parent, ignore_errors=True)
        except OSError:
            pass


def analyze_image_path(
    path: str | Path,
    annotated_output_path: str | Path | None = None,
    crops_output_dir: str | Path | None = None,
    model_path: str | None = None,
    target_labels: list[str] | tuple[str, ...] | None = None,
) -> dict:
    path = Path(path)
    if not path.is_file():
        return {"error": "Image introuvable."}
    frame = cv2.imread(str(path))
    if frame is None:
        return {"error": "Impossible de lire l'image."}

    model = get_weapon_test_model(model_path=model_path)
    alert_thr = float(getattr(settings, "WEAPON_ALERT_MIN_CONF", 0.82))
    draw_thr = float(getattr(settings, "WEAPON_DRAW_MIN_CONF", alert_thr))
    dets, frame_max = _detect_weapons_in_frame(model, frame, target_labels=target_labels)
    has_weapon = bool(dets and frame_max >= alert_thr)

    annotated_image_path = None
    if annotated_output_path is not None:
        ann = Path(annotated_output_path)
        ann.parent.mkdir(parents=True, exist_ok=True)
        drawn = _draw_weapon_markers(frame, dets, min_conf=draw_thr)
        if cv2.imwrite(str(ann), drawn):
            annotated_image_path = str(ann)

    knife_crops: list[dict] = []
    max_crops = int(getattr(settings, "WEAPON_TEST_MAX_CROP_SNAPSHOTS", 24))
    cdir = Path(crops_output_dir) if crops_output_dir else None
    _maybe_save_knife_crops(
        frame,
        dets,
        draw_thr,
        cdir,
        0,
        0.0,
        knife_crops,
        max_crops,
    )

    compact = ", ".join(f"{x['label']} {x['conf'] * 100:.0f}%" for x in dets[:8]) if dets else ""
    return {
        "video_label": path.name,
        "is_image": True,
        "frames_analyzed": 1,
        "weapon_frames": 1 if has_weapon else 0,
        "weapon_ratio": 1.0 if has_weapon else 0.0,
        "max_conf": round(frame_max, 4),
        "verdict": "knife" if has_weapon else "no_knife",
        "verdict_label": "Knife detecte" if has_weapon else "Aucun knife detecte",
        "details": [
            {
                "t": 0.0,
                "has_weapon": has_weapon,
                "weapon_max": round(frame_max, 4),
                "weapon_compact": compact,
            }
        ],
        "timeline": [],
        "annotated_image_path": annotated_image_path,
        "knife_crops": knife_crops,
    }
