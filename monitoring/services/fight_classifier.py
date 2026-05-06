"""
Analyse vidéo fusionnée : combat (YOLO classify, fight / nonfight)
+ armes (YOLO detect, gun / knife) sur les mêmes images.
"""

from __future__ import annotations

import base64
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from django.conf import settings

from .fight_backends import build_fight_backend

_models_fight_cache: dict[str, object] = {}
_models_weapon_cache: dict[str, object] = {}
_models_person_cache: dict[str, object] = {}
_models_fight_lock = threading.Lock()
_models_weapon_lock = threading.Lock()
_models_person_lock = threading.Lock()

YOUTUBE_HOST_RE = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/(watch\?|embed/|shorts/)|youtu\.be/)",
    re.I,
)


def is_youtube_url(url: str) -> bool:
    u = (url or "").strip()
    return bool(u and YOUTUBE_HOST_RE.search(u))


def extract_youtube_video_id(url: str) -> str | None:
    """Retourne l’ID vidéo pour embed (youtube.com / youtu.be / shorts)."""
    from urllib.parse import parse_qs, urlparse

    u = (url or "").strip()
    if not u:
        return None
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    if not is_youtube_url(u):
        return None
    p = urlparse(u)
    host = (p.netloc or "").lower().split(":")[0]
    path = (p.path or "").strip("/")

    if "youtu.be" in host:
        seg = path.split("/")[0] if path else ""
        return seg[:11] if len(seg) >= 6 else None

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        if path == "watch" or path.startswith("watch"):
            v = (parse_qs(p.query).get("v") or [None])[0]
            if v:
                return v[:11]
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] in ("embed", "v", "shorts", "live"):
            vid = parts[1]
            return vid[:11] if len(vid) >= 6 else None
    return None


def _friendly_ytdlp_error(raw_msg: str) -> str:
    msg = (raw_msg or "").strip()
    low = msg.lower()
    if "this video is not available" in low or "video unavailable" in low:
        return (
            "Cette video YouTube n'est pas disponible (supprimee, privee, geobloquee ou reservee a certains comptes). "
            "Essayez un autre lien public."
        )
    if "private video" in low:
        return "Cette video YouTube est privee. Utilisez une video publique."
    if "sign in to confirm your age" in low or "age-restricted" in low:
        return "Cette video est age-restricted. Utilisez une video publique sans restriction d'age."
    return msg[:800] if msg else "Erreur YouTube (yt-dlp)."


def _yt_dlp_base_args() -> list[str]:
    args = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings"]
    extractor_args = (
        getattr(settings, "FIGHT_YT_EXTRACTOR_ARGS", "")
        or "youtube:player_client=android,web"
    ).strip()
    if extractor_args:
        args.extend(["--extractor-args", extractor_args])
    js_runtimes = (getattr(settings, "FIGHT_YT_JS_RUNTIMES", "") or "").strip()
    if js_runtimes:
        args.extend(["--js-runtimes", js_runtimes])
    return args


def get_fight_model(yolo_weights: str | None = None):  # retour backend YOLO / Keras / PyTorch
    """
    Cache YOLO par chaîne exactement comme passée à Ultralytics YOLO()
    (chemin absolu projet ou hub « yolov8n.pt »).
    """
    w = (
        str(yolo_weights).strip()
        if yolo_weights
        else str(getattr(settings, "FIGHT_CLASSIFIER_PATH", "") or "")
    ).strip()
    if not w:
        raise FileNotFoundError("Modèle combat : chemin vide.")
    with _models_fight_lock:
        if w in _models_fight_cache:
            return _models_fight_cache[w]
        bk = build_fight_backend(w)
        _models_fight_cache[w] = bk
        return bk


def get_weapon_model(yolo_weights: str | None = None):
    w = (
        str(yolo_weights).strip()
        if yolo_weights
        else str(getattr(settings, "WEAPON_DETECTOR_PATH", "") or "")
    ).strip()
    if not w:
        raise FileNotFoundError("Modèle armes : chemin vide.")
    with _models_weapon_lock:
        if w in _models_weapon_cache:
            return _models_weapon_cache[w]
        from ultralytics import YOLO

        m = YOLO(w)
        _models_weapon_cache[w] = m
        return m


def get_person_detector(weights: str | None = None):
    """YOLO detect pré-entraîné (ex. COCO) pour la classe « person »."""
    w = (
        str(weights).strip()
        if weights
        else str(getattr(settings, "FIGHT_PERSON_MODEL_PATH", "yolov8n.pt") or "yolov8n.pt")
    ).strip()
    if not w:
        w = "yolov8n.pt"
    with _models_person_lock:
        if w in _models_person_cache:
            return _models_person_cache[w]
        from ultralytics import YOLO

        m = YOLO(w)
        _models_person_cache[w] = m
        return m


def _bbox_name_is_person(raw: str | int) -> bool:
    return str(raw).strip().lower() == "person"


def _xyxy_iou(a: list | tuple, b: list | tuple) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in b[:4]]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter + 1e-9
    return float(inter / union)


def _clamp_xyxy(xyxy: list[float], fw: int, fh: int) -> list[float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(fw - 1), x1))
    x2 = max(0.0, min(float(fw), x2))
    y1 = max(0.0, min(float(fh - 1), y1))
    y2 = max(0.0, min(float(fh), y2))
    if x2 <= x1 + 1:
        x2 = min(float(fw), x1 + 2.0)
    if y2 <= y1 + 1:
        y2 = min(float(fh), y1 + 2.0)
    return [x1, y1, x2, y2]


def _expand_pad_xyxy(
    bbox: list[float],
    fw: int,
    fh: int,
    pad_frac: float,
) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    px = w * pad_frac
    py = h * pad_frac
    return _clamp_xyxy([x1 - px, y1 - py, x2 + px, y2 + py], fw, fh)


def _crop_bgr_roi(frame_bgr: np.ndarray, xyxy: list[float]) -> np.ndarray | None:
    fh, fw = frame_bgr.shape[:2]
    xy = _clamp_xyxy([float(v) for v in xyxy], fw, fh)
    x1, y1, x2, y2 = [int(round(v)) for v in xy]
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


def _detect_persons_in_frame(det_model, frame_bgr: np.ndarray) -> list[dict]:
    conf = float(getattr(settings, "FIGHT_PERSON_YOLO_CONF", 0.35))
    iou = float(getattr(settings, "WEAPON_YOLO_IOU", 0.45))
    imgsz = int(getattr(settings, "FIGHT_PERSON_YOLO_IMGSZ", 640))
    r = det_model.predict(
        source=frame_bgr,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        verbose=False,
    )[0]
    out: list[dict] = []
    if r.boxes is None or len(r.boxes) == 0:
        return out
    cls_t = r.boxes.cls
    conf_t = r.boxes.conf
    xyxy_t = r.boxes.xyxy
    cls_arr = cls_t.cpu().numpy() if hasattr(cls_t, "cpu") else np.asarray(cls_t)
    conf_arr = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
    xyxy_arr = xyxy_t.cpu().numpy() if hasattr(xyxy_t, "cpu") else np.asarray(xyxy_t)
    for i in range(len(r.boxes)):
        cls_id = int(cls_arr[i])
        cf = float(conf_arr[i])
        name = str(det_model.names.get(cls_id, cls_id))
        if not _bbox_name_is_person(name):
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy_arr[i]]
        out.append({"label": "person", "conf": cf, "bbox": [x1, y1, x2, y2]})
    return out


def _rank_persons_near_weapons(
    persons: list[dict],
    weapons_hi: list[dict],
    *,
    fw: int,
    max_n: int,
) -> list[dict]:
    """Priorise IoU avec l’arme, sinon meilleure confiance globale."""
    max_n = max(1, int(max_n))
    if not persons:
        return []
    if weapons_hi:
        scored: list[tuple[float, float, dict]] = []
        for p in persons:
            pb = p["bbox"]
            best = 0.0
            for w in weapons_hi:
                wb = w["bbox"]
                best = max(best, _xyxy_iou(pb, wb))
                cx = (wb[0] + wb[2]) / 2.0
                cy = (wb[1] + wb[3]) / 2.0
                if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]:
                    best = max(best, 0.2)
            scored.append((best, float(p["conf"]), p))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        with_overlap = [s[2] for s in scored if s[0] > 1e-6][:max_n]
        if len(with_overlap) >= max_n:
            return with_overlap
        merged = list(with_overlap)
        for _s, __, p in scored:
            if p not in merged:
                merged.append(p)
                if len(merged) >= max_n:
                    break
        return merged[:max_n]
    _ = fw
    return sorted(persons, key=lambda x: -float(x["conf"]))[:max_n]


def _encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 82) -> str | None:
    if img_bgr is None or img_bgr.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _resize_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0 or img.size == 0:
        return img
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (max_w, nh), interpolation=cv2.INTER_AREA)


def _yt_dlp_stream_url(url: str) -> str:
    fmt = getattr(
        settings,
        "FIGHT_YT_FORMAT",
        "best[height<=720][ext=mp4]/best[ext=mp4]/best[height<=720]/best",
    )
    cmd = _yt_dlp_base_args() + ["-f", fmt, "-g", url.strip()]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=int(getattr(settings, "FIGHT_YT_TIMEOUT_SECONDS", 120)),
    )
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(_friendly_ytdlp_error(msg))
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("http://") or line.startswith("https://"):
            return line
    raise RuntimeError("Aucune URL de flux trouvée (yt-dlp).")


def _yt_dlp_download_capped(url: str) -> Path:
    """Télécharge une partie de la vidéo (taille plafonnée) pour OpenCV si le flux direct échoue."""
    d = tempfile.mkdtemp(prefix="fight_yt_")
    out_tmpl = str(Path(d) / "vid.%(ext)s")
    cmd = _yt_dlp_base_args() + [
        "-f",
        getattr(settings, "FIGHT_YT_FALLBACK_FORMAT", "best[height<=480]/best"),
        "-o",
        out_tmpl,
        "--max-filesize",
        getattr(settings, "FIGHT_YT_MAX_FILESIZE", "120m"),
        url.strip(),
    ]
    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=int(getattr(settings, "FIGHT_YT_DOWNLOAD_TIMEOUT", 600)),
    )
    if r.returncode != 0:
        raise RuntimeError(_friendly_ytdlp_error(r.stderr or r.stdout or "Téléchargement yt-dlp impossible"))
    files = sorted(Path(d).glob("vid.*"))
    if not files:
        raise RuntimeError("Fichier vidéo introuvable après téléchargement.")
    return files[0]


def _predict_frame(fight_backend, frame_bgr) -> dict | None:
    """Backend unifié : Ultralytics, Keras .h5, PyTorch .pth."""
    return fight_backend.predict_prob_fight(frame_bgr)


def _detect_weapons_in_frame(det_model, frame_bgr) -> tuple[list[dict], float]:
    """Retourne (liste {label, conf}, confiance max)."""
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
    for i in range(len(r.boxes)):
        cls_id = int(cls_arr[i])
        cf = float(conf_arr[i])
        max_conf = max(max_conf, cf)
        name = str(det_model.names.get(cls_id, cls_id))
        x1, y1, x2, y2 = [float(v) for v in xyxy_arr[i]]
        detections.append(
            {
                "label": name,
                "conf": round(cf, 4),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return detections, max_conf


def _draw_fight_weapon_overlay(
    frame_bgr: np.ndarray,
    fight_data: dict | None,
    weapon_dets: list[dict],
    *,
    weapon_draw_min_conf: float,
    show_fight_banner: bool = True,
    person_dets: list[dict] | None = None,
    face_dets: list[dict] | None = None,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    if person_dets:
        for p in person_dets:
            b = p.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = [int(v) for v in b]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, max(x1 + 1, x2))
            y2 = min(h - 1, max(y1 + 1, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), (60, 220, 100), 2)
            pc = float(p.get("conf", 0.0))
            cv2.putText(
                out,
                f"person {pc*100:.0f}%",
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 255, 120),
                2,
                cv2.LINE_AA,
            )
    if show_fight_banner and fight_data:
        p_fight = float(fight_data.get("p_fight", 0.0))
        is_fight = int(fight_data.get("top1", 1)) == 0
        txt = f"FIGHT {p_fight*100:.1f}%"
        txt_color = (0, 70, 255) if is_fight else (80, 220, 120)
        cv2.rectangle(out, (14, 14), (min(w - 14, 290), 58), (10, 10, 10), -1)
        cv2.putText(
            out,
            txt,
            (24, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            txt_color,
            2,
            cv2.LINE_AA,
        )
    for d in weapon_dets:
        cf = float(d.get("conf", 0.0))
        if cf < weapon_draw_min_conf:
            continue
        b = d.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = [int(v) for v in b]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w - 1, max(x1 + 1, x2))
        y2 = min(h - 1, max(y1 + 1, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        label = f"{d.get('label', 'weapon')} {cf*100:.0f}%"
        cv2.putText(
            out,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    if face_dets:
        cyan = (255, 215, 100)
        for p in face_dets:
            b = p.get("bbox") or [0, 0, 0, 0]
            x1, y1, x2, y2 = [int(v) for v in b]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, max(x1 + 1, x2))
            y2 = min(h - 1, max(y1 + 1, y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), cyan, 3)
            cv2.putText(
                out,
                str(p.get("label") or "face"),
                (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cyan,
                2,
                cv2.LINE_AA,
            )
    return out


def _video_duration_sec(cap) -> float:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    if total > 0 and fps > 0:
        return float(total) / fps
    return 0.0


_analysis_alert_last_ts: dict[tuple[int, str], float] = {}


def _analysis_alert_cooldown_ok(source_id: int, channel: str, cooldown: float) -> bool:
    import time

    key = (source_id, channel)
    now = time.time()
    last = _analysis_alert_last_ts.get(key, 0.0)
    if now - last < cooldown:
        return False
    _analysis_alert_last_ts[key] = now
    return True


def _get_fight_analysis_source():
    from monitoring.models import VideoSource

    obj, _ = VideoSource.objects.get_or_create(
        name="Analyse combat (site web)",
        defaults={
            "source_type": VideoSource.SourceType.FILE,
            "is_active": False,
            "url": "",
        },
    )
    return obj


def _emit_fight_site_alert(source, t_sec, p_fight, top1conf, frame_bgr, video_label: str) -> bool:
    from django.core.files.base import ContentFile

    from monitoring.models import Alert

    cooldown = float(getattr(settings, "FIGHT_ALERT_COOLDOWN_SECONDS", 45))
    if not _analysis_alert_cooldown_ok(source.id, "fight", cooldown):
        return False
    msg = (
        f"Combat détecté — {video_label} · t≈{t_sec:.1f}s · "
        f"P(combat)={p_fight:.1%} (conf. {top1conf:.1%})"
    )
    alert = Alert.objects.create(
        source=source,
        alert_type=Alert.AlertType.FIGHT,
        message=msg[:500],
        person_count=None,
        confidence_max=float(p_fight),
        metadata={
            "t_sec": round(float(t_sec), 2),
            "video_label": video_label[:200],
            "p_fight": float(p_fight),
            "top1conf": float(top1conf),
        },
    )
    if frame_bgr is not None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            alert.thumbnail.save(f"fight_{alert.id}.jpg", ContentFile(buf.tobytes()), save=True)
    return True


def _emit_weapon_site_alert(
    source,
    t_sec: float,
    detections: list[dict],
    max_conf: float,
    frame_bgr,
    video_label: str,
) -> bool:
    from django.core.files.base import ContentFile

    from monitoring.models import Alert

    cooldown = float(getattr(settings, "WEAPON_ALERT_COOLDOWN_SECONDS", 45))
    if not _analysis_alert_cooldown_ok(source.id, "weapon", cooldown):
        return False
    parts = ", ".join(f"{d['label']} ({d['conf']:.0%})" for d in detections[:6])
    msg = f"Arme détectée — {video_label} · t≈{t_sec:.1f}s · {parts}"[:500]
    alert = Alert.objects.create(
        source=source,
        alert_type=Alert.AlertType.WEAPON,
        message=msg,
        person_count=len(detections),
        confidence_max=float(max_conf),
        metadata={
            "t_sec": round(float(t_sec), 2),
            "video_label": video_label[:200],
            "detections": detections[:12],
        },
    )
    if frame_bgr is not None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            alert.thumbnail.save(f"weapon_{alert.id}.jpg", ContentFile(buf.tobytes()), save=True)
    return True


def _cv2_browser_friendly_writer(path: Path, fps: float, frame_size: tuple[int, int]):
    """
    VideoWriter pour export annoté (.mp4 en général).

    Sur Windows, avc1 / H.264 via FFmpeg tente souvent OpenH264 (openh264*.dll) : échecs bruyants
    si la DLL est absente ou incompatible. On privilège donc **mp4v** (MPEG-4 Part 2),
    lisible dans la plupart des navigateurs sous Windows.
    """
    w, h = frame_size
    if w <= 0 or h <= 0:
        return None
    fps = max(1.0, float(fps))
    suffix = path.suffix.lower()
    explicit = getattr(settings, "FIGHT_ANNOTATED_VIDEO_CODECS", None)
    if isinstance(explicit, (list, tuple)) and explicit:
        codec_order = tuple((str(c).strip() + "    ")[:4] for c in explicit)
    elif sys.platform.startswith("win"):
        if suffix == ".avi":
            codec_order = ("MJPG", "XVID", "mp4v")
        else:
            # .mp4 + MJPG ouvre rarement ; évite H.264 (OpenH264) par défaut.
            codec_order = ("mp4v",)
    else:
        codec_order = ("avc1", "H264", "X264", "mp4v")
    writer = None
    for codec in codec_order:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if candidate.isOpened():
            writer = candidate
            break
        try:
            candidate.release()
        except Exception:
            pass
    if writer is None:
        logging.getLogger(__name__).warning(
            "Aucun VideoWriter disponible pour %s (essayé: %s). "
            "Sous Windows, installez un build OpenCV + codecs ou utilisez ffmpeg en post-traitement.",
            path.name,
            codec_order,
        )
    return writer


def scan_video_fight_full(
    cap,
    video_label: str = "Vidéo",
    annotated_output_path: str | Path | None = None,
    *,
    enable_fight: bool = True,
    enable_weapon: bool = True,
    fight_yolo_weights: str | None = None,
    weapon_yolo_weights: str | None = None,
) -> dict:
    """
    Parcourt toute la vidéo (stride), comme une caméra image par image.
    Sur chaque image : combat (classify) + armes (detect) fusionnés dans la timeline.
    Alertes Django si combat ou arme au-dessus des seuils.
    """
    if not enable_fight and not enable_weapon:
        return {"error": "Activez au moins un modèle (combat ou armes)."}

    model = None
    if enable_fight:
        try:
            model = get_fight_model(fight_yolo_weights)
        except Exception as exc:
            return {"error": f"Modèle combat : {exc}"}
        model.reset_sequence_buffer()

    weapon_model = None
    if enable_weapon:
        try:
            weapon_model = get_weapon_model(weapon_yolo_weights)
        except Exception as exc:
            return {"error": f"Modèle armes : {exc}"}

    stride = max(1, int(getattr(settings, "FIGHT_FULL_SCAN_STRIDE", 1)))
    max_frames = int(getattr(settings, "FIGHT_FULL_SCAN_MAX_FRAMES", 20000))
    alert_thr = float(getattr(settings, "FIGHT_ALERT_MIN_CONFIDENCE", 0.99))
    top1_thr = float(getattr(settings, "FIGHT_ALERT_MIN_TOP1_CONF", 0.99))
    timeline_max = int(getattr(settings, "FIGHT_TIMELINE_MAX_POINTS", 800))

    source = _get_fight_analysis_source()

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = None
    annotated_video_path = None
    weapon_draw_thr = float(
        getattr(settings, "FIGHT_WEAPON_DRAW_MIN_CONF", getattr(settings, "WEAPON_DRAW_MIN_CONF", 0.5))
    )
    if annotated_output_path is not None:
        out_path = Path(annotated_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if w > 0 and h > 0:
            writer = _cv2_browser_friendly_writer(
                out_path, max(1.0, fps), (w, h)
            )

    frame_idx = 0
    peak_p_fight = -1.0
    peak_frame_bgr: np.ndarray | None = None
    p_fights: list[float] = []
    timeline_raw: list[dict] = []
    details_tail: list[dict] = []
    alerts_created = 0
    weapon_alerts_created = 0
    n_top1_fight = 0
    weapon_hit_frames = 0

    weapon_alert_thr = float(getattr(settings, "WEAPON_ALERT_MIN_CONF", 0.82))

    dual_evidence_gallery: list[dict] = []
    snapshot_count = 0
    dual_p_thr = float(getattr(settings, "FIGHT_DUAL_EVIDENCE_FIGHT_P", 0.52))
    dual_w_thr = float(getattr(settings, "FIGHT_DUAL_EVIDENCE_WEAPON_CONF", 0.42))
    max_snap = max(1, int(getattr(settings, "FIGHT_DUAL_EVIDENCE_MAX_SNAPSHOTS", 16)))
    crop_pad = float(getattr(settings, "FIGHT_DUAL_CROP_PAD", 0.14))
    max_ppl = max(1, int(getattr(settings, "FIGHT_PERSON_MAX_PER_FRAME", 2)))
    person_model = None
    evidence_root: Path | None = None
    run_slug = ""
    if enable_fight and enable_weapon:
        try:
            person_model = get_person_detector()
            run_slug = uuid4().hex
            evidence_root = Path(settings.MEDIA_ROOT) / "fusion_dual_evidence" / run_slug
            evidence_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            person_model = None
            evidence_root = None
            run_slug = ""

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx % stride != 0:
            if writer is not None:
                writer.write(frame)
            frame_idx += 1
            continue

        if enable_fight and model is not None:
            d = _predict_frame(model, frame)
            frame_idx += 1
            if not d:
                continue
        else:
            d = {"p_fight": 0.0, "top1": 1, "top1conf": 1.0, "label": "nonfight"}
            frame_idx += 1

        t = (frame_idx - 1) / fps if fps > 0 else 0.0
        p_fight = float(d["p_fight"])
        if p_fight > peak_p_fight:
            peak_p_fight = p_fight
            peak_frame_bgr = frame.copy()
        if enable_fight:
            p_fights.append(p_fight)
            if int(d["top1"]) == 0:
                n_top1_fight += 1

        t_round = round(float(t), 2)
        weapon_dets: list[dict] = []
        weapon_max = 0.0
        weapon_compact = ""
        if weapon_model is not None:
            weapon_dets, weapon_max = _detect_weapons_in_frame(weapon_model, frame)
            if weapon_dets:
                weapon_hit_frames += 1
                weapon_compact = ", ".join(
                    f"{x['label']} {x['conf'] * 100:.0f}%" for x in weapon_dets[:4]
                )

        fh, fw = frame.shape[:2]
        weapons_hi = [
            x for x in weapon_dets if float(x.get("conf", 0.0)) >= dual_w_thr
        ]
        person_draw: list[dict] | None = None
        dual_highlight = False
        if (
            person_model is not None
            and evidence_root is not None
            and snapshot_count < max_snap
            and enable_fight
            and enable_weapon
            and int(d["top1"]) == 0
            and p_fight >= dual_p_thr
            and weapons_hi
        ):
            dual_highlight = True
            preds = _detect_persons_in_frame(person_model, frame)
            highlighted = _rank_persons_near_weapons(
                preds, weapons_hi, fw=fw, max_n=max_ppl
            )
            person_draw = highlighted
            mu = settings.MEDIA_URL.rstrip("/")
            crops_rel: list[str] = []
            crops_bgr_list: list[np.ndarray] = []
            if highlighted:
                for j, per in enumerate(highlighted):
                    pad_bb = _expand_pad_xyxy(per["bbox"], fw, fh, crop_pad)
                    crop = _crop_bgr_roi(frame, pad_bb)
                    if crop is None:
                        continue
                    crops_bgr_list.append(np.ascontiguousarray(crop))
                    fn = f"dual_{snapshot_count:03d}_p{j}.jpg"
                    fp = evidence_root / fn
                    cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    crops_rel.append(
                        f"{mu}/fusion_dual_evidence/{run_slug}/{fn}"
                    )
            if not crops_rel:
                pad_bb = _expand_pad_xyxy(weapons_hi[0]["bbox"], fw, fh, 0.55)
                crop = _crop_bgr_roi(frame, pad_bb)
                if crop is not None:
                    crops_bgr_list.append(np.ascontiguousarray(crop))
                    fn = f"dual_{snapshot_count:03d}_arme_ctx.jpg"
                    fp = evidence_root / fn
                    cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    crops_rel.append(
                        f"{mu}/fusion_dual_evidence/{run_slug}/{fn}"
                    )
            if crops_rel:
                snapshot_count += 1
                face_mx: list[dict] = []
                if getattr(settings, "FACE_RECOGNITION_ENABLED", True):
                    try:
                        from monitoring.services.face_recognition_gallery import (
                            face_matches_roundtrip_json,
                            match_faces_in_bgr_crops,
                        )

                        face_mx = face_matches_roundtrip_json(
                            match_faces_in_bgr_crops(crops_bgr_list)
                        )
                    except Exception:
                        face_mx = []
                dual_evidence_gallery.append(
                    {
                        "t": t_round,
                        "p_fight": round(float(p_fight), 4),
                        "weapon_max": round(float(weapon_max), 4),
                        "crop_urls": crops_rel,
                        "n_persons": len(highlighted),
                        "face_matches": face_mx,
                    }
                )

        row = {**d, "t": t_round, "weapon_compact": weapon_compact}
        details_tail.append(row)
        if len(details_tail) > 128:
            details_tail.pop(0)

        timeline_raw.append(
            {
                "t": t_round,
                "p_fight": round(p_fight, 4),
                "label": d["label"],
                "top1": d["top1"],
                "weapon_compact": weapon_compact,
                "weapon_max": round(weapon_max, 4),
                "dual_highlight": bool(dual_highlight),
            }
        )

        if enable_fight and (
            int(d["top1"]) == 0
            and p_fight >= alert_thr
            and float(d["top1conf"]) >= top1_thr
        ):
            if _emit_fight_site_alert(
                source, t, p_fight, float(d["top1conf"]), frame, video_label
            ):
                alerts_created += 1

        if weapon_model is not None and weapon_dets and weapon_max >= weapon_alert_thr:
            if _emit_weapon_site_alert(
                source, t, weapon_dets, weapon_max, frame, video_label
            ):
                weapon_alerts_created += 1
        if writer is not None:
            ann = _draw_fight_weapon_overlay(
                frame,
                d if enable_fight else None,
                weapon_dets,
                weapon_draw_min_conf=weapon_draw_thr,
                show_fight_banner=enable_fight,
                person_dets=person_draw,
            )
            writer.write(ann)

    duration_sec = _video_duration_sec(cap)
    if writer is not None:
        writer.release()
        out_path = Path(annotated_output_path)
        if out_path.is_file() and out_path.stat().st_size > 0:
            annotated_video_path = str(out_path)

    if not timeline_raw:
        return {"error": "Impossible d'analyser des images (flux vide ou modèle)."}

    face_snap: dict[str, Any] = {}
    face_scan_gallery: list[dict] = []
    if getattr(settings, "FACE_RECOGNITION_ENABLED", True) and peak_frame_bgr is not None:
        try:
            face_snap = _build_face_scan_gallery_for_frame(peak_frame_bgr, uuid4().hex)
            face_scan_gallery = face_snap.get("face_scan_gallery") or []
        except Exception:
            face_scan_gallery = []
            face_snap = {}

    timeline = timeline_raw
    if len(timeline) > timeline_max:
        idx = np.linspace(0, len(timeline) - 1, timeline_max, dtype=int)
        timeline = [timeline[i] for i in idx]

    if enable_fight and p_fights:
        avg_fight = float(np.mean(p_fights))
        verdict = "fight" if avg_fight >= 0.5 else "nonfight"
        verdict_label = "Combat probable" if verdict == "fight" else "Pas de combat détecté"
    elif not enable_fight:
        avg_fight = None
        verdict = "n/a"
        verdict_label = "Analyse combat désactivée"
    else:
        avg_fight = None
        verdict = "n/a"
        verdict_label = "Données combat indisponibles"

    return {
        "frames_analyzed": len(timeline_raw),
        "avg_p_fight": avg_fight,
        "fight_frames": n_top1_fight,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "details": details_tail[-64:],
        "timeline": timeline,
        "duration_sec": round(duration_sec, 1) if duration_sec > 0 else None,
        "fight_alerts_created": alerts_created,
        "weapon_alerts_created": weapon_alerts_created,
        "weapon_frames_detected": weapon_hit_frames,
        "weapon_model_used": bool(weapon_model),
        "fight_model_used": bool(model),
        "annotated_video_path": annotated_video_path,
        "enable_fight_used": enable_fight,
        "enable_weapon_used": enable_weapon,
        "dual_evidence_gallery": dual_evidence_gallery,
        "face_scan_gallery": face_scan_gallery,
        "faces_detected": (face_snap.get("faces_detected") if face_snap else None) or [],
        "face_matches": (face_snap.get("face_matches") if face_snap else None) or [],
        "face_gallery_empty": bool((face_snap or {}).get("gallery_empty")),
        "face_matching_disabled": bool((face_snap or {}).get("matching_disabled")),
    }


def analyze_local_video_path(
    path: str | Path,
    annotated_output_path: str | Path | None = None,
    *,
    enable_fight: bool = True,
    enable_weapon: bool = True,
    fight_yolo_weights: str | None = None,
    weapon_yolo_weights: str | None = None,
) -> dict:
    """Analyse un fichier vidéo sur disque (upload depuis le PC)."""
    path = Path(path)
    if not path.is_file():
        return {"error": "Fichier vidéo introuvable."}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "Impossible d'ouvrir la vidéo (format non pris en charge ou fichier corrompu)."}
    try:
        return scan_video_fight_full(
            cap,
            video_label=path.name,
            annotated_output_path=annotated_output_path,
            enable_fight=enable_fight,
            enable_weapon=enable_weapon,
            fight_yolo_weights=fight_yolo_weights,
            weapon_yolo_weights=weapon_yolo_weights,
        )
    finally:
        cap.release()


def analyze_youtube_url(
    url: str,
    annotated_output_path: str | Path | None = None,
    *,
    enable_fight: bool = True,
    enable_weapon: bool = True,
    fight_yolo_weights: str | None = None,
    weapon_yolo_weights: str | None = None,
) -> dict:
    stream_err = None
    try:
        stream = _yt_dlp_stream_url(url)
        cap = cv2.VideoCapture(stream)
        if cap.isOpened():
            try:
                out = scan_video_fight_full(
                    cap,
                    video_label="YouTube",
                    annotated_output_path=annotated_output_path,
                    enable_fight=enable_fight,
                    enable_weapon=enable_weapon,
                    fight_yolo_weights=fight_yolo_weights,
                    weapon_yolo_weights=weapon_yolo_weights,
                )
                if out.get("frames_analyzed", 0) > 0 and "error" not in out:
                    return out
            finally:
                cap.release()
    except Exception as e:
        stream_err = e

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
        msg = str(stream_err) if stream_err else "Ouverture vidéo impossible"
        return {"error": msg}

    try:
        out = scan_video_fight_full(
            cap,
            video_label="YouTube",
            annotated_output_path=annotated_output_path,
            enable_fight=enable_fight,
            enable_weapon=enable_weapon,
            fight_yolo_weights=fight_yolo_weights,
            weapon_yolo_weights=weapon_yolo_weights,
        )
        if "error" in out and stream_err:
            out["error"] = f"{out['error']} (flux : {stream_err})"
        return out
    finally:
        cap.release()
        try:
            parent = path.parent
            path.unlink(missing_ok=True)
            shutil.rmtree(parent, ignore_errors=True)
        except OSError:
            pass


def _build_face_scan_gallery_for_frame(
    frame_bgr: np.ndarray,
    run_slug: str,
) -> dict[str, Any]:
    """
    Extrait les visages (DeepFace détecteur + repli Haar), crops dans media/face_crops/<slug>/,
    correspondance Cosine contre la galerie Django.
    """
    try:
        from monitoring.services.face_biometric import build_face_biometric_snapshot
    except Exception:
        return {
            "face_scan_gallery": [],
            "faces_detected": [],
            "face_matches": [],
            "face_overlay_boxes": [],
            "gallery_empty": True,
            "matching_disabled": True,
        }

    return build_face_biometric_snapshot(frame_bgr, run_slug, subdir="face_crops")


def read_video_frame_bgr_at_time(
    video_path: str | Path,
    t_sec: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """
    Extrait une frame BGR à un instant (secondes), avec durée / fps si disponibles.
    """
    path = Path(video_path)
    meta: dict[str, Any] = {"duration_sec": None, "fps": None, "t_used": float(t_sec)}
    if not path.is_file():
        return None, meta
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, meta
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = (nframes / fps) if nframes > 0 and fps > 0 else None
        meta["fps"] = fps
        meta["duration_sec"] = duration
        t = max(0.0, float(t_sec))
        if duration is not None:
            t = min(t, max(0.0, duration - (1.0 / fps)))
        meta["t_used"] = t
        frame_idx = int(round(t * fps))
        if nframes > 0:
            frame_idx = min(max(0, frame_idx), max(0, nframes - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)
            ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            return None, meta
        return np.ascontiguousarray(frame), meta
    finally:
        cap.release()


def analyze_uploaded_image_fusion(
    path: str | Path | None = None,
    annotated_output_path: str | Path | None = None,
    *,
    frame_bgr: np.ndarray | None = None,
    t_round: float = 0.0,
    media_label: str | None = None,
    suppress_site_alerts: bool = False,
    enable_fight: bool = True,
    enable_weapon: bool = True,
    fight_yolo_weights: str | None = None,
    weapon_yolo_weights: str | None = None,
) -> dict:
    """
    Analyse fusion (combat + armes) sur une seule image, compatible avec les clés résultat « vidéo »
    (timeline à 1 point, dual_evidence_gallery lorsque seuils correspondants).

    Si ``frame_bgr`` est fourni, ``path`` peut être omis (frame déjà décodée, ex. vidéo progressive).
    """
    if not enable_fight and not enable_weapon:
        return {"error": "Activez au moins un modèle (combat ou armes)."}
    if frame_bgr is not None:
        if frame_bgr.size == 0:
            return {"error": "Image vide."}
        frame = np.ascontiguousarray(frame_bgr)
        lbl = media_label or "frame"
    else:
        if path is None:
            return {"error": "Chemin image ou frame_bgr requis."}
        img_path = Path(path)
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            return {"error": "Impossible de lire l'image (format non pris en charge ou fichier corrompu)."}
        lbl = media_label or (img_path.name or "Image")

    model = None
    if enable_fight:
        try:
            model = get_fight_model(fight_yolo_weights)
            model.reset_sequence_buffer()
        except Exception as exc:
            return {"error": f"Modèle combat : {exc}"}

    weapon_model = None
    if enable_weapon:
        try:
            weapon_model = get_weapon_model(weapon_yolo_weights)
        except Exception as exc:
            return {"error": f"Modèle armes : {exc}"}

    source = _get_fight_analysis_source()
    fh, fw = frame.shape[:2]

    dual_p_thr = float(getattr(settings, "FIGHT_DUAL_EVIDENCE_FIGHT_P", 0.52))
    dual_w_thr = float(getattr(settings, "FIGHT_DUAL_EVIDENCE_WEAPON_CONF", 0.42))
    crop_pad = float(getattr(settings, "FIGHT_DUAL_CROP_PAD", 0.14))
    max_ppl = max(1, int(getattr(settings, "FIGHT_PERSON_MAX_PER_FRAME", 2)))
    max_snap = max(1, int(getattr(settings, "FIGHT_DUAL_EVIDENCE_MAX_SNAPSHOTS", 16)))
    alert_thr = float(getattr(settings, "FIGHT_ALERT_MIN_CONFIDENCE", 0.99))
    top1_thr = float(getattr(settings, "FIGHT_ALERT_MIN_TOP1_CONF", 0.99))
    weapon_alert_thr = float(getattr(settings, "WEAPON_ALERT_MIN_CONF", 0.82))

    weapon_draw_thr = float(
        getattr(settings, "FIGHT_WEAPON_DRAW_MIN_CONF", getattr(settings, "WEAPON_DRAW_MIN_CONF", 0.5))
    )

    person_model = None
    evidence_root: Path | None = None
    run_slug = ""
    dual_evidence_gallery: list[dict] = []
    snapshot_count = 0
    if enable_fight and enable_weapon:
        try:
            person_model = get_person_detector()
            run_slug = uuid4().hex
            evidence_root = Path(settings.MEDIA_ROOT) / "fusion_dual_evidence" / run_slug
            evidence_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            person_model = None
            evidence_root = None

    alerts_created = 0
    weapon_alerts_created = 0

    if enable_fight and model is not None:
        dd = _predict_frame(model, frame)
        if not dd:
            d = {"p_fight": 0.0, "top1": 1, "top1conf": 1.0, "label": "nonfight"}
        else:
            d = dd
    else:
        d = {"p_fight": 0.0, "top1": 1, "top1conf": 1.0, "label": "nonfight"}

    weapon_dets: list[dict] = []
    weapon_max = 0.0
    if weapon_model is not None:
        weapon_dets, weapon_max = _detect_weapons_in_frame(weapon_model, frame)

    weapon_compact = ""
    if weapon_dets:
        weapon_compact = ", ".join(
            f"{x['label']} {float(x['conf']) * 100:.0f}%" for x in weapon_dets[:4]
        )

    weapons_hi = [x for x in weapon_dets if float(x.get("conf", 0.0)) >= dual_w_thr]
    person_draw: list[dict] | None = None
    dual_highlight = False

    mu = settings.MEDIA_URL.rstrip("/")
    if (
        person_model is not None
        and evidence_root is not None
        and snapshot_count < max_snap
        and enable_fight
        and enable_weapon
        and int(d.get("top1", 1)) == 0
        and float(d.get("p_fight", 0.0)) >= dual_p_thr
        and weapons_hi
    ):
        dual_highlight = True
        preds = _detect_persons_in_frame(person_model, frame)
        highlighted = _rank_persons_near_weapons(preds, weapons_hi, fw=fw, max_n=max_ppl)
        person_draw = highlighted
        crops_rel: list[str] = []
        crops_bgr_list: list[np.ndarray] = []
        if highlighted:
            for j, per in enumerate(highlighted):
                pad_bb = _expand_pad_xyxy(per["bbox"], fw, fh, crop_pad)
                crop = _crop_bgr_roi(frame, pad_bb)
                if crop is None:
                    continue
                crops_bgr_list.append(np.ascontiguousarray(crop))
                fn = f"dual_{snapshot_count:03d}_p{j}.jpg"
                fp = evidence_root / fn
                cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                crops_rel.append(f"{mu}/fusion_dual_evidence/{run_slug}/{fn}")
        if not crops_rel:
            wf = weapons_hi[0]["bbox"]
            pad_bb = _expand_pad_xyxy(wf, fw, fh, 0.55)
            crop = _crop_bgr_roi(frame, pad_bb)
            if crop is not None:
                crops_bgr_list.append(np.ascontiguousarray(crop))
                fn = f"dual_{snapshot_count:03d}_arme_ctx.jpg"
                fp = evidence_root / fn
                cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                crops_rel.append(f"{mu}/fusion_dual_evidence/{run_slug}/{fn}")
        if crops_rel:
            snapshot_count += 1
            face_mx: list[dict] = []
            if getattr(settings, "FACE_RECOGNITION_ENABLED", True):
                try:
                    from monitoring.services.face_recognition_gallery import (
                        face_matches_roundtrip_json,
                        match_faces_in_bgr_crops,
                    )

                    face_mx = face_matches_roundtrip_json(match_faces_in_bgr_crops(crops_bgr_list))
                except Exception:
                    face_mx = []
            dual_evidence_gallery.append(
                {
                    "t": t_round,
                    "p_fight": round(float(d["p_fight"]), 4),
                    "weapon_max": round(float(weapon_max), 4),
                    "crop_urls": crops_rel,
                    "n_persons": len(highlighted),
                    "face_matches": face_mx,
                }
            )

    face_scan_slug = uuid4().hex
    face_snap: dict[str, Any] = {}
    try:
        face_snap = _build_face_scan_gallery_for_frame(frame, face_scan_slug)
    except Exception:
        face_snap = {}
    face_scan_gallery = face_snap.get("face_scan_gallery") or []
    faces_detected = face_snap.get("faces_detected") or []
    face_matches = face_snap.get("face_matches") or []
    face_overlay_boxes = face_snap.get("face_overlay_boxes") or []

    ann_person_dets = person_draw if person_draw is not None else None

    p_fight = float(d.get("p_fight", 0.0))

    timeline = [
        {
            "t": t_round,
            "p_fight": round(p_fight, 4),
            "label": str(d["label"]),
            "top1": int(d["top1"]),
            "weapon_compact": weapon_compact,
            "weapon_max": round(float(weapon_max), 4),
            "dual_highlight": bool(dual_highlight),
        }
    ]

    annotated_image_path: str | None = None
    if annotated_output_path is not None:
        out_img = Path(annotated_output_path)
        out_img.parent.mkdir(parents=True, exist_ok=True)
        ann = _draw_fight_weapon_overlay(
            frame,
            d if enable_fight else None,
            weapon_dets,
            weapon_draw_min_conf=weapon_draw_thr,
            show_fight_banner=enable_fight,
            person_dets=ann_person_dets,
            face_dets=face_overlay_boxes,
        )
        cv2.imwrite(str(out_img), ann, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if out_img.is_file() and out_img.stat().st_size > 0:
            annotated_image_path = str(out_img)

    if (
        not suppress_site_alerts
        and enable_fight
        and int(d["top1"]) == 0
        and p_fight >= alert_thr
        and float(d["top1conf"]) >= top1_thr
    ):
        if _emit_fight_site_alert(
            source,
            t_round,
            p_fight,
            float(d["top1conf"]),
            frame,
            lbl,
        ):
            alerts_created += 1

    if (
        not suppress_site_alerts
        and weapon_model is not None
        and weapon_dets
        and weapon_max >= weapon_alert_thr
    ):
        if _emit_weapon_site_alert(source, t_round, weapon_dets, weapon_max, frame, lbl):
            weapon_alerts_created += 1

    n_top1_fight = 1 if enable_fight and int(d["top1"]) == 0 else 0
    weapon_hit_frames = 1 if weapon_dets else 0

    if enable_fight:
        verdict = "fight" if p_fight >= 0.5 or int(d["top1"]) == 0 else "nonfight"
        verdict_label = "Combat probable" if verdict == "fight" else "Pas de combat détecté"
        avg_fight = p_fight
    elif not enable_fight:
        avg_fight = None
        verdict = "n/a"
        verdict_label = "Analyse combat désactivée"
    else:
        avg_fight = None
        verdict = "n/a"
        verdict_label = "Données combat indisponibles"

    weapon_detection_boxes: list[dict] = []
    for det in weapon_dets:
        bb = det.get("bbox") or [0, 0, 0, 0]
        weapon_detection_boxes.append(
            {
                "label": str(det.get("label", "")),
                "confidence": round(float(det.get("conf", 0.0)), 4),
                "bbox": [round(float(x), 2) for x in bb[:4]],
            }
        )

    return {
        "frames_analyzed": 1,
        "avg_p_fight": avg_fight,
        "fight_frames": n_top1_fight,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "details": [
            {
                **dict(d),
                "t": t_round,
                "weapon_compact": weapon_compact,
                "weapon_max": round(float(weapon_max), 4),
            }
        ],
        "timeline": timeline,
        "duration_sec": None,
        "fight_alerts_created": alerts_created,
        "weapon_alerts_created": weapon_alerts_created,
        "weapon_frames_detected": weapon_hit_frames,
        "weapon_model_used": bool(weapon_model),
        "fight_model_used": bool(model),
        "annotated_video_path": None,
        "annotated_image_path": annotated_image_path,
        "enable_fight_used": enable_fight,
        "enable_weapon_used": enable_weapon,
        "dual_evidence_gallery": dual_evidence_gallery,
        "face_scan_gallery": face_scan_gallery,
        "faces_detected": faces_detected,
        "face_matches": face_matches,
        "face_gallery_empty": bool(face_snap.get("gallery_empty")),
        "face_matching_disabled": bool(face_snap.get("matching_disabled")),
        "weapon_detection_boxes": weapon_detection_boxes,
    }


def analyze_live_frame_bgr(
    frame_bgr: np.ndarray,
    *,
    enable_fight: bool = True,
    enable_weapon: bool = True,
    fight_yolo_weights: str | None = None,
    weapon_yolo_weights: str | None = None,
    viz_person_boxes: bool = False,
    viz_face_match: bool = False,
) -> dict:
    """
    Une image BGR (OpenCV) pour la caméra navigateur : résultats combat + armes en JSON.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return {"ok": False, "error": "image_vide"}
    if not enable_fight and not enable_weapon:
        return {"ok": False, "error": "aucun_modele"}

    weapon_dets: list[dict] = []
    weapon_max = 0.0
    fight_payload = None

    if enable_fight:
        try:
            d = _predict_frame(get_fight_model(fight_yolo_weights), frame_bgr)
        except Exception:
            d = None
        if d:
            fight_payload = {
                "p_fight": round(float(d["p_fight"]), 4),
                "label": d["label"],
                "top1": int(d["top1"]),
                "top1conf": round(float(d["top1conf"]), 4),
            }
            # Gros plan / faux positifs : top1 « fight » avec P trop bas → libellé live « nonfight »
            min_p_live = float(getattr(settings, "FUSION_LIVE_FIGHT_DISPLAY_MIN_P", 0.52))
            if int(fight_payload["top1"]) == 0 and float(fight_payload["p_fight"]) < min_p_live:
                fight_payload = dict(fight_payload)
                fight_payload["label"] = "nonfight"
                fight_payload["live_weak_fight_signal"] = True

    if enable_weapon:
        try:
            weapon_dets, weapon_max = _detect_weapons_in_frame(
                get_weapon_model(weapon_yolo_weights), frame_bgr
            )
        except Exception:
            weapon_dets, weapon_max = [], 0.0

    serialized = []
    for det in weapon_dets:
        bbox = det.get("bbox") or [0, 0, 0, 0]
        serialized.append(
            {
                "label": str(det.get("label", "")),
                "conf": round(float(det.get("conf", 0.0)), 4),
                "bbox": [round(float(x), 2) for x in bbox[:4]],
            }
        )

    dual_view: dict | None = None
    dual_pf = float(getattr(settings, "FIGHT_DUAL_EVIDENCE_FIGHT_P", 0.52))
    dual_wc = float(getattr(settings, "FIGHT_DUAL_EVIDENCE_WEAPON_CONF", 0.42))
    live_max_w = max(96, int(getattr(settings, "FIGHT_DUAL_LIVE_CROP_MAX_W", 288)))

    if (
        enable_fight
        and enable_weapon
        and fight_payload
        and serialized
        and weapon_max >= dual_wc
    ):
        weapons_hi_live = [
            {"bbox": list(x["bbox"]), "conf": float(x["conf"])}
            for x in serialized
            if float(x["conf"]) >= dual_wc
        ]
        if weapons_hi_live and int(fight_payload.get("top1", 1)) == 0:
            pf_live = float(fight_payload["p_fight"])
            if pf_live >= dual_pf:
                try:
                    pm = get_person_detector()
                    preds = _detect_persons_in_frame(pm, frame_bgr)
                    fh, fw = frame_bgr.shape[:2]
                    max_live_ppl = max(
                        1, int(getattr(settings, "FIGHT_PERSON_MAX_PER_FRAME", 2))
                    )
                    hl = _rank_persons_near_weapons(
                        preds, weapons_hi_live, fw=fw, max_n=max_live_ppl
                    )
                    pad_px = float(getattr(settings, "FIGHT_DUAL_CROP_PAD", 0.14))
                    crops_b64: list[str] = []
                    hl_out = []
                    live_bgr_crops: list[np.ndarray] = []
                    for per in hl:
                        bb_exp = _expand_pad_xyxy(per["bbox"], fw, fh, pad_px)
                        cr = _crop_bgr_roi(frame_bgr, bb_exp)
                        if cr is None:
                            continue
                        live_bgr_crops.append(np.ascontiguousarray(cr))
                        crs = _resize_max_width(cr, live_max_w)
                        enc = _encode_jpeg_b64(crs, quality=82)
                        if enc:
                            crops_b64.append(f"data:image/jpeg;base64,{enc}")
                        hl_out.append(
                            {
                                "conf": round(float(per["conf"]), 4),
                                "bbox": [round(float(v), 1) for v in per["bbox"][:4]],
                            }
                        )
                    if not crops_b64:
                        wf = weapons_hi_live[0]["bbox"]
                        bb_exp = _expand_pad_xyxy(wf, fw, fh, 0.55)
                        cr = _crop_bgr_roi(frame_bgr, bb_exp)
                        if cr is not None:
                            live_bgr_crops.append(np.ascontiguousarray(cr))
                            crs = _resize_max_width(cr, live_max_w)
                            enc = _encode_jpeg_b64(crs, quality=82)
                            if enc:
                                crops_b64.append(f"data:image/jpeg;base64,{enc}")
                    face_lin: list[dict] = []
                    if getattr(settings, "FACE_RECOGNITION_ENABLED", True) and live_bgr_crops:
                        try:
                            from monitoring.services.face_recognition_gallery import (
                                face_matches_roundtrip_json,
                                match_faces_in_bgr_crops,
                            )

                            face_lin = face_matches_roundtrip_json(
                                match_faces_in_bgr_crops(live_bgr_crops)
                            )
                        except Exception:
                            face_lin = []
                    dual_view = {
                        "show": True,
                        "persons": hl_out,
                        "crop_data_urls": crops_b64[:4],
                        "face_matches": face_lin,
                    }
                except Exception:
                    dual_view = None

    payload = {
        "ok": True,
        "fight": fight_payload,
        "weapon": {"detections": serialized, "max_conf": round(float(weapon_max), 4)},
    }
    if dual_view:
        payload["dual_view"] = dual_view

    fh, fw = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    payload["frame"] = {"h": fh, "w": fw}

    if viz_person_boxes or viz_face_match:
        try:
            if viz_person_boxes:
                pm = get_person_detector()
                preds = _detect_persons_in_frame(pm, frame_bgr)
                preds_sorted = sorted(
                    preds,
                    key=lambda x: -float(x["conf"]),
                )[:12]
                payload["person_boxes"] = [
                    {
                        "bbox": [round(float(v), 1) for v in p["bbox"][:4]],
                        "conf": round(float(p["conf"]), 4),
                    }
                    for p in preds_sorted
                ]
            if viz_face_match and getattr(settings, "FACE_RECOGNITION_ENABLED", True):
                from monitoring.services.face_biometric import (
                    extract_face_boxes_bgr,
                    live_face_row_from_crop,
                )

                live_mx = max(1, int(getattr(settings, "FUSION_FACE_DETECT_MAX", 8)))
                pairs_fb = extract_face_boxes_bgr(frame_bgr)
                faces_out = [
                    live_face_row_from_crop(crop, bbox, person_conf=1.0)
                    for bbox, crop in pairs_fb[:live_mx]
                ]
                if faces_out:
                    payload["live_faces"] = faces_out
        except Exception:
            payload.setdefault("person_boxes", [])
    return payload
