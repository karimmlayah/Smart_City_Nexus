"""
Détection temps réel : YOLOv8 + suivi (ByteTrack).
Classes COCO : personnes + sacs (sac à main, dos, valise).
Heuristique « piste vol » : si un sac reste stable près d’une personne A puis près d’une personne B → alerte.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Generator

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile

# COCO 80 (indices Ultralytics)
COCO_PERSON = 0
COCO_BACKPACK = 24
COCO_HANDBAG = 26
COCO_SUITCASE = 28
BAG_CLASS_IDS = frozenset({COCO_BACKPACK, COCO_HANDBAG, COCO_SUITCASE})
TRACK_CLASSES = [COCO_PERSON, COCO_BACKPACK, COCO_HANDBAG, COCO_SUITCASE]

_model = None
_model_lock = threading.Lock()

_last_alert_at: dict[tuple[int, str], float] = {}
_prev_centers: dict[tuple[int, int], tuple[float, float]] = {}
# (source_id, bag_track_id) -> deque des person_track_id les plus proches (None si trop loin)
_bag_proximity_ring: dict[tuple[int, int], deque] = {}
# Dernier propriétaire « stable » par sac (track personne)
_bag_stable_owner: dict[tuple[int, int], int] = {}
# source_id -> nb d’analyses consécutives avec max vitesse > seuil (anti faux positifs 1 frame)
_fast_move_streak: dict[int, int] = {}

_stats_lock = threading.Lock()
# Dernières métriques par source_id (pour l’UI web, pas sur la vidéo)
_live_stats: dict[int, dict] = {}


def get_live_stats(source_id: int) -> dict:
    """Compteurs temps réel pour le site (personnes, sacs, piste vol)."""
    enhance = bool(getattr(settings, "YOLO_LOW_LIGHT_ENHANCE", False))
    with _stats_lock:
        row = _live_stats.get(source_id)
    if not row:
        return {
            "persons": None,
            "bags": None,
            "theft_hint": False,
            "stale": True,
            "age_ms": None,
            "low_light_enhance": enhance,
        }
    age_ms = (time.time() - row["ts"]) * 1000
    stale = age_ms > 3000
    return {
        "persons": row["persons"],
        "bags": row["bags"],
        "theft_hint": bool(row.get("theft_hint")),
        "stale": stale,
        "age_ms": int(age_ms),
        "low_light_enhance": enhance,
    }


def _publish_live_stats(source_id: int, persons: int, bags: int, theft_hint: bool) -> None:
    with _stats_lock:
        _live_stats[source_id] = {
            "persons": persons,
            "bags": bags,
            "theft_hint": theft_hint,
            "ts": time.time(),
        }


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from ultralytics import YOLO

            path = getattr(settings, "YOLO_MODEL_PATH", "yolov8n.pt")
            _model = YOLO(path)
        return _model


def open_capture(source):
    st = source.source_type
    if st == source.SourceType.WEBCAM:
        cap = cv2.VideoCapture(source.webcam_index)
    elif st == source.SourceType.RTSP:
        cap = cv2.VideoCapture(source.url, cv2.CAP_FFMPEG)
    else:
        path = source.get_file_path()
        cap = cv2.VideoCapture(path)
    return cap


def _cooldown_ok(source_id: int, alert_type: str) -> bool:
    cd = getattr(settings, "ALERT_COOLDOWN_SECONDS", 45)
    key = (source_id, alert_type)
    now = time.time()
    if key in _last_alert_at and now - _last_alert_at[key] < cd:
        return False
    _last_alert_at[key] = now
    return True


def _bbox_overlap_ratio(b1, b2) -> float:
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    m = min(a1, a2)
    return float(inter / m) if m > 0 else 0.0


def _avg_pairwise_overlap(boxes: np.ndarray) -> float:
    n = len(boxes)
    if n < 2:
        return 0.0
    acc = 0.0
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            acc += _bbox_overlap_ratio(boxes[i], boxes[j])
            cnt += 1
    return acc / cnt if cnt else 0.0


def _emit_alert(source, alert_type, message, person_count=None, confidence_max=None, extra=None, frame_bgr=None) -> bool:
    from monitoring.models import Alert

    if not _cooldown_ok(source.id, alert_type):
        return False
    meta = extra or {}
    alert = Alert.objects.create(
        source=source,
        alert_type=alert_type,
        message=message[:500],
        person_count=person_count,
        confidence_max=confidence_max,
        metadata=meta,
    )
    if frame_bgr is not None:
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            alert.thumbnail.save(f"alert_{alert.id}.jpg", ContentFile(buf.tobytes()), save=True)
    return True


def _maybe_enhance_low_light(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Améliore visibilité en pièce sombre : gamma (rehausse les ombres) puis CLAHE sur la luminance.
    L’inférence et l’image affichée utilisent cette version (même géométrie que l’original).
    """
    if not getattr(settings, "YOLO_LOW_LIGHT_ENHANCE", False):
        return frame_bgr
    out = frame_bgr
    gamma = float(getattr(settings, "YOLO_LOW_LIGHT_GAMMA", 1.0))
    if 0.02 < gamma < 0.999:
        f = out.astype(np.float32) / 255.0
        f = np.clip(np.power(f, gamma), 0.0, 1.0)
        out = (f * 255.0).astype(np.uint8)
    clip = float(getattr(settings, "YOLO_CLAHE_CLIP_LIMIT", 2.5))
    grid = int(getattr(settings, "YOLO_CLAHE_TILE", 8))
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l2 = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l2, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _as_numpy(t):
    if t is None:
        return None
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def _nearest_person_track_for_bag(
    bag_xyxy: np.ndarray,
    person_xyxy: np.ndarray,
    person_ids: np.ndarray,
) -> int | None:
    """Retourne l’ID de suivi de la personne la plus proche du sac (si assez proche)."""
    if person_xyxy is None or len(person_xyxy) == 0:
        return None
    bx1, by1, bx2, by2 = bag_xyxy
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
    bw, bh = max(bx2 - bx1, 1), max(by2 - by1, 1)
    diag = float(np.hypot(bw, bh))
    max_dist = diag * float(getattr(settings, "BAG_PERSON_MAX_DIST_FACTOR", 2.8))

    best_tid: int | None = None
    best_d = max_dist + 1.0
    for i in range(len(person_xyxy)):
        px1, py1, px2, py2 = person_xyxy[i]
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        d = float(np.hypot(bcx - pcx, bcy - pcy))
        if d < best_d:
            best_d = d
            tid = int(person_ids[i])
            best_tid = tid
    if best_d > max_dist:
        return None
    return best_tid


def _analyze_bag_association(
    source,
    annotated_bgr,
    xyxy: np.ndarray,
    ids: np.ndarray | None,
    cls: np.ndarray,
    conf: np.ndarray | None,
):
    """Détecte sacs + changement de personne associée (heuristique vol / transfert)."""
    if ids is None or xyxy is None or len(xyxy) == 0:
        return

    person_m = cls == COCO_PERSON
    bag_m = np.isin(cls, list(BAG_CLASS_IDS))
    if not np.any(bag_m) or not np.any(person_m):
        return

    p_xy = xyxy[person_m]
    p_ids = ids[person_m]
    b_xy = xyxy[bag_m]
    b_ids = ids[bag_m]
    b_cls = cls[bag_m]

    sid = source.id
    stable_frames = int(getattr(settings, "BAG_OWNER_STABLE_FRAMES", 5))

    for i in range(len(b_xy)):
        bag_tid = int(b_ids[i])
        owner = _nearest_person_track_for_bag(b_xy[i], p_xy, p_ids)
        key = (sid, bag_tid)
        ring = _bag_proximity_ring.get(key)
        if ring is None:
            ring = deque(maxlen=max(stable_frames + 2, 8))
            _bag_proximity_ring[key] = ring
        ring.append(owner)

        if len(ring) < stable_frames:
            continue
        tail = list(ring)[-stable_frames:]
        if any(x is None for x in tail):
            continue
        if len(set(tail)) != 1:
            continue
        stable_now = int(tail[0])
        prev_stable = _bag_stable_owner.get(key)
        bag_label = "sac"
        if int(b_cls[i]) == COCO_HANDBAG:
            bag_label = "sac à main"
        elif int(b_cls[i]) == COCO_BACKPACK:
            bag_label = "sac à dos"
        elif int(b_cls[i]) == COCO_SUITCASE:
            bag_label = "valise"

        if prev_stable is not None and stable_now != prev_stable:
            conf_max = float(np.max(conf)) if conf is not None and len(conf) else None
            n_people = int(np.sum(cls == COCO_PERSON))
            _emit_alert(
                source,
                "theft_suspicious",
                f"Piste vol / transfert : {bag_label} (suivi #{bag_tid}) "
                f"associé à une autre personne (IDs suivi {prev_stable} → {stable_now}). À vérifier.",
                person_count=n_people,
                confidence_max=conf_max,
                extra={
                    "bag_track_id": bag_tid,
                    "person_track_before": prev_stable,
                    "person_track_after": stable_now,
                    "bag_class": bag_label,
                },
                frame_bgr=annotated_bgr,
            )
        _bag_stable_owner[key] = stable_now


def _analyze_tracked_frame(source, annotated_bgr, boxes_xyxy, boxes_id, boxes_conf, boxes_cls):
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return
    cls = _as_numpy(boxes_cls)
    xyxy = _as_numpy(boxes_xyxy)
    ids = _as_numpy(boxes_id) if boxes_id is not None else None
    conf = _as_numpy(boxes_conf) if boxes_conf is not None else None

    if ids is None:
        return
    valid = np.isfinite(ids.astype(float))
    if not np.any(valid):
        return
    xyxy = xyxy[valid]
    cls = cls[valid]
    ids = ids[valid].astype(np.int64)
    if conf is not None:
        conf = conf[valid]

    mask_person = cls == COCO_PERSON
    if not np.any(mask_person):
        _analyze_bag_association(source, annotated_bgr, xyxy, ids, cls, conf)
        return

    xy_p = xyxy[mask_person]
    conf_p = conf[mask_person] if conf is not None else None
    ids_p = ids[mask_person]

    n = len(xy_p)
    conf_max = float(np.max(conf_p)) if conf_p is not None and len(conf_p) else None

    if n >= source.crowd_threshold:
        _emit_alert(
            source,
            "crowd",
            f"Foule dense : {n} personnes détectées (seuil {source.crowd_threshold}).",
            person_count=n,
            confidence_max=conf_max,
            extra={"threshold": source.crowd_threshold},
            frame_bgr=annotated_bgr,
        )

    overlap = _avg_pairwise_overlap(xy_p)
    if n >= 3 and overlap > 0.35:
        _emit_alert(
            source,
            "high_overlap",
            f"Regroupement serré : {n} personnes, chevauchement moyen élevé ({overlap:.2f}).",
            person_count=n,
            confidence_max=conf_max,
            extra={"avg_overlap": round(overlap, 3)},
            frame_bgr=annotated_bgr,
        )

    speeds = []
    sid = source.id
    for i in range(n):
        tid = int(ids_p[i])
        x1, y1, x2, y2 = xy_p[i]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        key = (sid, tid)
        px, py = _prev_centers.get(key, (cx, cy))
        spd = float(np.hypot(cx - px, cy - py))
        speeds.append(spd)
        _prev_centers[key] = (cx, cy)
    if speeds:
        spd_arr = np.array(speeds, dtype=np.float64)
        max_spd = float(np.max(spd_arr))
        mean_spd = float(np.mean(spd_arr))
        thr = float(source.fast_movement_threshold)
        fastest_i = int(np.argmax(spd_arr))
        fastest_tid = int(ids_p[fastest_i])
        streak_need = max(1, int(getattr(settings, "FAST_MOVEMENT_STREAK", 2)))
        if max_spd > thr:
            _fast_move_streak[sid] = _fast_move_streak.get(sid, 0) + 1
        else:
            _fast_move_streak[sid] = 0
        if n >= 1 and max_spd > thr and _fast_move_streak[sid] >= streak_need:
            if _emit_alert(
                source,
                "fast_movement",
                f"Mouvement rapide : personne suivi #{fastest_tid} "
                f"(~{max_spd:.1f} px/frame, seuil {thr}).",
                person_count=n,
                confidence_max=conf_max,
                extra={
                    "max_speed_px": round(max_spd, 2),
                    "mean_speed_px": round(mean_spd, 2),
                    "fastest_track_id": fastest_tid,
                },
                frame_bgr=annotated_bgr,
            ):
                _fast_move_streak[sid] = 0

    _analyze_bag_association(source, annotated_bgr, xyxy, ids, cls, conf)


def stream_mjpeg_frames(source) -> Generator[bytes, None, None]:
    from monitoring.models import VideoSource

    if not isinstance(source, VideoSource):
        raise TypeError("source must be VideoSource")

    model = get_model()
    cap = open_capture(source)
    if not cap.isOpened():
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "Impossible d'ouvrir la source",
            (40, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        ok, jpeg = cv2.imencode(".jpg", blank)
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
        return

    conf = float(getattr(settings, "YOLO_CONFIDENCE", 0.25))
    iou = float(getattr(settings, "YOLO_IOU", 0.5))
    imgsz = int(getattr(settings, "YOLO_IMGSZ", 640))
    frame_idx = 0
    theft_flash_until = 0.0
    tracking_enabled = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break

            frame = _maybe_enhance_low_light(frame)

            if tracking_enabled:
                try:
                    results = model.track(
                        frame,
                        persist=True,
                        conf=conf,
                        iou=iou,
                        classes=TRACK_CLASSES,
                        imgsz=imgsz,
                        verbose=False,
                    )
                except Exception:
                    # Fallback robuste: si ByteTrack/lap indisponible, on garde l'affichage en detection seule.
                    tracking_enabled = False
                    results = model.predict(
                        frame,
                        conf=conf,
                        iou=iou,
                        classes=TRACK_CLASSES,
                        imgsz=imgsz,
                        verbose=False,
                    )
            else:
                results = model.predict(
                    frame,
                    conf=conf,
                    iou=iou,
                    classes=TRACK_CLASSES,
                    imgsz=imgsz,
                    verbose=False,
                )
            r0 = results[0]
            annotated = r0.plot()

            n_person = n_bag = 0
            if r0.boxes is not None and len(r0.boxes):
                c = _as_numpy(r0.boxes.cls)
                n_person = int(np.sum(c == COCO_PERSON))
                n_bag = int(np.sum(np.isin(c, list(BAG_CLASS_IDS))))

            frame_idx += 1
            if tracking_enabled and frame_idx % 3 == 0 and r0.boxes is not None and len(r0.boxes):
                before = _last_alert_at.get((source.id, "theft_suspicious"))
                _analyze_tracked_frame(
                    source,
                    annotated,
                    r0.boxes.xyxy,
                    r0.boxes.id,
                    r0.boxes.conf,
                    r0.boxes.cls,
                )
                after = _last_alert_at.get((source.id, "theft_suspicious"))
                if after is not None and after != before:
                    theft_flash_until = time.time() + 4.0

            show_theft = time.time() < theft_flash_until
            _publish_live_stats(source.id, n_person, n_bag, show_theft)

            ok, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
            )
    finally:
        with _stats_lock:
            _live_stats.pop(source.id, None)
        cap.release()
