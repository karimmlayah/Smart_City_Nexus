from collections import Counter
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

ZONE_COLORS_BGR = [
    (255, 80, 80),
    (80, 220, 80),
    (80, 170, 255),
    (220, 120, 255),
    (255, 220, 80),
]

# Orange urgence — boîtes ambulance (BGR)
AMBULANCE_BOX_BGR = (40, 180, 255)

CLASS_COLORS_BGR = {
    "ambulance": AMBULANCE_BOX_BGR,
    # Common congestion-style labels
    "congestion": (80, 80, 255),   # red
    "congested": (80, 80, 255),    # red
    "heavy": (80, 80, 255),        # red
    "high": (80, 80, 255),         # red
    "jam": (80, 80, 255),          # red
    "faible": (80, 220, 80),       # green
    "low": (80, 220, 80),          # green
    "light": (80, 220, 80),        # green
    "low_congestion": (80, 220, 80),   # green
    "congestion_faible": (80, 220, 80),# green
    "normal": (80, 220, 80),       # green
    "free": (80, 220, 80),         # green
    "medium": (0, 215, 255),       # amber
    "moderate": (0, 215, 255),     # amber
}

FALLBACK_CLASS_COLORS_BGR = [
    (80, 220, 80),    # green
    (80, 80, 255),    # red-ish
    (0, 215, 255),    # amber
    (255, 170, 80),   # blue-ish
    (220, 120, 255),  # purple-ish
]


def _is_ambulance_label(label: str) -> bool:
    lb = str(label).strip().lower()
    if not lb:
        return False
    if lb.startswith("ambulance:"):
        return True
    if "ambulance" in lb or "ambul" in lb:
        return True
    if lb in ("emergency", "ems") or ("emergency" in lb and "non-" not in lb):
        return True
    return False


def color_for_class(label: str, cls_id: int) -> tuple:
    lb_raw = str(label).strip()
    if _is_ambulance_label(lb_raw):
        return AMBULANCE_BOX_BGR
    # Explicit numeric-class mapping requested by user:
    # class 1 -> green, class 0 -> red
    if cls_id == 1 or label.strip() == "1":
        return (80, 220, 80)
    if cls_id == 0 or label.strip() == "0":
        return (80, 80, 255)
    key = label.strip().lower()
    if key in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[key]
    return FALLBACK_CLASS_COLORS_BGR[cls_id % len(FALLBACK_CLASS_COLORS_BGR)]


def _speed_label_from_detection(label: str) -> str:
    key = (label or "").strip().lower()
    if key in {"congestion", "congested", "heavy", "high", "jam"}:
        return "Slow"
    if key in {"medium", "moderate"}:
        return "Medium"
    return "Fast"


def _speed_color(label: str) -> tuple:
    speed = _speed_label_from_detection(label)
    if speed == "Slow":
        return (80, 80, 255)  # red
    if speed == "Medium":
        return (0, 165, 255)  # orange
    return (80, 220, 80)  # green


def _point_in_poly(poly: np.ndarray, x: int, y: int) -> bool:
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0


def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[int, int]]) -> bool:
    """Ray-casting point-in-polygon test (same coordinate space as ROI vertices)."""
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    for i in range(n):
        xi, yi = float(polygon[i][0]), float(polygon[i][1])
        xj, yj = float(polygon[(i + 1) % n][0]), float(polygon[(i + 1) % n][1])
        intersect = (yi > y) != (yj > y) and x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi
        if intersect:
            inside = not inside
    return inside


def get_detection_center(det: Dict) -> Tuple[int, int]:
    x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _assign_zone_for_bbox(polygons: List[np.ndarray], x1: int, y1: int, x2: int, y2: int) -> int:
    """Assign detection to ROI zone using bbox center point only."""
    cx, cy = get_detection_center({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    for i, poly in enumerate(polygons):
        if _point_in_poly(poly, cx, cy):
            return i
    return -1


def _filter_candidates_by_roi(
    candidates: List[Dict],
    roi_polygons: List[List[Tuple[int, int]]],
) -> Tuple[List[Dict], List[int]]:
    """
    Keep detections whose bbox center lies inside at least one ROI polygon.
    When no ROI is defined, all candidates are kept (single implicit full-frame zone).
    """
    if not roi_polygons:
        return list(candidates), [0] * len(candidates)

    polygons = [np.array(poly, dtype=np.int32) for poly in roi_polygons]
    kept: List[Dict] = []
    zone_indices: List[int] = []
    for cand in candidates:
        x1, y1, x2, y2 = int(cand["x1"]), int(cand["y1"]), int(cand["x2"]), int(cand["y2"])
        zone_idx = _assign_zone_for_bbox(polygons, x1, y1, x2, y2)
        if zone_idx >= 0:
            kept.append(cand)
            zone_indices.append(zone_idx)
    return kept, zone_indices


def _bbox_iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ab = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + ab - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_greedy_high_conf_first(candidates: List[Dict], iou_thresh: float) -> List[Dict]:
    """Réduit les doublons multi-libellés sur le même véhicule (NMS greedy par IoU)."""
    if len(candidates) <= 1:
        return list(candidates)
    items = sorted(candidates, key=lambda x: -float(x.get("conf", 0.0)))
    kept: List[Dict] = []
    while items:
        cur = items.pop(0)
        kept.append(cur)
        cb = (int(cur["x1"]), int(cur["y1"]), int(cur["x2"]), int(cur["y2"]))
        rest = []
        for c in items:
            ob = (int(c["x1"]), int(c["y1"]), int(c["x2"]), int(c["y2"]))
            if _bbox_iou_xyxy(cb, ob) < float(iou_thresh):
                rest.append(c)
        items = rest
    return kept


def _prefer_ambulance_over_vehicle_overlap(candidates: List[Dict], iou_thresh: float) -> List[Dict]:
    """Priorité ambulance : si IoU élevé avec une boîte générique, masquer le doublon générique."""
    ambs: List[Dict] = []
    others: List[Dict] = []
    for c in candidates:
        if _is_ambulance_label(str(c.get("label", ""))):
            ambs.append(c)
        else:
            others.append(c)
    if not ambs or not others:
        return candidates
    drop_idx: set[int] = set()
    for i, o in enumerate(others):
        ob = (int(o["x1"]), int(o["y1"]), int(o["x2"]), int(o["y2"]))
        for a in ambs:
            ab = (int(a["x1"]), int(a["y1"]), int(a["x2"]), int(a["y2"]))
            if _bbox_iou_xyxy(ob, ab) >= float(iou_thresh):
                drop_idx.add(i)
                break
    others_kept = [o for i, o in enumerate(others) if i not in drop_idx]
    return ambs + others_kept


def _ambulance_overlap_iou_from_settings() -> float:
    try:
        from django.conf import settings as dj_settings

        return float(getattr(dj_settings, "TRAFFIC_AMBULANCE_OVERLAP_IOU", 0.48))
    except Exception:
        return 0.48


@dataclass
class ZoneDetection:
    per_zone_counts: List[Counter]
    processed_frame: np.ndarray
    detections_total: int
    track_state: Optional[Dict] = None
    model_config: Optional[Dict] = None
    vehicle_detections: Optional[List[Dict]] = None


def _normalize_allowed_labels(allowed_labels: Optional[Set[str]]) -> Optional[Set[str]]:
    if not allowed_labels:
        return None
    norm = {str(x).strip().lower() for x in allowed_labels if str(x).strip()}
    if "motorcycle" in norm:
        norm.add("motorbike")
    if "motorbike" in norm:
        norm.add("motorcycle")
    if "truck" in norm:
        norm.add("lorry")
    if "car" in norm:
        norm.add("vehicle")
    if "ambulance" in norm:
        norm.update({"emergency", "ems"})
    return norm


def _label_excluded_from_traffic_detection(label: str) -> bool:
    """Libellés à ne jamais compter / afficher (ex. personne, train COCO)."""
    lb = str(label).strip().lower()
    if not lb:
        return False
    if ":" in lb:
        lb = lb.split(":", 1)[-1].strip()
    try:
        from django.conf import settings as dj_settings

        excluded = getattr(dj_settings, "TRAFFIC_YOLO_EXCLUDED_CLASSES", ("person", "train"))
    except Exception:
        excluded = ("person", "train")
    for ex in excluded:
        key = str(ex).strip().lower()
        if key and lb == key:
            return True
    return False


def _label_allowed_for_vehicle_filter(label: str, allowed_norm: Optional[Set[str]]) -> bool:
    """Filtrage classes véhicules ; élargit ambulance → libellés usuels des modèles custom."""
    if not allowed_norm:
        return True
    lb = str(label).strip().lower()
    if lb in allowed_norm:
        return True
    if "ambulance" in allowed_norm:
        if "ambul" in lb or lb in ("emergency", "ems") or "emergency" in lb:
            return True
    return False


def _run_model_candidates(
    model: YOLO,
    frame: np.ndarray,
    allowed_labels: Optional[Set[str]],
    imgsz: int,
    max_det: int,
) -> List[Dict]:
    allowed_norm = _normalize_allowed_labels(allowed_labels)
    results = model.predict(frame, verbose=False, imgsz=imgsz, max_det=max_det)[0]
    names = results.names
    out: List[Dict] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = str(names[cls_id]) if names is not None else str(cls_id)
        if _label_excluded_from_traffic_detection(label):
            continue
        if allowed_norm and not _label_allowed_for_vehicle_filter(label, allowed_norm):
            continue
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        out.append(
            {
                "cls_id": cls_id,
                "label": label,
                "conf": float(box.conf[0]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )
    return out


def filter_ambulance_candidates(
    candidates: List[Dict],
    keywords: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """Filtre ambulance (best.pt) : mots-clés + NMS greedy + fallback confiance pour classes anonymes."""
    nms_iou = 0.40
    fb_min = 0.20
    fb_max = 6
    amb_min = 0.85
    try:
        from django.conf import settings as dj_settings

        default_kw = getattr(dj_settings, "TRAFFIC_AMBULANCE_LABEL_KEYWORDS", ("ambulance", "emergency"))
        nms_iou = float(getattr(dj_settings, "TRAFFIC_AMBULANCE_NMS_IOU", nms_iou))
        fb_min = float(getattr(dj_settings, "TRAFFIC_AMBULANCE_FALLBACK_MIN_CONF", fb_min))
        fb_max = int(getattr(dj_settings, "TRAFFIC_AMBULANCE_FALLBACK_MAX_DET", fb_max))
        amb_min = float(getattr(dj_settings, "TRAFFIC_AMBULANCE_MIN_CONF", amb_min))
    except Exception:
        default_kw = ("ambulance", "emergency")
    keys = [str(k).strip().lower() for k in (keywords or default_kw) if str(k).strip()]
    if not keys:
        keys = ["ambulance", "emergency"]
    out: List[Dict] = []
    for c in candidates:
        lb = str(c.get("label", "")).strip().lower()
        if any(k in lb for k in keys):
            if float(c.get("conf", 0.0)) < float(amb_min):
                continue
            c2 = dict(c)
            if not lb.startswith("ambulance") and "ambulance" not in lb:
                c2["label"] = f"ambulance:{c.get('label', '?')}"
            out.append(c2)
    if out:
        return _nms_greedy_high_conf_first(out, nms_iou)
    if not candidates:
        return []
    eff_min = max(float(fb_min), float(amb_min))
    pool = [dict(c) for c in candidates if float(c.get("conf", 0.0)) >= eff_min]
    pool = _nms_greedy_high_conf_first(pool, max(0.25, nms_iou + 0.08))
    pool = sorted(pool, key=lambda x: -float(x.get("conf", 0.0)))[: max(1, fb_max)]
    for c in pool:
        lb = str(c.get("label", "")).strip().lower()
        if "ambulance" not in lb:
            c["label"] = f"ambulance:{c.get('label', '?')}"
    return pool


def _nms_by_label(candidates: List[Dict], iou_thresh: float = 0.5) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for c in candidates:
        grouped.setdefault(c["label"], []).append(c)
    kept: List[Dict] = []
    for _, items in grouped.items():
        boxes_xywh = [[d["x1"], d["y1"], max(1, d["x2"] - d["x1"]), max(1, d["y2"] - d["y1"])] for d in items]
        scores = [float(d["conf"]) for d in items]
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.01, nms_threshold=iou_thresh)
        if len(idxs) == 0:
            continue
        for idx in np.array(idxs).reshape(-1):
            kept.append(items[int(idx)])
    return kept


def apply_conf_and_size_filters(
    candidates: List[Dict],
    *,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
) -> List[Dict]:
    amb_conf = 0.85
    amb_area = 110
    try:
        from django.conf import settings as dj_settings

        amb_conf = float(getattr(dj_settings, "TRAFFIC_AMBULANCE_MIN_CONF", amb_conf))
        amb_area = int(getattr(dj_settings, "TRAFFIC_AMBULANCE_MIN_BOX_AREA", amb_area))
    except Exception:
        pass
    out: List[Dict] = []
    for cand in candidates:
        lb = str(cand.get("label", ""))
        is_amb = _is_ambulance_label(lb)
        conf = float(cand.get("conf", 0.0))
        thr = float(amb_conf) if is_amb else float(min_confidence)
        if conf < thr:
            continue
        x1, y1, x2, y2 = int(cand["x1"]), int(cand["y1"]), int(cand["x2"]), int(cand["y2"])
        min_area = int(amb_area) if is_amb else int(min_box_area)
        if max(0, x2 - x1) * max(0, y2 - y1) < min_area:
            continue
        out.append(cand)
    return out


def _speed_status_from_px(speed_px: float) -> str:
    if speed_px < 2.0:
        return "Slow"
    if speed_px < 8.0:
        return "Medium"
    return "Fast"


def update_tracks(
    detections: List[Dict],
    track_state: Optional[Dict] = None,
) -> tuple[List[Dict], Dict]:
    """
    Tracking léger centroid-based pour display_mode=speed_status.
    Ne modifie pas le format principal des détections.
    """
    st = dict(track_state or {})
    prev_raw = dict(st.get("tracks") or {})
    prev_tracks: Dict[int, tuple[int, int]] = {}
    for k, v in prev_raw.items():
        try:
            tid_k = int(k)
        except (TypeError, ValueError):
            continue
        if not isinstance(v, (list, tuple)) or len(v) < 2:
            continue
        try:
            prev_tracks[tid_k] = (int(v[0]), int(v[1]))
        except (TypeError, ValueError):
            continue
    next_id = int(st.get("next_id") or 1)
    assigned_prev: set[int] = set()
    tracked: List[Dict] = []

    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        best_tid = None
        best_d2 = None
        for tid, (px, py) in prev_tracks.items():
            if tid in assigned_prev:
                continue
            d2 = float((cx - px) ** 2 + (cy - py) ** 2)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_tid = tid
        if best_tid is None or (best_d2 is not None and best_d2 > (55.0 * 55.0)):
            tid = next_id
            next_id += 1
            spd = 0.0
        else:
            tid = int(best_tid)
            assigned_prev.add(tid)
            px, py = prev_tracks.get(tid, (cx, cy))
            spd = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5)

        det2 = dict(det)
        det2["track_id"] = tid
        det2["speed_px"] = round(spd, 3)
        det2["speed_status"] = _speed_status_from_px(spd)
        tracked.append(det2)

    new_tracks = {int(d["track_id"]): (int((d["x1"] + d["x2"]) / 2), int((d["y1"] + d["y2"]) / 2)) for d in tracked}
    new_state = {"tracks": new_tracks, "next_id": next_id}
    return tracked, new_state


def _draw_ambulance_corner_brackets(
    output: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Coins en L pour un cadrage lisible (boîte ambulance)."""
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    hl = max(10, min(36, bw // 4))
    vl = max(10, min(36, bh // 4))
    t = max(int(thickness), 2)
    cv2.line(output, (x1, y1), (x1 + hl, y1), color, t)
    cv2.line(output, (x1, y1), (x1, y1 + vl), color, t)
    cv2.line(output, (x2, y1), (x2 - hl, y1), color, t)
    cv2.line(output, (x2, y1), (x2, y1 + vl), color, t)
    cv2.line(output, (x1, y2), (x1 + hl, y2), color, t)
    cv2.line(output, (x1, y2), (x1, y2 - vl), color, t)
    cv2.line(output, (x2, y2), (x2 - hl, y2), color, t)
    cv2.line(output, (x2, y2), (x2, y2 - vl), color, t)


def _draw_normal_boxes(
    output: np.ndarray,
    dets: List[Dict],
    *,
    show_confidence: bool,
    line_thickness: int,
    font_scale: float,
    font_thickness: int,
) -> None:
    h, w = output.shape[:2]
    for det in dets:
        cls_id = int(det["cls_id"])
        label = str(det["label"])
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        is_amb = _is_ambulance_label(label)
        color = color_for_class(label, cls_id)
        box_th = max(line_thickness + 5, 5) if is_amb else line_thickness
        fs = min(0.95, font_scale * (1.15 if is_amb else 1.0))
        ft = max(font_thickness, font_thickness + (1 if is_amb else 0))
        if is_amb:
            pad = 6
            cv2.rectangle(output, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (255, 255, 255), 2)
            cv2.rectangle(output, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (55, 55, 55), 1)
            _draw_ambulance_corner_brackets(output, x1, y1, x2, y2, color, max(box_th - 1, 3))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_th)
        disp = "AMBULANCE" if is_amb else label
        text = f"{disp} {float(det.get('conf', 0.0)):.2f}" if show_confidence else disp
        (tw_m, th_txt), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
        label_y2 = max(th_txt + baseline + 2, y1)
        label_y1 = max(0, label_y2 - th_txt - baseline - 6)
        label_x2 = min(w - 1, x1 + tw_m + 10)
        cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            output,
            text,
            (x1 + 5, max(th_txt + 2, label_y2 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (255, 255, 255),
            ft,
        )


def _draw_speed_boxes(
    output: np.ndarray,
    tracked: List[Dict],
    *,
    show_speed_overlays: bool,
    draw_confidence: bool = False,
    line_thickness: int,
    font_scale: float,
    font_thickness: int,
) -> None:
    h, w = output.shape[:2]
    for det in tracked:
        lb_det = str(det.get("label", ""))
        is_amb = _is_ambulance_label(lb_det)
        status = str(det.get("speed_status") or "Medium")
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        fs_sp = min(0.95, font_scale * (1.12 if is_amb else 1.0))
        ft_sp = max(font_thickness, font_thickness + (1 if is_amb else 0))
        if is_amb:
            color = AMBULANCE_BOX_BGR
            box_th = max(line_thickness + 5, 5)
            pad = 6
            cv2.rectangle(output, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (255, 255, 255), 2)
            cv2.rectangle(output, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (55, 55, 55), 1)
            _draw_ambulance_corner_brackets(output, x1, y1, x2, y2, color, max(box_th - 1, 3))
        else:
            color = (80, 80, 255) if status == "Slow" else ((0, 165, 255) if status == "Medium" else (80, 220, 80))
            box_th = line_thickness
        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_th)
        text = None
        if show_speed_overlays:
            text = "AMBULANCE" if is_amb else status
            if draw_confidence:
                text = f"{text} {float(det.get('conf', 0.0)):.2f}"
        elif draw_confidence:
            text = f"{float(det.get('conf', 0.0)):.2f}"
        if text is None:
            continue
        (tw_m, th_txt), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs_sp, ft_sp)
        label_y2 = max(th_txt + baseline + 2, y1)
        label_y1 = max(0, label_y2 - th_txt - baseline - 6)
        label_x2 = min(w - 1, x1 + tw_m + 10)
        cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            output,
            text,
            (x1 + 5, max(th_txt + 2, label_y2 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs_sp,
            (255, 255, 255),
            ft_sp,
        )


def _draw_minimal_boxes(
    output: np.ndarray,
    dets: List[Dict],
    *,
    line_thickness: int,
    draw_confidence: bool = False,
    font_scale: float = 0.45,
    font_thickness: int = 1,
) -> None:
    h, w = output.shape[:2]
    neutral = (165, 188, 210)
    for det in dets:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        lb = str(det.get("label", ""))
        is_amb = _is_ambulance_label(lb)
        col = AMBULANCE_BOX_BGR if is_amb else neutral
        box_th = max(line_thickness + 5, 5) if is_amb else line_thickness
        fs_m = min(0.85, font_scale * (1.12 if is_amb else 1.0))
        ft_m = max(font_thickness, font_thickness + (1 if is_amb else 0))
        if is_amb:
            pad = 6
            cv2.rectangle(output, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (255, 255, 255), 2)
            cv2.rectangle(output, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (55, 55, 55), 1)
            _draw_ambulance_corner_brackets(output, x1, y1, x2, y2, col, max(box_th - 1, 3))
        cv2.rectangle(output, (x1, y1), (x2, y2), col, box_th)
        if draw_confidence:
            disp = "AMBULANCE" if is_amb else lb
            text = f"{disp} {float(det.get('conf', 0.0)):.2f}"
            (tw_txt, th_txt), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs_m, ft_m)
            label_y2 = max(th_txt + baseline + 2, y1)
            label_y1 = max(0, label_y2 - th_txt - baseline - 6)
            label_x2 = min(w - 1, x1 + tw_txt + 8)
            cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), col, -1)
            cv2.putText(
                output,
                text,
                (x1 + 5, max(th_txt + 2, label_y2 - baseline - 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs_m,
                (255, 255, 255),
                ft_m,
            )


def _render_and_count_candidates(
    candidates: List[Dict],
    frame: np.ndarray,
    roi_polygons: List[List[Tuple[int, int]]],
    draw_confidence: bool = True,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
    display_mode: str = "normal_detection",
    show_speed_overlays: bool = True,
    track_state: Optional[Dict] = None,
    draw_boxes: bool = True,
) -> ZoneDetection:
    output = frame.copy()
    h, w = output.shape[:2]
    line_thickness = max(2, int(round(min(h, w) / 360)))
    font_scale = max(0.4, min(0.8, min(h, w) / 900.0))
    font_thickness = max(1, line_thickness - 1)
    polygons = [np.array(poly, dtype=np.int32) for poly in roi_polygons]
    has_roi = len(polygons) > 0
    per_zone = [Counter() for _ in roi_polygons] if has_roi else [Counter()]
    detections_total = 0

    filtered = apply_conf_and_size_filters(
        candidates,
        min_confidence=min_confidence,
        min_box_area=min_box_area,
    )

    roi_candidates, roi_zone_indices = _filter_candidates_by_roi(filtered, roi_polygons)
    if has_roi and logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "ROI filter: zones=%d total_detections=%d inside_roi=%d",
            len(polygons),
            len(filtered),
            len(roi_candidates),
        )

    dm_raw = (display_mode or "normal_detection").strip().lower()
    dm_draw = "normal_detection" if dm_raw == "congestion_heatmap" else dm_raw

    ts_base = dict(track_state or {})
    prev_hm_pack = ts_base.pop("congestion_heatmap", None)
    tracked, new_track_state = update_tracks(roi_candidates, ts_base)
    merged_track: Dict[str, Any] = dict(ts_base)
    merged_track.update(new_track_state or {})

    for cand, zone_idx in zip(roi_candidates, roi_zone_indices):
        label = str(cand["label"])
        detections_total += 1
        per_zone[zone_idx][label] += 1

    if draw_boxes:
        if dm_draw == "speed_status":
            _draw_speed_boxes(
                output,
                tracked,
                show_speed_overlays=bool(show_speed_overlays),
                draw_confidence=bool(draw_confidence),
                line_thickness=line_thickness,
                font_scale=font_scale,
                font_thickness=font_thickness,
            )
        elif dm_draw == "minimal":
            _draw_minimal_boxes(
                output,
                roi_candidates,
                line_thickness=line_thickness,
                draw_confidence=bool(draw_confidence),
                font_scale=font_scale,
                font_thickness=font_thickness,
            )
        elif dm_draw != "congestion_heatmap":
            _draw_normal_boxes(
                output,
                roi_candidates,
                show_confidence=bool(draw_confidence),
                line_thickness=line_thickness,
                font_scale=font_scale,
                font_thickness=font_thickness,
            )

    for i, poly in enumerate(polygons):
        color = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]
        cv2.polylines(output, [poly], isClosed=True, color=(128, 184, 210), thickness=max(1, line_thickness - 1))
        px, py = int(poly[0][0]), int(poly[0][1])
        cv2.putText(
            output,
            f"Zone {i + 1}",
            (px, max(int(18 * font_scale), py - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
        )

    if dm_raw == "congestion_heatmap":
        from monitoring.traffic_tdss.congestion_heatmap import update_and_render_heatmap

        roi_list: List[List[Tuple[int, int]]] = [list(map(tuple, poly)) for poly in roi_polygons]
        output, hm_packed = update_and_render_heatmap(output, roi_candidates, roi_list, prev_hm_pack)
        merged_track["congestion_heatmap"] = hm_packed
    else:
        merged_track.pop("congestion_heatmap", None)

    vehicle_detections: List[Dict] = []
    for det_item, zone_idx in zip(tracked, roi_zone_indices):
        cls_id = int(det_item.get("cls_id") or 0)
        vehicle_detections.append(
            {
                "track_id": int(det_item.get("track_id") or 0),
                "label": str(det_item.get("label") or "vehicle"),
                "cls_id": cls_id,
                "conf": round(float(det_item.get("conf") or 0.0), 4),
                "x1": int(det_item["x1"]),
                "y1": int(det_item["y1"]),
                "x2": int(det_item["x2"]),
                "y2": int(det_item["y2"]),
                "zone_idx": int(zone_idx),
                "speed_px": round(float(det_item.get("speed_px") or 0.0), 3),
                "speed_status": str(det_item.get("speed_status") or "Medium"),
            }
        )

    return ZoneDetection(
        per_zone_counts=per_zone,
        processed_frame=output,
        detections_total=detections_total,
        track_state=merged_track,
        vehicle_detections=vehicle_detections,
    )


def detect_in_zones(
    model: YOLO,
    frame: np.ndarray,
    roi_polygons: List[List[Tuple[int, int]]],
    allowed_labels: Optional[Set[str]] = None,
    imgsz: int = 416,
    max_det: int = 80,
    draw_confidence: bool = True,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
    display_mode: str = "normal_detection",
    show_speed_overlays: bool = True,
    track_state: Optional[Dict] = None,
    draw_boxes: bool = True,
) -> ZoneDetection:
    candidates = _run_model_candidates(model, frame, allowed_labels, imgsz, max_det)
    return _render_and_count_candidates(
        candidates,
        frame,
        roi_polygons,
        draw_confidence=draw_confidence,
        min_confidence=min_confidence,
        min_box_area=min_box_area,
        display_mode=display_mode,
        show_speed_overlays=show_speed_overlays,
        track_state=track_state,
        draw_boxes=draw_boxes,
    )


def detect_in_zones_multi(
    models: List[YOLO],
    frame: np.ndarray,
    roi_polygons: List[List[Tuple[int, int]]],
    allowed_labels: Optional[Set[str]] = None,
    imgsz: int = 416,
    max_det: int = 80,
    draw_confidence: bool = True,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
    display_mode: str = "normal_detection",
    show_speed_overlays: bool = True,
    track_state: Optional[Dict] = None,
    draw_boxes: bool = True,
) -> ZoneDetection:
    candidates: List[Dict] = []
    for mi, model in enumerate(models):
        al = allowed_labels
        if len(models) >= 2 and mi == 0:
            al = None
        batch = _run_model_candidates(model, frame, al, imgsz, max_det)
        # Premier modèle = weights ambulance (ex. models_traffic/best.pt) : garder détections utiles ambulances.
        if len(models) >= 2 and mi == 0:
            batch = filter_ambulance_candidates(batch)
        candidates.extend(batch)
    ov_iou = _ambulance_overlap_iou_from_settings()
    candidates = _prefer_ambulance_over_vehicle_overlap(candidates, ov_iou)
    merged = _nms_by_label(candidates, iou_thresh=0.5)
    return _render_and_count_candidates(
        merged,
        frame,
        roi_polygons,
        draw_confidence=draw_confidence,
        min_confidence=min_confidence,
        min_box_area=min_box_area,
        display_mode=display_mode,
        show_speed_overlays=show_speed_overlays,
        track_state=track_state,
        draw_boxes=draw_boxes,
    )


def infer_candidates_multi(
    models: List[YOLO],
    frame: np.ndarray,
    allowed_labels: Optional[Set[str]] = None,
    imgsz: int = 416,
    max_det: int = 80,
) -> List[Dict]:
    """Run one or multiple models and return merged raw candidates."""
    candidates: List[Dict] = []
    for mi, model in enumerate(models):
        al = allowed_labels
        if len(models) >= 2 and mi == 0:
            al = None
        batch = _run_model_candidates(model, frame, al, imgsz, max_det)
        if len(models) >= 2 and mi == 0:
            batch = filter_ambulance_candidates(batch)
        candidates.extend(batch)
    ov_iou = _ambulance_overlap_iou_from_settings()
    candidates = _prefer_ambulance_over_vehicle_overlap(candidates, ov_iou)
    return _nms_by_label(candidates, iou_thresh=0.5)


def infer_candidates_congestion_ambulance(
    model_congestion: YOLO,
    model_ambulance: YOLO,
    frame: np.ndarray,
    imgsz: int = 416,
    max_det: int = 80,
) -> List[Dict]:
    """Congestion (toutes classes) + best.pt filtré ambulances, fusion NMS."""
    c_cong = _run_model_candidates(model_congestion, frame, None, imgsz, max_det)
    c_amb_raw = _run_model_candidates(model_ambulance, frame, None, imgsz, max_det)
    c_amb = filter_ambulance_candidates(c_amb_raw)
    merged = c_cong + c_amb
    ov_iou = _ambulance_overlap_iou_from_settings()
    merged = _prefer_ambulance_over_vehicle_overlap(merged, ov_iou)
    return _nms_by_label(merged, iou_thresh=0.45)


def detect_in_zones_congestion_ambulance(
    model_congestion: YOLO,
    model_ambulance: YOLO,
    frame: np.ndarray,
    roi_polygons: List[List[Tuple[int, int]]],
    imgsz: int = 416,
    max_det: int = 80,
    draw_confidence: bool = True,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
    display_mode: str = "normal_detection",
    show_speed_overlays: bool = True,
    track_state: Optional[Dict] = None,
    draw_boxes: bool = True,
) -> ZoneDetection:
    merged = infer_candidates_congestion_ambulance(
        model_congestion, model_ambulance, frame, imgsz=imgsz, max_det=max_det
    )
    return _render_and_count_candidates(
        merged,
        frame,
        roi_polygons,
        draw_confidence=draw_confidence,
        min_confidence=min_confidence,
        min_box_area=min_box_area,
        display_mode=display_mode,
        show_speed_overlays=show_speed_overlays,
        track_state=track_state,
        draw_boxes=draw_boxes,
    )


def detect_from_candidates(
    frame: np.ndarray,
    candidates: List[Dict],
    roi_polygons: List[List[Tuple[int, int]]],
    draw_confidence: bool = True,
    min_confidence: float = 0.35,
    min_box_area: int = 180,
    display_mode: str = "normal_detection",
    show_speed_overlays: bool = True,
    track_state: Optional[Dict] = None,
    draw_boxes: bool = True,
) -> ZoneDetection:
    """Build zone counts + annotated frame from precomputed candidates."""
    return _render_and_count_candidates(
        candidates,
        frame,
        roi_polygons,
        draw_confidence=draw_confidence,
        min_confidence=min_confidence,
        min_box_area=min_box_area,
        display_mode=display_mode,
        show_speed_overlays=show_speed_overlays,
        track_state=track_state,
        draw_boxes=draw_boxes,
    )

