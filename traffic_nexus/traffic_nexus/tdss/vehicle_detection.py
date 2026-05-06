from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

ZONE_COLORS_BGR = [
    (255, 80, 80),
    (80, 220, 80),
    (80, 170, 255),
    (220, 120, 255),
    (255, 220, 80),
]

CLASS_COLORS_BGR = {
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


def color_for_class(label: str, cls_id: int) -> tuple:
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


def _point_in_poly(poly: np.ndarray, x: int, y: int) -> bool:
    return cv2.pointPolygonTest(poly, (x, y), False) >= 0


def _assign_zone_for_bbox(polygons: List[np.ndarray], x1: int, y1: int, x2: int, y2: int) -> int:
    """Robust zone assignment: center + corners + edge midpoints."""
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    sample_points = [
        (cx, cy),
        (x1, y1),
        (x2, y1),
        (x1, y2),
        (x2, y2),
        (cx, y1),
        (cx, y2),
        (x1, cy),
        (x2, cy),
    ]
    for i, poly in enumerate(polygons):
        if any(_point_in_poly(poly, px, py) for px, py in sample_points):
            return i
    return -1


@dataclass
class ZoneDetection:
    per_zone_counts: List[Counter]
    processed_frame: np.ndarray
    detections_total: int


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
    if not allowed_norm:
        return True
    lb = str(label).strip().lower()
    if lb in allowed_norm:
        return True
    if "ambulance" in allowed_norm:
        if "ambul" in lb or lb in ("emergency", "ems") or "emergency" in lb:
            return True
    return False


def filter_ambulance_candidates(
    candidates: List[Dict],
    keywords: Optional[Sequence[str]] = None,
) -> List[Dict]:
    """Garde les détections ambulance si confiance >= TRAFFIC_AMBULANCE_MIN_CONF (défaut 0,85)."""
    amb_min = 0.85
    try:
        from django.conf import settings as dj_settings

        amb_min = float(getattr(dj_settings, "TRAFFIC_AMBULANCE_MIN_CONF", amb_min))
    except Exception:
        pass
    keys = [str(k).strip().lower() for k in (keywords or ("ambulance", "emergency", "ems")) if str(k).strip()]
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
    if not out and candidates:
        hi = [dict(c) for c in candidates if float(c.get("conf", 0.0)) >= float(amb_min)]
        return sorted(hi, key=lambda x: -float(x.get("conf", 0.0)))[:25]
    return out


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


def _render_and_count_candidates(
    candidates: List[Dict],
    frame: np.ndarray,
    roi_polygons: List[List[Tuple[int, int]]],
    draw_confidence: bool = True,
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

    for cand in candidates:
        cls_id = int(cand["cls_id"])
        label = str(cand["label"])
        x1, y1, x2, y2 = int(cand["x1"]), int(cand["y1"]), int(cand["x2"]), int(cand["y2"])
        if has_roi:
            zone_idx = _assign_zone_for_bbox(polygons, x1, y1, x2, y2)
            if zone_idx < 0:
                continue
        else:
            zone_idx = 0

        detections_total += 1
        per_zone[zone_idx][label] += 1
        color = color_for_class(label, cls_id)
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, line_thickness)
        text = f"{label} {float(cand['conf']):.2f}" if draw_confidence else label
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        label_y2 = max(th + baseline + 2, y1)
        label_y1 = max(0, label_y2 - th - baseline - 6)
        label_x2 = min(w - 1, x1 + tw + 8)
        cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(
            output,
            text,
            (x1 + 4, max(th + 2, label_y2 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )
        cv2.circle(output, (cx, cy), max(2, line_thickness), color, -1)

    for i, poly in enumerate(polygons):
        color = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]
        cv2.polylines(output, [poly], isClosed=True, color=color, thickness=line_thickness)
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

    return ZoneDetection(per_zone_counts=per_zone, processed_frame=output, detections_total=detections_total)


def detect_in_zones(
    model: YOLO,
    frame: np.ndarray,
    roi_polygons: List[List[Tuple[int, int]]],
    allowed_labels: Optional[Set[str]] = None,
    imgsz: int = 416,
    max_det: int = 80,
    draw_confidence: bool = True,
) -> ZoneDetection:
    candidates = _run_model_candidates(model, frame, allowed_labels, imgsz, max_det)
    return _render_and_count_candidates(candidates, frame, roi_polygons, draw_confidence=draw_confidence)


def detect_in_zones_multi(
    models: List[YOLO],
    frame: np.ndarray,
    roi_polygons: List[List[Tuple[int, int]]],
    allowed_labels: Optional[Set[str]] = None,
    imgsz: int = 416,
    max_det: int = 80,
    draw_confidence: bool = True,
) -> ZoneDetection:
    candidates: List[Dict] = []
    for mi, model in enumerate(models):
        al = allowed_labels
        if len(models) >= 2 and mi == 0:
            al = None
        batch = _run_model_candidates(model, frame, al, imgsz, max_det)
        if len(models) >= 2 and mi == 0:
            batch = filter_ambulance_candidates(batch)
        candidates.extend(batch)
    merged = _nms_by_label(candidates, iou_thresh=0.5)
    return _render_and_count_candidates(merged, frame, roi_polygons, draw_confidence=draw_confidence)


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
    return _nms_by_label(candidates, iou_thresh=0.5)


def detect_from_candidates(
    frame: np.ndarray,
    candidates: List[Dict],
    roi_polygons: List[List[Tuple[int, int]]],
    draw_confidence: bool = True,
) -> ZoneDetection:
    """Build zone counts + annotated frame from precomputed candidates."""
    return _render_and_count_candidates(candidates, frame, roi_polygons, draw_confidence=draw_confidence)

