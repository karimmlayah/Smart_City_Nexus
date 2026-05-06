"""Utilitaires Traffic Nexus (polygones ROI, état scène) — logique alignée sur Streamlit app.py."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple

import cv2
import numpy as np

VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle", "ambulance"}


def polygon_area(poly: List[Tuple[int, int]]) -> float:
    return float(abs(cv2.contourArea(np.array(poly, dtype=np.float32))))


def scale_polygon(
    poly: List[Tuple[int, int]],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> List[Tuple[int, int]]:
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return poly
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    out: List[Tuple[int, int]] = []
    for x, y in poly:
        nx = int(round(float(x) * sx))
        ny = int(round(float(y) * sy))
        nx = max(0, min(dst_w - 1, nx))
        ny = max(0, min(dst_h - 1, ny))
        out.append((nx, ny))
    return out


def _polygons_from_fabric_object(obj: dict) -> List[List[Tuple[int, int]]]:
    polys: List[List[Tuple[int, int]]] = []
    if obj.get("type") in {"group", "activeSelection"}:
        for child in obj.get("objects", []) or []:
            polys.extend(_polygons_from_fabric_object(child))
        return polys

    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))
    path_offset = obj.get("pathOffset", {}) or {}
    off_x = float(path_offset.get("x", 0))
    off_y = float(path_offset.get("y", 0))

    def to_abs(x: float, y: float) -> Tuple[int, int]:
        ax = left + (x - off_x) * scale_x
        ay = top + (y - off_y) * scale_y
        return int(round(ax)), int(round(ay))

    if obj.get("type") == "path" and obj.get("path"):
        pts = []
        for cmd in obj["path"]:
            if isinstance(cmd, list) and len(cmd) >= 3:
                x, y = cmd[-2], cmd[-1]
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    pts.append(to_abs(float(x), float(y)))
        if len(pts) >= 3:
            polys.append(pts)

    if obj.get("type") == "polygon" and obj.get("points"):
        pts = []
        for p in obj["points"]:
            x = float(p.get("x", 0))
            y = float(p.get("y", 0))
            pts.append(to_abs(x, y))
        if len(pts) >= 3:
            polys.append(pts)

    if obj.get("type") == "rect":
        left_i = int(round(left))
        top_i = int(round(top))
        width = int(round(float(obj.get("width", 0)) * scale_x))
        height = int(round(float(obj.get("height", 0)) * scale_y))
        if width > 2 and height > 2:
            polys.append(
                [
                    (left_i, top_i),
                    (left_i + width, top_i),
                    (left_i + width, top_i + height),
                    (left_i, top_i + height),
                ]
            )

    if not polys and all(k in obj for k in ("left", "top", "width", "height")):
        left_i = int(round(left))
        top_i = int(round(top))
        width = int(round(float(obj.get("width", 0)) * scale_x))
        height = int(round(float(obj.get("height", 0)) * scale_y))
        if width > 2 and height > 2:
            polys.append(
                [
                    (left_i, top_i),
                    (left_i + width, top_i),
                    (left_i + width, top_i + height),
                    (left_i, top_i + height),
                ]
            )
    return polys


def extract_polygon_points(canvas_data: dict) -> List[List[Tuple[int, int]]]:
    """Extrait les polygones depuis le JSON Fabric.js (streamlit_drawable_canvas compatible)."""
    if not canvas_data or "objects" not in canvas_data:
        return []
    out: List[List[Tuple[int, int]]] = []
    for obj in canvas_data["objects"]:
        out.extend(_polygons_from_fabric_object(obj))
    cleaned: List[List[Tuple[int, int]]] = []
    seen = set()
    for poly in out:
        if len(poly) < 3:
            continue
        area = abs(cv2.contourArea(np.array(poly, dtype=np.float32)))
        if area < 20:
            continue
        key = tuple(poly)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(poly)
    return cleaned


def build_scene_state(
    zone_states: list,
    decisions: list,
    per_zone_counts: List[Counter],
    mock_emergency: bool,
) -> Dict:
    zones: List[Dict] = []
    for i, zs in enumerate(zone_states):
        z_dec = decisions[i] if i < len(decisions) else None
        z_counts = per_zone_counts[i] if i < len(per_zone_counts) else Counter()
        detections: List[str] = []
        for label, count in z_counts.items():
            detections.extend([label] * int(count))
        zones.append(
            {
                "id": f"Zone {i + 1}",
                "congestion": float(zs.density),
                "vehicle_count": int(zs.vehicle_count),
                "waiting_time": float(z_dec.estimated_waiting_time_min if z_dec else 0.0),
                "criticality": int(z_dec.traffic_criticality_score if z_dec else 0),
                "traffic_light": "adaptive",
                "detections": detections,
            }
        )
    return {"zones": zones, "mock_emergency": mock_emergency}


def counter_list_serializable(per_zone: List[Counter]) -> List[Dict[str, int]]:
    return [dict(c) for c in per_zone]


def resize_frame_max_side(frame_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    if frame_bgr is None or max_side <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame_bgr
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def bgr_to_jpeg_base64(frame_bgr: np.ndarray, quality: int = 88) -> str:
    import base64

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")
