import time
import json
import io
import hashlib
import os
from collections import Counter, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib import request as urllib_request

import cv2
import numpy as np
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image
from streamlit.elements.lib.image_utils import image_to_url
from streamlit.elements.lib.layout_utils import LayoutConfig
from streamlit_drawable_canvas import st_canvas
from ultralytics import YOLO

from tdss.congestion_analysis import compute_zone_states
from tdss.decision_engine import DecisionEngine
from tdss.agentic_system import AgenticOrchestrator
from tdss.ui_dashboard import (
    inject_modern_css,
    render_alert_if_needed,
    render_agentic_control_center,
    render_congestion_class_cards,
    render_class_counts_table,
    render_global_overview,
    render_xai,
    render_zone_counts_table,
    render_zone_decision,
)
from tdss.vehicle_detection import (
    ZONE_COLORS_BGR,
    color_for_class,
    detect_from_candidates,
    detect_in_zones,
    detect_in_zones_multi,
    infer_candidates_multi,
)
from tdss.video_processing import read_first_frame, resolve_video_source
from tdss.xai import explain_decision
from tdss.esp32_bridge import (
    build_esp32_payload,
    esp32_decision_signature,
    format_esp32_log_line,
    infer_emergency_zone,
    new_emergency_confirmation_state,
    post_json,
    step_emergency_confirmation,
)


def _canvas_compatible_image_to_url(image, width, clamp, channels, output_format, image_id):
    # Try legacy width-based signature first (best rendering behavior when available),
    # then fallback to layout_config signature used by newer Streamlit internals.
    try:
        return image_to_url(
            image=image,
            width=width,
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )
    except TypeError:
        return image_to_url(
            image=image,
            layout_config=LayoutConfig(width=width),
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )


st_image.image_to_url = _canvas_compatible_image_to_url


def _resolve_traffic_best_weights() -> str:
    """best.pt pour ambulances — même ordre que Django : models_traffic/, paquet, racine projet."""
    root = Path(__file__).resolve().parent.parent.parent
    inner_pkg = Path(__file__).resolve().parent
    for candidate in (
        root / "models_traffic" / "best.pt",
        inner_pkg / "best.pt",
        root / "best.pt",
    ):
        if candidate.is_file():
            return str(candidate)
    return str(root / "models_traffic" / "best.pt")
st.set_page_config(page_title="Traffic Decision Support System", layout="wide")

DEFAULT_MODEL = "yolov8n.pt"
DEFAULT_CONGESTION_MODEL = "yolo_congestion.pt"
VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}


def _polygons_from_fabric_object(obj: dict) -> List[List[Tuple[int, int]]]:
    polys: List[List[Tuple[int, int]]] = []
    if obj.get("type") in {"group", "activeSelection"}:
        for child in obj.get("objects", []) or []:
            polys.extend(_polygons_from_fabric_object(child))
        return polys

    # Fabric often stores points relative to object origin.
    # Convert to absolute canvas coordinates for reliable multi-zone support.
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
    # Fallback: some Fabric objects still carry a bbox even if type is unexpected.
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


def extract_polygon_points(canvas_data: Dict) -> List[List[Tuple[int, int]]]:
    if not canvas_data or "objects" not in canvas_data:
        return []
    out: List[List[Tuple[int, int]]] = []
    for obj in canvas_data["objects"]:
        out.extend(_polygons_from_fabric_object(obj))
    # Keep only valid non-degenerate polygons and avoid duplicates.
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


def polygon_area(poly: List[Tuple[int, int]]) -> float:
    return float(abs(cv2.contourArea(np.array(poly, dtype=np.float32))))


def _scale_polygon(
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


def build_zone_heatmap(frame: np.ndarray, polygons: List[List[Tuple[int, int]]], densities: List[float]) -> np.ndarray:
    h, w = frame.shape[:2]
    heat_float = np.zeros((h, w), dtype=np.float32)
    for i, poly in enumerate(polygons):
        density = float(max(0.0, min(1.0, densities[i]))) if i < len(densities) else 0.0
        if density <= 0:
            continue

        zone_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(zone_mask, [np.array(poly, dtype=np.int32)], 255)

        # Build a smooth radial gradient inside each zone:
        # low values near borders, high values toward zone center.
        dist = cv2.distanceTransform(zone_mask, cv2.DIST_L2, 3)
        max_dist = float(dist.max())
        if max_dist <= 1e-6:
            continue
        zone_grad = (dist / max_dist) * density
        heat_float = np.maximum(heat_float, zone_grad)

    # Global blur to avoid blocky edges and create a real heatmap feel.
    heat_float = cv2.GaussianBlur(heat_float, (0, 0), sigmaX=18, sigmaY=18)
    mask = np.clip(heat_float * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(mask, cv2.COLORMAP_TURBO)

    # Blend with frame for a more natural visualization.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    alpha = np.expand_dims(np.clip(heat_float * 0.75, 0.0, 0.75), axis=2)
    blended = (frame_rgb.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return blended


@st.cache_resource
def load_model(path: str) -> YOLO:
    return YOLO(path)


def _bgr_to_hex(color_bgr: Tuple[int, int, int]) -> str:
    b, g, r = color_bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _build_scene_state(
    zone_states: List,
    decisions: List,
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


def _auto_precompute_results(
    source: str,
    models: List[YOLO],
    allowed_labels: Optional[Set[str]],
    imgsz: int,
    max_det: int,
    frame_step: int,
    progress_cb=None,
) -> Dict[int, List[Dict]]:
    out: Dict[int, List[Dict]] = {}
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return out
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % max(1, frame_step) != 0:
            if progress_cb is not None and total_frames > 0 and frame_idx % 20 == 0:
                progress_cb(frame_idx, total_frames, processed)
            continue
        out[frame_idx] = infer_candidates_multi(
            models=models,
            frame=frame,
            allowed_labels=allowed_labels,
            imgsz=imgsz,
            max_det=max_det,
        )
        processed += 1
        if progress_cb is not None:
            progress_cb(frame_idx, total_frames, processed)
    cap.release()
    return out


def _append_report_record(
    records: deque,
    total_vehicles: int,
    avg_density: float,
    global_level: str,
    global_criticality: int,
    final_decision: str,
    final_recommendation: str,
) -> None:
    records.append(
        {
            "ts": datetime.utcnow().isoformat(),
            "vehicles": int(total_vehicles),
            "avg_density": float(avg_density),
            "congestion_level": str(global_level),
            "criticality": int(global_criticality),
            "final_decision": str(final_decision),
            "final_recommendation": str(final_recommendation),
        }
    )


def _filter_records_by_days(records: List[Dict], days: int) -> List[Dict]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    out: List[Dict] = []
    for r in records:
        try:
            ts = datetime.fromisoformat(str(r.get("ts", "")))
            if ts >= cutoff:
                out.append(r)
        except Exception:
            continue
    return out


def _report_summary(records: List[Dict]) -> Dict:
    if not records:
        return {
            "samples": 0,
            "avg_vehicles": 0.0,
            "avg_density": 0.0,
            "avg_criticality": 0.0,
            "peak_vehicles": 0,
            "peak_density": 0.0,
            "peak_criticality": 0,
            "top_decision": "-",
            "top_level": "-",
        }
    decisions = Counter([str(r.get("final_decision", "-")) for r in records])
    levels = Counter([str(r.get("congestion_level", "-")) for r in records])
    return {
        "samples": len(records),
        "avg_vehicles": float(np.mean([float(r.get("vehicles", 0)) for r in records])),
        "avg_density": float(np.mean([float(r.get("avg_density", 0.0)) for r in records])),
        "avg_criticality": float(np.mean([float(r.get("criticality", 0)) for r in records])),
        "peak_vehicles": int(max(float(r.get("vehicles", 0)) for r in records)),
        "peak_density": float(max(float(r.get("avg_density", 0.0)) for r in records)),
        "peak_criticality": int(max(float(r.get("criticality", 0)) for r in records)),
        "top_decision": decisions.most_common(1)[0][0] if decisions else "-",
        "top_level": levels.most_common(1)[0][0] if levels else "-",
    }


def _report_xai_features(records: List[Dict], summary: Dict) -> Dict[str, float | str]:
    avg_vehicles = float(summary.get("avg_vehicles", 0.0))
    peak_vehicles = max(1.0, float(summary.get("peak_vehicles", 0)))
    avg_density = float(summary.get("avg_density", 0.0))
    avg_criticality = float(summary.get("avg_criticality", 0.0))
    waiting_proxy = min(100.0, max(0.0, avg_criticality * 1.2))
    volume_pressure = min(100.0, max(0.0, (avg_vehicles / peak_vehicles) * 100.0))
    congestion_pressure = min(100.0, max(0.0, avg_density * 100.0))
    criticality_pressure = min(100.0, max(0.0, avg_criticality))

    factors = {
        "Congestion pressure": congestion_pressure,
        "Vehicle pressure": volume_pressure,
        "Estimated waiting impact": waiting_proxy,
        "Criticality pressure": criticality_pressure,
    }
    dominant = max(factors.items(), key=lambda kv: kv[1])[0] if factors else "Congestion pressure"
    return {
        "congestion_pressure": congestion_pressure,
        "vehicle_pressure": volume_pressure,
        "waiting_impact": waiting_proxy,
        "criticality_pressure": criticality_pressure,
        "dominant_driver": dominant,
    }


def _build_local_narrative(records: List[Dict], summary: Dict, period_title: str) -> Dict[str, str]:
    decisions = Counter([str(r.get("final_decision", "-")) for r in records])
    levels = Counter([str(r.get("congestion_level", "-")) for r in records])
    xai = _report_xai_features(records, summary)

    top_decision = decisions.most_common(1)[0][0] if decisions else "-"
    top_level = levels.most_common(1)[0][0] if levels else "-"
    peak_vehicles = int(summary.get("peak_vehicles", 0))
    peak_criticality = int(summary.get("peak_criticality", 0))

    executive = (
        f"Over **{period_title.lower()}**, the system analyzed **{int(summary.get('samples', 0))} traffic snapshots**. "
        f"Average load remains around **{float(summary.get('avg_vehicles', 0.0)):.1f} vehicles** with a mean density "
        f"of **{float(summary.get('avg_density', 0.0)):.2f}** and average criticality of **{float(summary.get('avg_criticality', 0.0)):.1f}/100**."
    )
    operations = (
        f"The dominant control strategy is **{top_decision}**, while the most frequent traffic state is **{top_level.upper()}**. "
        f"Peak stress reaches **{peak_vehicles} vehicles** and **{peak_criticality}/100 criticality**, indicating periods where adaptive control is required."
    )
    xai_txt = (
        f"The XAI layer indicates that **{xai['dominant_driver']}** is the strongest contributor in this period. "
        f"Influence scores show congestion at **{float(xai['congestion_pressure']):.1f}%**, vehicle pressure at "
        f"**{float(xai['vehicle_pressure']):.1f}%**, waiting impact at **{float(xai['waiting_impact']):.1f}%**, "
        f"and criticality at **{float(xai['criticality_pressure']):.1f}%**."
    )
    conclusion = (
        "Operationally, this suggests keeping adaptive signal plans active during medium/high phases, "
        "prioritizing zones with recurring peaks, and monitoring transitions from low to medium congestion to prevent escalation."
    )
    return {
        "executive": executive,
        "operations": operations,
        "xai": xai_txt,
        "conclusion": conclusion,
    }


def _generate_local_report(records: List[Dict], summary: Dict, period_title: str) -> str:
    decisions = Counter([str(r.get("final_decision", "-")) for r in records])
    levels = Counter([str(r.get("congestion_level", "-")) for r in records])
    recos = Counter([str(r.get("final_recommendation", "-")) for r in records if str(r.get("final_recommendation", "")).strip()])
    top_decisions = ", ".join([f"{k} ({v})" for k, v in decisions.most_common(3)]) or "-"
    top_levels = ", ".join([f"{k} ({v})" for k, v in levels.most_common(3)]) or "-"
    top_recos = ", ".join([f"{k} ({v})" for k, v in recos.most_common(3)]) or "-"
    narrative = _build_local_narrative(records, summary, period_title)
    return (
        f"### Traffic Report ({period_title})\n\n"
        f"### Executive Summary\n\n{narrative['executive']}\n\n"
        f"### Operational Dynamics\n\n{narrative['operations']}\n\n"
        f"Top decision patterns: **{top_decisions}**.\n\n"
        f"Top congestion levels: **{top_levels}**.\n\n"
        f"Frequent recommendations: **{top_recos}**.\n\n"
        f"### XAI Interpretation\n\n{narrative['xai']}\n\n"
        f"### Conclusion\n\n{narrative['conclusion']}"
    )


def _generate_groq_report(
    records: List[Dict],
    summary: Dict,
    period_title: str,
) -> Tuple[Optional[str], Optional[str]]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        # Fallback to Streamlit secrets if available.
        try:
            api_key = str(st.secrets.get("GROQ_API_KEY", "")).strip()  # type: ignore[arg-type]
        except Exception:
            api_key = ""
    if not api_key:
        return None, "GROQ_API_KEY introuvable."

    recent = records[-30:]
    decisions = Counter([str(r.get("final_decision", "-")) for r in records])
    recos = Counter([str(r.get("final_recommendation", "-")) for r in records if str(r.get("final_recommendation", "-")).strip()])
    top_decisions = ", ".join([f"{k} ({v})" for k, v in decisions.most_common(5)]) or "-"
    top_recos = ", ".join([f"{k} ({v})" for k, v in recos.most_common(5)]) or "-"

    payload = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an intelligent traffic analyst. Write a professional report in English, "
                    "clear, structured, and operations-focused."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Period: {period_title}\n"
                    f"KPI summary: {json.dumps(summary, ensure_ascii=False)}\n"
                    f"Top decisions: {top_decisions}\n"
                    f"Top recommendations: {top_recos}\n"
                    f"Latest records: {json.dumps(recent, ensure_ascii=False)}\n\n"
                    "Write a report with these sections:\n"
                    "1) Executive summary\n"
                    "2) Key KPIs\n"
                    "3) Peaks and notable events\n"
                    "4) Decisions taken and expected impact\n"
                    "5) Actionable recommendations (short-term / mid-term)\n"
                    "6) Overall risk level\n"
                    "Stay factual and do not invent data that is not provided."
                ),
            },
        ],
    }

    try:
        req = urllib_request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        report = (
            body.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        ) or None
        if not report:
            return None, "Reponse Groq vide."
        return report, None
    except Exception as exc:
        return None, f"Echec appel Groq: {exc}"


def _status_from_criticality(avg_criticality: float) -> Tuple[str, str]:
    if avg_criticality > 70:
        return "CRITICAL", "#ef4444"
    if avg_criticality >= 40:
        return "WARNING", "#f59e0b"
    return "NORMAL", "#22c55e"


def _record_observation(level: str, criticality: int) -> str:
    lvl = (level or "").lower()
    if lvl == "high" or criticality > 70:
        return "High-risk traffic phase"
    if lvl == "medium" and criticality >= 40:
        return "Potential escalation"
    if lvl == "medium":
        return "Medium congestion detected"
    return "Stable flow"


def _ai_interpretation(summary: Dict) -> Dict[str, str]:
    avg_criticality = float(summary.get("avg_criticality", 0.0))
    avg_density = float(summary.get("avg_density", 0.0))
    avg_vehicles = float(summary.get("avg_vehicles", 0.0))
    peak_vehicles = int(summary.get("peak_vehicles", 0))
    peak_criticality = int(summary.get("peak_criticality", 0))

    if avg_criticality > 70:
        headline = "Critical traffic stress detected across the analyzed period."
        detail = (
            f"Average criticality is {avg_criticality:.1f}/100, with peaks up to {peak_criticality}/100. "
            "This indicates sustained high operational risk and limited resilience."
        )
        conclusion = "Immediate mitigation is required with aggressive adaptive signal plans and escalation alerts."
    elif avg_criticality >= 40:
        headline = "Moderate pressure requires continuous surveillance."
        detail = (
            f"Average criticality is {avg_criticality:.1f}/100 and average density is {avg_density:.2f}. "
            f"Traffic stays manageable but recurrent peaks (vehicles up to {peak_vehicles}) suggest instability windows."
        )
        conclusion = "Maintain adaptive control and proactively protect high-density intersections."
    else:
        headline = "Traffic flow remains globally stable."
        detail = (
            f"Average criticality is {avg_criticality:.1f}/100 with average load around {avg_vehicles:.1f} vehicles. "
            "The network remains in a normal operating envelope."
        )
        conclusion = "Keep baseline operations and maintain monitoring to detect early trend changes."

    return {
        "headline": headline,
        "detail": detail,
        "conclusion": conclusion,
    }


def _dynamic_actions(summary: Dict, xai: Dict[str, float | str]) -> List[str]:
    actions: List[str] = []
    avg_criticality = float(summary.get("avg_criticality", 0.0))
    avg_density = float(summary.get("avg_density", 0.0))
    peak_vehicles = int(summary.get("peak_vehicles", 0))
    dominant = str(xai.get("dominant_driver", "Congestion pressure"))

    actions.append("Adjust signal cycles dynamically according to real-time queue pressure.")
    if avg_density >= 0.30:
        actions.append("Prioritize intersections with high sustained density to reduce spillback risk.")
    if peak_vehicles >= 8:
        actions.append("Monitor zones with recurring vehicle peaks and pre-allocate green windows.")
    if avg_criticality >= 40:
        actions.append("Trigger supervisory alerts when criticality remains above warning threshold.")
    if avg_criticality > 70:
        actions.append("Activate short-term emergency plan: stricter inflow control and rerouting.")
    else:
        actions.append("Keep a short-term monitoring plan with periodic reassessment every control cycle.")

    if dominant == "Vehicle pressure":
        actions.append("Dominant factor action: smooth incoming platoons using staged green offsets.")
    elif dominant == "Congestion pressure":
        actions.append("Dominant factor action: increase clearance times for congested corridors.")
    elif dominant == "Estimated waiting impact":
        actions.append("Dominant factor action: rebalance cycle splits to reduce average waiting time.")
    else:
        actions.append("Dominant factor action: enforce proactive risk alerts for control operators.")
    return actions


def _xai_feature_explanations() -> Dict[str, str]:
    return {
        "Vehicle pressure": "Impact of detected vehicle volume on signal control urgency.",
        "Congestion pressure": "Impact of measured congestion intensity across monitored zones.",
        "Waiting impact": "Impact of estimated waiting time on user experience and queue buildup.",
        "Criticality pressure": "Impact of operational risk level on priority and mitigation actions.",
    }


def _safe_text(value: object, fallback: str = "Not available") -> str:
    txt = str(value).strip() if value is not None else ""
    return txt if txt else fallback


def _wrap_cell_text(text: str, width: int = 22) -> str:
    s = _safe_text(text)
    if len(s) <= width:
        return s
    chunks = [s[i : i + width] for i in range(0, len(s), width)]
    return "\n".join(chunks[:3])


def _build_pdf_report(records: List[Dict], summary: Dict, period_title: str) -> Optional[bytes]:
    narrative = _build_local_narrative(records, summary, period_title)
    xai = _report_xai_features(records, summary)
    ai_interp = _ai_interpretation(summary)
    status_label, status_color = _status_from_criticality(float(summary.get("avg_criticality", 0.0)))
    actions = _dynamic_actions(summary, xai)
    xai_explain = _xai_feature_explanations()
    try:
        import textwrap
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception:
        lines = [
            "Smart City Traffic Intelligence Report",
            f"Analyzed period: {period_title}",
            f"Generated (UTC): {datetime.utcnow().isoformat()}",
            f"Global status: {status_label}",
            "",
            "Executive Summary",
            narrative["executive"],
            "",
            "AI Interpretation",
            ai_interp["headline"],
            ai_interp["detail"],
            "",
            "XAI",
            narrative["xai"],
            "",
            "Recommended Actions",
        ]
        lines.extend([f"- {a}" for a in actions])
        lines.extend(["", "Recent records"])
        for r in records[-40:]:
            lines.append(
                f"{r.get('ts','')} | veh={r.get('vehicles',0)} | dens={float(r.get('avg_density',0.0)):.2f} | "
                f"lvl={r.get('congestion_level','-')} | crit={r.get('criticality',0)} | obs={_record_observation(str(r.get('congestion_level','-')), int(r.get('criticality',0)))}"
            )
        return _build_minimal_pdf(lines)

    def _theme(fig):
        fig.patch.set_facecolor("#0b1220")

    def _darken(ax):
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    out = io.BytesIO()
    with PdfPages(out) as pdf:
        recent = records[-40:] if records else []
        x = list(range(1, len(recent) + 1))
        veh = [int(r.get("vehicles", 0)) for r in recent] or [0]
        dens = [float(r.get("avg_density", 0.0)) for r in recent] or [0.0]
        crit = [int(r.get("criticality", 0)) for r in recent] or [0]
        risk = [max(0.0, min(100.0, (d * 45.0) + (c * 0.55))) for d, c in zip(dens, crit)] or [0.0]

        # Cover page
        fig = plt.figure(figsize=(8.27, 11.69), dpi=160)
        _theme(fig)
        ax = fig.add_axes([0.06, 0.05, 0.88, 0.90])
        ax.axis("off")
        ax.text(
            0.0,
            0.96,
            "Smart City Traffic\nIntelligence Report",
            fontsize=22,
            color="#67e8f9",
            fontweight="bold",
            va="top",
            linespacing=1.15,
        )
        ax.text(0.0, 0.90, f"Analyzed period: {period_title}", fontsize=11, color="#cbd5e1", va="top")
        ax.text(0.0, 0.86, f"Generated UTC: {datetime.utcnow().isoformat()}", fontsize=10, color="#94a3b8", va="top")
        ax.add_patch(plt.Rectangle((0.72, 0.80), 0.24, 0.11, facecolor="#1e293b", edgecolor="#334155", linewidth=1.2))
        ax.text(0.86, 0.92, "CITY LOGO", fontsize=10, color="#93c5fd", ha="center", va="center")
        ax.add_patch(plt.Rectangle((0.0, 0.78), 0.30, 0.06, facecolor=status_color, alpha=0.22, edgecolor=status_color, linewidth=1.5))
        ax.text(0.015, 0.81, f"GLOBAL STATUS: {status_label}", fontsize=12, color=status_color, fontweight="bold", va="center")

        kpi_cards = [
            ("Samples", f"{int(summary.get('samples', 0))}"),
            ("Avg Vehicles", f"{float(summary.get('avg_vehicles', 0.0)):.1f}"),
            ("Avg Density", f"{float(summary.get('avg_density', 0.0)):.2f}"),
            ("Avg Criticality", f"{float(summary.get('avg_criticality', 0.0)):.1f}/100"),
            ("Peak Vehicles", f"{int(summary.get('peak_vehicles', 0))}"),
            ("Peak Criticality", f"{int(summary.get('peak_criticality', 0))}/100"),
        ]
        x0, y0 = 0.0, 0.58
        w, h = 0.30, 0.09
        for idx, (title, value) in enumerate(kpi_cards):
            row = idx // 3
            col = idx % 3
            cx = x0 + col * 0.32
            cy = y0 - row * 0.12
            ax.add_patch(plt.Rectangle((cx, cy), w, h, facecolor="#0f172a", edgecolor="#334155", linewidth=1.2))
            ax.text(cx + 0.02, cy + 0.06, title, fontsize=9.5, color="#93c5fd", va="center")
            ax.text(cx + 0.02, cy + 0.025, value, fontsize=14, color="#f8fafc", fontweight="bold", va="center")

        ax.text(0.0, 0.33, "AI Interpretation", fontsize=15, color="#f8fafc", fontweight="bold", va="top")
        ax.text(0.0, 0.29, "\n".join(textwrap.wrap(ai_interp["headline"], width=100)), fontsize=12, color="#f59e0b", va="top")
        ax.text(0.0, 0.24, "\n".join(textwrap.wrap(ai_interp["detail"], width=94)), fontsize=10.8, color="#e2e8f0", va="top")
        ax.text(0.0, 0.14, "Operational conclusion", fontsize=12.5, color="#f8fafc", fontweight="bold", va="top")
        ax.text(0.0, 0.10, "\n".join(textwrap.wrap(ai_interp["conclusion"], width=100)), fontsize=10.8, color="#e2e8f0", va="top")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # KPI charts page
        fig, axs = plt.subplots(2, 2, figsize=(8.27, 11.69), dpi=160)
        _theme(fig)
        for row in axs:
            for ax in row:
                _darken(ax)

        axs[0, 0].plot(range(1, len(veh) + 1), veh, color="#22d3ee", linewidth=2.2)
        if veh:
            p_idx = int(np.argmax(veh))
            axs[0, 0].scatter(p_idx + 1, veh[p_idx], color="#f59e0b", s=40, zorder=3)
            axs[0, 0].annotate(f"Peak={veh[p_idx]}", (p_idx + 1, veh[p_idx]), textcoords="offset points", xytext=(8, 8), color="#f8fafc", fontsize=8)
        axs[0, 0].set_title("Vehicle Load Trend", color="#f8fafc")
        axs[0, 0].set_xlabel("Snapshot index", color="#cbd5e1")
        axs[0, 0].set_ylabel("Vehicles", color="#cbd5e1")
        axs[0, 0].grid(alpha=0.25, color="#475569")

        axs[0, 1].plot(range(1, len(crit) + 1), crit, color="#fb923c", linewidth=2.2)
        if crit:
            c_idx = int(np.argmax(crit))
            axs[0, 1].scatter(c_idx + 1, crit[c_idx], color="#ef4444", s=40, zorder=3)
            axs[0, 1].annotate(f"Peak={crit[c_idx]}", (c_idx + 1, crit[c_idx]), textcoords="offset points", xytext=(8, 8), color="#f8fafc", fontsize=8)
        axs[0, 1].set_title("Criticality Trend", color="#f8fafc")
        axs[0, 1].set_xlabel("Snapshot index", color="#cbd5e1")
        axs[0, 1].set_ylabel("Criticality / 100", color="#cbd5e1")
        axs[0, 1].grid(alpha=0.25, color="#475569")

        axs[1, 0].fill_between(range(1, len(dens) + 1), dens, color="#38bdf8", alpha=0.35)
        axs[1, 0].plot(range(1, len(dens) + 1), dens, color="#0ea5e9", linewidth=2.0)
        axs[1, 0].set_title("Density Evolution", color="#f8fafc")
        axs[1, 0].set_xlabel("Snapshot index", color="#cbd5e1")
        axs[1, 0].set_ylabel("Density", color="#cbd5e1")
        axs[1, 0].grid(alpha=0.25, color="#475569")

        axs[1, 1].plot(range(1, len(risk) + 1), risk, color="#ef4444", linewidth=2.2)
        axs[1, 1].fill_between(range(1, len(risk) + 1), risk, color="#ef4444", alpha=0.20)
        axs[1, 1].set_ylim(0, 100)
        axs[1, 1].set_title("Overall Traffic Risk", color="#f8fafc")
        axs[1, 1].set_xlabel("Snapshot index", color="#cbd5e1")
        axs[1, 1].set_ylabel("Risk score / 100", color="#cbd5e1")
        axs[1, 1].grid(alpha=0.25, color="#475569")

        fig.suptitle("Traffic Operations Dashboard", color="#67e8f9", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # XAI page
        fig = plt.figure(figsize=(8.27, 11.69), dpi=160)
        _theme(fig)
        ax1 = fig.add_axes([0.08, 0.55, 0.84, 0.34])
        _darken(ax1)
        factor_names = ["Vehicle pressure", "Congestion pressure", "Waiting impact", "Criticality pressure"]
        factor_vals = [
            float(xai["vehicle_pressure"]),
            float(xai["congestion_pressure"]),
            float(xai["waiting_impact"]),
            float(xai["criticality_pressure"]),
        ]
        colors = ["#22d3ee", "#38bdf8", "#f59e0b", "#ef4444"]
        bars = ax1.barh(factor_names, factor_vals, color=colors)
        ax1.set_xlim(0, 100)
        ax1.set_title("XAI Feature Influence Scores", color="#f8fafc", fontsize=14)
        ax1.set_xlabel("Influence (%)", color="#cbd5e1")
        ax1.grid(axis="x", alpha=0.25, color="#475569")
        for b, v in zip(bars, factor_vals):
            ax1.text(min(98, v + 1.5), b.get_y() + b.get_height() / 2, f"{v:.1f}%", color="#f8fafc", va="center", fontsize=9)

        ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.40])
        ax2.axis("off")
        ax2.text(0.0, 0.98, "XAI Decision Analysis", fontsize=16, color="#67e8f9", fontweight="bold", va="top")
        ax2.text(0.0, 0.88, "\n".join(textwrap.wrap(narrative["xai"], width=95)), fontsize=10.8, color="#e2e8f0", va="top")
        y0 = 0.66
        desc_map = [
            ("Vehicle pressure", xai_explain["Vehicle pressure"]),
            ("Congestion pressure", xai_explain["Congestion pressure"]),
            ("Waiting impact", xai_explain["Waiting impact"]),
            ("Criticality pressure", xai_explain["Criticality pressure"]),
        ]
        for name, desc in desc_map:
            ax2.text(0.0, y0, f"- {name}: {desc}", fontsize=9.8, color="#cbd5e1", va="top")
            y0 -= 0.08

        dominant = str(xai.get("dominant_driver", "Not available"))
        dominant_action = _dynamic_actions(summary, xai)[-1] if actions else "Not available"
        ax2.add_patch(plt.Rectangle((0.0, 0.16), 0.98, 0.16, facecolor="#111827", edgecolor="#334155", linewidth=1.2))
        ax2.text(0.02, 0.27, "Dominant Factor Card", fontsize=12, color="#f8fafc", fontweight="bold", va="top")
        ax2.text(0.02, 0.22, f"Dominant factor: {dominant}", fontsize=10.5, color="#93c5fd", va="top")
        ax2.text(0.02, 0.17, f"Linked recommendation: {dominant_action}", fontsize=9.8, color="#e2e8f0", va="top")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Recommended actions page
        fig = plt.figure(figsize=(8.27, 11.69), dpi=160)
        _theme(fig)
        ax = fig.add_axes([0.06, 0.07, 0.88, 0.88])
        ax.axis("off")
        ax.text(0.0, 0.97, "Recommended Actions", fontsize=18, color="#67e8f9", fontweight="bold", va="top")
        ax.text(0.0, 0.90, "Dynamic actions generated from KPI and XAI signals.", fontsize=10.5, color="#cbd5e1", va="top")
        yy = 0.84
        for idx, action in enumerate(actions[:8], start=1):
            ax.add_patch(plt.Rectangle((0.0, yy - 0.05), 0.96, 0.06, facecolor="#0f172a", edgecolor="#334155", linewidth=1.0))
            wrapped_action = "\n".join(textwrap.wrap(f"{idx}. {action}", width=95))
            ax.text(0.015, yy - 0.012, wrapped_action, fontsize=9.9, color="#e2e8f0", va="center", linespacing=1.15)
            yy -= 0.085
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        # Records table page
        fig = plt.figure(figsize=(8.27, 11.69), dpi=160)
        _theme(fig)
        ax = fig.add_axes([0.04, 0.05, 0.92, 0.90])
        ax.axis("off")
        ax.text(0.0, 0.98, "Recent Operational Records", fontsize=16, color="#67e8f9", fontweight="bold", va="top")
        table_rows = []
        for r in records[-24:]:
            ts = str(r.get("ts", ""))
            short_ts = ts.replace("T", " ")[:19] if ts else "Not available"
            lvl = str(r.get("congestion_level", "Not available"))
            crit_val = int(r.get("criticality", 0))
            decision_txt = _wrap_cell_text(_safe_text(r.get("final_decision", "Not available")), width=16)
            obs_txt = _wrap_cell_text(_record_observation(lvl, crit_val), width=24)
            table_rows.append(
                [
                    short_ts,
                    int(r.get("vehicles", 0)),
                    f"{float(r.get('avg_density', 0.0)):.2f}",
                    lvl,
                    crit_val,
                    decision_txt,
                    obs_txt,
                ]
            )

        if table_rows:
            labels = ["Time (UTC)", "Veh", "Density", "Level", "Crit", "Decision", "Observation"]
            table = ax.table(
                cellText=table_rows,
                colLabels=labels,
                cellLoc="center",
                colLoc="center",
                loc="upper left",
                bbox=[0.0, 0.02, 1.0, 0.91],
                colWidths=[0.19, 0.07, 0.09, 0.08, 0.08, 0.17, 0.32],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7.2)
            table.scale(1, 1.55)
            for (r_idx, c_idx), cell in table.get_celld().items():
                if r_idx == 0:
                    cell.set_text_props(color="#f8fafc", weight="bold")
                    cell.set_facecolor("#1e293b")
                    cell.set_edgecolor("#334155")
                    continue
                row = table_rows[r_idx - 1]
                level = str(row[3]).lower()
                crit_val = int(row[4])
                base_bg = "#0f172a" if r_idx % 2 == 0 else "#111827"
                fg = "#e2e8f0"
                if c_idx == 3:
                    if level == "high":
                        base_bg = "#7f1d1d"
                    elif level == "medium":
                        base_bg = "#78350f"
                    elif level == "low":
                        base_bg = "#14532d"
                if c_idx == 4:
                    if crit_val > 70:
                        base_bg = "#7f1d1d"
                    elif crit_val >= 40:
                        base_bg = "#78350f"
                    else:
                        base_bg = "#14532d"
                if c_idx in (5, 6):
                    cell.set_text_props(ha="left", va="center", color=fg)
                cell.set_facecolor(base_bg)
                cell.set_edgecolor("#334155")
        else:
            ax.text(0.0, 0.90, "Not available: no records for selected period.", color="#e2e8f0", fontsize=11, va="top")
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    return out.getvalue()


def _escape_pdf_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def _build_minimal_pdf(lines: List[str]) -> bytes:
    # Tiny PDF writer fallback (single page, Helvetica).
    y = 800
    content_parts: List[str] = ["BT", "/F1 10 Tf", "40 820 Td"]
    first = True
    for raw in lines[:70]:
        txt = _escape_pdf_text(str(raw))
        if first:
            content_parts.append(f"({txt}) Tj")
            first = False
        else:
            content_parts.append("0 -12 Td")
            content_parts.append(f"({txt}) Tj")
        y -= 12
        if y < 40:
            break
    content_parts.append("ET")
    stream = "\n".join(content_parts).encode("latin-1", errors="replace")

    objects: List[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    out = bytearray(b"%PDF-1.4\n")
    xref_offsets = [0]
    for i, obj in enumerate(objects, start=1):
        xref_offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_start = len(out)
    out.extend(f"xref\n0 {len(xref_offsets)}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in xref_offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        (
            f"trailer\n<< /Size {len(xref_offsets)} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(out)


def main() -> None:
    inject_modern_css()
    st.markdown('<div class="main-title">Traffic Decision Support System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Real-time congestion analysis, smart recommendations, and explainable AI.</div>',
        unsafe_allow_html=True,
    )

    if "run" not in st.session_state:
        st.session_state["run"] = False
    if "editing" not in st.session_state:
        st.session_state["editing"] = True
    if "preview_frame" not in st.session_state:
        st.session_state["preview_frame"] = None
    if "canvas_json" not in st.session_state:
        st.session_state["canvas_json"] = None
    if "roi_polygons" not in st.session_state:
        st.session_state["roi_polygons"] = []
    if "decision_engine" not in st.session_state:
        st.session_state["decision_engine"] = DecisionEngine(history_size=20)
    if "video_cap" not in st.session_state:
        st.session_state["video_cap"] = None
    if "video_source_resolved" not in st.session_state:
        st.session_state["video_source_resolved"] = None
    if "frame_idx" not in st.session_state:
        st.session_state["frame_idx"] = 0
    if "canvas_version" not in st.session_state:
        st.session_state["canvas_version"] = 0
    if "uploaded_video_path" not in st.session_state:
        st.session_state["uploaded_video_path"] = None
    if "uploaded_video_sig" not in st.session_state:
        st.session_state["uploaded_video_sig"] = None
    if "agentic_runner" not in st.session_state:
        st.session_state["agentic_runner"] = AgenticOrchestrator()
    if "report_records" not in st.session_state:
        st.session_state["report_records"] = deque(maxlen=20000)
    if "groq_report_text" not in st.session_state:
        st.session_state["groq_report_text"] = ""
    if "report_pdf_bytes" not in st.session_state:
        st.session_state["report_pdf_bytes"] = b""
    if "report_pdf_name" not in st.session_state:
        st.session_state["report_pdf_name"] = "traffic_report.pdf"
    if "precomputed_map" not in st.session_state:
        st.session_state["precomputed_map"] = {}
    if "precomputed_key" not in st.session_state:
        st.session_state["precomputed_key"] = ""
    if "last_canvas_live_json" not in st.session_state:
        st.session_state["last_canvas_live_json"] = None
    if "esp32_last_post_signature" not in st.session_state:
        st.session_state["esp32_last_post_signature"] = None
    if "esp32_stored_url" not in st.session_state:
        st.session_state["esp32_stored_url"] = None
    if "esp32_emergency_confirm" not in st.session_state:
        st.session_state["esp32_emergency_confirm"] = new_emergency_confirmation_state()
    if "esp32_skip_unchanged_log_at" not in st.session_state:
        st.session_state["esp32_skip_unchanged_log_at"] = 0.0

    with st.sidebar:
        st.subheader("Configuration")
        mode = st.radio("Analysis mode", ["Vehicle counting", "Congestion detection"], index=0)
        processing_mode = st.radio("Processing mode", ["Live", "Optimized stream (auto-cache)"], index=0)
        source_mode = st.radio("Video source type", ["YouTube/URL", "Local file path", "Upload file"], index=0)
        source = ""
        if source_mode == "YouTube/URL":
            source = st.text_input("Video URL", value="https://www.youtube.com/watch?v=MNn9qKG2UFI")
        elif source_mode == "Local file path":
            source = st.text_input("Local video path", value="")
        else:
            uploaded_video = st.file_uploader(
                "Upload local video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                accept_multiple_files=False,
            )
            if uploaded_video is not None:
                upload_bytes = uploaded_video.getvalue()
                sig = hashlib.md5(upload_bytes).hexdigest()
                if st.session_state["uploaded_video_sig"] != sig:
                    upload_dir = Path.cwd() / ".streamlit_uploads"
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    ext = Path(uploaded_video.name).suffix or ".mp4"
                    target = upload_dir / f"uploaded_{sig[:12]}{ext}"
                    target.write_bytes(upload_bytes)
                    st.session_state["uploaded_video_path"] = str(target)
                    st.session_state["uploaded_video_sig"] = sig
                source = st.session_state["uploaded_video_path"] or ""
                st.caption(f"Using uploaded file: {uploaded_video.name}")
            else:
                source = st.session_state["uploaded_video_path"] or ""
        if mode == "Vehicle counting":
            model_name = st.selectbox("YOLO model", [DEFAULT_MODEL, "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
            custom_model = st.text_input("Custom model path (.pt)", value="")
            labels_raw = st.text_input("Vehicle labels", value="car,bus,truck,motorcycle,bicycle")
            roi_tool = st.radio("ROI drawing method", ["Boxes", "Points (connect)"], index=0)
            use_dual_models = True
            st.caption("Dual model mode ON: best.pt + yolov8n.pt")
        else:
            model_name = str(Path.cwd() / DEFAULT_CONGESTION_MODEL)
            custom_model = ""
            labels_raw = st.text_input("Congestion labels (optional)", value="")
            roi_tool = "Polygons"
            st.caption(f"Congestion model in use: {model_name}")
            use_dual_models = False
        rebuild_cache = False
        if processing_mode == "Optimized stream (auto-cache)":
            rebuild_cache = st.checkbox("Rebuild optimized cache", value=False)
            st.caption("The app auto-generates cached detections internally (no manual JSON file).")
        conf = st.slider("Confidence", 0.1, 0.95, 0.35, 0.05)
        frame_skip = st.slider("Frame skip", 1, 5, 2)
        imgsz = st.slider("Inference size", 160, 960, 416, 32)
        max_det = st.slider("Max detections", 5, 200, 80, 5)
        ui_refresh_every = st.slider("Dashboard refresh (frames)", 1, 10, 3)
        turbo_local = st.checkbox("Turbo local playback", value=True)
        show_label_conf = st.checkbox("Show labels and probabilities", value=True)
        mock_emergency = st.checkbox(
            "Simulate emergency vehicle (test mode)",
            value=False,
            help="When enabled, emergency is forced ON for demo/testing, even without real ambulance detection.",
        )
        st.divider()
        st.caption("ESP32 / feux (POST JSON sur WiFi)")
        esp32_enabled = st.checkbox("Activer lien ESP32", value=False)
        esp32_url = st.text_input(
            "URL (ex. http://IP_ESP32/traffic)",
            value="http://192.168.0.164/traffic",
            help="Même réseau WiFi que le PC. L'ESP32 doit accepter POST JSON sur cette route.",
        )
        if not esp32_enabled:
            st.session_state["esp32_stored_url"] = None
            st.session_state["esp32_last_post_signature"] = None
            st.session_state["esp32_emergency_confirm"] = new_emergency_confirmation_state()
        esp32_timeout = st.slider("Timeout HTTP (s)", 0.5, 10.0, 2.0, 0.5)
        esp32_sim_zone = st.number_input(
            "Zone prioritaire si urgence simulée (0 = 1re zone)",
            min_value=0,
            max_value=7,
            value=0,
            help="Utilisé avec la case urgence simulée : feu vert prioritaire sur cette zone.",
        )
        if mode == "Vehicle counting":
            if roi_tool == "Boxes":
                st.caption("ROI selection: draw rectangles then click Save zone.")
            else:
                st.caption("ROI selection: click points to build polygon, then close the shape and Save zone.")
        else:
            st.caption("ROI selection: polygon (right-click to close each zone)")
        capture_btn = st.button("Capture frame")
        toggle_btn = st.button("Start detection" if not st.session_state["run"] else "Pause detection")
        clear_btn = st.button("Clear zones")
        stop_btn = st.button("Stop")

    if stop_btn:
        st.session_state["run"] = False
    if clear_btn:
        st.session_state["roi_polygons"] = []
        st.session_state["canvas_json"] = None
        st.session_state["canvas_version"] += 1

    active_model = custom_model.strip() or model_name
    if mode == "Congestion detection":
        active_model = str(Path.cwd() / DEFAULT_CONGESTION_MODEL)
    allowed_labels: Optional[Set[str]] = {s.strip() for s in labels_raw.split(",") if s.strip()} or None
    if mode == "Vehicle counting" and use_dual_models and allowed_labels is not None:
        allowed_labels = set(allowed_labels) | {"ambulance", "emergency-vehicle", "emergency_vehicle", "emergency"}
    resolved_source = resolve_video_source(source)
    if processing_mode == "Optimized stream (auto-cache)" and rebuild_cache:
        st.session_state["precomputed_map"] = {}
        st.session_state["precomputed_key"] = ""

    if capture_btn or (mode == "Vehicle counting" and st.session_state["preview_frame"] is None):
        try:
            first = read_first_frame(resolved_source)
            if first is not None:
                st.session_state["preview_frame"] = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    # Single main area: video on top, smart dashboard below.
    left = st.container()
    canvas_result = None
    with left:
        editor_box = st.empty()
        if mode == "Vehicle counting" and st.session_state["preview_frame"] is not None and not st.session_state["run"]:
            with editor_box.container():
                if roi_tool == "Boxes":
                    st.write("Draw one box, click Save zone, then draw the next one.")
                else:
                    st.write("Place points to form a polygon, close it, click Save zone, then draw the next one.")
                c1, c2, c3, c4 = st.columns(4)
                save_zones_btn = c1.button("Save zone")
                remove_last_zone_btn = c2.button("Remove last zone")
                reset_btn = c3.button("Clear zones")
                if reset_btn:
                    st.session_state["canvas_json"] = None
                    st.session_state["roi_polygons"] = []
                    st.session_state["canvas_version"] += 1
                if remove_last_zone_btn and st.session_state["roi_polygons"]:
                    st.session_state["roi_polygons"].pop()

                ph, pw = st.session_state["preview_frame"].shape[:2]
                # Standard fixed canvas width for consistent rendering.
                canvas_w = min(960, max(640, int(pw)))
                canvas_h = int(round((ph / max(1, pw)) * canvas_w))
                bg_img = Image.fromarray(st.session_state["preview_frame"]).resize((canvas_w, canvas_h))

                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0.0)",
                    background_color="#0b1220",
                    stroke_width=2,
                    stroke_color="#38bdf8",
                    background_image=bg_img,
                    update_streamlit=True,
                    height=canvas_h,
                    width=canvas_w,
                    drawing_mode="rect" if roi_tool == "Boxes" else "polygon",
                    display_toolbar=True if roi_tool != "Boxes" else False,
                    initial_drawing=None,
                    key=f"canvas_main_{st.session_state['canvas_version']}",
                )

                if canvas_result is not None and getattr(canvas_result, "json_data", None):
                    try:
                        obj_count = len((canvas_result.json_data or {}).get("objects", []) or [])
                        if obj_count > 0:
                            st.session_state["last_canvas_live_json"] = canvas_result.json_data
                    except Exception:
                        pass

                if save_zones_btn:
                    latest_json = canvas_result.json_data if canvas_result is not None else None
                    if not latest_json or not (latest_json.get("objects", []) if isinstance(latest_json, dict) else []):
                        latest_json = st.session_state.get("last_canvas_live_json")
                    polygons = extract_polygon_points(latest_json or {})
                    if polygons:
                        # Keep only the latest drawn zone for deterministic save.
                        poly = polygons[-1]
                        # Canvas is resized for UI; convert ROI back to original frame coordinates.
                        poly = _scale_polygon(
                            poly=poly,
                            src_w=canvas_w,
                            src_h=canvas_h,
                            dst_w=pw,
                            dst_h=ph,
                        )
                        key = tuple(poly)
                        existing_keys = {tuple(p) for p in st.session_state["roi_polygons"]}
                        added = 0
                        if key not in existing_keys:
                            st.session_state["roi_polygons"].append(poly)
                            added = 1
                        st.session_state["canvas_json"] = None
                        st.session_state["canvas_version"] += 1
                        if added > 0:
                            st.success(f"{added} zone(s) added. Total: {len(st.session_state['roi_polygons'])}.")
                        else:
                            st.info("Zone already saved (duplicate ignored).")
                    else:
                        if roi_tool == "Boxes":
                            st.warning("No zone detected. Draw a rectangle and click Save zone.")
                        else:
                            st.warning(
                                "No zone detected. Close polygon first (double-click/right-click), then click Save zone."
                            )

                st.caption(f"Zones: {len(st.session_state['roi_polygons'])}")
        else:
            editor_box.empty()
        video_box = st.empty()
        dashboard_box = st.empty()

    if toggle_btn:
        if not st.session_state["run"]:
            if mode == "Vehicle counting" and not st.session_state["roi_polygons"]:
                st.warning("Save at least one zone before starting detection.")
            else:
                st.session_state["run"] = True
        else:
            st.session_state["run"] = False
            if st.session_state["video_cap"] is not None:
                st.session_state["video_cap"].release()
                st.session_state["video_cap"] = None

    st.subheader("Traffic Operations Dashboard")

    if st.session_state["run"]:
        try:
            dual_models: List[YOLO] = []
            model: Optional[YOLO] = None
            if processing_mode == "Live":
                model = load_model(active_model)
                model.overrides["conf"] = conf
                dual_models = [model]
                if mode == "Vehicle counting" and use_dual_models:
                    model_best = load_model(_resolve_traffic_best_weights())
                    model_yolo = load_model("yolov8n.pt")
                    model_best.overrides["conf"] = conf
                    model_yolo.overrides["conf"] = conf
                    dual_models = [model_best, model_yolo]
            else:
                # Auto-cache mode still uses real models once to build an internal cache.
                if mode == "Vehicle counting" and use_dual_models:
                    model_best = load_model(_resolve_traffic_best_weights())
                    model_yolo = load_model("yolov8n.pt")
                    model_best.overrides["conf"] = conf
                    model_yolo.overrides["conf"] = conf
                    dual_models = [model_best, model_yolo]
                else:
                    model = load_model(active_model)
                    model.overrides["conf"] = conf
                    dual_models = [model]
            # Recreate capture only when source changes or not initialized.
            if (
                st.session_state["video_cap"] is None
                or st.session_state["video_source_resolved"] != resolved_source
            ):
                if st.session_state["video_cap"] is not None:
                    st.session_state["video_cap"].release()
                st.session_state["video_cap"] = cv2.VideoCapture(resolved_source)
                # Reduce buffering for local files/cams when backend supports it.
                if source_mode != "YouTube/URL":
                    st.session_state["video_cap"].set(cv2.CAP_PROP_BUFFERSIZE, 1)
                st.session_state["video_source_resolved"] = resolved_source
                st.session_state["frame_idx"] = 0
            cap = st.session_state["video_cap"]
            if not cap or not cap.isOpened():
                st.error("Cannot open video source.")
                st.session_state["run"] = False
                st.session_state["video_cap"] = None
                return

            if processing_mode == "Optimized stream (auto-cache)":
                cache_key = "|".join(
                    [
                        str(resolved_source),
                        str(mode),
                        str(imgsz),
                        str(max_det),
                        str(sorted(list(allowed_labels or []))),
                        str(conf),
                        str(use_dual_models),
                    ]
                )
                if cache_key != st.session_state.get("precomputed_key", ""):
                    progress_box = st.empty()
                    progress = progress_box.progress(0.0, text="Building optimized cache... 0%")

                    def _progress_cb(done_frames: int, total_frames: int, processed_steps: int) -> None:
                        if total_frames > 0:
                            pct = max(0.0, min(1.0, done_frames / float(total_frames)))
                            progress.progress(
                                pct,
                                text=f"Building optimized cache... {int(pct*100)}% ({done_frames}/{total_frames} frames, sampled {processed_steps})",
                            )
                        else:
                            # Fallback when total frame count is unavailable.
                            pseudo = min(0.95, processed_steps / 1000.0)
                            progress.progress(
                                pseudo,
                                text=f"Building optimized cache... sampled {processed_steps} frame(s)",
                            )

                    with st.spinner("Building optimized cache from current video source..."):
                        st.session_state["precomputed_map"] = _auto_precompute_results(
                            source=resolved_source,
                            models=dual_models,
                            allowed_labels=allowed_labels if mode == "Congestion detection" else allowed_labels or VEHICLE_LABELS,
                            imgsz=imgsz,
                            max_det=max_det,
                            frame_step=max(1, frame_skip),
                            progress_cb=_progress_cb,
                        )
                        st.session_state["precomputed_key"] = cache_key
                    progress.progress(1.0, text="Building optimized cache... 100% complete")
                    progress_box.empty()

            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = float(fps) if fps and fps > 0 else 25.0
            effective_frame_skip = frame_skip
            effective_ui_refresh = ui_refresh_every
            draw_confidence = True
            if turbo_local and source_mode != "YouTube/URL":
                # Turbo mode prioritizes smooth playback over per-frame detail.
                effective_frame_skip = max(frame_skip, 2)
                effective_ui_refresh = max(ui_refresh_every, 4)
            draw_confidence = show_label_conf
            # Continuous playback without page refresh/rerun.
            st.markdown("### Live Video")
            esp32_log_box = st.empty()
            st.session_state["esp32_emergency_confirm"] = new_emergency_confirmation_state()
            curr_esp_u = esp32_url.strip()
            if esp32_enabled and curr_esp_u:
                if st.session_state.get("esp32_stored_url") != curr_esp_u:
                    st.session_state["esp32_last_post_signature"] = None
                st.session_state["esp32_stored_url"] = curr_esp_u
            while st.session_state["run"]:
                loop_start = time.time()
                frame = None
                for _ in range(max(1, effective_frame_skip)):
                    ok, candidate = cap.read()
                    if not ok:
                        st.session_state["run"] = False
                        cap.release()
                        st.session_state["video_cap"] = None
                        break
                    frame = candidate
                    st.session_state["frame_idx"] += 1
                if frame is None:
                    break

                # Keep ROI coordinate space consistent with canvas space.
                if mode == "Vehicle counting" and st.session_state.get("preview_frame") is not None:
                    ph, pw = st.session_state["preview_frame"].shape[:2]
                    if pw > 0 and ph > 0:
                        frame = cv2.resize(frame, (pw, ph))

                if processing_mode == "Optimized stream (auto-cache)":
                    candidates = st.session_state.get("precomputed_map", {}).get(st.session_state["frame_idx"], [])
                    det = detect_from_candidates(
                        frame=frame,
                        candidates=candidates,
                        roi_polygons=st.session_state["roi_polygons"] if mode == "Vehicle counting" else [],
                        draw_confidence=draw_confidence,
                    )
                elif mode == "Vehicle counting" and use_dual_models:
                    det = detect_in_zones_multi(
                        models=dual_models,
                        frame=frame,
                        roi_polygons=st.session_state["roi_polygons"],
                        allowed_labels=allowed_labels or VEHICLE_LABELS,
                        imgsz=imgsz,
                        max_det=max_det,
                        draw_confidence=draw_confidence,
                    )
                else:
                    det = detect_in_zones(
                        model=model,  # type: ignore[arg-type]
                        frame=frame,
                        roi_polygons=st.session_state["roi_polygons"] if mode == "Vehicle counting" else [],
                        allowed_labels=allowed_labels if mode == "Congestion detection" else allowed_labels or VEHICLE_LABELS,
                        imgsz=imgsz,
                        max_det=max_det,
                        draw_confidence=draw_confidence,
                    )
                polygons = st.session_state["roi_polygons"] if mode == "Vehicle counting" else []
                zone_areas = [polygon_area(p) for p in polygons] if polygons else [frame.shape[0] * frame.shape[1]]
                zone_states = compute_zone_states(det.per_zone_counts, zone_areas)
                decisions = st.session_state["decision_engine"].decide_many(zone_states)
                wave = st.session_state["decision_engine"].green_wave(decisions)

                esp32_payload_for_ui = None
                if esp32_enabled and decisions:
                    raw_em, raw_zone = infer_emergency_zone(
                        list(det.per_zone_counts),
                        mock_emergency,
                        int(esp32_sim_zone),
                    )
                    now_m = time.monotonic()
                    em_st = st.session_state["esp32_emergency_confirm"]
                    conf_em, conf_zone, em_logs = step_emergency_confirmation(
                        em_st, raw_em, raw_zone, now_m
                    )
                    esp32_payload_for_ui = build_esp32_payload(
                        decisions,
                        conf_em,
                        conf_zone if conf_em else None,
                    )
                    if esp32_url.strip():
                        sig = esp32_decision_signature(esp32_payload_for_ui)
                        prev_sig = st.session_state.get("esp32_last_post_signature")
                        decision_changed = prev_sig is None or sig != prev_sig

                        log_chunks: List[str] = []
                        for msg in em_logs:
                            log_chunks.append(f"**{msg}**")

                        if decision_changed:
                            ok, _msg = post_json(
                                esp32_url.strip(),
                                esp32_payload_for_ui,
                                timeout=float(esp32_timeout),
                            )
                            st.session_state["esp32_last_ok"] = ok
                            line = format_esp32_log_line(esp32_payload_for_ui)
                            st.session_state["esp32_last_log"] = line
                            if ok:
                                st.session_state["esp32_last_post_signature"] = sig
                            log_chunks.append(
                                "**ESP32 sent: confirmed decision changed**  "
                                f"`{line}`  ·  HTTP **{'OK' if ok else 'FAIL'}**"
                            )
                            esp32_log_box.markdown("\n\n".join(log_chunks))
                        else:
                            if (
                                now_m - float(st.session_state.get("esp32_skip_unchanged_log_at") or 0.0)
                            ) >= 1.0:
                                st.session_state["esp32_skip_unchanged_log_at"] = now_m
                                if log_chunks:
                                    log_chunks.append(
                                        "**ESP32 skipped: confirmed decision unchanged**"
                                    )
                                    esp32_log_box.markdown("\n\n".join(log_chunks))
                                else:
                                    esp32_log_box.markdown(
                                        "**ESP32 skipped: confirmed decision unchanged**"
                                    )

                total_vehicles = sum(s.vehicle_count for s in zone_states)
                avg_density = float(np.mean([s.density for s in zone_states])) if zone_states else 0.0
                global_level = max((d.congestion_level for d in decisions), default="low")
                global_criticality = int(np.mean([d.traffic_criticality_score for d in decisions])) if decisions else 0

                should_refresh_ui = st.session_state["frame_idx"] % effective_ui_refresh == 0
                if should_refresh_ui:
                    dashboard_box.empty()
                    with dashboard_box.container():
                        if esp32_enabled and esp32_payload_for_ui is not None:
                            pz = esp32_payload_for_ui.get("priority_zone")
                            em = esp32_payload_for_ui.get("emergency")
                            st.caption(
                                f"**ESP32 live** · emergency={em} · priority_zone={pz} · "
                                f"{format_esp32_log_line(esp32_payload_for_ui)}"
                            )
                        st.markdown("### Overview")
                        render_global_overview(
                            total_vehicles=total_vehicles,
                            avg_density=avg_density,
                            level=global_level,
                            criticality=global_criticality,
                        )
                        if mode == "Vehicle counting":
                            zone_count_rows = []
                            for i, zc in enumerate(det.per_zone_counts):
                                zone_count_rows.append(
                                    {
                                        "Zone": f"Zone {i+1}",
                                        "Vehicles": int(sum(zc.values())),
                                        "Congestion": zone_states[i].level if i < len(zone_states) else "-",
                                        "Types": ", ".join([f"{k}:{v}" for k, v in sorted(zc.items())]) or "-",
                                    }
                                )
                            render_zone_counts_table(zone_count_rows)
                            if polygons:
                                st.markdown("**Congestion Heatmap**")
                                hm = build_zone_heatmap(det.processed_frame, polygons, [s.density for s in zone_states])
                                st.image(hm, channels="RGB", use_container_width=True)

                            st.markdown("### Smart Recommendations")
                            for i, d in enumerate(decisions):
                                color = "#{:02x}{:02x}{:02x}".format(
                                    ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)][2],
                                    ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)][1],
                                    ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)][0],
                                )
                                render_zone_decision(i, d, color)
                            if wave:
                                st.markdown(f"**Green wave plan:** {' -> '.join(wave)}")
                            if decisions:
                                render_alert_if_needed(global_level, decisions[0].driver_alert)

                            if decisions and zone_states:
                                st.markdown("### XAI")
                                render_xai(explain_decision(zone_states[0], decisions[0]))
                        else:
                            st.markdown("### Congestion Detection Dashboard")
                            class_rows = []
                            congestion_counts = det.per_zone_counts[0] if det.per_zone_counts else {}
                            for idx, (label, count) in enumerate(
                                sorted(congestion_counts.items(), key=lambda kv: kv[1], reverse=True)
                            ):
                                class_rows.append(
                                    {
                                        "Class": label,
                                        "Count": int(count),
                                        "Color": _bgr_to_hex(color_for_class(label, idx)),
                                    }
                                )
                            render_class_counts_table(class_rows)
                            c_left, c_right = st.columns([1, 1])
                            with c_left:
                                render_congestion_class_cards(class_rows)
                            with c_right:
                                st.markdown("#### Dominant class")
                                if class_rows:
                                    top = class_rows[0]
                                    st.markdown(
                                        f"""
                                        <div class="card" style="border-left:6px solid {top['Color']};">
                                            <div style="font-size:1.1rem;">
                                                <b>{top['Class']}</b>
                                            </div>
                                            <div style="margin-top:6px;color:#cbd5e1;">
                                                Detections: <b>{top['Count']}</b>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.info("No dominant class yet.")
                            st.markdown("### Congestion Decision")
                            if decisions:
                                color = "#{:02x}{:02x}{:02x}".format(
                                    ZONE_COLORS_BGR[0][2], ZONE_COLORS_BGR[0][1], ZONE_COLORS_BGR[0][0]
                                )
                                render_zone_decision(0, decisions[0], color)
                                render_alert_if_needed(global_level, decisions[0].driver_alert)

                        scene_state = _build_scene_state(
                            zone_states=zone_states,
                            decisions=decisions,
                            per_zone_counts=det.per_zone_counts,
                            mock_emergency=mock_emergency,
                        )
                        agentic = st.session_state["agentic_runner"].run_agentic_system(scene_state)
                        render_agentic_control_center(agentic)
                        _append_report_record(
                            records=st.session_state["report_records"],
                            total_vehicles=total_vehicles,
                            avg_density=avg_density,
                            global_level=global_level,
                            global_criticality=global_criticality,
                            final_decision=str(agentic.get("final_decision", global_level)),
                            final_recommendation=str(agentic.get("final_recommendation", "")),
                        )

                video_box.image(
                    cv2.cvtColor(det.processed_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                target_frame_time = (1.0 / fps) * max(1, effective_frame_skip)
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, target_frame_time - elapsed))
        except Exception as exc:
            st.error(f"Runtime error: {exc}")
            st.session_state["run"] = False
            if st.session_state.get("video_cap") is not None:
                st.session_state["video_cap"].release()
                st.session_state["video_cap"] = None

    st.markdown("### Report Generator (PDF / CSV)")
    period_label = st.selectbox("Report period", ["Daily (last 24h)", "Weekly (last 7 days)"], index=0)
    days = 1 if period_label.startswith("Daily") else 7
    report_rows = _filter_records_by_days(list(st.session_state["report_records"]), days=days)
    summary = _report_summary(report_rows)
    if not report_rows:
        st.info("No report data yet. Run detection to collect KPI snapshots.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples", int(summary["samples"]))
        m2.metric("Avg vehicles", f"{summary['avg_vehicles']:.1f}")
        m3.metric("Peak vehicles", int(summary["peak_vehicles"]))
        m4.metric("Peak criticality", int(summary["peak_criticality"]))
        narrative = _build_local_narrative(report_rows, summary, period_label)
        xai_features = _report_xai_features(report_rows, summary)

        st.markdown("#### Report Narrative")
        st.markdown(narrative["executive"])
        st.markdown(narrative["operations"])
        st.markdown(narrative["conclusion"])

        try:
            import pandas as pd

            chart_df = pd.DataFrame(
                [
                    {
                        "time": str(r.get("ts", ""))[11:19] if str(r.get("ts", "")) else "-",
                        "vehicles": int(r.get("vehicles", 0)),
                        "density": float(r.get("avg_density", 0.0)),
                        "criticality": int(r.get("criticality", 0)),
                    }
                    for r in report_rows[-40:]
                ]
            )
            if not chart_df.empty:
                st.markdown("#### KPI Trends")
                left_chart, right_chart = st.columns(2)
                with left_chart:
                    st.line_chart(
                        chart_df.set_index("time")[["vehicles", "criticality"]],
                        use_container_width=True,
                    )
                with right_chart:
                    st.area_chart(
                        chart_df.set_index("time")[["density"]],
                        use_container_width=True,
                    )
        except Exception:
            pass

        level_counts = Counter([str(r.get("congestion_level", "-")) for r in report_rows])
        if level_counts:
            st.markdown("#### Congestion Distribution")
            try:
                import pandas as pd

                level_df = pd.DataFrame(
                    {
                        "level": ["low", "medium", "high"],
                        "count": [
                            int(level_counts.get("low", 0)),
                            int(level_counts.get("medium", 0)),
                            int(level_counts.get("high", 0)),
                        ],
                    }
                ).set_index("level")
                st.bar_chart(level_df, use_container_width=True)
            except Exception:
                st.write(
                    {
                        "low": int(level_counts.get("low", 0)),
                        "medium": int(level_counts.get("medium", 0)),
                        "high": int(level_counts.get("high", 0)),
                    }
                )

        st.markdown("#### XAI Visual Factors")
        st.markdown(narrative["xai"])
        st.progress(
            min(1.0, float(xai_features["congestion_pressure"]) / 100.0),
            text=f"Congestion pressure: {float(xai_features['congestion_pressure']):.1f}%",
        )
        st.progress(
            min(1.0, float(xai_features["vehicle_pressure"]) / 100.0),
            text=f"Vehicle pressure: {float(xai_features['vehicle_pressure']):.1f}%",
        )
        st.progress(
            min(1.0, float(xai_features["waiting_impact"]) / 100.0),
            text=f"Estimated waiting impact: {float(xai_features['waiting_impact']):.1f}%",
        )
        st.progress(
            min(1.0, float(xai_features["criticality_pressure"]) / 100.0),
            text=f"Criticality pressure: {float(xai_features['criticality_pressure']):.1f}%",
        )

        st.caption("AI narrative report powered by Groq.")
        c1, c2 = st.columns([1, 1])
        generate_ai_btn = c1.button("Generate AI Report (Groq)")
        clear_ai_btn = c2.button("Clear AI Report")
        if clear_ai_btn:
            st.session_state["groq_report_text"] = ""
            st.session_state["report_pdf_bytes"] = b""

        if generate_ai_btn:
            with st.spinner("Generating report with Groq..."):
                report_text, report_error = _generate_groq_report(
                    report_rows,
                    summary,
                    period_label,
                )
                if report_text:
                    st.session_state["groq_report_text"] = report_text
                else:
                    st.warning(f"Groq report unavailable: {report_error or 'unknown error'}.")
                    st.session_state["groq_report_text"] = _generate_local_report(report_rows, summary, period_label)
                # Build and persist PDF bytes at generation time to avoid broken download links on rerender.
                built_pdf = _build_pdf_report(report_rows, summary, period_label) or b""
                st.session_state["report_pdf_bytes"] = built_pdf
                st.session_state["report_pdf_name"] = f"traffic_report_{'daily' if days == 1 else 'weekly'}.pdf"

        if st.session_state.get("groq_report_text"):
            st.markdown("#### AI Report")
            st.markdown(st.session_state["groq_report_text"])
            if not st.session_state.get("report_pdf_bytes"):
                st.session_state["report_pdf_bytes"] = _build_pdf_report(report_rows, summary, period_label) or b""
                st.session_state["report_pdf_name"] = f"traffic_report_{'daily' if days == 1 else 'weekly'}.pdf"
            st.download_button(
                "Download PDF report",
                data=st.session_state["report_pdf_bytes"],
                file_name=st.session_state.get("report_pdf_name", "traffic_report.pdf"),
                mime="application/pdf",
                key="download_pdf_report_btn",
            )


if __name__ == "__main__":
    main()

