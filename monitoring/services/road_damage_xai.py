from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import requests
from django.conf import settings

from .road_damage_tester import get_road_damage_model


def _max_damage_conf_for_frame(frame_bgr: np.ndarray) -> float:
    model = get_road_damage_model()
    conf = float(getattr(settings, "ROAD_DAMAGE_YOLO_CONF", 0.25))
    iou = float(getattr(settings, "ROAD_DAMAGE_YOLO_IOU", 0.45))
    imgsz = int(getattr(settings, "ROAD_DAMAGE_YOLO_IMGSZ", 640))
    pred = model.predict(source=frame_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
    boxes = getattr(pred, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0
    conf_t = boxes.conf
    conf_arr = conf_t.cpu().numpy() if hasattr(conf_t, "cpu") else np.asarray(conf_t)
    if conf_arr.size == 0:
        return 0.0
    return float(np.max(conf_arr))


def build_occlusion_heatmap(image_path: str | Path) -> dict:
    image_path = Path(image_path)
    frame = cv2.imread(str(image_path))
    if frame is None:
        return {"error": "Impossible de lire l'image pour XAI occlusion."}

    h, w = frame.shape[:2]
    if h < 16 or w < 16:
        return {"error": "Image trop petite pour une analyse XAI occlusion."}

    baseline = _max_damage_conf_for_frame(frame)
    grid_y = int(getattr(settings, "ROAD_DAMAGE_XAI_GRID_Y", 10))
    grid_x = int(getattr(settings, "ROAD_DAMAGE_XAI_GRID_X", 10))
    occ_ratio = float(getattr(settings, "ROAD_DAMAGE_XAI_OCCLUSION_RATIO", 0.18))
    grid_y = max(4, min(24, grid_y))
    grid_x = max(4, min(24, grid_x))
    occ_ratio = max(0.08, min(0.35, occ_ratio))

    occ_h = max(12, int(h * occ_ratio))
    occ_w = max(12, int(w * occ_ratio))
    step_y = h / grid_y
    step_x = w / grid_x

    heat_small = np.zeros((grid_y, grid_x), dtype=np.float32)
    regions: list[dict] = []

    for gy in range(grid_y):
        cy = int((gy + 0.5) * step_y)
        for gx in range(grid_x):
            cx = int((gx + 0.5) * step_x)
            x1 = max(0, cx - occ_w // 2)
            y1 = max(0, cy - occ_h // 2)
            x2 = min(w, x1 + occ_w)
            y2 = min(h, y1 + occ_h)

            occ = frame.copy()
            occ[y1:y2, x1:x2] = 127
            conf_occ = _max_damage_conf_for_frame(occ)
            impact = max(0.0, baseline - conf_occ)
            heat_small[gy, gx] = impact
            regions.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "impact": round(float(impact), 4),
                }
            )

    m = float(np.max(heat_small)) if heat_small.size else 0.0
    if m > 1e-8:
        heat_small = heat_small / m
    heat = cv2.resize(heat_small, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_uint8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.55, colored, 0.45, 0)

    out_dir = Path(settings.MEDIA_ROOT) / "road_damage_xai"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"occlusion_{uuid4().hex}.jpg"
    ok = cv2.imwrite(str(out_path), overlay)
    if not ok:
        return {"error": "Impossible d'ecrire la heatmap XAI."}

    top_regions = sorted(regions, key=lambda r: r["impact"], reverse=True)[:5]
    return {
        "heatmap_path": str(out_path),
        "baseline_conf": round(float(baseline), 4),
        "top_regions": top_regions,
    }


def generate_road_damage_report_with_groq(result: dict, xai_data: dict) -> dict:
    openai_key = (getattr(settings, "OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")).strip()
    groq_key = (getattr(settings, "GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")).strip()
    if not openai_key and not groq_key:
        return {
            "error": "Configurez OPENAI_API_KEY (ou GROQ_API_KEY) dans vos variables d'environnement."
        }

    provider = "openai" if openai_key else "groq"
    api_key = openai_key if provider == "openai" else groq_key
    model = (
        (getattr(settings, "OPENAI_MODEL", "") or "gpt-4o-mini").strip()
        if provider == "openai"
        else (getattr(settings, "GROQ_MODEL", "") or "llama-3.3-70b-versatile").strip()
    )
    endpoint = (
        "https://api.openai.com/v1/chat/completions"
        if provider == "openai"
        else "https://api.groq.com/openai/v1/chat/completions"
    )

    payload_data = {
        "verdict": result.get("verdict"),
        "verdict_label": result.get("verdict_label"),
        "max_conf": result.get("max_conf"),
        "damage_ratio": result.get("damage_ratio"),
        "frames_analyzed": result.get("frames_analyzed"),
        "damage_frames": result.get("damage_frames"),
        "details": (result.get("details") or [])[:10],
        "xai_baseline_conf": xai_data.get("baseline_conf"),
        "xai_top_regions": xai_data.get("top_regions") or [],
    }

    prompt = (
        "Tu es un expert en inspection routiere.\n"
        "A partir des donnees JSON suivantes, redige:\n"
        "1) un resume executif (3 lignes max),\n"
        "2) 5 recommandations priorisees (maintenance),\n"
        "3) un mini rapport technique (niveau risque, causes probables, actions immediates),\n"
        "4) une note de confiance (haute/moyenne/faible) avec justification.\n"
        "Reponds en francais clair, structure en sections Markdown.\n\n"
        f"JSON:\n{json.dumps(payload_data, ensure_ascii=True)}"
    )

    body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "Tu produis des recommandations techniques fiables et actionnables."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            endpoint,
            headers=headers,
            json=body,
            timeout=45,
        )
    except requests.RequestException as exc:
        return {"error": f"Appel Groq impossible: {exc}"}
    if resp.status_code >= 300:
        return {"error": f"{provider.upper()} API error {resp.status_code}: {resp.text[:400]}"}

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return {"error": f"Reponse {provider.upper()} invalide (choices vide)."}
    content = (((choices[0] or {}).get("message") or {}).get("content") or "").strip()
    if not content:
        return {"error": f"Reponse {provider.upper()} vide."}
    return {"report_markdown": content, "model": model, "provider": provider}
