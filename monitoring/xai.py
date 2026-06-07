"""
Explainabilité IA — occlusion (carte rouge-jaune), cache, messages honnêtes en français.
Pas de fausse « attention modèle » par défaut.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from django.conf import settings

from . import vision_analysis

# Grille max 16×16 = 256 patches (contrat performances)
XAI_GRID_MAX = 16
XAI_MIN_CONF_FOR_OCCLUSION = 0.5
XAI_OCC_RATIO_DEFAULT = 0.10
XAI_OVERLAY_ALPHA = 0.52
XAI_BOX_COLOR = (0, 255, 255)  # jaune BGR
XAI_BOX_THICK = 2


def _xai_out_dir() -> Path:
    d = Path(settings.MEDIA_ROOT) / "xai"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _xai_cache_dir() -> Path:
    d = _xai_out_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _quality_badge(pipeline_conf_pct: float | None) -> tuple[str, str]:
    """(clé, libellé FR) pour l’UI."""
    if pipeline_conf_pct is None:
        return "unavailable", "Indisponible"
    x = float(pipeline_conf_pct)
    if x >= 70:
        return "reliable", "Fiable"
    if x >= 50:
        return "indicative", "Indicatif"
    return "unavailable", "Indisponible"


def _french_explanation(
    *,
    mode: str,
    max_conf: float,
) -> str:
    """Textes dynamiques honnêtes (FR)."""
    if mode == "high":
        return (
            "La carte XAI montre que le modèle se concentre principalement sur une zone irrégulière "
            "de la chaussée correspondant à un nid de poule. La confiance élevée rend cette "
            "explication fiable."
        )
    if mode == "medium":
        return (
            "L’explication est indicative : le modèle détecte une zone potentiellement endommagée, "
            "mais la confiance reste modérée."
        )
    if mode == "none":
        return (
            "Aucune explication visuelle fiable n’est générée car le modèle n’a pas confirmé "
            "un dégât avec une confiance suffisante."
        )
    if mode == "low_conf":
        return (
            "XAI indisponible : confiance modèle inférieure au seuil requis pour une carte "
            "d’occlusion fiable."
        )
    if mode == "no_model":
        return (
            "Superposition visuelle à titre d’aide à la lecture — ce n’est pas une carte "
            "d’attention du modèle (YOLO non chargé ou indisponible)."
        )
    return (
        "Impossible de calculer une carte d’explication pour ce média."
    )


def _opencv_visual_overlay_only(image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Contour léger — étiqueté explicitement comme non-XAI modèle."""
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    edges_col = cv2.applyColorMap(edges, cv2.COLORMAP_BONE)
    overlay = cv2.addWeighted(image_bgr, 0.68, edges_col, 0.32, 0)
    return overlay, (
        "Superposition visuelle (pas une explication du modèle) — contours pour repérer les zones "
        "contrastées."
    )


def _apply_red_yellow_heatmap(heat_01: np.ndarray) -> np.ndarray:
    """heat_01 float32 H×W dans [0,1] → BGR couleur rouge (chaud) → jaune (froid)."""
    t = np.clip(heat_01.astype(np.float32), 0.0, 1.0)
    h, w = t.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    g = (255.0 * (1.0 - t)).astype(np.uint8)
    rch = np.full((h, w), 255, dtype=np.uint8)
    bgr[:, :, 0] = 0
    bgr[:, :, 1] = g
    bgr[:, :, 2] = rch
    return bgr


def _draw_boxes(frame_bgr: np.ndarray, boxes: list[dict[str, Any]] | None) -> None:
    import cv2

    if not boxes:
        return
    for b in boxes:
        xy = b.get("xyxy")
        if not xy or len(xy) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in xy[:4]]
        except (TypeError, ValueError):
            continue
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), XAI_BOX_COLOR, XAI_BOX_THICK)


def _cache_key(
    path_str: str,
    model_key: str | None,
    infer_conf: float,
    grid_n: int,
    occ_ratio: float,
    mtime: float,
) -> str:
    raw = f"{path_str}|{model_key}|{infer_conf:.4f}|{grid_n}|{occ_ratio:.4f}|{mtime}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:40]


def _try_load_cache(cache_key: str) -> dict[str, Any] | None:
    meta_path = _xai_cache_dir() / f"{cache_key}.json"
    img_path = _xai_cache_dir() / f"{cache_key}.jpg"
    if not meta_path.is_file() or not img_path.is_file():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        mu = settings.MEDIA_URL.rstrip("/")
        rel = img_path.relative_to(Path(settings.MEDIA_ROOT))
        url = f"{mu}/{rel.as_posix().replace(chr(92), '/')}"
        meta["heatmap_url"] = url
        meta["overlay_url"] = url
        meta["cached"] = True
        return meta
    except Exception:
        return None


def _save_cache(cache_key: str, result: dict[str, Any], img_bgr: np.ndarray) -> None:
    import cv2

    meta_path = _xai_cache_dir() / f"{cache_key}.json"
    img_path = _xai_cache_dir() / f"{cache_key}.jpg"
    try:
        cv2.imwrite(str(img_path), img_bgr)
        store = {k: v for k, v in result.items() if k not in ("heatmap_url", "overlay_url", "cached")}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def occlusion_sensitivity_image(
    image_abs_path: str | Path,
    model_key: str | None = None,
    *,
    inference_conf: float | None = None,
    yolo_used: bool = False,
    max_confidence: float = 0.0,
    boxes: list[dict[str, Any]] | None = None,
    pipeline_confidence: float | None = None,
) -> dict[str, Any]:
    """
    Occlusion uniquement si YOLO utilisé et max_confidence >= 0.5.
    Sinon message honnête sans heatmap trompeuse.
    """
    import cv2

    image_abs_path = Path(image_abs_path)
    path_str = str(image_abs_path.resolve())
    frame = cv2.imread(path_str)
    infer_c = float(
        inference_conf
        if inference_conf is not None
        else getattr(settings, "ROAD_DAMAGE_YOLO_CONF", 0.25)
    )

    badge_key, badge_fr = _quality_badge(pipeline_confidence)
    mc = float(max_confidence or 0.0)

    if frame is None:
        return {
            "ok": False,
            "error": "read_failed",
            "heatmap_url": None,
            "overlay_url": None,
            "explanation": "Impossible de lire l’image.",
            "method": "error",
            "xai_quality_badge": badge_key,
            "xai_quality_label_fr": badge_fr,
            "model_confidence_pct": round(mc * 100, 1),
            "legend_fr": "",
            "cached": False,
            "skip_reason_fr": "Fichier image illisible ou manquant.",
            "detection_label_fr": "—",
        }

    model, backend, _req, _used, _err = vision_analysis._load_vision_model(model_key)

    # Pas de modèle : superposition honnête (pas vendue comme attention YOLO)
    if model is None:
        overlay, note = _opencv_visual_overlay_only(frame)
        out = _xai_out_dir() / f"visual_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(str(out), overlay)
        rel = out.relative_to(Path(settings.MEDIA_ROOT))
        url = f"{settings.MEDIA_URL.rstrip('/')}/{rel.as_posix().replace(chr(92), '/')}"
        return {
            "ok": True,
            "fallback": True,
            "visual_overlay_only": True,
            "fallback_label": "Superposition visuelle (pas une explication du modèle)",
            "heatmap_url": url,
            "overlay_url": url,
            "baseline_conf": None,
            "explanation": _french_explanation(mode="no_model", max_conf=mc)
            + " "
            + note,
            "method": "visual_overlay_only",
            "xai_quality_badge": "unavailable",
            "xai_quality_label_fr": "Indisponible",
            "model_confidence_pct": round(mc * 100, 1),
            "legend_fr": "",
            "cached": False,
        }

    # Confiance insuffisante pour occlusion fiable (pas de heatmap trompeuse)
    if not yolo_used:
        return {
            "ok": True,
            "fallback": False,
            "heatmap_url": None,
            "overlay_url": None,
            "baseline_conf": round(mc, 4),
            "explanation": _french_explanation(mode="none", max_conf=mc)
            + " XAI ne fournit pas de carte d’occlusion sans inférence YOLO sur ce signalement.",
            "method": "skipped_low_confidence",
            "xai_quality_badge": badge_key,
            "xai_quality_label_fr": badge_fr,
            "model_confidence_pct": round(mc * 100, 1),
            "legend_fr": "",
            "cached": False,
            "skip_reason_fr": "Modèle YOLO non utilisé pour le score (règles seules).",
            "detection_label_fr": "—",
        }
    if mc < XAI_MIN_CONF_FOR_OCCLUSION:
        return {
            "ok": True,
            "fallback": False,
            "heatmap_url": None,
            "overlay_url": None,
            "baseline_conf": round(mc, 4),
            "explanation": _french_explanation(mode="low_conf", max_conf=mc),
            "method": "skipped_low_confidence",
            "xai_quality_badge": badge_key,
            "xai_quality_label_fr": badge_fr,
            "model_confidence_pct": round(mc * 100, 1),
            "legend_fr": "",
            "cached": False,
            "skip_reason_fr": "Confiance de détection < 50 % — pas de carte d’occlusion.",
            "detection_label_fr": ", ".join(
                str(b.get("label", "")) for b in (boxes or [])[:5]
            )
            if boxes
            else "—",
        }

    try:
        mtime = image_abs_path.stat().st_mtime
    except OSError:
        mtime = 0.0

    occ_ratio = float(getattr(settings, "ROAD_DAMAGE_XAI_OCCLUSION_RATIO", XAI_OCC_RATIO_DEFAULT))
    grid_n = min(XAI_GRID_MAX, max(8, int(getattr(settings, "ROAD_DAMAGE_XAI_GRID_N", XAI_GRID_MAX))))

    ck = _cache_key(path_str, model_key, infer_c, grid_n, occ_ratio, mtime)
    cached = _try_load_cache(ck)
    if cached:
        return cached

    baseline = vision_analysis.baseline_predict_confidence(
        frame, model, conf=infer_c, backend=backend
    )
    h, w = frame.shape[:2]
    grid_y = grid_n
    grid_x = grid_n
    occ_h = max(8, int(h * occ_ratio))
    occ_w = max(8, int(w * occ_ratio))
    step_y = h / grid_y
    step_x = w / grid_x

    heat_small = np.zeros((grid_y, grid_x), dtype=np.float32)
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
            conf_occ = vision_analysis.baseline_predict_confidence(
                occ, model, conf=infer_c, backend=backend
            )
            impact = max(0.0, float(baseline) - float(conf_occ))
            heat_small[gy, gx] = impact

    heat_small = cv2.GaussianBlur(heat_small, (3, 3), 0)

    m = float(np.max(heat_small)) if heat_small.size else 0.0
    if m > 1e-8:
        heat_small = heat_small / m
    else:
        heat_small = np.zeros_like(heat_small)

    heat_full = cv2.resize(heat_small, (w, h), interpolation=cv2.INTER_CUBIC)
    heat_full = np.clip(heat_full.astype(np.float32), 0.0, 1.0)
    heat_full = cv2.GaussianBlur(heat_full, (11, 11), 0)
    hm = float(np.max(heat_full))
    if hm > 1e-8:
        heat_full = heat_full / hm
    else:
        heat_full = np.zeros_like(heat_full)

    colored = _apply_red_yellow_heatmap(heat_full)
    overlay_base = cv2.addWeighted(frame, 1.0 - XAI_OVERLAY_ALPHA, colored, XAI_OVERLAY_ALPHA, 0)
    _draw_boxes(overlay_base, boxes)

    legend_fr = "Zones chaudes = forte influence sur la décision du modèle (occlusion)."

    explain_mode = "high" if mc >= 0.7 else "medium"
    explanation = _french_explanation(mode=explain_mode, max_conf=mc)

    det_label = ", ".join(str(b.get("label", "")) for b in (boxes or [])[:5]) if boxes else "—"

    result: dict[str, Any] = {
        "ok": True,
        "fallback": False,
        "visual_overlay_only": False,
        "fallback_label": None,
        "heatmap_url": "",
        "overlay_url": "",
        "baseline_conf": round(float(baseline), 4),
        "explanation": explanation,
        "method": "occlusion_sensitivity",
        "xai_quality_badge": badge_key,
        "xai_quality_label_fr": badge_fr,
        "model_confidence_pct": round(mc * 100, 1),
        "legend_fr": legend_fr,
        "cached": False,
        "detection_label_fr": det_label,
    }
    _save_cache(ck, result, overlay_base)
    mu = settings.MEDIA_URL.rstrip("/")
    rel = (_xai_cache_dir() / f"{ck}.jpg").relative_to(Path(settings.MEDIA_ROOT))
    url = f"{mu}/{rel.as_posix().replace(chr(92), '/')}"
    result["heatmap_url"] = url
    result["overlay_url"] = url
    return result


def explain_reclamation_image(
    image_abs_path: str | Path,
    model_key: str | None = None,
    *,
    inference_conf: float | None = None,
    yolo_used: bool = False,
    max_confidence: float = 0.0,
    boxes: list[dict[str, Any]] | None = None,
    pipeline_confidence: float | None = None,
) -> dict[str, Any]:
    """Point d’entrée vue détail (images)."""
    try:
        return occlusion_sensitivity_image(
            image_abs_path,
            model_key,
            inference_conf=inference_conf,
            yolo_used=yolo_used,
            max_confidence=max_confidence,
            boxes=boxes,
            pipeline_confidence=pipeline_confidence,
        )
    except Exception as exc:
        bk, bf = _quality_badge(pipeline_confidence)
        return {
            "ok": False,
            "fallback": True,
            "visual_overlay_only": False,
            "heatmap_url": None,
            "overlay_url": None,
            "explanation": f"Erreur XAI ({exc}). Utiliser la priorité basée sur les règles.",
            "method": "error",
            "xai_quality_badge": bk,
            "xai_quality_label_fr": bf,
            "model_confidence_pct": round(float(max_confidence or 0) * 100, 1),
            "legend_fr": "",
            "cached": False,
            "skip_reason_fr": str(exc),
            "detection_label_fr": "—",
        }


def grad_cam_placeholder() -> dict[str, Any]:
    return {
        "available": False,
        "message": "Grad-CAM non exposé pour ce modèle YOLO dans ce build.",
    }
