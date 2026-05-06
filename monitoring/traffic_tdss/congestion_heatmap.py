"""Heatmap de congestion : accumulation temps réel des véhicules détectés (densité spatiale)."""

from __future__ import annotations

import base64
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def _unpack(prev: Optional[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], int, int]:
    if not prev or not isinstance(prev, dict):
        return None, 0, 0
    z = prev.get("z")
    gw = int(prev.get("gw", 0))
    gh = int(prev.get("gh", 0))
    fw = int(prev.get("frame_w", 0))
    fh = int(prev.get("frame_h", 0))
    if not z or gw <= 0 or gh <= 0:
        return None, 0, 0
    try:
        raw = zlib.decompress(base64.standard_b64decode(z.encode("ascii") if isinstance(z, str) else z))
        acc = np.frombuffer(raw, dtype=np.float16).reshape(gh, gw).astype(np.float32)
        return acc, fw, fh
    except Exception:
        return None, 0, 0


def _pack(acc: np.ndarray, gw: int, gh: int, frame_w: int, frame_h: int) -> Dict[str, Any]:
    flat = acc.astype(np.float16).tobytes()
    z = base64.standard_b64encode(zlib.compress(flat, 6)).decode("ascii")
    return {
        "v": 1,
        "gw": gw,
        "gh": gh,
        "frame_w": frame_w,
        "frame_h": frame_h,
        "z": z,
    }


def _grid_size(frame_h: int, frame_w: int) -> Tuple[int, int]:
    gh = max(36, frame_h // 6)
    gw = max(48, frame_w // 6)
    return gh, gw


def _roi_mask_full(frame_h: int, frame_w: int, roi_polygons: Sequence[Sequence[Tuple[int, int]]]) -> np.ndarray:
    m = np.ones((frame_h, frame_w), dtype=np.float32)
    if not roi_polygons:
        return m
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    for poly in roi_polygons:
        if len(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    if not mask.any():
        return m
    return (mask.astype(np.float32) / 255.0)


def update_and_render_heatmap(
    frame_bgr: np.ndarray,
    filtered_detections: List[Dict[str, Any]],
    roi_polygons: List[List[Tuple[int, int]]],
    prev_packed: Optional[Dict[str, Any]],
    *,
    decay: float = 0.87,
    alpha: float = 0.48,
    spot_scale: float = 1.15,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Met à jour une grille basse résolution (mémoire temporelle) et fusionne une couche INFERNO sur l'image.

    Les centres des boîtes pondèrent la carte ; léger flou gaussien pour lisser le rendu « vrai » heatmap.
    """
    fh, fw = frame_bgr.shape[:2]
    gh, gw = _grid_size(fh, fw)

    prev_acc, pfw, pfh = _unpack(prev_packed)
    if prev_acc is None or prev_acc.shape != (gh, gw) or pfw != fw or pfh != fh:
        acc = np.zeros((gh, gw), dtype=np.float32)
    else:
        acc = prev_acc * float(decay)

    inv_x = gw / max(1, fw)
    inv_y = gh / max(1, fh)

    for d in filtered_detections:
        try:
            x1 = int(d["x1"])
            y1 = int(d["y1"])
            x2 = int(d["x2"])
            y2 = int(d["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = (x1 + x2) // 2
        cy = min(y2 - max(2, bh // 12), fh - 1)
        gx = int(cx * inv_x)
        gy = int(cy * inv_y)
        gx = int(np.clip(gx, 0, gw - 1))
        gy = int(np.clip(gy, 0, gh - 1))
        wgt = float(np.clip(float(d.get("conf", 0.5)), 0.08, 1.0)) * spot_scale
        rad = max(2, int((bw / max(1.0, fw)) * gw * 0.35))
        rad = int(np.clip(rad, 2, max(5, gw // 6)))
        cv2.circle(acc, (gx, gy), rad, wgt, thickness=-1)

    k = min(31, max(5, (gw // 28) * 2 + 1))
    if k % 2 == 0:
        k += 1
    sm = cv2.GaussianBlur(acc, (k, k), 0)

    mx = float(sm.max())
    if mx < 1e-6:
        return frame_bgr.copy(), _pack(acc, gw, gh, fw, fh)

    norm = (sm / mx * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    colored_full = cv2.resize(colored, (fw, fh), interpolation=cv2.INTER_LINEAR)

    roi_m = _roi_mask_full(fh, fw, roi_polygons)
    a = float(np.clip(alpha, 0.12, 0.72))
    blend_w = roi_m * a
    base = frame_bgr.astype(np.float32)
    heat = colored_full.astype(np.float32)
    w3 = np.stack([blend_w, blend_w, blend_w], axis=-1)
    out = base * (1.0 - w3) + heat * w3
    out = np.clip(out, 0, 255).astype(np.uint8)

    return out, _pack(acc, gw, gh, fw, fh)
