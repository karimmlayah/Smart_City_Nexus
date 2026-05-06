from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import time

import cv2
from django.conf import settings as django_settings

from .sightengine_client import SightengineError, analyze_image_file


def analyze_video_with_sightengine_as_images(
    video_path: Path,
    frames_subdir: str,
    frame_stride: int = 30,
    max_frames: int = 60,
) -> Dict[str, Any]:
    """
    Coupe une vidéo en frames JPEG et envoie chaque image à Sightengine (API images).
    Retourne un résumé avec la frame la plus "weapon" et la liste détaillée.

    - frame_stride: on garde 1 frame sur `frame_stride` (ex. 30 ≈ 1 fps à 30 fps)
    - max_frames: nombre maximum de frames envoyées à Sightengine
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise SightengineError("Vidéo introuvable pour l'analyse par images Sightengine.")

    media_root = Path(django_settings.MEDIA_ROOT)
    out_dir = media_root / frames_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SightengineError("Impossible d'ouvrir la vidéo pour extraire des images.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    throttle_delay = float(
        getattr(django_settings, "SIGHTENGINE_THROTTLE_DELAY_SEC", 1.0)
    )
    results: List[Dict[str, Any]] = []
    frame_idx = 0
    kept = 0

    try:
        while kept < max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            img_name = f"frame_{frame_idx:06d}.jpg"
            img_path = out_dir / img_name
            ok_write = cv2.imwrite(str(img_path), frame)
            if not ok_write:
                frame_idx += 1
                continue

            try:
                api_raw = analyze_image_file(img_path)
                weapon_data = api_raw.get("weapon") or {}
                prob = float(weapon_data.get("prob", 0.0))
                label = weapon_data.get("type", "weapon")
            except SightengineError as exc:
                # On arrête proprement en remontant l'erreur (clé invalide, etc.).
                raise SightengineError(str(exc)) from exc

            t_sec = frame_idx / float(fps or 1.0)
            rel = img_path.relative_to(media_root)
            results.append(
                {
                    "frame_idx": frame_idx,
                    "t_sec": t_sec,
                    "rel_path": rel.as_posix(),
                    "prob": prob,
                    "label": label,
                }
            )

            kept += 1
            frame_idx += 1

            # Throttling doux pour ne pas dépasser la limite de requêtes/secondes du plan gratuit.
            if throttle_delay > 0:
                time.sleep(throttle_delay)
    finally:
        cap.release()

    if not results:
        return {"frames": [], "best": None}

    best = max(results, key=lambda r: r.get("prob", 0.0))
    return {"frames": results, "best": best}

