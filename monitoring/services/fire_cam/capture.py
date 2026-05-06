"""Capture JPEG depuis l’ESP (port 80)."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


def fetch_jpeg_capture(base_url: str) -> np.ndarray | None:
    """Télécharge `/capture` et retourne une image BGR ou None."""
    bu = (base_url or "").rstrip("/")
    if not bu:
        return None
    try:
        r = requests.get(f"{bu}/capture", timeout=5)
        r.raise_for_status()
        data = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.debug("capture fail: %s", exc)
        return None
