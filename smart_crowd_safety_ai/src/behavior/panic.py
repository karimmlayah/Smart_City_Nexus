"""Détection de mouvement anormal (panique / course) à partir des vitesses des tracks."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..utils.config import PANIC_VELOCITY_PX_PER_S


class PanicDetector:
    def __init__(
        self,
        velocity_threshold: float = PANIC_VELOCITY_PX_PER_S,
        fps: float = 30.0,
    ) -> None:
        self.velocity_threshold = velocity_threshold
        self.fps = max(fps, 1e-3)
        self._prev: Dict[int, Tuple[float, float]] = {}

    def reset(self) -> None:
        self._prev.clear()

    def update(self, tracks: List[dict]) -> Tuple[float, List[int]]:
        """
        Retourne (score_panic 0-1, liste d'ids suspects).
        Score = min(1, max_vitesse / (2 * seuil)) pour lisser.
        """
        max_v = 0.0
        suspicious: List[int] = []
        dt = 1.0 / self.fps

        for tr in tracks:
            tid = tr["id"]
            xyxy = np.array(tr["xyxy"])
            cx = float((xyxy[0] + xyxy[2]) / 2)
            cy = float((xyxy[1] + xyxy[3]) / 2)
            if tid in self._prev:
                px, py = self._prev[tid]
                dist = float(np.hypot(cx - px, cy - py))
                v = dist / dt
                if v > max_v:
                    max_v = v
                if v >= self.velocity_threshold:
                    suspicious.append(tid)
            self._prev[tid] = (cx, cy)

        # Normalise en score 0-1
        score = float(min(1.0, max_v / (2.0 * self.velocity_threshold)))
        return score, suspicious
