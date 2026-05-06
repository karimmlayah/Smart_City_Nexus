"""
Détection « combat » via modèle Ultralytics **classify** pré-entraîné (.pt),
sans entraînement local — même famille que la page « Combat » du site Smart City.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class FightYoloClassifier:
    """Charge un YOLO `task=classify` et retourne P(fight) sur une frame BGR."""

    def __init__(self, weights: str | Path) -> None:
        from ultralytics import YOLO

        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"Modèle classify introuvable: {self.weights}")
        self.model = YOLO(str(self.weights))

    def predict_fight_probability(self, frame_bgr: np.ndarray) -> float:
        """P(fight) : indice 0 = fight dans le projet existant (fight_classifier)."""
        r = self.model.predict(source=frame_bgr, verbose=False)[0]
        if r.probs is None:
            return 0.0
        raw = r.probs.data
        if raw is None:
            return 0.0
        if hasattr(raw, "cpu"):
            raw = raw.cpu().numpy()
        arr = np.asarray(raw).ravel()
        return float(arr[0]) if len(arr) > 0 else 0.0

    @property
    def label(self) -> str:
        return str(self.weights.name)
