"""Détection de personnes avec YOLOv8 (Ultralytics)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO

from ..utils.config import CLASS_PERSON


class YoloPersonDetector:
    """Encapsule YOLOv8 pour la classe personne uniquement."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device

    def predict_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """
        Retourne le frame annoté et une liste de détections:
        {xyxy, conf, cls}
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=[CLASS_PERSON],
            verbose=False,
            device=self.device,
        )[0]
        dets: List[dict] = []
        if results.boxes is None or len(results.boxes) == 0:
            return frame_bgr, dets
        boxes = results.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for i in range(len(xyxy)):
            dets.append(
                {
                    "xyxy": xyxy[i].astype(float),
                    "conf": float(confs[i]),
                    "cls": int(clss[i]),
                }
            )
        plotted = results.plot()
        return plotted, dets

    @staticmethod
    def count_people(dets: List[dict]) -> int:
        return len(dets)
