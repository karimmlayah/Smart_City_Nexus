"""
Objet abandonné (sac / valise) : détection YOLO + proximité personne puis éloignement.

Logique: si un bbox sac n'intersecte plus aucune personne depuis N frames
alors qu'il était proche d'une personne avant -> alerte.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from ultralytics import YOLO

from ..utils.config import ABANDON_DISTANCE_PX, ABANDON_FRAMES, BAG_CLASSES, CLASS_PERSON


def _center(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _min_distance_bag_to_persons(bag_xyxy: np.ndarray, person_xyxys: List[np.ndarray]) -> float:
    if not person_xyxys:
        return 1e9
    bc = np.array(_center(bag_xyxy))
    best = 1e9
    for p in person_xyxys:
        pc = np.array(_center(p))
        d = float(np.linalg.norm(bc - pc))
        if d < best:
            best = d
    return best


class AbandonedObjectDetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        abandon_frames: int = ABANDON_FRAMES,
        abandon_distance_px: float = ABANDON_DISTANCE_PX,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.abandon_frames = abandon_frames
        self.abandon_distance_px = abandon_distance_px
        self.device = device
        # id sac (si track disponible) ou index interne -> compteur frames sans voisinage
        self._no_person_streak: Dict[int, int] = defaultdict(int)
        self._was_near_person: Dict[int, bool] = defaultdict(bool)

    def reset(self) -> None:
        self._no_person_streak.clear()
        self._was_near_person.clear()

    def step(self, frame_bgr: np.ndarray) -> Tuple[float, List[str], List[dict]]:
        """
        Retourne (score 0-1, raisons texte, événements détail).
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            classes=[CLASS_PERSON] + list(BAG_CLASSES),
            verbose=False,
            device=self.device,
        )[0]
        events: List[dict] = []
        reasons: List[str] = []

        if results.boxes is None or len(results.boxes) == 0:
            return 0.0, reasons, events

        boxes = results.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        persons = [xyxy[i] for i in range(len(clss)) if clss[i] == CLASS_PERSON]
        bags: List[Tuple[int, np.ndarray]] = []
        for i in range(len(clss)):
            if clss[i] in BAG_CLASSES:
                bid = int(ids[i]) if ids is not None else i + 10000
                bags.append((bid, xyxy[i]))

        max_score = 0.0
        for bid, bxy in bags:
            d_min = _min_distance_bag_to_persons(bxy, persons)
            near = d_min < self.abandon_distance_px
            if near:
                self._was_near_person[bid] = True
                self._no_person_streak[bid] = 0
            else:
                if self._was_near_person[bid]:
                    self._no_person_streak[bid] += 1
                else:
                    self._no_person_streak[bid] = min(self._no_person_streak[bid] + 1, self.abandon_frames)

            if self._was_near_person[bid] and self._no_person_streak[bid] >= self.abandon_frames:
                s = 1.0
                reasons.append(
                    f"Objet (sac/valise) id={bid} éloigné des personnes depuis "
                    f"{self._no_person_streak[bid]} frames."
                )
                events.append({"type": "abandoned_object", "bag_id": bid, "score": s})
                max_score = max(max_score, s)

        return float(max_score), reasons, events
