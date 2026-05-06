"""
Suivi multi-objets via YOLOv8 track (ByteTrack / BoT-SORT intégrés à Ultralytics).

Les trajectoires sont stockées par ID de track.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from ultralytics import YOLO


def _centroid(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


class MultiObjectTracker:
    """
    Utilise model.track() pour personnes + éventuellement sacs (classes COCO).
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.35,
        iou: float = 0.45,
        device: str | None = None,
        track_classes: Tuple[int, ...] | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        # None = toutes les classes demandées dans track()
        self.track_classes = track_classes

        # id -> liste de (frame_idx, cx, cy)
        self.trajectories: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
        self._frame_idx = 0

    def reset(self) -> None:
        self.trajectories.clear()
        self._frame_idx = 0

    def step(
        self,
        frame_bgr: np.ndarray,
        persist: bool = True,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Une frame de tracking. Retourne frame annoté et tracks:
        {id, xyxy, cls, conf}
        """
        classes = list(self.track_classes) if self.track_classes is not None else None
        results = self.model.track(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            classes=classes,
            persist=persist,
            verbose=False,
            device=self.device,
        )[0]

        tracks: List[dict] = []
        if results.boxes is None or len(results.boxes) == 0:
            self._frame_idx += 1
            return results.plot(), tracks

        boxes = results.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id
        if ids is None:
            self._frame_idx += 1
            return results.plot(), tracks
        ids_np = ids.cpu().numpy().astype(int)

        for i in range(len(xyxy)):
            tid = int(ids_np[i])
            c = _centroid(xyxy[i])
            self.trajectories[tid].append((self._frame_idx, c[0], c[1]))
            tracks.append(
                {
                    "id": tid,
                    "xyxy": xyxy[i].astype(float),
                    "cls": int(clss[i]),
                    "conf": float(confs[i]),
                }
            )

        self._frame_idx += 1
        return results.plot(), tracks

    def get_trajectory(self, track_id: int) -> List[Tuple[int, float, float]]:
        return list(self.trajectories.get(track_id, []))
