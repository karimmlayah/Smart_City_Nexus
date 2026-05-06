"""
Pipeline bout-en-bout : tracking YOLO, combat (YOLO classify pré-entraîné),
panique, objets abandonnés, score.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .behavior.abandoned import AbandonedObjectDetector
from .behavior.fight_pretrained import FightYoloClassifier
from .behavior.panic import PanicDetector
from .tracking.tracker import MultiObjectTracker
from .utils.anomaly_scorer import AnomalyScorer
from .utils.config import CLASS_PERSON, FIGHT_PROB_ROLLING_MAX, default_fight_classifier_path
from .utils.explainability import build_alert_explanation


def _person_tracks(tracks: List[dict]) -> List[dict]:
    return [t for t in tracks if t.get("cls") == CLASS_PERSON]


def load_fight_classifier(path: Optional[Path] = None) -> tuple[Optional[FightYoloClassifier], str]:
    """
    Charge uniquement un modèle **pré-entraîné** (YOLO classify).
    Pas d'entraînement local requis.
    """
    p = path or default_fight_classifier_path()
    if not p.is_file():
        return None, f"aucun fichier: {p}"
    try:
        return FightYoloClassifier(p), str(p.name)
    except Exception as exc:
        return None, f"chargement impossible ({exc}): {p}"


def run_video_pipeline(
    video_path: str | Path,
    output_path: Optional[str | Path] = None,
    yolo_weights: str = "yolov8n.pt",
    fight_classifier_path: Optional[Path] = None,
    show_preview: bool = True,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Traite une vidéo complète. Retourne statistiques + dernier état explicable.

    ``fight_classifier_path`` : chemin vers .pt YOLO classify (défaut : best_fight_classifier.pt
    à la racine du dépôt smartcity, ou variable SMARTCROWD_FIGHT_CLASSIFIER).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = MultiObjectTracker(
        weights=yolo_weights,
        track_classes=(CLASS_PERSON, 24, 26, 28),
    )
    panic_det = PanicDetector(fps=fps)
    abandoned_det = AbandonedObjectDetector(weights=yolo_weights)
    fight_clf, fight_source = load_fight_classifier(fight_classifier_path)
    scorer = AnomalyScorer()

    fight_roll: deque = deque(maxlen=FIGHT_PROB_ROLLING_MAX)

    writer: Optional[cv2.VideoWriter] = None
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_p), fourcc, fps, (w, h))

    stats = {
        "frames": 0,
        "max_people": 0,
        "alerts": 0,
        "max_risk": 0.0,
    }
    last_expl: Dict[str, Any] = {}
    fight_warn_printed = False

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            plotted, tracks = tracker.step(frame)
            persons = _person_tracks(tracks)
            n_people = len(persons)
            stats["max_people"] = max(stats["max_people"], n_people)

            panic_score, panic_ids = panic_det.update(persons)

            ab_score, ab_reasons, _ = abandoned_det.step(frame)

            if fight_clf is not None:
                p_frame = fight_clf.predict_fight_probability(frame)
                fight_roll.append(p_frame)
                fight_p = max(fight_roll) if fight_roll else 0.0
            else:
                if not fight_warn_printed:
                    fight_warn_printed = True
                fight_p = 0.0

            risk = scorer.score(fight_p, panic_score, ab_score)
            stats["max_risk"] = max(stats["max_risk"], risk)

            expl = build_alert_explanation(
                fight_p,
                panic_score,
                ab_score,
                risk,
                panic_ids,
                ab_reasons,
            )
            last_expl = expl
            if expl["alert"]:
                stats["alerts"] += 1

            hud = (
                f"People:{n_people} | Fight:{fight_p:.2f} | Panic:{panic_score:.2f} | "
                f"Aband:{ab_score:.2f} | RISK:{risk:.2f}"
            )
            cv2.putText(
                plotted,
                hud,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            if fight_clf is None:
                cv2.putText(
                    plotted,
                    f"Fight: no model ({fight_source})",
                    (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 165, 255),
                    1,
                )
            if expl["alert"]:
                cv2.putText(
                    plotted,
                    "ALERT",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )

            if writer:
                writer.write(plotted)
            if show_preview:
                cv2.imshow("Smart Crowd Safety AI", plotted)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            stats["frames"] += 1
            frame_idx += 1
    finally:
        cap.release()
        if writer:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    return {
        "stats": stats,
        "explanation": last_expl,
        "video": str(video_path),
        "output": str(output_path) if output_path else None,
        "fight_model": fight_clf.label if fight_clf else fight_source,
        "fight_backend": "yolo_classify" if fight_clf else "none",
    }
