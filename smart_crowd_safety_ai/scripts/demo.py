#!/usr/bin/env python3
"""Démo vidéo sur un fichier — modèles pré-entraînés uniquement."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run_video_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Smart Crowd Safety AI — démo vidéo")
    p.add_argument("video", type=str, help="Chemin vers la vidéo (mp4, etc.)")
    p.add_argument("-o", "--output", type=str, default="", help="Vidéo sortie annotée")
    p.add_argument("--no-show", action="store_true", help="Ne pas afficher la fenêtre OpenCV")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--yolo", type=str, default="yolov8n.pt")
    p.add_argument(
        "--fight-classifier",
        default="",
        help="Chemin .pt YOLO classify (défaut: best_fight_classifier.pt à la racine smartcity)",
    )
    args = p.parse_args()

    out = Path(args.output) if args.output else None
    fc = Path(args.fight_classifier).expanduser() if args.fight_classifier else None

    result = run_video_pipeline(
        args.video,
        output_path=out,
        yolo_weights=args.yolo,
        fight_classifier_path=fc,
        show_preview=not args.no_show,
        max_frames=args.max_frames,
    )
    print("Stats:", result["stats"])
    print("Fight backend:", result.get("fight_backend"), result.get("fight_model"))
    print("Explication:", result["explanation"])


if __name__ == "__main__":
    main()
