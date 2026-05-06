#!/usr/bin/env python3
"""
Point d'entrée CLI du projet Smart Crowd Safety AI.

Exemples:
  python main.py demo --video path/to/video.mp4 --output out.mp4
  python main.py train --epochs 12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def cmd_demo(args: argparse.Namespace) -> None:
    from pathlib import Path

    from src.pipeline import run_video_pipeline

    out = Path(args.output) if args.output else None
    fc = Path(args.fight_classifier).expanduser() if args.fight_classifier else None
    r = run_video_pipeline(
        args.video,
        output_path=out,
        yolo_weights=args.yolo,
        fight_classifier_path=fc,
        show_preview=not args.no_show,
        max_frames=args.max_frames,
    )
    print(r)


def cmd_train(args: argparse.Namespace) -> None:
    from scripts.train_fight import train
    from src.utils.config import MODELS_DIR

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "fight_cnn_lstm.pt"
    train(args.epochs, args.batch_size, args.lr, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="smartcrowd")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_demo = sub.add_parser("demo", help="Lancer la démo sur une vidéo")
    p_demo.add_argument("--video", required=True)
    p_demo.add_argument("--output", default="")
    p_demo.add_argument(
        "--fight-classifier",
        default="",
        help="Chemin .pt YOLO classify combat (défaut: best_fight_classifier.pt à la racine smartcity)",
    )
    p_demo.add_argument("--no-show", action="store_true")
    p_demo.add_argument("--max-frames", type=int, default=None)
    p_demo.add_argument("--yolo", default="yolov8n.pt")
    p_demo.set_defaults(func=cmd_demo)

    p_train = sub.add_parser("train", help="[Optionnel] Entraîner CNN+LSTM sur données synthétiques")
    p_train.add_argument("--epochs", type=int, default=12)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
