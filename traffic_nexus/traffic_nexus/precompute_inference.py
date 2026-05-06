import argparse
import json
from pathlib import Path
from typing import List, Set

import cv2
from ultralytics import YOLO

from tdss.vehicle_detection import infer_candidates_multi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute detections for replay-optimized mode.")
    p.add_argument("--video", required=True, help="Video path")
    p.add_argument("--output", default="precomputed_results.jsonl", help="Output JSONL file")
    p.add_argument("--models", nargs="+", default=["best.pt", "yolov8n.pt"], help="YOLO model list")
    p.add_argument("--labels", default="car,bus,truck,motorcycle,bicycle,ambulance,emergency-vehicle", help="Comma-separated labels")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--max-det", type=int, default=80)
    p.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    allowed: Set[str] = {s.strip() for s in args.labels.split(",") if s.strip()}

    models: List[YOLO] = [YOLO(m) for m in args.models]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 0 else 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_idx = 0
    saved = 0
    with out_path.open("w", encoding="utf-8") as f:
        header = {
            "type": "meta",
            "video": str(video_path),
            "fps": fps,
            "width": width,
            "height": height,
            "models": args.models,
            "imgsz": args.imgsz,
            "max_det": args.max_det,
            "frame_step": args.frame_step,
        }
        f.write(json.dumps(header) + "\n")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % max(1, args.frame_step) != 0:
                continue
            candidates = infer_candidates_multi(
                models=models,
                frame=frame,
                allowed_labels=allowed,
                imgsz=args.imgsz,
                max_det=args.max_det,
            )
            row = {"type": "frame", "frame_idx": frame_idx, "candidates": candidates}
            f.write(json.dumps(row) + "\n")
            saved += 1
            if saved % 100 == 0:
                print(f"Processed {saved} frames...")

    cap.release()
    print(f"Done. Saved {saved} frames to {out_path}")


if __name__ == "__main__":
    main()

