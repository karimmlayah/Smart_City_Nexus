import argparse
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from yt_dlp import YoutubeDL


VEHICLE_LABELS = {"car", "bus", "truck", "motorcycle", "bicycle"}
ZONE_COLORS = [
    (255, 80, 80),
    (80, 220, 80),
    (80, 170, 255),
    (220, 120, 255),
    (255, 220, 80),
]


@dataclass
class SignalDecision:
    score: float
    level: str
    green_s: int
    red_s: int


class TrafficDecisionEngine:
    """Simple rule-based traffic-light decision engine."""

    def __init__(self, cycle_s: int = 90, min_green_s: int = 15, max_green_s: int = 70):
        self.cycle_s = cycle_s
        self.min_green_s = min_green_s
        self.max_green_s = max_green_s
        # Higher weight for heavy vehicles.
        self.weights = {"car": 1.0, "motorcycle": 0.5, "bicycle": 0.3, "bus": 2.0, "truck": 2.0}

    def _score(self, counts: Counter) -> float:
        if not counts:
            return 0.0
        return float(sum(self.weights.get(k, 1.0) * v for k, v in counts.items()))

    def decide_for_zone(self, counts: Counter) -> SignalDecision:
        score = self._score(counts)
        if score >= 10:
            level = "forte"
            green = 60
        elif score >= 5:
            level = "moyenne"
            green = 45
        else:
            level = "faible"
            green = 25
        green = max(self.min_green_s, min(self.max_green_s, green))
        red = max(10, self.cycle_s - green)
        return SignalDecision(score=score, level=level, green_s=green, red_s=red)

    def green_wave(self, decisions: List[SignalDecision]) -> List[Tuple[int, int]]:
        """Return [(zone_index, offset_s)] sorted by priority score."""
        if not decisions:
            return []
        order = sorted(range(len(decisions)), key=lambda i: decisions[i].score, reverse=True)
        return [(zone_idx, idx * 8) for idx, zone_idx in enumerate(order)]


def resolve_video_source(source: str) -> str:
    s = source.strip()
    if "youtube.com" not in s and "youtu.be" not in s:
        return s
    with YoutubeDL({"quiet": True, "format": "best[ext=mp4]/best", "noplaylist": True}) as ydl:
        info = ydl.extract_info(s, download=False)
        return info["url"]


class RoiDrawer:
    def __init__(self) -> None:
        self.rois: List[Tuple[int, int, int, int]] = []
        self.dragging = False
        self.start = (0, 0)
        self.current = (0, 0)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            x1, y1 = self.start
            x2, y2 = x, y
            l, r = sorted((x1, x2))
            t, b = sorted((y1, y2))
            if r - l > 8 and b - t > 8:
                self.rois.append((l, t, r, b))


def inside_any_roi(cx: int, cy: int, rois: List[Tuple[int, int, int, int]]) -> int:
    for i, (l, t, r, b) in enumerate(rois):
        if l <= cx <= r and t <= cy <= b:
            return i
    return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="YouTube URL, RTSP, or local video path")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path (.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--all-classes", action="store_true", help="Do not filter to vehicle classes")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.overrides["conf"] = args.conf
    video_source = resolve_video_source(args.source)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir le flux video.")

    window = "YOLO Live ROI"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    drawer = RoiDrawer()
    cv2.setMouseCallback(window, drawer.mouse_cb)
    cumulative_by_roi: List[Counter] = []
    decision_engine = TrafficDecisionEngine()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, verbose=False, imgsz=640, max_det=100)[0]
        names = results.names
        current_by_roi = [Counter() for _ in drawer.rois]
        if len(cumulative_by_roi) != len(drawer.rois):
            cumulative_by_roi = [Counter() for _ in drawer.rois]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = str(names[cls_id]) if names is not None else str(cls_id)
            if not args.all_classes and label not in VEHICLE_LABELS:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            roi_idx = inside_any_roi(cx, cy, drawer.rois) if drawer.rois else 0
            if drawer.rois and roi_idx < 0:
                continue

            color = ZONE_COLORS[roi_idx % len(ZONE_COLORS)] if drawer.rois else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {float(box.conf[0]):.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
            cv2.circle(frame, (cx, cy), 3, color, -1)

            if drawer.rois:
                current_by_roi[roi_idx][label] += 1
                cumulative_by_roi[roi_idx][label] += 1

        for i, (l, t, r, b) in enumerate(drawer.rois):
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, f"Zone {i+1}", (l, max(15, t - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            realtime = ", ".join([f"{k}:{v}" for k, v in sorted(current_by_roi[i].items())]) or "-"
            cumul = ", ".join([f"{k}:{v}" for k, v in sorted(cumulative_by_roi[i].items())]) or "-"
            cv2.putText(frame, f"R: {realtime}", (10, 24 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(frame, f"C: {cumul}", (10, 42 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Decision support: dynamic signal timings + green wave.
        if drawer.rois:
            decisions = [decision_engine.decide_for_zone(zc) for zc in current_by_roi]
            y0 = 24 + max(0, len(drawer.rois) * 38) + 16
            cv2.putText(frame, "DECISION FEUX:", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            for i, d in enumerate(decisions):
                text = (
                    f"Zone {i+1} | congestion {d.level} | score {d.score:.1f} | "
                    f"vert {d.green_s}s | rouge {d.red_s}s"
                )
                color = ZONE_COLORS[i % len(ZONE_COLORS)]
                cv2.putText(frame, text, (10, y0 + 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            wave = decision_engine.green_wave(decisions)
            if wave:
                wave_txt = "Green wave: " + " -> ".join(
                    [f"Z{z+1}(+{off}s)" for z, off in wave]
                )
                cv2.putText(
                    frame,
                    wave_txt,
                    (10, y0 + 24 + len(decisions) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 255, 180),
                    1,
                )

        if drawer.dragging:
            x1, y1 = drawer.start
            x2, y2 = drawer.current
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        cv2.putText(
            frame,
            "Draw ROI: drag mouse | d: delete last | c: clear | q: quit",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d") and drawer.rois:
            drawer.rois.pop()
            if cumulative_by_roi:
                cumulative_by_roi.pop()
        if key == ord("c"):
            drawer.rois.clear()
            cumulative_by_roi.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
