"""Torchvision Faster R-CNN inference for road-damage .pth weights."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision

_model_lock = threading.Lock()
_model_cache: dict[str, torch.nn.Module] = {}


def _infer_num_classes(state_dict: dict) -> int:
    w = state_dict.get("roi_heads.box_predictor.cls_score.weight")
    if w is not None and hasattr(w, "shape"):
        return int(w.shape[0])
    return 4


def _class_label_map(num_classes: int) -> dict[int, str]:
    from django.conf import settings

    custom = getattr(settings, "ROAD_VISION_FASTERRCNN_LABELS", None)
    if isinstance(custom, dict) and custom:
        return {int(k): str(v) for k, v in custom.items()}
    # Faster R-CNN index 0 = background; remaining indices are detection classes.
    defaults = ["background", "crack", "pothole", "damage"]
    return {i: defaults[i] if i < len(defaults) else f"class_{i}" for i in range(num_classes)}


def load_fasterrcnn(path: str) -> torch.nn.Module:
    """Load a Faster R-CNN ResNet50-FPN checkpoint (.pth state_dict)."""
    resolved = str(Path(path).resolve())
    with _model_lock:
        if resolved in _model_cache:
            return _model_cache[resolved]

        ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            raise ValueError("fasterrcnn_checkpoint_invalid")

        num_classes = _infer_num_classes(ckpt)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes,
        )
        model.load_state_dict(ckpt)
        model.eval()
        _model_cache[resolved] = model
        return model


def predict_fasterrcnn_bgr(
    model: torch.nn.Module,
    frame_bgr: np.ndarray,
    conf: float | None = None,
) -> dict[str, Any]:
    """Run Faster R-CNN on a BGR OpenCV frame."""
    conf_threshold = float(conf if conf is not None else 0.25)
    if frame_bgr is None or frame_bgr.size == 0:
        return {
            "boxes": [],
            "labels": [],
            "max_confidence": 0.0,
            "annotated_bgr": frame_bgr,
        }

    num_classes = int(getattr(model, "roi_heads").box_predictor.cls_score.out_features)
    names = _class_label_map(num_classes)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        outputs = model([tensor])[0]

    boxes_out: list[dict[str, Any]] = []
    max_conf = 0.0
    boxes = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    annotated = frame_bgr.copy()
    for i in range(len(boxes)):
        score = float(scores[i])
        if score < conf_threshold:
            continue
        cls_id = int(labels[i])
        if cls_id == 0:
            continue
        max_conf = max(max_conf, score)
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        label = names.get(cls_id, f"class_{cls_id}")
        boxes_out.append(
            {
                "label": label,
                "class_id": cls_id,
                "confidence": round(score, 4),
                "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 140, 255), 2)
        cv2.putText(
            annotated,
            f"{label} {score:.2f}",
            (int(x1), max(int(y1) - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 140, 255),
            1,
            cv2.LINE_AA,
        )

    return {
        "boxes": boxes_out,
        "labels": sorted({b["label"] for b in boxes_out}),
        "max_confidence": round(max_conf, 4),
        "annotated_bgr": annotated,
    }
