"""Inférence YOLO ONNX (640) — même logique que l’export Ultralytics."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def letterbox(
    im: np.ndarray,
    new_shape: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    shape = im.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    if (shape[1], shape[0]) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (float(dw), float(dh))


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def scale_boxes_back(
    xyxy: np.ndarray,
    pad_w: float,
    pad_h: float,
    ratio: float,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    out = xyxy.copy().astype(np.float32)
    out[:, [0, 2]] -= pad_w
    out[:, [1, 3]] -= pad_h
    out /= ratio
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out


class YoloOnnx:
    def __init__(
        self,
        model_path: str | Path,
        conf_thres: float = 0.65,
        iou_thres: float = 0.45,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.inp_name = self.session.get_inputs()[0].name
        out = self.session.get_outputs()[0]
        self.out_name = out.name
        self.nc = int(out.shape[1] - 4)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_names = [f"cl{i}" for i in range(self.nc)]
        txt = model_path.parent / "classes.txt"
        if txt.is_file():
            try:
                lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if len(lines) == self.nc:
                    self.class_names = lines
            except OSError:
                pass
        names_env = os.environ.get("YOLO_CLASS_NAMES", "").strip()
        if names_env:
            parts = [p.strip() for p in names_env.split(",")]
            if len(parts) == self.nc:
                self.class_names = parts

    def _preprocess(self, bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float]]:
        im, r, (pad_w, pad_h) = letterbox(bgr, 640)
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = im_rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return x, r, (pad_w, pad_h)

    def _postprocess(
        self,
        output: np.ndarray,
        orig_hw: tuple[int, int],
        r: float,
        pad: tuple[float, float],
    ) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        oh, ow = orig_hw
        pad_w, pad_h = pad
        pred = output[0].transpose(1, 0)
        boxes_cxcywh = pred[:, :4]
        cls_logits = pred[:, 4:]
        cls_prob = 1.0 / (1.0 + np.exp(-cls_logits))
        conf = cls_prob.max(axis=1)
        cls_ids = cls_prob.argmax(axis=1)

        m = conf >= self.conf_thres
        if not np.any(m):
            return []
        boxes_cxcywh = boxes_cxcywh[m]
        conf = conf[m]
        cls_ids = cls_ids[m]

        xyxy = xywh2xyxy(boxes_cxcywh)
        xyxy = scale_boxes_back(xyxy, pad_w, pad_h, r, ow, oh)

        boxes_wh: list[list[float]] = []
        scores_list: list[float] = []
        classes_list: list[int] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w, h = x2 - x1, y2 - y1
            if w < 1 or h < 1:
                continue
            boxes_wh.append([float(x1), float(y1), float(w), float(h)])
            scores_list.append(float(conf[i]))
            classes_list.append(int(cls_ids[i]))

        if not boxes_wh:
            return []

        idx = cv2.dnn.NMSBoxes(boxes_wh, scores_list, self.conf_thres, self.iou_thres)
        if idx is None or len(idx) == 0:
            return []
        idx = np.array(idx).flatten()
        out_list: list[tuple[int, float, tuple[int, int, int, int]]] = []
        for j in idx:
            x1, y1, w, h = boxes_wh[j]
            x2, y2 = x1 + w, y1 + h
            out_list.append(
                (
                    classes_list[j],
                    scores_list[j],
                    (int(x1), int(y1), int(x2), int(y2)),
                )
            )
        return out_list

    def predict_boxes(self, bgr: np.ndarray) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        h0, w0 = bgr.shape[:2]
        x, r, pad = self._preprocess(bgr)
        out = self.session.run([self.out_name], {self.inp_name: x})[0]
        return self._postprocess(out, (h0, w0), r, pad)

    def draw_boxes(
        self,
        bgr: np.ndarray,
        dets: list[tuple[int, float, tuple[int, int, int, int]]],
    ) -> np.ndarray:
        out = bgr.copy()
        for cls_id, score, (x1, y1, x2, y2) in dets:
            label = f"{self.class_names[cls_id]} {score:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                out,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return out
