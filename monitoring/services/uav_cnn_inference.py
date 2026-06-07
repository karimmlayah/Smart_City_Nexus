"""
Inférence CNN structure UAV — modèle Keras (224×224, 3 classes).

Configurer dans settings.py / .env :
  UAV_MODEL_PATH — chemin absolu ou relatif à BASE_DIR vers best_CNN.keras

Les cartes « Grad-CAM » affichées utilisent un gradient par rapport à l’image d’entrée,
lissé spatialement (compatible Keras 3 ; les gradients vers couches conv peuvent être absents).
"""
from __future__ import annotations

import logging
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model = None
_last_load_error: str | None = None

MODEL_IMG_SIZE = 224
UAV_MODEL_FILENAME = "best_CNN.keras"

_UAV_FALLBACK_RELATIVE = (
    "models/best_CNN.keras",
    "media/models/best_CNN.keras",
    "static/models/best_CNN.keras",
    "smartcity/models/best_CNN.keras",
    "ai_models/best_CNN.keras",
)


def _base_dir() -> Path:
    return Path(getattr(settings, "BASE_DIR", Path.cwd()))


def _path_from_setting(raw: str) -> Path:
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        p = _base_dir() / p
    return p


def _collect_uav_model_candidates() -> list[Path]:
    """Ordered unique candidate paths (configured path first, then fallbacks)."""
    seen: set[str] = set()
    out: list[Path] = []

    def _add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        out.append(path)

    configured = (getattr(settings, "UAV_MODEL_PATH", "") or "").strip()
    if configured:
        _add(_path_from_setting(configured))
    for rel in _UAV_FALLBACK_RELATIVE:
        _add(_base_dir() / rel)
    return out


@lru_cache(maxsize=1)
def resolve_uav_model_path() -> Path | None:
    """Return the first existing UAV model file among configured + fallback paths."""
    for candidate in _collect_uav_model_candidates():
        if candidate.is_file():
            logger.debug("UAV model resolved at %s", candidate)
            return candidate
    logger.warning(
        "UAV model not found; checked: %s",
        ", ".join(str(p) for p in _collect_uav_model_candidates()),
    )
    return None


def uav_model_status(*, verify_load: bool = False) -> dict[str, Any]:
    """Backend model readiness for dashboard / API."""
    attempted = [str(p) for p in _collect_uav_model_candidates()]
    path = resolve_uav_model_path()
    if path is None:
        return {
            "ready": False,
            "model_path": "",
            "message": "UAV model missing",
            "attempted_paths": attempted,
        }

    if not verify_load:
        return {
            "ready": True,
            "model_path": str(path),
            "message": "UAV model ready",
            "attempted_paths": attempted,
        }

    model = get_uav_model()
    if model is None:
        return {
            "ready": False,
            "model_path": str(path),
            "message": "UAV model found but not loaded",
            "load_error": _last_load_error or "",
            "attempted_paths": attempted,
        }

    return {
        "ready": True,
        "model_path": str(path),
        "message": "UAV model ready",
        "input_shape": getattr(model, "input_shape", None),
        "attempted_paths": attempted,
    }


def get_uav_model():
    """Load the model once (thread-safe)."""
    global _model, _last_load_error
    path = resolve_uav_model_path()
    if path is None:
        _last_load_error = None
        return None
    with _model_lock:
        if _model is None:
            try:
                import tensorflow as tf

                try:
                    tf.keras.utils.disable_interactive_logging()
                except Exception:
                    pass
                _model = tf.keras.models.load_model(str(path), compile=False)
                _last_load_error = None
                logger.info("UAV Keras model loaded from %s", path)
            except Exception as exc:
                _last_load_error = str(exc)
                logger.exception("UAV model load failed for %s: %s", path, exc)
                return None
        return _model


def class_labels() -> tuple[str, str, str]:
    lab = getattr(settings, "UAV_CLASS_LABELS", None)
    if lab and len(lab) == 3:
        return (str(lab[0]), str(lab[1]), str(lab[2]))
    return ("Intact", "Damaged", "Collapsed")


def preprocess_bgr_for_model(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Retourne (tensor_batch 1×224×224×3 float32 [0,1], rgb_original_uint8 pour overlay).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("empty_image")
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    orig = rgb.copy()
    resized = cv2.resize(rgb, (MODEL_IMG_SIZE, MODEL_IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    batch = np.expand_dims(x, axis=0)
    return batch, orig


def _overlay_heatmap_on_rgb(
    rgb_uint8: np.ndarray,
    heatmap_small: np.ndarray,
    intensity: float = 0.48,
) -> np.ndarray:
    """heatmap_small H×W float ≥0, upscaled to rgb shape."""
    h, w = rgb_uint8.shape[:2]
    hm = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_CUBIC)
    hm = np.maximum(hm, 0)
    hm = hm / (np.max(hm) + 1e-8)
    hm_u8 = np.uint8(255 * hm)
    jet = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    blend = np.clip(
        rgb_uint8.astype(np.float32) * (1 - intensity) + jet_rgb.astype(np.float32) * intensity,
        0,
        255,
    ).astype(np.uint8)
    return blend


def _input_gradient_map(batch_224: np.ndarray, pred_idx: int) -> np.ndarray:
    """Gradient du score de la classe prédite par rapport à l’entrée (224×224)."""
    import tensorflow as tf

    model = get_uav_model()
    if model is None:
        raise RuntimeError("model_missing")

    img_tf = tf.Variable(batch_224.astype(np.float32), dtype=tf.float32)
    num_classes = int(model.output_shape[-1])
    idx = min(max(int(pred_idx), 0), num_classes - 1)

    with tf.GradientTape() as tape:
        tape.watch(img_tf)
        preds = model(img_tf, training=False)
        loss = preds[0, idx]

    grads = tape.gradient(loss, img_tf)
    if grads is None:
        raise RuntimeError("input_grad_failed")

    return tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()


def compute_gradcam_proxy(batch_224: np.ndarray, pred_idx: int) -> np.ndarray:
    """
    Carte type Grad-CAM : même signal que la salience entrée, lissée (passe-haut spatial).
    Sous Keras 3, certains graphes ne rendent pas le gradient vers les couches conv ; cette approche reste fidèle à « où l’IA regarde ».
    """
    g224 = _input_gradient_map(batch_224, pred_idx)
    g = np.maximum(g224, 0)
    g = cv2.GaussianBlur(g.astype(np.float32), (21, 21), 0)
    return g


def compute_saliency_map(batch_224: np.ndarray, pred_idx: int) -> np.ndarray:
    """Carte de salience brute sur l’entrée."""
    return np.maximum(_input_gradient_map(batch_224, pred_idx), 0)


def _model_missing_payload() -> dict[str, Any]:
    status = uav_model_status()
    attempted = status.get("attempted_paths") or []
    path = status.get("model_path") or ""
    load_error = (status.get("load_error") or "").strip()

    if path and load_error:
        return {
            "success": False,
            "error": "model_load_failed",
            "message": f"UAV model found at {path} but failed to load. Check TensorFlow/Keras installation.",
            "model_path": path,
            "load_error": load_error,
            "attempted_paths": attempted,
        }

    return {
        "success": False,
        "error": "model_missing",
        "message": "Set UAV_MODEL_PATH to the best_CNN.keras file.",
        "attempted_paths": attempted,
    }


def predict_structural_state(frame_bgr: np.ndarray) -> dict[str, Any]:
    """
    Prédit les probabilités + Grad-CAM + salience + URLs relatives sauvegardées sous MEDIA_ROOT/uav_analysis/.
    """
    model = get_uav_model()
    if model is None:
        return _model_missing_payload()

    batch, rgb_orig = preprocess_bgr_for_model(frame_bgr)
    probs = model.predict(batch, verbose=0)[0]
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / (np.sum(probs) + 1e-12)
    pred_idx = int(np.argmax(probs))
    labels = class_labels()
    label = labels[pred_idx]
    confidence = float(probs[pred_idx])

    try:
        cam = compute_gradcam_proxy(batch, pred_idx)
        sal = compute_saliency_map(batch, pred_idx)
        grad_overlay = _overlay_heatmap_on_rgb(rgb_orig, cam)
        sal_overlay = _overlay_heatmap_on_rgb(rgb_orig, sal)
    except Exception as exc:
        logger.warning("UAV XAI failed: %s", exc)
        grad_overlay = rgb_orig.copy()
        sal_overlay = rgb_orig.copy()

    media_root = Path(settings.MEDIA_ROOT)
    out_dir = media_root / "uav_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex

    def _write_rel(path_local: Path, arr_rgb: np.ndarray) -> str:
        bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        path_local.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path_local), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        rel = path_local.relative_to(media_root)
        return f"{settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"

    base_path = out_dir / uid
    original_url = _write_rel(Path(str(base_path) + "_orig.jpg"), rgb_orig)
    gradcam_url = _write_rel(Path(str(base_path) + "_gradcam.jpg"), grad_overlay)
    saliency_url = _write_rel(Path(str(base_path) + "_sal.jpg"), sal_overlay)

    badge_key = ("intact", "damaged", "collapsed")[pred_idx] if pred_idx < 3 else "unknown"

    return {
        "success": True,
        "probabilities": [float(p) for p in probs],
        "probabilities_dict": {labels[i]: float(probs[i]) for i in range(min(3, len(labels)))},
        "predicted_index": pred_idx,
        "predicted_label": label,
        "badge_key": badge_key,
        "confidence": confidence,
        "original_image_url": original_url,
        "gradcam_image_url": gradcam_url,
        "saliency_image_url": saliency_url,
        "labels_order": list(labels),
    }
