"""
Backends classification « combat » : Ultralytics (.pt) ou personnalisés (.h5 / .pth).

- .pth : pickle d’un nn.Module, ou dict avec valeur nn.Module parmi les clés
  model|module|net|teacher|student (pas uniquement « state_dict »).
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
from collections import deque
from pathlib import Path
from typing import Callable, TypeVar

import cv2
import numpy as np
from django.conf import settings

_TLoad = TypeVar("_TLoad")


def predict_yolo_cls(model, frame_bgr: np.ndarray) -> dict | None:
    r = model.predict(source=frame_bgr, verbose=False)[0]
    if r.probs is None:
        return None
    probs = r.probs
    raw = probs.data
    if raw is None:
        return None
    if hasattr(raw, "cpu"):
        raw = raw.cpu().numpy()
    arr = np.asarray(raw).ravel()
    p_fight = float(arr[0]) if len(arr) > 0 else 0.0
    top1 = int(probs.top1)
    tc = probs.top1conf
    top1conf = float(tc.cpu().item()) if hasattr(tc, "cpu") else float(tc)
    return {
        "p_fight": p_fight,
        "top1": top1,
        "top1conf": top1conf,
        "label": "fight" if top1 == 0 else "nonfight",
    }


class YoloFightBackend:
    __slots__ = ("_weights", "_model")

    def __init__(self, weights: str):
        from ultralytics import YOLO

        self._weights = weights
        self._model = YOLO(weights)

    def reset_sequence_buffer(self) -> None:
        """Réservé aux backends à mémoire temporelle (.h5 séquence)."""
        pass

    def predict_prob_fight(self, frame_bgr: np.ndarray) -> dict | None:
        return predict_yolo_cls(self._model, frame_bgr)


def _keras_post_logits(y_flat: np.ndarray) -> dict:
    if y_flat.size <= 1:
        raw = float(y_flat[0]) if y_flat.size else 0.0
        p_fight = float(1.0 / (1.0 + np.exp(-raw))) if np.isfinite(raw) else 0.5
    else:
        e = np.exp(y_flat - np.max(y_flat))
        sm = e / (np.sum(e) + 1e-12)
        p_fight = float(sm[0]) if len(sm) else 0.5

    top1 = 0 if p_fight >= 0.5 else 1
    top1conf = float(max(p_fight, 1.0 - p_fight))
    return {
        "p_fight": float(p_fight),
        "top1": top1,
        "top1conf": top1conf,
        "label": "fight" if top1 == 0 else "nonfight",
    }


def _keras_first_input_shape_tuple(model) -> tuple:
    ish = getattr(model, "input_shape", None)
    if isinstance(ish, list):
        if len(ish) != 1:
            raise RuntimeError(
                ".h5 : entrées multiples non encore prises en charge pour le combat."
            )
        ish = ish[0]
    if ish is None or len(ish) < 2:
        raise RuntimeError(".h5 : forme d'entrée illisible pour le modèle Keras.")
    return tuple(ish)


def _keras_home_models_dir() -> Path:
    root = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras"))
    d = root / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _keras_unlink_maybe_corrupt(weights_name: str) -> None:
    p = _keras_home_models_dir() / weights_name
    try:
        if p.is_file():
            p.unlink()
    except OSError:
        pass


def _keras_is_incomplete_fetch_error(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, urllib.error.ContentTooShortError):
        return True
    low = type(exc).__name__.lower()
    if "contenttooshort" in low:
        return True
    msg = str(exc).lower()
    return "retrieval incomplete" in msg or "url fetch failure" in msg


def _keras_exc_chain_has_incomplete(exc: BaseException) -> bool:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if _keras_is_incomplete_fetch_error(cur):
            return True
        cur = cur.__cause__ or getattr(cur, "__context__", None)
    return False


def _keras_load_imagenet_backbone_with_retries(
    load_fn: Callable[[], _TLoad],
    cache_weight_filenames: tuple[str, ...],
    *,
    label: str,
) -> _TLoad:
    """
    Keras télécharge les poids sur GCS ; sur réseau instable le fichier .h5
    peut rester tronqué → on supprime les restes et on retente quelques fois.
    """

    n_try = max(
        1,
        int(getattr(settings, "FIGHT_KERAS_IMAGENET_WEIGHT_RETRIES", 5) or 5),
    )
    delays = tuple(min(12.0, 1.0 * (2**i)) for i in range(n_try))

    models_dir = _keras_home_models_dir()
    last_err: BaseException | None = None
    for i in range(n_try):
        try:
            return load_fn()
        except Exception as e:
            last_err = e
            if _keras_exc_chain_has_incomplete(e):
                for w in cache_weight_filenames:
                    _keras_unlink_maybe_corrupt(w)
                time.sleep(delays[i] if i < len(delays) else delays[-1])
                continue
            raise
    hint = models_dir.resolve()
    raise RuntimeError(
        f"Poids ImageNet (« {label} ») : téléchargement interrompu ({hint}). Réseau, proxy ou "
        "antivirus coupe souvent les gros fichiers (~95 Mo pour ResNet50). Supprimez manuellement "
        "les .h5 tronqués dans ce dossier, vérifiez la connexion, puis relancez. "
        "Vous pouvez aussi télécharger le même fichier depuis le dépôt "
        "`tensorflow/keras-applications` (Google Storage) avec un gestionnaire de téléchargement "
        "qui reprend après coupure."
    ) from last_err


def _keras_resolve_seq_embedding(
    keras,
    *,
    feat_dim: int,
    forced_name: str,
) -> tuple[object, object, tuple[int, int], int]:
    """
    Retourne (sous-modèle pooling ImageNet sans tête, preprocess_input, (H,W), dim_sortie).

    La dimension doit coïncider avec celle utilisée lors de l'entraînement du fichier .h5.
    """

    pref = forced_name.strip().lower()

    def _resnet50():
        def _once():
            m = keras.applications.ResNet50(
                include_top=False, pooling="avg", weights="imagenet"
            )
            return m, keras.applications.resnet.preprocess_input, (224, 224)

        return _keras_load_imagenet_backbone_with_retries(
            _once,
            ("resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",),
            label="ResNet50",
        )

    def _mobilenet_v2():
        def _once():
            m = keras.applications.MobileNetV2(
                include_top=False, pooling="avg", weights="imagenet"
            )
            return m, keras.applications.mobilenet_v2.preprocess_input, (224, 224)

        return _keras_load_imagenet_backbone_with_retries(
            _once,
            (
                # alpha=1.0, rows=224 par défaut (voir keras mobilenet_v2.py)
                "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
            ),
            label="MobileNetV2",
        )

    def _inception_resnet_v2():
        def _once():
            m = keras.applications.InceptionResNetV2(
                include_top=False, pooling="avg", weights="imagenet"
            )
            return (
                m,
                keras.applications.inception_resnet_v2.preprocess_input,
                (299, 299),
            )

        return _keras_load_imagenet_backbone_with_retries(
            _once,
            (
                "inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5",
            ),
            label="InceptionResNetV2",
        )

    def _densenet121():
        def _once():
            m = keras.applications.DenseNet121(
                include_top=False, pooling="avg", weights="imagenet"
            )
            return m, keras.applications.densenet.preprocess_input, (224, 224)

        return _keras_load_imagenet_backbone_with_retries(
            _once,
            ("densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",),
            label="DenseNet121",
        )

    rows: tuple[tuple[str, int, callable], ...] = (
        ("resnet50", 2048, _resnet50),
        ("mobilenet_v2", 1280, _mobilenet_v2),
        ("inception_resnet_v2", 1536, _inception_resnet_v2),
        ("densenet121", 1024, _densenet121),
    )

    names_rows = [(n, o, _) for (n, o, _) in rows]
    avail = ", ".join(f"{n}→{o}" for n, o, _ in names_rows)

    if pref not in ("", "auto"):
        sel = [(n, o, lb) for n, o, lb in rows if n == pref]
        if not sel:
            raise RuntimeError(
                f"FIGHT_CUSTOM_KERAS_SEQ_EMBEDDING={forced_name!r} inconnu. "
                f"Attendus : {avail} ou chaîne vide (auto)."
            )
        nm, out_dim_expect, loader = sel[0]
        if feat_dim != out_dim_expect:
            raise RuntimeError(
                f"l'embedding « {nm} » sort {out_dim_expect} dimensions alors que le graphe "
                f".h5 attend feat_dim={feat_dim}. Ajustez FIGHT_CUSTOM_* ou le modèle."
            )
    else:
        sel = [(n, o, lb) for n, o, lb in rows if o == feat_dim]
        if not sel:
            raise RuntimeError(
                f".h5 séquence : feat_dim={feat_dim}, sans backbone ImageNet prévu ({avail}). "
                "Ajoutez un nouveau couple (nom,dim→loader) dans _keras_resolve_seq_embedding "
                "si votre fichier utilise un backbone custom projeté dessus."
            )
        nm, out_dim_expect, loader = sel[0]

    embed_model, preprocess, hw = loader()
    embed_model.trainable = False
    actual = int(embed_model.output_shape[-1])
    if actual != feat_dim:
        raise RuntimeError(
            f"Embedding {nm} ({actual}d) différent du tenseur attendu par le graphe ({feat_dim}d)."
        )
    return embed_model, preprocess, hw, actual


def _keras_rnn_compat_custom_objects(keras_mod):
    """
    Anciens .h5 enregistrent encore time_major=False sur LSTM/GRU/SimpleRNN ;
    Keras 3 (TF 2.16+) ne l’accepte plus en argument de couche — on le retire.
    """

    def _make_compat(base):
        class _Compat(base):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.pop("time_major", None)
                super().__init__(*args, **kwargs)

            @classmethod
            def from_config(cls, config):
                c = dict(config) if isinstance(config, dict) else config
                if isinstance(c, dict):
                    c.pop("time_major", None)
                return super().from_config(c)

        _Compat.__name__ = getattr(base, "__name__", "RNNCompat")
        return _Compat

    layers = keras_mod.layers
    out = {}
    for cls_name in ("LSTM", "GRU", "SimpleRNN"):
        if hasattr(layers, cls_name):
            out[cls_name] = _make_compat(getattr(layers, cls_name))
    return out


def _keras_fight_merge_custom_objects(keras_mod) -> dict:
    """HDF5 legacy : class_name « model », « functional », etc. hors Module_objects."""
    merged = dict(_keras_rnn_compat_custom_objects(keras_mod))
    mdl = keras_mod.models
    try:
        from keras.src.models import functional as _kfun

        functional_cls = _kfun.Functional
    except ImportError:
        functional_cls = mdl.Model
    for alias, cls_obj in (
        ("model", mdl.Model),
        ("Functional", functional_cls),
        ("functional", functional_cls),
        ("Sequential", mdl.Sequential),
        ("sequential", mdl.Sequential),
    ):
        merged.setdefault(alias, cls_obj)
    return merged


def _keras_load_fight_weights_h5(keras_mod, path: Path):
    """
    Charge un classify .h5 local (fichier de confiance utilisateur).
    safe_mode=False : lambdas / configs legacy parfois rejetés en mode strict.
    """
    p = str(path.resolve())
    co = _keras_fight_merge_custom_objects(keras_mod)
    try:
        return keras_mod.models.load_model(
            p,
            compile=False,
            custom_objects=co,
            safe_mode=False,
        )
    except KeyError as err:
        key = repr(err.args[0]) if err.args else "?"
        raise RuntimeError(
            "Fichier .h5 combat illisible ou tronqué : élément manquant "
            f"{key} (souvent un groupe de poids « model » absent dans le HDF5). "
            f"Vérifiez l’intégrité de « {path.name} » (téléchargement complet), "
            "ou réexportez le modèle depuis TensorFlow/Keras avec model.save() complet. "
            "Les .h5 partiels ou produits par une autre version de Keras peuvent "
            "provoquer cette erreur."
        ) from err


class KerasFightBackend:
    __slots__ = (
        "_keras",
        "_model",
        "_seq_lock",
        "_input_kind",
        "_seq_len",
        "_feat_dim",
        "_embed_model",
        "_embed_preprocess",
        "_embed_hw",
        "_seq_buf",
        "_image_hw",
    )

    def __init__(self, path: Path):
        try:
            import tensorflow as tf

            keras = tf.keras  # pylint: disable=import-error,no-name-in-module
        except ImportError as exc:
            raise RuntimeError(
                "Poids .h5 : ajoutez TensorFlow (pip install tensorflow)."
            ) from exc

        self._keras = keras
        self._model = _keras_load_fight_weights_h5(keras, path)
        self._seq_lock = threading.Lock()
        self._seq_buf = deque()
        self._embed_model = None
        self._embed_preprocess = None  # callable
        self._embed_hw: tuple[int, int] | None = None
        self._image_hw = (224, 224)

        shape = _keras_first_input_shape_tuple(self._model)

        ov_len = getattr(settings, "FIGHT_CUSTOM_KERAS_SEQ_LEN", 0) or 0
        ov_feat = getattr(settings, "FIGHT_CUSTOM_KERAS_SEQ_FEAT_DIM", 0) or 0

        if len(shape) == 3:
            raw_t = shape[1]
            raw_f = shape[2]
            t_dim = ov_len if ov_len > 0 else raw_t
            f_dim = ov_feat if ov_feat > 0 else raw_f
            if isinstance(t_dim, tuple) or t_dim is None:
                raise RuntimeError(
                    "Longueur de séquence introuvable dans le graphe .h5 "
                    "(dim T=None). Définissez FIGHT_CUSTOM_KERAS_SEQ_LEN (>0)."
                )
            if isinstance(f_dim, tuple) or f_dim is None:
                raise RuntimeError(
                    "Taille du vecteur de features introuvable "
                    "(dim=None). Définissez FIGHT_CUSTOM_KERAS_SEQ_FEAT_DIM (>0)."
                )
            seq_len_i = int(t_dim)
            feat_dim_i = int(f_dim)
            self._input_kind = "sequence"
            self._seq_len = seq_len_i
            self._feat_dim = feat_dim_i

            pref = getattr(settings, "FIGHT_CUSTOM_KERAS_SEQ_EMBEDDING", "") or ""
            self._embed_model, self._embed_preprocess, hw, odim = (
                _keras_resolve_seq_embedding(
                    keras, feat_dim=self._feat_dim, forced_name=pref.strip().lower()
                )
            )
            self._embed_hw = hw
            self._embed_model.trainable = False
            assert odim == self._feat_dim
            self._seq_buf = deque(maxlen=self._seq_len)

        elif len(shape) == 4:
            self._input_kind = "image"
            h, w = shape[1], shape[2]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                self._image_hw = (int(h), int(w))
        else:
            raise RuntimeError(
                f".h5 : entrée de rang {len(shape)-1} non supportée ({shape})."
            )

    def reset_sequence_buffer(self) -> None:
        """À appeler avant chaque nouvelle vidéo (cache global du backend)."""

        if self._input_kind != "sequence":
            return
        with self._seq_lock:
            self._seq_buf.clear()

    def predict_prob_fight(self, frame_bgr: np.ndarray) -> dict | None:
        if self._input_kind == "sequence":
            return self._predict_sequence_frame(frame_bgr)
        return self._predict_single_image(frame_bgr)

    def _stack_seq_window(self) -> np.ndarray:
        buf_list = list(self._seq_buf)
        if not buf_list:
            raise RuntimeError("buffer vide")
        if len(buf_list) < self._seq_len:
            pad = self._seq_len - len(buf_list)
            head = buf_list[0]
            stacked = np.vstack(
                [np.broadcast_to(head, (pad, head.size)), np.stack(buf_list, axis=0)]
            ).astype(np.float32)
        else:
            stacked = np.stack(buf_list, axis=0).astype(np.float32)

        batch = stacked[None, ...]
        assert batch.shape[-1] == self._feat_dim
        assert batch.shape[1] == self._seq_len
        return np.ascontiguousarray(batch)

    def _predict_sequence_frame(self, frame_bgr: np.ndarray) -> dict | None:
        assert (
            self._embed_model is not None
            and self._embed_preprocess is not None
            and self._embed_hw is not None
        )
        eh, ew = self._embed_hw

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_s = cv2.resize(rgb, (ew, eh), interpolation=cv2.INTER_AREA)
        x_rgb = rgb_s.astype(np.float32)
        x_b = np.expand_dims(x_rgb, axis=0)
        x_prep = np.ascontiguousarray(self._embed_preprocess(x_b.copy()))

        with self._seq_lock:
            feats = np.asarray(self._embed_model.predict(x_prep, verbose=0)).squeeze(
                axis=0
            )
            vec = feats.reshape(-1).astype(np.float32)
            if vec.size != self._feat_dim:
                raise RuntimeError(
                    "Embedding et entrée séquence désalignées : "
                    f"attendu {self._feat_dim}, obtenu {vec.size}"
                )
            self._seq_buf.append(vec)
            batch_seq = self._stack_seq_window()

        out = self._model.predict(batch_seq, verbose=0)
        y_flat = np.atleast_1d(np.asarray(out).squeeze().flatten()).astype(np.float64)
        return _keras_post_logits(y_flat)

    def _predict_single_image(self, frame_bgr: np.ndarray) -> dict | None:
        th, tw = self._image_hw

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        x = np.expand_dims(rgb / 255.0, axis=0).astype(np.float32)
        norm = (getattr(settings, "FIGHT_CUSTOM_KERAS_NORM_MODE", "") or "").strip().lower()
        if norm in ("imagenet", "mobilenet", "std"):
            mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
            std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
            x = (x - mean) / std

        batch = np.ascontiguousarray(x)
        out = self._model.predict(batch, verbose=0)
        y_flat = np.atleast_1d(np.asarray(out).squeeze().flatten()).astype(np.float64)
        return _keras_post_logits(y_flat)


def _guess_torch_hw() -> tuple[int, int]:
    pair = getattr(settings, "FIGHT_CUSTOM_TORCH_INPUT_HW", ()) or ()
    return (int(pair[0]), int(pair[1])) if len(pair) == 2 else (224, 224)


class TorchNnFightBackend:
    __slots__ = ("_torch", "_module", "_hw")

    def reset_sequence_buffer(self) -> None:
        pass

    def __init__(self, path: Path):
        import torch
        import torch.nn as nn  # pylint: disable=import-error

        self._torch = torch

        try:
            ckpt = torch.load(
                str(path.resolve()),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            ckpt = torch.load(str(path.resolve()), map_location="cpu")
        module = None

        if isinstance(ckpt, nn.Module):
            module = ckpt
        elif isinstance(ckpt, dict):
            for key in ("model", "module", "net", "teacher", "student"):
                cand = ckpt.get(key)
                if isinstance(cand, nn.Module):
                    module = cand
                    break
            if module is None and isinstance(ckpt.get("model_state_dict"), dict):
                raise RuntimeError(
                    ".pth : fichier state_dict brut — rechargez avec le nn.Module pickled "
                    "ou utilisez ultralytics (.pt)."
                )

        if module is None:
            raise RuntimeError(
                f"{path.name} : fichier .pytorch hors format (voir docstring)."
            )

        module.eval()
        self._module = module
        self._hw = _guess_torch_hw()

    def predict_prob_fight(self, frame_bgr: np.ndarray) -> dict | None:
        th, tw = self._hw
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(x, (2, 0, 1))[None, ...]

        mode = getattr(settings, "FIGHT_CUSTOM_TORCH_NORMALIZE_TO_N", "").strip().lower()
        if mode == "neg1":
            tensor = tensor * 2.0 - 1.0

        t_in = self._torch.from_numpy(np.ascontiguousarray(tensor)).float()
        with self._torch.no_grad():
            out = self._module(t_in)

        if isinstance(out, (list, tuple)) and len(out):
            out = out[0]

        if not isinstance(out, self._torch.Tensor):
            return None

        logits = out.detach().cpu().float().squeeze()

        logits_np = logits.numpy().flatten().astype(np.float64)

        if logits_np.size == 2:
            e = np.exp(logits_np - np.max(logits_np))
            sm = e / (np.sum(e) + 1e-12)
            p_fight = float(sm[0])
            top1 = 0 if p_fight >= 0.5 else 1
            top1conf = float(max(p_fight, 1.0 - p_fight))
            return {
                "p_fight": p_fight,
                "top1": top1,
                "top1conf": top1conf,
                "label": "fight" if top1 == 0 else "nonfight",
            }

        if logits_np.size == 1:
            v = float(logits.squeeze().detach().cpu().item())
            if 0.0 <= v <= 1.0001:
                p_fight = float(v)
            else:
                p_fight = float(1.0 / (1.0 + np.exp(-v)))

            top1 = 0 if p_fight >= 0.5 else 1
            top1conf = float(max(p_fight, 1.0 - p_fight))
            return {
                "p_fight": float(p_fight),
                "top1": top1,
                "top1conf": top1conf,
                "label": "fight" if top1 == 0 else "nonfight",
            }

        e = np.exp(logits_np - np.max(logits_np))
        sm = e / (np.sum(e) + 1e-12)
        p_fight = float(sm[0])
        top1 = 0 if p_fight >= 0.5 else 1
        top1conf = float(max(p_fight, 1.0 - p_fight))
        return {
            "p_fight": p_fight,
            "top1": top1,
            "top1conf": top1conf,
            "label": "fight" if top1 == 0 else "nonfight",
        }


def build_fight_backend(weights_path_str: str) -> YoloFightBackend | KerasFightBackend | TorchNnFightBackend:
    pth = Path(weights_path_str)
    suf = pth.suffix.lower()

    if suf == ".h5":
        return KerasFightBackend(pth)

    if suf == ".pth":
        return TorchNnFightBackend(pth)

    return YoloFightBackend(weights_path_str)
