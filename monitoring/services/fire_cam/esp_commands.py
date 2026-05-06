"""Commandes HTTP vers l’ESP32-CAM — `/control` (Arduino CameraWebServer) ou `/cmd` (autres firmwares)."""

from __future__ import annotations

import logging
from typing import Any

import requests

from monitoring.services.fire_cam.esp_net import base_url, clean_host

logger = logging.getLogger(__name__)

FS_OPTIONS: list[tuple[int, str]] = [
    (5, "QVGA  320×240"),
    (6, "CIF   400×296"),
    (8, "VGA   640×480"),
    (9, "SVGA  800×600"),
    (10, "XGA  1024×768"),
    (11, "HD   1280×720"),
    (12, "SXGA 1280×1024"),
    (13, "UXGA 1600×1200"),
]

FS_NAMES: dict[int, str] = {v: label.split()[0] for v, label in FS_OPTIONS}

GAIN_LABELS: list[str] = ["2x", "4x", "8x", "16x", "32x", "64x", "128x"]

WB_OPTIONS: list[tuple[int, str]] = [
    (0, "Auto"),
    (1, "Ensoleillé"),
    (2, "Nuageux"),
    (3, "Bureau"),
    (4, "Maison"),
    (5, "Nuit"),
]

FX_OPTIONS: list[tuple[int, str]] = [
    (0, "Normal"),
    (1, "Négatif"),
    (2, "Noir & Blanc"),
    (3, "Rougeâtre"),
    (4, "Verdâtre"),
    (5, "Bleuâtre"),
    (6, "Sépia"),
]

BOOL_KEYS = frozenset(
    {
        "awb",
        "awb_gain",
        "aec",
        "aec2",
        "agc",
        "bpc",
        "wpc",
        "raw_gma",
        "lenc",
        "dcw",
        "hmirror",
        "vflip",
    }
)

INT_KEYS = frozenset(
    {
        "framesize",
        "quality",
        "brightness",
        "contrast",
        "saturation",
        "sharpness",
        "wb_mode",
        "ae_level",
        "gainceiling",
        "special_effect",
        "led_intensity",
    }
)

# Ordre d’envoi recommandé (proche firmware ESP32-CAM web UI)
_CMD_ORDER: list[str] = [
    "framesize",
    "quality",
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "awb",
    "awb_gain",
    "wb_mode",
    "aec",
    "aec2",
    "ae_level",
    "agc",
    "gainceiling",
    "bpc",
    "wpc",
    "raw_gma",
    "lenc",
    "dcw",
    "hmirror",
    "vflip",
    "special_effect",
    "led_intensity",
]


def default_esp_prefs() -> dict[str, Any]:
    """
    Préréglage « qualité image » (Arduino CameraWebServer / même plage de paramètres).

    - ``framesize`` 13 = UXGA 1600×1200 (résolution max proposée dans l’UI).
    - ``quality`` 4 = JPEG le plus fin (4 = max qualité, 63 = compression max).
    - Luminosité / contraste / saturation / netteté légèrement relevés pour un rendu net et vivant.
    """
    return {
        "framesize": 13,
        "quality": 4,
        "brightness": 0,
        "contrast": 1,
        "saturation": 1,
        "sharpness": 2,
        "awb": True,
        "awb_gain": True,
        "wb_mode": 0,
        "aec": True,
        "aec2": True,
        "ae_level": 0,
        "agc": True,
        "gainceiling": 2,
        "bpc": True,
        "wpc": True,
        "raw_gma": True,
        "lenc": True,
        "dcw": True,
        "hmirror": False,
        "vflip": False,
        "special_effect": 0,
        "led_intensity": 0,
    }


def merge_prefs(stored: dict[str, Any] | None) -> dict[str, Any]:
    out = default_esp_prefs()
    if not stored:
        return out
    for k, v in stored.items():
        if k not in out:
            continue
        if k in BOOL_KEYS:
            if isinstance(v, bool):
                out[k] = v
            else:
                s = str(v).strip().lower()
                out[k] = s in ("1", "true", "yes", "on")
        elif k in INT_KEYS:
            try:
                out[k] = int(v)
            except (TypeError, ValueError):
                pass
        else:
            out[k] = v
    return out


def normalize_pref(var: str, val: Any) -> Any:
    if var in BOOL_KEYS:
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        return s in ("1", "true", "on", "yes")
    if var in INT_KEYS:
        return int(round(float(val)))
    return val


def send_cmd(host: str, var: str, val: int | str | bool) -> bool:
    """
    Envoie var/val au firmware caméra.
    Ordre : `/control` (ex. Arduino « CameraWebServer »), puis `/cmd` (ex. forks Ultralytics).
    """
    bu = base_url(host)
    if not bu:
        return False
    if isinstance(val, bool):
        val = 1 if val else 0
    paths = ("/control", "/cmd")
    last_status: int | None = None
    for path in paths:
        try:
            r = requests.get(
                f"{bu.rstrip('/')}{path}",
                params={"var": var, "val": str(val)},
                timeout=(2, 3),
            )
            last_status = r.status_code
            if r.ok:
                return True
            if r.status_code == 404:
                continue
            logger.warning("ESP %s %s=%s → HTTP %s", path, var, val, r.status_code)
            return False
        except requests.RequestException as exc:
            logger.warning("ESP %s %s=%s : %s", path, var, val, exc)
            return False
    logger.warning("ESP /control et /cmd %s=%s → dernier HTTP %s", var, val, last_status)
    return False


def apply_prefs_to_esp(host_raw: str, prefs: dict[str, Any]) -> None:
    host = clean_host(host_raw)
    if not host:
        return
    for var in _CMD_ORDER:
        if var not in prefs:
            continue
        val = prefs[var]
        send_cmd(host, var, normalize_pref(var, val))


def prefs_from_post(post: Any) -> dict[str, Any]:
    """Construit un dict prefs depuis request.POST Django."""
    merged = merge_prefs(None)
    for var in INT_KEYS:
        if var not in post:
            continue
        raw = post.get(var)
        if raw is None or raw == "":
            continue
        try:
            merged[var] = int(raw)
        except (TypeError, ValueError):
            pass
    for var in BOOL_KEYS:
        merged[var] = post.get(var) == "1"
    return merged
