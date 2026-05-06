from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import requests
from django.conf import settings


SIGHTENGINE_API_USER = getattr(settings, "SIGHTENGINE_API_USER", "")
SIGHTENGINE_API_KEY = getattr(settings, "SIGHTENGINE_API_KEY", "")
SIGHTENGINE_IMAGE_URL = "https://api.sightengine.com/1.0/check.json"
SIGHTENGINE_VIDEO_URL = "https://api.sightengine.com/1.0/video/check.json"
SIGHTENGINE_VIDEO_BYID_URL = "https://api.sightengine.com/1.0/video/byid.json"
# Vidéo : modèles autorisés par l'API video (ex. "weapon") — pas "context" (réservé aux images).
SIGHTENGINE_VIDEO_MODELS = getattr(settings, "SIGHTENGINE_VIDEO_MODELS", "weapon")


class SightengineError(Exception):
    """Erreur lors de l'appel à l'API Sightengine."""


def _ensure_creds() -> None:
    if not (SIGHTENGINE_API_USER and SIGHTENGINE_API_KEY):
        raise SightengineError("Identifiants Sightengine manquants (voir settings.py).")


SIGHTENGINE_IMAGE_MODELS = getattr(settings, "SIGHTENGINE_IMAGE_MODELS", "weapon")


def analyze_image_file(path: str | Path) -> Dict[str, Any]:
    """Analyse une image locale (armes, etc.)."""
    _ensure_creds()
    path = Path(path)
    if not path.is_file():
        raise SightengineError("Fichier image introuvable pour Sightengine.")

    ctype, _ = mimetypes.guess_type(path.name)
    if not ctype:
        ctype = "image/jpeg"

    with open(path, "rb") as fh:
        files = {"media": (path.name, fh, ctype)}
        data = {
            "models": SIGHTENGINE_IMAGE_MODELS,
            "api_user": SIGHTENGINE_API_USER,
            "api_secret": SIGHTENGINE_API_KEY,
        }
        resp = requests.post(SIGHTENGINE_IMAGE_URL, data=data, files=files, timeout=40)

    if resp.status_code != 200:
        # Essaye d'extraire un message d'erreur lisible du JSON renvoyé.
        try:
            err_payload = resp.json()
            err_block = (err_payload or {}).get("error") or {}
            msg = err_block.get("message") or str(err_payload)
        except Exception:
            msg = resp.text[:500]
        raise SightengineError(f"Erreur HTTP Sightengine {resp.status_code}: {msg}")
    out = resp.json()
    if out.get("status") != "success":
        raise SightengineError(str(out))
    return out


def analyze_image_url(url: str) -> Dict[str, Any]:
    """Analyse une image via URL publique."""
    _ensure_creds()
    resp = requests.get(
        SIGHTENGINE_IMAGE_URL,
        params={
            "models": SIGHTENGINE_IMAGE_MODELS,
            "url": url,
            "api_user": SIGHTENGINE_API_USER,
            "api_secret": SIGHTENGINE_API_KEY,
        },
        timeout=40,
    )
    if resp.status_code != 200:
        try:
            err_payload = resp.json()
            err_block = (err_payload or {}).get("error") or {}
            msg = err_block.get("message") or str(err_payload)
        except Exception:
            msg = resp.text[:500]
        raise SightengineError(f"Erreur HTTP Sightengine {resp.status_code}: {msg}")
    out = resp.json()
    if out.get("status") != "success":
        raise SightengineError(str(out))
    return out


def submit_video_file(path: str | Path, callback_url: str) -> str:
    """
    Soumet une vidéo locale (<= 50 Mo). Retourne media.id (med_...).
    https://sightengine.com/docs/moderate-stored-video-asynchronously
    """
    _ensure_creds()
    path = Path(path)
    if not path.is_file():
        raise SightengineError("Fichier vidéo introuvable.")

    max_mb = int(getattr(settings, "SIGHTENGINE_VIDEO_MAX_UPLOAD_MB", 50))
    if path.stat().st_size > max_mb * 1024 * 1024:
        raise SightengineError(f"Vidéo trop volumineuse (max {max_mb} Mo pour Sightengine).")

    ctype, _ = mimetypes.guess_type(path.name)
    if not ctype:
        ctype = "video/mp4"

    with open(path, "rb") as fh:
        files = {"media": (path.name, fh, ctype)}
        data = {
            "models": SIGHTENGINE_VIDEO_MODELS,
            "callback_url": callback_url,
            "api_user": SIGHTENGINE_API_USER,
            "api_secret": SIGHTENGINE_API_KEY,
        }
        resp = requests.post(SIGHTENGINE_VIDEO_URL, data=data, files=files, timeout=120)

    if resp.status_code != 200:
        # Erreurs structurelles HTTP (hors JSON Sightengine).
        raise SightengineError(f"Soumission vidéo HTTP {resp.status_code}: {resp.text[:500]}")
    out = resp.json()
    if out.get("status") != "success":
        err = out.get("error") or {}
        if (err.get("type") == "usage_limit") and str(err.get("code")) == "3701":
            # Cas fréquent : plan gratuit sans droit vidéo.
            raise SightengineError(
                "Votre clé Sightengine n'a pas accès à l'analyse vidéo "
                "(feature réservée aux offres payantes). "
                "Désactivez l'envoi vidéo ou passez à un plan payant."
            )
        raise SightengineError(str(out))
    media = out.get("media") or {}
    mid = media.get("id")
    if not mid:
        raise SightengineError(f"Pas de media id: {out}")
    return str(mid)


def submit_video_stream_url(stream_url: str, callback_url: str) -> str:
    """Soumet une vidéo via URL directe (mp4, etc.). Retourne media.id."""
    _ensure_creds()
    data = {
        "stream_url": stream_url,
        "models": SIGHTENGINE_VIDEO_MODELS,
        "callback_url": callback_url,
        "api_user": SIGHTENGINE_API_USER,
        "api_secret": SIGHTENGINE_API_KEY,
    }
    resp = requests.post(SIGHTENGINE_VIDEO_URL, data=data, timeout=60)
    if resp.status_code != 200:
        raise SightengineError(f"Soumission vidéo URL HTTP {resp.status_code}: {resp.text[:500]}")
    out = resp.json()
    if out.get("status") != "success":
        err = out.get("error") or {}
        if (err.get("type") == "usage_limit") and str(err.get("code")) == "3701":
            raise SightengineError(
                "Votre clé Sightengine n'a pas accès à l'analyse vidéo "
                "(feature réservée aux offres payantes). "
                "Utilisez plutôt l'analyse d'images ou passez à un plan payant."
            )
        raise SightengineError(str(out))
    media = out.get("media") or {}
    mid = media.get("id")
    if not mid:
        raise SightengineError(f"Pas de media id: {out}")
    return str(mid)


def poll_video_job(media_id: str) -> Dict[str, Any]:
    """Une requête de statut / résultat partiel."""
    _ensure_creds()
    resp = requests.get(
        SIGHTENGINE_VIDEO_BYID_URL,
        params={
            "id": media_id,
            "api_user": SIGHTENGINE_API_USER,
            "api_secret": SIGHTENGINE_API_KEY,
        },
        timeout=40,
    )
    if resp.status_code != 200:
        raise SightengineError(f"Poll vidéo HTTP {resp.status_code}")
    return resp.json()


def wait_for_video_and_summarize_weapon(media_id: str) -> Tuple[Dict[str, Any], float, str]:
    """
    Attend la fin du job (polling) et extrait P(weapon) max sur les frames connues.
    Retourne (dernier_json_complet, prob_max, label).
    """
    max_sec = int(getattr(settings, "SIGHTENGINE_VIDEO_POLL_MAX_SEC", 120))
    interval = float(getattr(settings, "SIGHTENGINE_VIDEO_POLL_INTERVAL_SEC", 2.0))
    deadline = time.time() + max_sec
    last: Dict[str, Any] = {}
    best_prob = 0.0
    best_label = "weapon"

    while time.time() < deadline:
        last = poll_video_job(media_id)
        if last.get("status") != "success":
            raise SightengineError(str(last))

        out_block = last.get("output") or {}
        data = out_block.get("data") or {}
        st = (data.get("status") or "").lower()
        frames = data.get("frames") or []
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            w = fr.get("weapon")
            if isinstance(w, dict):
                p = float(w.get("prob", 0.0))
                if p > best_prob:
                    best_prob = p
                    best_label = str(w.get("type") or best_label)

        if st in ("finished", "failure", "stopped", "error"):
            break
        time.sleep(interval)

    return last, best_prob, best_label


def weapon_verdict_from_prob(prob: float) -> str:
    return "weapon" if prob >= 0.5 else "no_weapon"
