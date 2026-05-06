"""Réseau ESP32-CAM (HTTP) — IP lue depuis la base Django."""

from __future__ import annotations

import os
import socket

import requests

from monitoring.services.fire_cam.store import get_fire_config


def read_saved_ip() -> str:
    return (get_fire_config().esp_ip or "").strip()


def get_default_ip() -> str:
    """IP sauvegardée en base en priorité, puis DRONE_HOST (.env), puis valeur par défaut."""
    ip = read_saved_ip()
    if ip:
        return ip
    env = (os.environ.get("DRONE_HOST") or "").strip()
    if env:
        return env
    return "192.168.0.148"


def resolve_esp_host(cfg) -> str:
    """
    IP utilisée pour flux + /capture : modèle FireCameraConfig d’abord,
    puis variable d’environnement DRONE_HOST si la base est vide.
    """
    for raw in ((getattr(cfg, "esp_ip", None) or "").strip(), (os.environ.get("DRONE_HOST") or "").strip()):
        h = clean_host(raw)
        if h:
            return h
    return ""


def clean_host(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    s = s.replace("https://", "").replace("http://", "")
    s = s.split("/")[0]
    if ":" in s:
        host, _, port = s.partition(":")
        if port in ("80", "81"):
            return host.strip()
    return s.strip()


def save_ip(raw: str) -> None:
    h = clean_host(raw)
    if not h:
        return
    cfg = get_fire_config()
    cfg.esp_ip = h
    cfg.save(update_fields=["esp_ip", "updated_at"])


def base_url(host: str) -> str | None:
    h = clean_host(host)
    if not h:
        return None
    return f"http://{h}"


def stream_url_exact(host: str) -> str | None:
    """Premier URL flux MJPEG (compat UI / liens) — voir ``stream_url_candidates`` pour les repli."""
    h = clean_host(host)
    if not h:
        return None
    return f"http://{h}:81/stream"


def stream_url_candidates(host: str) -> list[str]:
    """
    Essais dans l’ordre : beaucoup de firmwares Arduino utilisent :81/stream ;
    d’autres n’exposent le MJPEG que sur le port 80 à ``/stream``.
    """
    h = clean_host(host)
    if not h:
        return []
    return [f"http://{h}:81/stream", f"http://{h}/stream"]


def tcp_reachable(host: str, port: int, timeout: float = 2.5) -> bool:
    h = clean_host(host)
    if not h:
        return False
    try:
        with socket.create_connection((h, port), timeout=timeout):
            return True
    except OSError:
        return False


def connection_help_markdown(host: str, err: Exception | None = None) -> str:
    h = clean_host(host) or "?"
    ok80 = tcp_reachable(host, 80)
    ok81 = tcp_reachable(host, 81)
    lines = [
        "### Réseau",
        f"Port **80** : **{'joignable' if ok80 else 'injoignable'}** · Port **81** : **{'joignable' if ok81 else 'injoignable'}**",
        "Vérifie l’IP (Serial Monitor / box), le **même Wi‑Fi** que le PC, pas le Wi‑Fi invité.",
    ]
    if err:
        lines.append(f"*Détail :* `{type(err).__name__}: {err}`")
    return "\n".join(lines)


def test_port_80(host: str) -> tuple[bool, str]:
    bu = base_url(host)
    if not bu:
        return False, "Adresse IP vide."
    try:
        r = requests.get(f"{bu}/", timeout=(5, 12))
        return True, f"Port 80 OK — HTTP {r.status_code} sur {bu}/"
    except requests.RequestException as e:
        if "timed out" in str(e).lower() or "ConnectTimeout" in type(e).__name__:
            return False, f"**Port 80 inaccessible** (`{bu}/`)\n\n{connection_help_markdown(host, e)}"
        return False, f"Port 80 inaccessible : `{e}`"


def test_port_81_stream(host: str) -> tuple[bool, str]:
    su = stream_url_exact(host)
    if not su:
        return False, "Adresse IP vide."
    try:
        with requests.get(su, stream=True, timeout=(8, 25)) as r:
            r.raise_for_status()
            chunk = next(r.iter_content(chunk_size=2048))
        ct = (r.headers.get("Content-Type") or "").lower()
        if "multipart" in ct or b"jpeg" in chunk.lower() or b"JFIF" in chunk:
            return True, f"Port 81 OK — MJPEG sur {su}"
        return True, f"Port 81 répond (HTTP {r.status_code}) : {r.headers.get('Content-Type')}"
    except requests.RequestException as e:
        return False, f"**Port 81 inaccessible** (`{su}`)\n\n{connection_help_markdown(host, e)}"
