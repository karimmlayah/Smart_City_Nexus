"""Geocoding helpers (OpenStreetMap Nominatim by default)."""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from django.conf import settings

logger = logging.getLogger(__name__)

DEFAULT_NOMINATIM = "https://nominatim.openstreetmap.org"


def _user_agent() -> str:
    return (getattr(settings, "GEOCODING_USER_AGENT", "") or "MedinaMind/1.0").strip()


def _nominatim_base() -> str:
    return (getattr(settings, "GEOCODING_NOMINATIM_URL", "") or DEFAULT_NOMINATIM).rstrip("/")


def _fetch_json(url: str) -> list | dict | None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": _user_agent(), "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        logger.warning("Geocoding request failed: %s", exc)
        return None


def geocode_address(query: str) -> dict | None:
    """Forward geocode an address string → {address, latitude, longitude}."""
    q = (query or "").strip()
    if not q:
        return None
    params = urllib.parse.urlencode({"q": q, "format": "json", "limit": 1})
    data = _fetch_json(f"{_nominatim_base()}/search?{params}")
    if not isinstance(data, list) or not data:
        return None
    hit = data[0]
    try:
        lat = float(hit["lat"])
        lng = float(hit["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    label = (hit.get("display_name") or q).strip()
    return {"address": label, "latitude": lat, "longitude": lng}


def reverse_geocode(latitude: float, longitude: float) -> dict | None:
    """Reverse geocode coordinates → {address, latitude, longitude}."""
    try:
        lat = float(latitude)
        lng = float(longitude)
    except (TypeError, ValueError):
        return None
    params = urllib.parse.urlencode({"lat": lat, "lon": lng, "format": "json", "limit": 1})
    data = _fetch_json(f"{_nominatim_base()}/reverse?{params}")
    if not isinstance(data, dict):
        return None
    label = (data.get("display_name") or "").strip()
    if not label:
        return None
    return {"address": label, "latitude": lat, "longitude": lng}
