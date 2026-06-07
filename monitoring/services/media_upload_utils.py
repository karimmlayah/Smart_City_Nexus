"""Upload helpers — HEIC/HEIF → JPEG conversion for citizen reports."""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

from django.core.files.base import ContentFile

logger = logging.getLogger(__name__)

_HEIF_REGISTERED = False


def register_heif_opener() -> None:
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        from pillow_heif import register_heif_opener as _register

        _register()
        _HEIF_REGISTERED = True
    except ImportError:
        logger.warning("pillow-heif not installed — HEIC server-side conversion unavailable.")


def _heic_extension(name: str, content_type: str = "") -> bool:
    ext = Path(name or "").suffix.lower().lstrip(".")
    if ext in {"heic", "heif"}:
        return True
    ct = (content_type or "").lower()
    return "heic" in ct or "heif" in ct


def convert_uploaded_heic_to_jpeg(uploaded_file) -> ContentFile:
    """Convert an uploaded HEIC/HEIF file to JPEG bytes."""
    register_heif_opener()
    from PIL import Image

    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    original = getattr(uploaded_file, "name", "") or "upload.heic"
    stem = Path(original).stem or "upload"
    return ContentFile(buf.read(), name=f"{stem}.jpg")


def maybe_convert_heic_upload(uploaded_file):
    """Return JPEG ContentFile if HEIC/HEIF, otherwise the original upload."""
    name = getattr(uploaded_file, "name", "") or ""
    content_type = getattr(uploaded_file, "content_type", "") or ""
    if not _heic_extension(name, content_type):
        return uploaded_file
    try:
        return convert_uploaded_heic_to_jpeg(uploaded_file)
    except Exception as exc:
        logger.exception("HEIC conversion failed for %s: %s", name, exc)
        raise ValueError(
            "Could not convert HEIC/HEIF image. Please upload JPG or PNG."
        ) from exc
