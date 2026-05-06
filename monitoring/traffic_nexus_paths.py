"""Validation des chemins modèles / médias pour Traffic Nexus (anti path traversal)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from django.conf import settings


def _base_roots() -> list[Path]:
    base = Path(settings.BASE_DIR).resolve()
    roots = [base]
    tx = base / "traffic_nexus" / "traffic_nexus"
    if tx.is_dir():
        roots.append(tx.resolve())
    media = base / "media"
    if media.is_dir():
        roots.append(media.resolve())
    models = base / "models"
    if models.is_dir():
        roots.append(models.resolve())
    mt = base / "models_traffic"
    if mt.is_dir():
        roots.append(mt.resolve())
    return roots


def is_under_allowed_root(path: Path) -> bool:
    rp = path.resolve()
    for root in _base_roots():
        try:
            rp.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def resolve_trusted_media_file(path_str: str) -> Optional[str]:
    """Fichier vidéo/image lisible par OpenCV, sous la racine projet."""
    if not path_str or not str(path_str).strip():
        return None
    p = Path(str(path_str).strip()).expanduser()
    if not p.is_absolute():
        p = (Path(settings.BASE_DIR) / p).resolve()
    else:
        p = p.resolve()
    if not is_under_allowed_root(p):
        return None
    if p.is_file():
        return str(p)
    return None


def resolve_trusted_model_weights(path_str: str) -> Optional[str]:
    """Fichier .pt / .onnx sous la racine projet."""
    if not path_str or not str(path_str).strip():
        return None
    p = Path(str(path_str).strip()).expanduser()
    if not p.is_absolute():
        p = (Path(settings.BASE_DIR) / p).resolve()
    else:
        p = p.resolve()
    if not is_under_allowed_root(p):
        return None
    if p.is_file() and p.suffix.lower() in {".pt", ".onnx"}:
        return str(p)
    return None
