from __future__ import annotations

from pathlib import Path
from typing import Optional

from django.conf import settings as django_settings
from yt_dlp import YoutubeDL


def download_youtube_to_mp4(url: str, subdir: str = "sightengine_youtube") -> Path:
    """
    Télécharge une vidéo YouTube en MP4 dans MEDIA_ROOT / subdir et
    retourne le chemin du fichier.
    """
    media_root = Path(django_settings.MEDIA_ROOT)
    out_dir = media_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fichier de sortie : <subdir>/<videoid>.mp4 (yt-dlp remplira la partie id)
    out_tpl = str(out_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "outtmpl": out_tpl,
        "quiet": True,
        "noprogress": True,
    }

    last_path: Optional[Path] = None
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        final_path = ydl.prepare_filename(info)
        last_path = Path(final_path)

    if not last_path or not last_path.is_file():
        raise RuntimeError("Échec du téléchargement YouTube pour Sightengine.")

    return last_path

