"""Lightweight YouTube URL helpers (no OpenCV / ML imports)."""
from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

YOUTUBE_HOST_RE = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/(watch\?|embed/|shorts/)|youtu\.be/)",
    re.I,
)


def is_youtube_url(url: str) -> bool:
    u = (url or "").strip()
    return bool(u and YOUTUBE_HOST_RE.search(u))


def extract_youtube_video_id(url: str) -> str | None:
    u = (url or "").strip()
    if not u:
        return None
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    if not is_youtube_url(u):
        return None
    p = urlparse(u)
    host = (p.netloc or "").lower().split(":")[0]
    path = (p.path or "").strip("/")

    if "youtu.be" in host:
        seg = path.split("/")[0] if path else ""
        return seg[:11] if len(seg) >= 6 else None

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        if path == "watch" or path.startswith("watch"):
            q = parse_qs(p.query)
            vid = (q.get("v") or [None])[0]
            return str(vid)[:11] if vid else None
        if path.startswith("embed/"):
            seg = path.split("/")[1] if "/" in path else path.replace("embed/", "")
            return seg[:11] if seg else None
        if path.startswith("shorts/"):
            seg = path.split("/")[1] if len(path.split("/")) > 1 else ""
            return seg[:11] if seg else None
    return None
