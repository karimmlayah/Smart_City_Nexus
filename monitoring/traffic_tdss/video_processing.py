from typing import Optional

import cv2


class _SilentYtdlpLogger:
    def debug(self, _msg: str) -> None:
        return

    def warning(self, _msg: str) -> None:
        return

    def error(self, _msg: str) -> None:
        return


def resolve_video_source(input_source: str) -> str:
    source = input_source.strip()
    if "youtube.com" not in source and "youtu.be" not in source:
        return source

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError(
            "YouTube sources require yt-dlp. Install with: pip install yt-dlp"
        ) from exc

    with YoutubeDL(
        {
            "quiet": True,
            "no_warnings": True,
            "logger": _SilentYtdlpLogger(),
            "format": "best[ext=mp4]/best",
            "noplaylist": True,
            "extractor_args": {
                "youtube": {
                    # Prefer clients that usually avoid heavy JS code paths.
                    "player_client": ["android", "web"],
                }
            },
        }
    ) as ydl:
        info = ydl.extract_info(source, download=False)
        return info["url"]


def read_first_frame(source: str) -> Optional[cv2.typing.MatLike]:
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame

