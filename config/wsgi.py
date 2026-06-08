import os
import sys
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
_MANIFEST = _BASE / "staticfiles" / "staticfiles.json"

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

if os.environ.get("VERCEL") == "1":
    os.environ.setdefault("DEBUG", "False")
    _static_src = _BASE / "static"
    if not _MANIFEST.is_file() and _static_src.is_dir():
        print("[wsgi] staticfiles.json missing — running collectstatic", file=sys.stderr)
        import subprocess

        result = subprocess.run(
            [sys.executable, str(_BASE / "manage.py"), "collectstatic", "--noinput"],
            cwd=str(_BASE),
            capture_output=True,
            text=True,
        )
        print(f"[wsgi] collectstatic exit={result.returncode}", file=sys.stderr)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "")[-2000:]
            if tail:
                print(tail, file=sys.stderr)

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
app = application
