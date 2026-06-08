import logging
import os
import sys

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

logger = logging.getLogger("medinamind.startup")

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()
app = application

logger.info(
    "WSGI ready | Running on Vercel: %s | DEBUG: %s | DATABASE_URL configured: %s | OPENAI_API_KEY configured: %s",
    os.environ.get("VERCEL") == "1",
    os.environ.get("DEBUG", "?"),
    bool((os.environ.get("DATABASE_URL") or "").strip()),
    bool((os.environ.get("OPENAI_API_KEY") or "").strip()),
)
