"""
Envoi email alertes déchets — SMTP Django ou mode mock.

Configurer dans settings.py / .env (voir django.core.mail) :
  EMAIL_HOST, EMAIL_PORT, EMAIL_HOST_USER, EMAIL_HOST_PASSWORD
  EMAIL_USE_TLS / EMAIL_USE_SSL
  DEFAULT_FROM_EMAIL

Si EMAIL_HOST est vide → simulation (log console).
"""
from __future__ import annotations

import logging
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)


def send_waste_alert_email(
    *,
    to_email: str,
    subject: str,
    body: str,
) -> dict[str, Any]:
    to_email = (to_email or "").strip()
    if not to_email:
        return {"success": False, "mode": "skipped", "message": "Missing recipient email"}

    host = (getattr(settings, "EMAIL_HOST", "") or "").strip()
    if not host:
        logger.info("[waste_email MOCK] To=%s Subject=%s … %s", to_email, subject, body[:200])
        return {
            "success": True,
            "mode": "mock",
            "message": "Email simulated (configure EMAIL_HOST / SMTP in settings).",
        }

    try:
        from django.core.mail import send_mail

        send_mail(
            subject=subject[:200],
            message=body[:8000],
            from_email=getattr(settings, "DEFAULT_FROM_EMAIL", None) or "noreply@smartcity.local",
            recipient_list=[to_email],
            fail_silently=False,
        )
        return {"success": True, "mode": "smtp", "message": "Email sent successfully"}
    except Exception as exc:
        logger.exception("waste email failed")
        err = str(exc or "").strip()
        if "WinError 10061" in err or "Connection refused" in err:
            err = (
                "SMTP connection refused. Verify EMAIL_HOST / EMAIL_PORT and provider access "
                "(for Yahoo use smtp.mail.yahoo.com:587 with TLS + app password)."
            )
        return {"success": False, "mode": "smtp", "message": err}
