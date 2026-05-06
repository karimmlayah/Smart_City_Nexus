"""
Envoi SMS municipalité (Twilio) ou simulation si clés absentes.

Configuration settings.py / .env :
  TWILIO_ACCOUNT_SID — https://console.twilio.com
  TWILIO_AUTH_TOKEN
  TWILIO_PHONE_NUMBER — émetteur Twilio (E.164) ; sinon TWILIO_FROM_NUMBER
  MUNICIPALITY_PHONE_NUMBER — destinataire mairie (E.164) ; sinon WASTE_MUNICIPALITY_PHONE
"""
from __future__ import annotations

import logging
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)


def send_waste_alert_sms(
    *,
    location: str,
    count: int,
    confidence_pct: float,
    severity: str = "",
    recommended_action: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """
    Retourne {success, mode: 'twilio'|'mock', message, sid?, payload_preview?}.
    """
    sev = (severity or "Unknown").strip()
    action = (recommended_action or "Assessment pending.").strip()
    body = (
        f"Smart City Alert: Waste detected at {location}. "
        f"Severity: {sev}. Objects detected: {count}. "
        f"Confidence: {confidence_pct:.1f}%. "
        f"Recommended action: {action[:240]} "
        f"Please send cleaning team."
    )
    if notes.strip():
        body = body + f" Notes: {notes.strip()[:120]}"

    account_sid = (getattr(settings, "TWILIO_ACCOUNT_SID", "") or "").strip()
    auth_token = (getattr(settings, "TWILIO_AUTH_TOKEN", "") or "").strip()
    from_num = (
        getattr(settings, "TWILIO_PHONE_NUMBER", "")
        or getattr(settings, "TWILIO_FROM_NUMBER", "")
        or ""
    ).strip()
    to_num = (
        getattr(settings, "MUNICIPALITY_PHONE_NUMBER", "")
        or getattr(settings, "WASTE_MUNICIPALITY_PHONE", "")
        or ""
    ).strip()

    if not all([account_sid, auth_token, from_num, to_num]):
        logger.info(
            "[waste_sms MOCK] To=%s Body=%s",
            to_num or "(non configuré)",
            body[:220],
        )
        return {
            "success": True,
            "mode": "mock",
            "message": "SMS simulated (Twilio / phone numbers not fully configured).",
            "sid": None,
            "payload_preview": body,
        }

    try:
        from twilio.rest import Client

        client = Client(account_sid, auth_token)
        msg = client.messages.create(
            body=body[:1600],
            from_=from_num,
            to=to_num,
        )
        return {
            "success": True,
            "mode": "twilio",
            "message": "SMS sent successfully",
            "sid": getattr(msg, "sid", None),
            "payload_preview": body,
        }
    except ImportError:
        logger.warning("twilio package not installed — pip install twilio")
        return {
            "success": True,
            "mode": "mock",
            "message": "SMS simulated (twilio package not installed).",
            "sid": None,
            "payload_preview": body,
        }
    except Exception as exc:
        logger.exception("Twilio SMS failed")
        err_txt = str(exc or "").strip()
        err_low = err_txt.lower()
        if "20003" in err_txt or "authenticate" in err_low:
            return {
                "success": False,
                "mode": "twilio",
                "message": (
                    "Twilio authentication failed (error 20003). "
                    "Check TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN in model_road_damage/.env."
                ),
                "sid": None,
            }
        return {
            "success": False,
            "mode": "twilio",
            "message": err_txt or "Unknown Twilio error",
            "sid": None,
        }
