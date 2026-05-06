"""Appel vocal Twilio (TwiML Say fr-FR)."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

DEFAULT_ALERT_MESSAGE = (
    "Attention. Un risque de feu ou de fumée a été détecté par le système de surveillance. "
    "Vérifiez la zone immédiatement."
)

_TWILIO_KEYS = [
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "TWILIO_FROM",
    "TWILIO_TO",
    "TWILIO_ALERT_MESSAGE",
]


def twilio_env_configured() -> bool:
    return all(os.environ.get(k) for k in _TWILIO_KEYS[:4])


def twilio_env_report() -> list[tuple[str, bool, str]]:
    def _mask(s: str, *, head: int = 4, tail: int = 4) -> str:
        s = (s or "").strip()
        if not s:
            return "-"
        if len(s) <= head + tail + 3:
            return "***"
        return f"{s[:head]}...{s[-tail:]}"

    rows: list[tuple[str, bool, str]] = []
    for key in _TWILIO_KEYS:
        raw = (os.environ.get(key) or "").strip()
        ok = bool(raw)
        if key == "TWILIO_AUTH_TOKEN":
            preview = _mask(raw, head=2, tail=4) if ok else "-"
        elif key in ("TWILIO_ACCOUNT_SID",):
            preview = _mask(raw, head=6, tail=4) if ok else "-"
        elif key in ("TWILIO_FROM", "TWILIO_TO"):
            preview = _mask(raw.replace(" ", ""), head=4, tail=4) if ok else "-"
        else:
            preview = (raw[:60] + "...") if len(raw) > 60 else raw if ok else "-"
        rows.append((key, ok, preview))
    return rows


def twilio_rest_ping() -> tuple[bool, str]:
    try:
        from twilio.rest import Client
    except ImportError:
        return False, "paquet 'twilio' non installé (pip install twilio)"

    if not twilio_env_configured():
        return False, "variables TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO requises"

    sid = os.environ["TWILIO_ACCOUNT_SID"].strip()
    token = os.environ["TWILIO_AUTH_TOKEN"].strip()

    try:
        client = Client(sid, token)
        account = client.api.accounts(sid).fetch()
        name = getattr(account, "friendly_name", None) or sid
        status = getattr(account, "status", "?")
        return True, f"REST OK — compte {name!r} (statut: {status})"
    except Exception as exc:
        return False, f"REST échec: {exc}"


def twiml_say_french(message: str) -> str:
    root = ET.Element("Response")
    say = ET.SubElement(root, "Say")
    say.set("language", "fr-FR")
    say.text = escape(message)
    return ET.tostring(root, encoding="unicode")


def place_twilio_alert_call() -> tuple[bool, str]:
    try:
        from twilio.rest import Client
    except ImportError:
        return False, "paquet 'twilio' non installé (pip install twilio)"

    if not twilio_env_configured():
        return (
            False,
            "variables TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO requises",
        )

    sid = os.environ["TWILIO_ACCOUNT_SID"]
    token = os.environ["TWILIO_AUTH_TOKEN"]
    from_num = os.environ["TWILIO_FROM"].replace(" ", "")
    to_num = os.environ["TWILIO_TO"].replace(" ", "")
    body = os.environ.get("TWILIO_ALERT_MESSAGE", DEFAULT_ALERT_MESSAGE)

    try:
        client = Client(sid, token)
        twiml = twiml_say_french(body)
        call = client.calls.create(twiml=twiml, from_=from_num, to=to_num)
        return True, f"appel lancé (sid {call.sid})"
    except Exception as exc:
        return False, str(exc)
