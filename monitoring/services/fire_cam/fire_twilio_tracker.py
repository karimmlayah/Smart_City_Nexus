"""Suivi feu stable + appel Twilio (état dict, ex. cache Django)."""

from __future__ import annotations

import os
import re
import time
from typing import Any, MutableMapping

import cv2
import numpy as np

from monitoring.services.fire_cam.twilio_voice_alert import place_twilio_alert_call, twilio_env_configured

_FIRE_NAME_KEYWORDS = ("fire", "feu", "fumee", "fumée", "smoke", "flame")
_STATE_PREFIX = "_fta_"
_GENERIC_CL = re.compile(r"^cl\d+$", re.IGNORECASE)


def _names_are_default_cl_labels(class_names: list[str]) -> bool:
    if not class_names:
        return False
    return all(_GENERIC_CL.match((n or "").strip()) for n in class_names)


def is_fire_detection(cls_id: int, class_names: list[str]) -> bool:
    raw = os.environ.get("TWILIO_FIRE_CLASS_IDS", "").strip()
    if raw:
        ids: set[int] = set()
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                ids.add(int(part))
        return cls_id in ids
    if cls_id < 0 or cls_id >= len(class_names):
        return False
    name = class_names[cls_id].lower()
    if any(k in name for k in _FIRE_NAME_KEYWORDS):
        return True
    if _names_are_default_cl_labels(class_names):
        nc = len(class_names)
        if nc == 1:
            return cls_id == 0
        if nc == 2:
            return cls_id in (0, 1)
        return cls_id == 0
    return False


def draw_fire_alert_overlay(frame_bgr: np.ndarray) -> None:
    h, w = frame_bgr.shape[:2]
    red_layer = np.zeros_like(frame_bgr)
    red_layer[:] = (20, 20, 220)
    cv2.addWeighted(frame_bgr, 0.52, red_layer, 0.48, 0, frame_bgr)
    thick = max(10, min(w, h) // 64)
    cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), (0, 0, 255), thickness=thick)
    band = frame_bgr.copy()
    cv2.rectangle(band, (0, max(0, h // 2 - 90)), (w, min(h, h // 2 + 110)), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.45, frame_bgr, 0.55, 0, frame_bgr)
    cv2.putText(
        frame_bgr,
        "ALERTE FEU / FUMEE",
        (40, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


def reset_fire_tracker_state(ss: MutableMapping[str, Any]) -> None:
    for k in list(ss.keys()):
        if k.startswith(_STATE_PREFIX) or k in ("_fta_err_toast", "_twilio_dlg_open"):
            del ss[k]


def step_fire_twilio(
    ss: MutableMapping[str, Any],
    dets: list[tuple[int, float, tuple[int, int, int, int]]],
    class_names: list[str],
    *,
    twilio_enabled: bool,
    alert_after_sec: float = 2.0,
    grace_sec: float = 0.8,
) -> dict[str, Any]:
    now = time.monotonic()
    p = _STATE_PREFIX
    has_fire = any(is_fire_detection(d[0], class_names) for d in dets)

    t0 = ss.get(f"{p}t0")
    last = ss.get(f"{p}last")
    sent = bool(ss.get(f"{p}sent", False))

    if has_fire:
        ss[f"{p}last"] = now
        if t0 is None:
            ss[f"{p}t0"] = now
            t0 = now
    else:
        if last is not None and (now - last) > grace_sec:
            ss[f"{p}t0"] = None
            ss[f"{p}last"] = None
            ss[f"{p}sent"] = False
            ss.pop("_fta_err_toast", None)
            ss.pop(f"{p}twilio_outcome", None)
            ss.pop(f"{p}twilio_outcome_detail", None)
            ss.pop(f"{p}modal_done", None)
            t0 = None
            sent = False

    risk_seconds = 0.0
    if t0 is not None and has_fire:
        risk_seconds = now - t0

    status = "OK"
    if has_fire:
        status = f"Détection feu… {risk_seconds:.1f}s / {alert_after_sec:.1f}s"

    show_alert = risk_seconds >= alert_after_sec and has_fire
    if show_alert:
        status = "ALERTE"

    twilio_called = False
    twilio_detail = ""
    twilio_error = ""
    twilio_popup = False
    twilio_popup_l1 = ""
    twilio_popup_l2 = ""
    twilio_modal_kind = ""

    if show_alert and twilio_enabled and not sent:
        ss[f"{p}sent"] = True
        first_modal = not ss.get(f"{p}modal_done")
        if first_modal:
            ss[f"{p}modal_done"] = True
            twilio_popup = True

        if not twilio_env_configured():
            ss[f"{p}twilio_outcome"] = "no_env"
            ss[f"{p}twilio_outcome_detail"] = "Variables TWILIO_* manquantes dans .env"
            twilio_error = ""
            if first_modal:
                twilio_popup_l1 = "Twilio non configuré"
                twilio_popup_l2 = (
                    "Ajoutez dans le fichier .env à la racine du projet : TWILIO_ACCOUNT_SID, "
                    "TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO (et optionnellement TWILIO_ALERT_MESSAGE). "
                    "Enregistrez le fichier puis redémarrez le serveur Django."
                )
                twilio_modal_kind = "warning"
        else:
            ok, detail = place_twilio_alert_call()
            twilio_called = ok
            twilio_detail = detail
            if not ok:
                twilio_error = detail or "Erreur Twilio"
            ss[f"{p}twilio_outcome"] = "success" if ok else "fail"
            ss[f"{p}twilio_outcome_detail"] = detail if ok else (twilio_error or detail)
            if first_modal:
                if ok:
                    twilio_popup_l1 = "Appel Twilio lancé"
                    twilio_popup_l2 = (detail or "La passerelle Twilio a accepté la demande d’appel.").strip()[:450]
                    twilio_modal_kind = "success"
                else:
                    twilio_popup_l1 = "Échec de l’appel Twilio"
                    twilio_popup_l2 = (twilio_error or detail or "Erreur inconnue.")[:450]
                    twilio_modal_kind = "danger"
    elif show_alert and not twilio_enabled:
        status = "ALERTE (Twilio désactivé)"
        ss[f"{p}twilio_outcome"] = "twilio_off"
        ss[f"{p}twilio_outcome_detail"] = "Appels Twilio désactivés"

    if show_alert:
        last_beep = float(ss.get(f"{p}beep", 0.0))
        if now - last_beep > 2.0:
            try:
                import winsound

                winsound.MessageBeep(winsound.MB_ICONWARNING)
            except Exception:
                pass
            ss[f"{p}beep"] = now

    return {
        "status": status,
        "show_alert": show_alert,
        "has_fire": has_fire,
        "risk_seconds": risk_seconds,
        "twilio_called": twilio_called,
        "twilio_detail": twilio_detail,
        "twilio_error": twilio_error,
        "twilio_outcome": ss.get(f"{p}twilio_outcome"),
        "twilio_outcome_detail": ss.get(f"{p}twilio_outcome_detail") or "",
        "twilio_popup": twilio_popup,
        "twilio_popup_l1": twilio_popup_l1,
        "twilio_popup_l2": twilio_popup_l2,
        "twilio_modal_kind": twilio_modal_kind,
    }
