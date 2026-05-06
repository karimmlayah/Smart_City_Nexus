"""
ESP32 hardware bridge: POST JSON with demo-friendly short cycle timings.

Emergency: set emergency=true and priority_zone; firmware should hold GREEN on that
zone until emergency=false (ignore green_s/red_s for cycling during emergency).

JSON shape (stable):
{
  "emergency": true/false,
  "priority_zone": null | 0 | 1,
  "zones": [
    {"index": 0, "green_s": X, "red_s": Y},
    {"index": 1, "green_s": X, "red_s": Y}
  ]
}
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

from .decision_engine import ZoneDecision

# Demo: full cycle split across zones (sum of greens <= this)
DEMO_CYCLE_SECONDS = 10
DEMO_MIN_GREEN = 2
DEMO_MAX_GREEN = 8
YELLOW_S = 1  # partner phase buffer for red_s

# Raw detection must hold this long before confirmed emergency on/off or zone change.
EMERGENCY_CONFIRM_HOLD_S = 1.0


def infer_emergency_zone(
    per_zone_counts: List[Union[Counter, Dict[str, int]]],
    mock_emergency: bool,
    mock_priority_zone: int,
) -> Tuple[bool, Optional[int]]:
    if mock_emergency:
        return True, max(0, mock_priority_zone)
    for idx, ctr in enumerate(per_zone_counts):
        for label, _n in ctr.items():
            low = str(label).lower()
            if "ambulance" in low or "emergency" in low:
                return True, idx
    return False, None


def _allocate_demo_greens(criticalities: List[int]) -> List[int]:
    """Split DEMO_CYCLE_SECONDS across zones by criticality (integer seconds)."""
    n = len(criticalities)
    if n == 0:
        return []
    if n == 1:
        g = max(DEMO_MIN_GREEN, min(DEMO_MAX_GREEN, DEMO_CYCLE_SECONDS // 2))
        return [g]
    if n == 2:
        c0, c1 = criticalities[0], criticalities[1]
        w0, w1 = max(1, int(c0)), max(1, int(c1))
        g0 = int(round(DEMO_CYCLE_SECONDS * w0 / (w0 + w1)))
        g0 = max(DEMO_MIN_GREEN, min(DEMO_MAX_GREEN, g0))
        g1 = DEMO_CYCLE_SECONDS - g0
        g1 = max(DEMO_MIN_GREEN, min(DEMO_MAX_GREEN, g1))
        if g0 + g1 != DEMO_CYCLE_SECONDS:
            g0 = DEMO_CYCLE_SECONDS - g1
            g0 = max(DEMO_MIN_GREEN, min(DEMO_MAX_GREEN, g0))
        return [g0, g1]

    ws = [max(1, int(c)) for c in criticalities]
    ssum = sum(ws)
    raw = [DEMO_CYCLE_SECONDS * w / ssum for w in ws]
    floors = [int(x) for x in raw]
    rem = DEMO_CYCLE_SECONDS - sum(floors)
    order = sorted(range(n), key=lambda i: raw[i] - floors[i], reverse=True)
    greens = floors[:]
    for r in range(rem):
        greens[order[r % n]] += 1
    for i in range(n):
        greens[i] = max(DEMO_MIN_GREEN, min(DEMO_MAX_GREEN, greens[i]))
    return greens


def _demo_red_when_other_green(greens: List[int]) -> List[int]:
    """red_s for zone i ≈ sum of other greens + yellow (opposing phase length)."""
    n = len(greens)
    reds: List[int] = []
    for i in range(n):
        other = sum(greens[j] for j in range(n) if j != i)
        reds.append(int(other + YELLOW_S))
    return reds


def build_esp32_payload(
    decisions: List[ZoneDecision],
    emergency: bool,
    priority_zone: Optional[int],
) -> Dict[str, Any]:
    """
    Build strict JSON for ESP32. Uses proportional short timings from criticality.
    When emergency: firmware should force GREEN on priority_zone until cleared.
    """
    if not decisions:
        return {
            "emergency": bool(emergency),
            "priority_zone": None,
            "zones": [],
        }

    n = len(decisions)
    crits = [d.traffic_criticality_score for d in decisions]

    if emergency and priority_zone is not None:
        pz = int(priority_zone)
        if pz < 0 or pz >= n:
            pz = 0
        zones_out: List[Dict[str, Any]] = []
        for i in range(n):
            if i == pz:
                zones_out.append({"index": i, "green_s": DEMO_MAX_GREEN, "red_s": 0})
            else:
                zones_out.append({"index": i, "green_s": 0, "red_s": DEMO_MAX_GREEN})
        return {
            "emergency": True,
            "priority_zone": pz,
            "zones": zones_out,
        }

    greens = _allocate_demo_greens(crits)
    reds = _demo_red_when_other_green(greens)
    zones_out = [
        {"index": i, "green_s": int(greens[i]), "red_s": int(reds[i])}
        for i in range(n)
    ]
    return {
        "emergency": False,
        "priority_zone": None,
        "zones": zones_out,
    }


def new_emergency_confirmation_state() -> Dict[str, Any]:
    """Mutable state for step_emergency_confirmation (store in st.session_state)."""
    return {
        "active": False,
        "zone": None,
        "rise_since": None,
        "rise_zone": None,
        "fall_since": None,
        "pz_cand": None,
        "pz_since": None,
    }


def step_emergency_confirmation(
    st: Dict[str, Any],
    raw_emergency: bool,
    raw_zone: Optional[int],
    now: float,
    hold_s: float = EMERGENCY_CONFIRM_HOLD_S,
) -> Tuple[bool, Optional[int], List[str]]:
    """
    Debounce emergency/priority zone from single-frame detections.
    Requires continuous raw detection for hold_s before confirming ON or OFF.
    While active, priority_zone changes require hold_s at the new zone.
    """
    logs: List[str] = []
    rz = int(raw_zone) if raw_zone is not None else None

    if not st["active"]:
        st["fall_since"] = None
        st["pz_cand"] = None
        st["pz_since"] = None
        if raw_emergency:
            if rz is None:
                rz = 0
            if st["rise_since"] is None:
                st["rise_since"] = now
                st["rise_zone"] = rz
                logs.append("Emergency candidate detected")
            elif rz != st["rise_zone"]:
                st["rise_since"] = now
                st["rise_zone"] = rz
            elif now - float(st["rise_since"]) >= hold_s:
                st["active"] = True
                st["zone"] = rz
                st["rise_since"] = None
                st["rise_zone"] = None
                logs.append("Emergency confirmed")
        else:
            st["rise_since"] = None
            st["rise_zone"] = None
    else:
        st["rise_since"] = None
        st["rise_zone"] = None
        if raw_emergency:
            st["fall_since"] = None
            if rz is None:
                rz = st["zone"]
            if rz is not None and st["zone"] is not None and rz != int(st["zone"]):
                if st["pz_cand"] != rz:
                    st["pz_cand"] = rz
                    st["pz_since"] = now
                elif st["pz_since"] is not None and now - float(st["pz_since"]) >= hold_s:
                    st["zone"] = rz
                    st["pz_cand"] = None
                    st["pz_since"] = None
            elif rz is not None and st["zone"] is not None and rz == int(st["zone"]):
                st["pz_cand"] = None
                st["pz_since"] = None
        else:
            st["pz_cand"] = None
            st["pz_since"] = None
            if st["fall_since"] is None:
                st["fall_since"] = now
                logs.append("Emergency false candidate")
            elif now - float(st["fall_since"]) >= hold_s:
                st["active"] = False
                st["zone"] = None
                st["fall_since"] = None
                logs.append("Emergency cleared confirmed")

    z = int(st["zone"]) if st["zone"] is not None else None
    return bool(st["active"]), z, logs


def esp32_decision_signature(payload: Dict[str, Any]) -> str:
    """Stable string for comparing confirmed POST payloads (emergency, zones timings)."""
    zones = payload.get("zones") or []
    slim = {
        "emergency": bool(payload.get("emergency")),
        "priority_zone": payload.get("priority_zone"),
        "zones": [
            {"index": int(z.get("index", i)), "green_s": int(z.get("green_s", 0)), "red_s": int(z.get("red_s", 0))}
            for i, z in enumerate(zones)
        ],
    }
    return json.dumps(slim, sort_keys=True, separators=(",", ":"))


def format_esp32_log_line(payload: Dict[str, Any]) -> str:
    """Single-line summary for Streamlit."""
    em = payload.get("emergency")
    pz = payload.get("priority_zone")
    zones = payload.get("zones") or []
    parts = [f"emergency={em}", f"priority_zone={pz}"]
    for z in zones:
        parts.append(f"Z{z.get('index', '?')}: G{z.get('green_s')}/R{z.get('red_s')}")
    return " | ".join(parts)


def post_json(url: str, payload: Dict[str, Any], timeout: float = 2.0) -> Tuple[bool, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")[:500]
            return True, body or "ok"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except urllib.error.URLError as e:
        return False, str(e.reason if hasattr(e, "reason") else e)
    except Exception as e:
        return False, str(e)
