"""Risk score, operational level, indicative CO₂ and contextual recommendations (UAV)."""
from __future__ import annotations

from typing import Any


def zone_criticality(zone: str) -> float:
    z = (zone or "B").strip().upper()
    if z == "A":
        return 1.25
    if z == "C":
        return 0.88
    return 1.0


def operational_level_from_score(score: float) -> tuple[str, str]:
    """Return (display label, CSS key)."""
    if score >= 8.0:
        return "Critical", "critical"
    if score >= 6.0:
        return "Urgent", "urgent"
    if score >= 4.0:
        return "Moderate", "moderate"
    return "Low", "low"


def compute_risk_score(
    pred_idx: int,
    confidence: float,
    probabilities: list[float],
    zone: str,
) -> float:
    """Score out of 10 — weights class + confidence + zone criticality."""
    base = {0: 2.2, 1: 5.5, 2: 8.4}.get(pred_idx, 5.0)
    margin = 0.0
    if probabilities and len(probabilities) >= 3:
        sorted_p = sorted(probabilities, reverse=True)
        margin = max(0.0, sorted_p[0] - sorted_p[1])
    conf_boost = (confidence - 0.34) * 2.8
    raw = base + conf_boost * 0.35 + margin * 1.6
    raw *= zone_criticality(zone)
    return float(max(0.0, min(10.0, raw)))


def estimate_co2_kg(pred_idx: int, surface_m2: float, confidence: float) -> float:
    """Indicative order of magnitude (repair / reconstruction)."""
    s = max(0.0, float(surface_m2 or 0.0))
    factors = {0: 8.0, 1: 42.0, 2: 95.0}
    base = factors.get(pred_idx, 35.0)
    return round(base * max(50.0, s) * (0.65 + 0.35 * confidence) / 1000.0, 2)


def build_recommendations(
    pred_idx: int,
    confidence: float,
    zone: str,
    risk_score: float,
    operational: str,
) -> dict[str, Any]:
    """Action plan P1–P4 + AI limitations."""
    z = (zone or "B").strip().upper()

    if risk_score >= 8 or pred_idx == 2:
        prio = "P1"
    elif risk_score >= 6 or pred_idx == 1:
        prio = "P2"
    elif risk_score >= 4:
        prio = "P3"
    else:
        prio = "P4"

    immediate: list[str] = []
    short_term: list[str] = []
    watch: list[str] = []
    ai_limits: list[str] = [
        "Interpretation is based on a single UAV image; on-site inspection remains mandatory.",
        "Attention and saliency maps reflect what the AI observes in the image; they do not replace geotechnical measurement.",
        "CO₂ estimates are order-of-magnitude values for preliminary decision support.",
    ]

    if pred_idx == 2:
        immediate = [
            "Mark a safety perimeter and suspend all pedestrian / vehicle access.",
            "Notify emergency services and infrastructure management.",
            "Document the scene (supplementary low-altitude flights if safe).",
        ]
        short_term = [
            "Structural inspection by a qualified engineer within 24–48 hours.",
            "Insurance damage assessment and provisional stabilization.",
        ]
        watch = ["Monitor vibrations / movement until independent review."]
    elif pred_idx == 1:
        immediate = [
            "Restrict access to areas under suspect structural spans.",
            "Ground marking and communication to residents / operators.",
        ]
        short_term = [
            "Detailed inspection (cracking, supports, roofing) by structure.",
            "Schedule a new UAV acquisition after intervention.",
        ]
        watch = ["Compare with a historical image if available.", "Follow up after severe weather."]
    else:
        immediate = ["Maintain watch; no emergency structural measure required by AI alone."]
        short_term = ["Log the mission in the registry and plan the next scheduled overflight."]
        watch = ["Reassess if the zone is sensitive (zone " + z + ")."]

    if z == "A" and confidence >= 0.55:
        immediate.insert(0, "Zone A — strengthen coordination with urban project management.")

    if confidence < 0.5:
        ai_limits.insert(
            0,
            "Moderate confidence: prioritize human review before any closure decision.",
        )

    return {
        "priority": prio,
        "operational_level": operational,
        "immediate": immediate,
        "short_term": short_term,
        "watch": watch,
        "ai_limits": ai_limits,
    }
