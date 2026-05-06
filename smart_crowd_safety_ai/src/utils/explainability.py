"""Explications textuelles type XAI (règles + contributions)."""
from __future__ import annotations

from typing import Dict, List


def build_alert_explanation(
    fight_prob: float,
    panic_score: float,
    abandoned_score: float,
    risk: float,
    panic_ids: List[int],
    abandoned_reasons: List[str],
    fight_threshold: float = 0.55,
    panic_threshold: float = 0.35,
    abandoned_threshold: float = 0.5,
    risk_threshold: float = 0.45,
) -> Dict:
    """Construit un résumé lisible pour l'utilisateur."""
    triggers: List[str] = []
    if fight_prob >= fight_threshold:
        triggers.append(
            f"Scène de lutte probable (YOLO classify pré-entraîné): P(fight)={fight_prob:.2f} "
            f"(seuil {fight_threshold})."
        )
    if panic_score >= panic_threshold:
        ids_txt = ", ".join(str(i) for i in panic_ids[:8]) or "—"
        triggers.append(
            f"Mouvement brusque / course anormale: score={panic_score:.2f}, "
            f"tracks suspects: {ids_txt}."
        )
    if abandoned_score >= abandoned_threshold:
        for r in abandoned_reasons[:3]:
            triggers.append(r)

    if risk >= risk_threshold and not triggers:
        triggers.append(
            f"Score de risque élevé ({risk:.2f}) sans détail dominant — "
            "vérifier la combinaison des signaux."
        )

    return {
        "risk": risk,
        "signals": {
            "fight_prob": fight_prob,
            "panic_score": panic_score,
            "abandoned_score": abandoned_score,
        },
        "triggers": triggers,
        "alert": risk >= risk_threshold,
    }
