"""Score de risque, niveau opérationnel, CO₂ indicatif et recommandations contextuelles (UAV)."""
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
    """Retourne (libellé FR, clé CSS)."""
    if score >= 8.0:
        return "Critique", "critical"
    if score >= 6.0:
        return "Urgent", "urgent"
    if score >= 4.0:
        return "Modéré", "moderate"
    return "Faible", "low"


def compute_risk_score(
    pred_idx: int,
    confidence: float,
    probabilities: list[float],
    zone: str,
) -> float:
    """Score sur 10 — pondère classe + confiance + criticité zone."""
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
    """Ordre de grandeur indicatif (réparation / reconstruction)."""
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
    """Plan d’action P1–P4 + limites IA."""
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
        "L’interprétation repose sur une image UAV unique ; une inspection terrain reste obligatoire.",
        "Les cartes Grad-CAM / salience sont des indications d’attention du réseau, pas une mesure géotechnique.",
        "Les estimations CO₂ sont des ordres de grandeur à usage décisionnel préliminaire.",
    ]

    if pred_idx == 2:
        immediate = [
            "Baliser un périmètre de sécurité et suspendre tout accès piéton / véhicule.",
            "Notifier les services de secours et la direction des infrastructures.",
            "Documenter la scène (vols complémentaires à basse altitude si sécurité OK).",
        ]
        short_term = [
            "Inspection structurale par ingénieur habilité sous 24–48 h.",
            "Évaluation dommages assurances et consolidation provisoire.",
        ]
        watch = ["Surveillance des vibrations / mouvements jusqu’à contre-expertise."]
    elif pred_idx == 1:
        immediate = [
            "Limiter l’accès aux zones sous travées suspectes.",
            "Marquage au sol et communication aux résidents / exploitants.",
        ]
        short_term = [
            "Inspection détaillée (fissuration, appuis, toiture) par structure.",
            "Planifier une nouvelle acquisition UAV après intervention.",
        ]
        watch = ["Comparer avec une image historique si disponible.", "Suivi après intempéries."]
    else:
        immediate = ["Maintenir la veille ; aucune mesure d’urgence structurelle imposée par l’IA seule."]
        short_term = ["Enregistrer la mission dans le registre et planifier le prochain survol programmé."]
        watch = ["Réévaluer si la zone est sensible (zone " + z + ")."]

    if z == "A" and confidence >= 0.55:
        immediate.insert(0, "Zone A — renforcer la coordination avec la maîtrise d’ouvrage urbaine.")

    if confidence < 0.5:
        ai_limits.insert(
            0,
            "Confiance modérée : privilégier une analyse humaine avant toute décision de fermeture.",
        )

    return {
        "priority": prio,
        "operational_level": operational,
        "immediate": immediate,
        "short_term": short_term,
        "watch": watch,
        "ai_limits": ai_limits,
    }
