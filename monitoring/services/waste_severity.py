"""
Calcul gravité déchets + action recommandée (Smart City Waste).

Règles :
- Low : ≤1 objet OU confiance max < 50 %
- Medium : 2–4 objets OU confiance entre 50 % et 75 %
- High : ≥5 objets OU confiance > 75 %

Si le texte localisation contient une zone sensible (école, hôpital, marché…),
gravité augmentée d’un cran (Low→Medium, Medium→High).
"""
from __future__ import annotations

from django.conf import settings


def _conf_ratio(max_confidence: float) -> float:
    """Confiance en ratio 0–1."""
    if max_confidence <= 0:
        return 0.0
    return max_confidence if max_confidence <= 1.0 else max_confidence / 100.0


_DEFAULT_SENSITIVE = (
    "school",
    "école",
    "ecole",
    "college",
    "lycée",
    "hospital",
    "hôpital",
    "hopital",
    "clinic",
    "clinique",
    "market",
    "marché",
    "marche",
    "souk",
    "main road",
    "route principale",
    "avenue principale",
    "centre-ville",
    "downtown",
)


def location_contains_sensitive_area(location_text: str) -> bool:
    raw = (location_text or "").lower()
    keywords = getattr(settings, "WASTE_SENSITIVE_AREA_KEYWORDS", None)
    if not keywords:
        keywords = _DEFAULT_SENSITIVE
    return any(str(k).lower() in raw for k in keywords)


def compute_severity_and_action(
    *,
    count: int,
    max_confidence: float,
    location_hint: str = "",
) -> tuple[str, str, bool]:
    """
    Retourne (severity_label, recommended_action_fr, sensitive_boost).
    severity_label ∈ {'Low','Medium','High'}.
    """
    conf = _conf_ratio(max_confidence)

    if count >= 5 or conf > 0.75:
        base = "High"
    elif (2 <= count <= 4) or (0.5 <= conf <= 0.75):
        base = "Medium"
    else:
        base = "Low"

    sensitive = location_contains_sensitive_area(location_hint)
    order = ["Low", "Medium", "High"]
    idx = order.index(base)
    if sensitive and base != "High":
        idx = min(2, idx + 1)
    severity = order[idx]

    actions = {
        "Low": "Planifier un passage équipe propreté sous 7 jours ; surveillance citoyenne.",
        "Medium": "Envoyer une équipe de nettoyage sous 48–72 h ; signaler aux services voirie.",
        "High": "Intervention urgente équipe nettoyage sous 24 h ; inspection terrain recommandée.",
    }
    return severity, actions[severity], sensitive
