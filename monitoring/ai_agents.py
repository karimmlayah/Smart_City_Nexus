"""
Agents IA « multi-agents » (règles déterministes, hors ligne) pour CitizenReclamation.
Scores calculés dynamiquement — pas de migration.

Vision YOLO (quand fournie) prime pour les signalements « pothole » : faux positifs catégorie réduits.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from django.conf import settings as django_settings

# Rayon « proche » pour LocationAgent (~1,2 km à latitude Gabès)
NEARBY_DEG_LAT = 0.011
NEARBY_DEG_LON = 0.011

# Pondération agrégée — le signal modèle domine (66 % confiance de base)
AI_W_SEVERITY = 0.17
AI_W_LOCATION = 0.17
AI_W_CONFIDENCE = 0.66

# Bonus score quand YOLO voit beaucoup de dégâts chaussée × confiance (plafonné)
MODEL_VOLUME_BOOST_CAP = 24.0
MODEL_VOLUME_BOOST_PER_UNIT = 1.12  # ~ n × conf_max × ce facteur, plafonné

# Fragments de libellés YOLO considérés comme dégât / nid de poule (insensible à la casse)
POTHOLE_LABEL_FRAGMENTS = (
    "pothole",
    "nid",
    "poth",
    "crack",
    "damage",
    "orniere",
    "ornière",
    "dégât",
    "degat",
    "alligator",
    "hole",
    "trou",
    "fissure",
    "d00",
    "d10",
    "d20",
)

STRONG_SEVERITY_KEYWORDS = (
    "danger",
    "profond",
    "accident",
    "route principale",
    "urgent",
    "critique",
)

# Monastir / est tunisien (fallback carte sans points GPS)
DEFAULT_MAP_CENTER = (35.7643, 10.8113)

KEYWORDS_POTHOLE = (
    "pothole",
    "nid",
    "poule",
    "hole",
    "hofra",
    "fosse",
    "ornière",
    "nids",
    "route",
    "chaussée",
    "asphalte",
    "bitume",
    "trou",
)
KEYWORDS_GARBAGE = (
    "déchet",
    "dechet",
    "ordures",
    "poubelle",
    "sac",
    "dépôt",
    "dépotoir",
    "plastique",
    "recycl",
)
KEYWORDS_SMOKE = (
    "fumée",
    "fumee",
    "smoke",
    "incendie",
    "feu",
    "brûle",
    "brule",
)
KEYWORDS_FLOOD = (
    "inond",
    "eau",
    "flood",
    "submer",
    "ruissel",
    "drainage",
)

SEVERITY_BASE = {
    "pothole": 72,
    "garbage": 55,
    "smoke": 78,
    "flood": 82,
    "unknown": 45,
}


def _norm_text(s: str) -> str:
    return (s or "").lower()


def _label_matches_pothole(label: str) -> bool:
    ln = (label or "").lower()
    if not ln.strip():
        return False
    targets = getattr(django_settings, "ROAD_DAMAGE_TARGET_LABELS", None)
    if targets:
        return any(str(t).lower() in ln or ln in str(t).lower() for t in targets)
    return any(frag in ln for frag in POTHOLE_LABEL_FRAGMENTS)


def summarize_vision_for_scoring(
    vision: dict[str, Any] | None, category: str = ""
) -> dict[str, Any]:
    """
    Agrège ce que le modèle a réellement produit (boîtes, libellés type nid de poule, confiances).
    Sert à afficher des pourcentages / comptes cohérents avec la sortie YOLO.
    """
    out: dict[str, Any] = {
        "yolo_ready": False,
        "pothole_like_count": 0,
        "total_detections": 0,
        "max_confidence_pct": 0,
        "pothole_max_confidence_pct": None,
        "model_key": None,
        "model_filename": None,
        "notes_fr": [],
    }
    if not vision or vision.get("stub") or not vision.get("ok"):
        out["notes_fr"].append(
            "YOLO non utilisé ou indisponible : le pourcentage « Conf » du score vient "
            "d’estimations par règles (plafonné à 60 %), pas d’inférence directe sur le média."
        )
        return out

    out["yolo_ready"] = True
    out["model_key"] = vision.get("model_key_resolved") or vision.get("model_key_requested")
    path_used = (vision.get("model_path_used") or "").strip()
    if path_used:
        try:
            out["model_filename"] = Path(path_used.replace("\\", "/")).name
        except Exception:
            out["model_filename"] = path_used.split("/")[-1].split("\\")[-1] or path_used

    pothole_confs: list[float] = []
    total = 0

    def _consume_boxes(boxes: list[Any]) -> None:
        nonlocal total, pothole_confs
        for b in boxes or []:
            if not isinstance(b, dict):
                continue
            total += 1
            lab = str(b.get("label", "") or "")
            if _label_matches_pothole(lab):
                try:
                    pothole_confs.append(float(b.get("confidence") or 0.0))
                except (TypeError, ValueError):
                    pothole_confs.append(0.0)

    _consume_boxes(list(vision.get("boxes") or []))
    for fr in vision.get("frame_results") or []:
        _consume_boxes(list(fr.get("boxes") or []))

    out["pothole_like_count"] = len(pothole_confs)
    out["total_detections"] = total
    mc = float(vision.get("max_confidence") or 0.0)
    out["max_confidence_pct"] = int(round(max(0.0, min(1.0, mc)) * 100))
    if pothole_confs:
        out["pothole_max_confidence_pct"] = int(
            round(max(0.0, min(1.0, max(pothole_confs))) * 100)
        )

    pct_sev = int(round(AI_W_SEVERITY * 100))
    pct_loc = int(round(AI_W_LOCATION * 100))
    pct_conf = int(round(AI_W_CONFIDENCE * 100))
    out["notes_fr"].append(
        "Les chiffres du panneau ci-dessus viennent directement de YOLO. "
        f"Le score « AI » utilise en base {pct_sev} % gravité, {pct_loc} % localisation, "
        f"{pct_conf} % la confiance modèle ; un bonus « volume de détections » (nombre × confiance) "
        "peut s’ajouter quand le modèle signale beaucoup de dégâts chaussée."
    )
    cat = (category or "").strip()
    if cat and cat != "pothole":
        out["notes_fr"].append(
            "Catégorie autre que « nid de poule » : la valeur « Conf » du bandeau principal "
            "suit les règles texte / catégorie, pas la branche YOLO « dégât chaussée ». "
            "Les comptages YOLO ci-dessus restent informatifs sur le média."
        )
    return out


def vision_has_pothole_detection(vision: dict[str, Any]) -> bool:
    """True si au moins une boîte / label YOLO correspond à un dégât chaussée."""
    if not vision or vision.get("stub") or not vision.get("ok"):
        return False
    for L in vision.get("labels") or []:
        if _label_matches_pothole(str(L)):
            return True
    for b in vision.get("boxes") or []:
        if _label_matches_pothole(str(b.get("label", ""))):
            return True
    for fr in vision.get("frame_results") or []:
        for b in fr.get("boxes") or []:
            if _label_matches_pothole(str(b.get("label", ""))):
                return True
    return False


def stub_model_out_no_yolo(reclamation) -> dict[str, Any]:
    """
    Pas de sortie YOLO utilisable : score de confiance plafonné (jamais 85).
    base 40 +10 catégorie nid +10 mots-clés forts → max 60.
    """
    cat = getattr(reclamation, "category", "") or ""
    desc = _norm_text(getattr(reclamation, "description", ""))
    media_type = getattr(reclamation, "media_type", "") or ""

    base = 40
    if cat == "pothole":
        base += 10
    if any(k in desc for k in STRONG_SEVERITY_KEYWORDS):
        base += 10
    confidence_score = int(max(0, min(60, base)))

    # Issue déduite texte / catégorie (sans CNN)
    if cat == "pothole" or any(k in desc for k in KEYWORDS_POTHOLE):
        detected_issue = "pothole"
    elif cat in ("garbage", "smoke", "flood"):
        detected_issue = cat
    elif any(k in desc for k in KEYWORDS_GARBAGE):
        detected_issue = "garbage"
    elif any(k in desc for k in KEYWORDS_SMOKE):
        detected_issue = "smoke"
    elif any(k in desc for k in KEYWORDS_FLOOD):
        detected_issue = "flood"
    else:
        detected_issue = cat or "unknown"

    return {
        "detected_issue": detected_issue,
        "confidence_score": confidence_score,
        "model_analysis_ready": bool(getattr(reclamation, "media", None)),
        "model_note": "Rule-based estimation (YOLO not used or unavailable)",
        "media_signal": {"image": 3, "video": 6}.get(media_type, 0),
        "yolo_used": False,
        "vision_verdict": None,
    }


def build_model_out_from_vision(reclamation, vision: dict[str, Any] | None) -> dict[str, Any]:
    """
    Fusionne la sortie média : YOLO prime pour catégorie « pothole », sinon stub borné.
    """
    cat = getattr(reclamation, "category", "") or ""

    if vision is None:
        return stub_model_out_no_yolo(reclamation)

    if vision.get("stub") or not vision.get("ok"):
        return stub_model_out_no_yolo(reclamation)

    # Vision OK
    if cat == "pothole":
        if vision_has_pothole_detection(vision):
            mc = float(vision.get("max_confidence") or 0.0)
            conf = int(round(max(0.0, min(100.0, mc * 100.0))))
            return {
                "detected_issue": "pothole",
                "confidence_score": conf,
                "model_analysis_ready": True,
                "model_note": "YOLO: pothole / road damage class detected",
                "media_signal": 8 if getattr(reclamation, "media_type", "") == "video" else 5,
                "yolo_used": True,
                "vision_verdict": "pothole",
            }
        return {
            "detected_issue": "none",
            "confidence_score": 10,
            "model_analysis_ready": True,
            "model_note": "YOLO: no pothole / damage class detected in image",
            "media_signal": 2,
            "yolo_used": True,
            "vision_verdict": "no_damage",
        }

    return stub_model_out_no_yolo(reclamation)


def analyze_media_with_model(reclamation) -> dict[str, Any]:
    """Compat API : même logique que stub sans vision (tests / ancien code)."""
    return stub_model_out_no_yolo(reclamation)


class PerceptionAgent:
    """Lit catégorie + description + type média → type de problème détecté."""

    @staticmethod
    def run(reclamation, model_hint: dict[str, Any]) -> dict[str, Any]:
        desc = _norm_text(getattr(reclamation, "description", ""))
        cat = (getattr(reclamation, "category", "") or "").strip()

        mh_issue = model_hint.get("detected_issue")
        if mh_issue == "none":
            signals = ["model:yolo_no_damage"]
            if cat:
                signals.insert(0, f"category:{cat}")
            return {"issue_type": "none", "signals": signals[:12]}

        if mh_issue == "pothole":
            issue = "pothole"
        elif cat in ("pothole", "garbage", "smoke", "flood"):
            issue = cat
        elif any(k in desc for k in KEYWORDS_POTHOLE):
            issue = "pothole"
        elif any(k in desc for k in KEYWORDS_GARBAGE):
            issue = "garbage"
        elif any(k in desc for k in KEYWORDS_SMOKE):
            issue = "smoke"
        elif any(k in desc for k in KEYWORDS_FLOOD):
            issue = "flood"
        else:
            issue = "unknown"

        signals = []
        if cat:
            signals.append(f"category:{cat}")
        if getattr(reclamation, "media_type", None):
            signals.append(f"media:{reclamation.media_type}")
        for label, keys in (
            ("pothole", KEYWORDS_POTHOLE),
            ("garbage", KEYWORDS_GARBAGE),
            ("smoke", KEYWORDS_SMOKE),
            ("flood", KEYWORDS_FLOOD),
        ):
            if any(k in desc for k in keys):
                signals.append(f"keyword:{label}")
        if model_hint.get("yolo_used"):
            signals.append("model:yolo_ready")
        elif model_hint.get("model_analysis_ready"):
            signals.append("model:rules_ready")

        return {"issue_type": issue, "signals": signals[:12]}


class SeverityAgent:
    """Score 0–100 : gravité issue des catégories, mots-clés, type média."""

    @staticmethod
    def run(reclamation, perception: dict[str, Any]) -> dict[str, Any]:
        desc = _norm_text(getattr(reclamation, "description", ""))
        issue = perception.get("issue_type") or "unknown"

        if issue == "none":
            return {
                "severity_score": 20,
                "severity_reasons": [
                    "Vision: no road damage class — severity capped for false-positive control",
                ],
            }

        base = float(SEVERITY_BASE.get(issue, SEVERITY_BASE["unknown"]))

        if not (desc and desc.strip()):
            base -= 20.0
        if any(k in desc for k in STRONG_SEVERITY_KEYWORDS):
            base += 20.0

        word_hits = sum(
            1
            for w in re.findall(r"[a-zàâäéèêëïîôùûç\-]{4,}", desc)
            if len(w) >= 5
        )
        base += min(8, word_hits * 2)

        if getattr(reclamation, "media_type", "") == "video":
            base += 4.0

        severity_score = int(max(0, min(100, round(base))))
        reasons = [
            f"Issue class: {issue}",
            f"Base severity table → {SEVERITY_BASE.get(issue, SEVERITY_BASE['unknown'])}",
        ]
        if not (desc and desc.strip()):
            reasons.append("No description text (−20)")
        if any(k in desc for k in STRONG_SEVERITY_KEYWORDS):
            reasons.append("Strong severity keywords (+20)")
        if word_hits:
            reasons.append(f"Descriptive keywords (+{min(8, word_hits * 2)})")
        return {"severity_score": severity_score, "severity_reasons": reasons}


class LocationAgent:
    """Priorité géographique : GPS valide + densité de signalements proches."""

    @staticmethod
    def count_nearby(reclamation, others: list[Any]) -> int:
        lat = getattr(reclamation, "latitude", None)
        lon = getattr(reclamation, "longitude", None)
        if lat is None or lon is None:
            return 0
        n = 0
        pk = getattr(reclamation, "pk", None)
        for o in others:
            if pk and getattr(o, "pk", None) == pk:
                continue
            olat = getattr(o, "latitude", None)
            olon = getattr(o, "longitude", None)
            if olat is None or olon is None:
                continue
            if abs(olat - lat) <= NEARBY_DEG_LAT and abs(olon - lon) <= NEARBY_DEG_LON:
                n += 1
        return n

    @staticmethod
    def run(reclamation, others: list[Any]) -> dict[str, Any]:
        lat = getattr(reclamation, "latitude", None)
        lon = getattr(reclamation, "longitude", None)
        if lat is None or lon is None:
            return {
                "location_score": 15,
                "nearby_count": 0,
                "location_reasons": ["Missing coordinates — location score reduced"],
            }

        nearby = LocationAgent.count_nearby(reclamation, others)
        score = 20 + nearby * 5
        score = int(max(0, min(40, score)))
        return {
            "location_score": score,
            "nearby_count": nearby,
            "location_reasons": [
                f"GPS OK ({lat:.5f}, {lon:.5f})",
                f"Nearby reports within ~1.3 km: {nearby} (+5 each, max 40)",
            ],
        }


class DecisionAgent:
    """Priorité finale à partir du score IA agrégé."""

    @staticmethod
    def run(ai_score: float) -> dict[str, Any]:
        x = float(ai_score)
        if x >= 85:
            lvl = "Critical"
            tone = "critical"
        elif x >= 70:
            lvl = "High"
            tone = "high"
        elif x >= 45:
            lvl = "Medium"
            tone = "medium"
        else:
            lvl = "Low"
            tone = "low"
        return {"final_priority": lvl, "priority_tone": tone}

    @staticmethod
    def apply_confidence_safety(
        confidence_score: float, decision: dict[str, Any], ai_score: float
    ) -> dict[str, Any]:
        """Anti faux positifs : confiance modèle < 50 → pas Critical / High."""
        if float(confidence_score) >= 50.0:
            return dict(decision)
        lvl = decision.get("final_priority")
        if lvl not in ("Critical", "High"):
            return dict(decision)
        if float(ai_score) >= 45.0:
            return {"final_priority": "Medium", "priority_tone": "medium"}
        return {"final_priority": "Low", "priority_tone": "low"}


class RecommendationAgent:
    """Recommandations opérationnelles + propriétaire d'action suggéré."""

    RECS = {
        "pothole": [
            "Sécuriser la zone (balisage temporaire, ralentissement).",
            "Marquage provisoire haute visibilité avant intervention.",
            "Réparation enrobé / scalpage selon diagnostic route.",
            "Contrôle du drainage et des regards alentour.",
            "Équipe d'inspection chaussée sous 48–72 h.",
        ],
        "garbage": [
            "Mobiliser équipe nettoyage municipale.",
            "Évaluer besoin en corbeilles / collecte renforcée.",
            "Tri et valorisation si déchets recyclables.",
        ],
        "smoke": [
            "Vérifier la source (industriel, véhicule, feu ouvert).",
            "Coordination pompiers / autorités si risque sanitaire.",
            "Communication aux riverains / protection respiratoire.",
        ],
        "flood": [
            "Intervention drainage / curage buses prioritaire.",
            "Signalisation danger et déviation si nécessaire.",
            "Pompage ou évacuation d'eau stagnante.",
        ],
        "unknown": [
            "Inspection terrain générale.",
            "Compléter le signalement avec photos supplémentaires.",
        ],
        "none": [
            "Vérifier terrain si le citoyen maintient la plainte.",
            "Considérer reclassification si nouvelles preuves.",
            "Pas d’anomalie chaussée détectée par le modèle sur ce média.",
        ],
    }

    OWNERS = {
        "pothole": "Road maintenance team",
        "garbage": "Environmental service",
        "smoke": "Emergency service",
        "flood": "Emergency service",
        "unknown": "Municipality",
        "none": "Municipality",
    }

    @classmethod
    def run(cls, perception: dict[str, Any]) -> dict[str, Any]:
        issue = perception.get("issue_type") or "unknown"
        recs = list(cls.RECS.get(issue, cls.RECS["unknown"]))
        owner = cls.OWNERS.get(issue, "Municipality")
        return {"recommendations": recs, "suggested_owner": owner}


def workflow_status_badge(reclamation, final_priority: str) -> str:
    """Badge UX : New / Analyzing / … (règles simples)."""
    from django.utils import timezone

    created = getattr(reclamation, "created_at", None)
    if not created:
        return "New"
    age = timezone.now() - created
    hours = age.total_seconds() / 3600
    if hours < 24:
        return "New"
    if hours < 72 and final_priority in ("Medium", "Low"):
        return "Analyzing"
    if final_priority == "Critical":
        return "Critical"
    if final_priority == "Medium":
        return "Medium"
    if final_priority == "Low":
        return "Low"
    return "Analyzing"


def compute_ai_pipeline(
    reclamation,
    all_reclamations: list[Any],
    vision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Orchestre tous les agents + formules de score (pondération centrée sur la confiance modèle)."""
    model_out = build_model_out_from_vision(reclamation, vision)
    perception = PerceptionAgent.run(reclamation, model_out)
    sev = SeverityAgent.run(reclamation, perception)
    loc = LocationAgent.run(reclamation, list(all_reclamations))
    confidence_score = int(max(0, min(100, model_out.get("confidence_score", 40))))

    severity_score = sev["severity_score"]
    location_score = loc["location_score"]

    cat = getattr(reclamation, "category", "") or ""
    model_scoring = summarize_vision_for_scoring(vision, cat)

    ai_score_base = (
        severity_score * AI_W_SEVERITY
        + location_score * AI_W_LOCATION
        + confidence_score * AI_W_CONFIDENCE
    )
    model_volume_boost = 0.0
    if (
        model_out.get("yolo_used")
        and model_out.get("vision_verdict") == "pothole"
        and model_scoring.get("yolo_ready")
    ):
        n = int(model_scoring.get("pothole_like_count") or 0)
        mpc = float(model_scoring.get("max_confidence_pct") or 0) / 100.0
        if n >= 1 and mpc > 0:
            model_volume_boost = min(
                MODEL_VOLUME_BOOST_CAP,
                float(n) * mpc * MODEL_VOLUME_BOOST_PER_UNIT,
            )

    ai_score = round(
        max(0.0, min(100.0, ai_score_base + model_volume_boost)),
        1,
    )

    force_low_no_damage = (
        cat == "pothole"
        and model_out.get("vision_verdict") == "no_damage"
        and model_out.get("yolo_used")
    )
    if force_low_no_damage:
        ai_score = min(ai_score, 39.0)

    decision = DecisionAgent.run(ai_score)
    decision = DecisionAgent.apply_confidence_safety(confidence_score, decision, ai_score)

    if force_low_no_damage:
        decision = {"final_priority": "Low", "priority_tone": "low"}

    rec = RecommendationAgent.run(perception)

    src_note = (
        "YOLO detections (max conf ×100)"
        if model_out.get("yolo_used")
        else "rule-based estimation (max 60)"
    )
    why_lines = [
        f"severity_score={severity_score}",
        f"location_score={location_score} (GPS + nearby ×5, max 40)",
        f"confidence_score={confidence_score} ({src_note})",
        (
            f"ai_score_base = {AI_W_SEVERITY}×sev + {AI_W_LOCATION}×loc + "
            f"{AI_W_CONFIDENCE}×conf → {round(ai_score_base, 1)}"
        ),
    ]
    if model_volume_boost > 0:
        why_lines.append(
            f"model_volume_boost=+{round(model_volume_boost, 1)} "
            f"(min({MODEL_VOLUME_BOOST_CAP}, pothole_boxes×max_conf×{MODEL_VOLUME_BOOST_PER_UNIT}))"
        )
    why_lines.append(f"ai_score_final → {ai_score}")
    if force_low_no_damage:
        why_lines.append("Forced Low: pothole category but no damage class in YOLO output")

    wf = workflow_status_badge(reclamation, decision["final_priority"])

    verdict = model_out.get("vision_verdict")
    if verdict == "pothole":
        model_label_display = "pothole"
    elif verdict == "no_damage":
        model_label_display = "none"
    elif model_out.get("yolo_used"):
        model_label_display = "none"
    else:
        model_label_display = "not_analyzed"

    return {
        "perception": perception,
        "severity_score": severity_score,
        "severity_reasons": sev["severity_reasons"],
        "location_score": location_score,
        "nearby_count": loc["nearby_count"],
        "location_reasons": loc["location_reasons"],
        "confidence_score": confidence_score,
        "ai_score_base": round(ai_score_base, 1),
        "model_volume_boost": round(model_volume_boost, 1),
        "ai_score": ai_score,
        "final_priority": decision["final_priority"],
        "priority_tone": decision["priority_tone"],
        "recommendations": rec["recommendations"],
        "suggested_owner": rec["suggested_owner"],
        "model_analysis": model_out,
        "why_score": why_lines,
        "workflow_status": wf,
        "detected_issue": perception["issue_type"],
        "detected_signals": perception.get("signals", []),
        "yolo_used": bool(model_out.get("yolo_used")),
        "vision_verdict": verdict,
        "model_detection_label": model_label_display,
        "score_stub_warning": not bool(model_out.get("yolo_used")),
        "model_scoring": model_scoring,
    }


def map_marker_color(priority_tone: str) -> str:
    if priority_tone in ("critical", "high"):
        return "#e53935"
    if priority_tone == "medium":
        return "#fb8c00"
    return "#43a047"


def build_map_payload(enriched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Centre auto + marqueurs + points heatmap (lat, lon, intensity)."""
    markers = []
    heat = []
    lats = []
    lons = []

    for row in enriched_rows:
        r = row["rec"]
        lat = getattr(r, "latitude", None)
        lon = getattr(r, "longitude", None)
        if lat is None or lon is None:
            continue
        lats.append(float(lat))
        lons.append(float(lon))
        pipe = row["pipeline"]
        intensity = max(0.15, min(1.0, float(pipe["ai_score"]) / 100.0))
        heat.append([float(lat), float(lon), intensity])
        markers.append(
            {
                "lat": float(lat),
                "lng": float(lon),
                "color": map_marker_color(pipe["priority_tone"]),
                "rid": r.pk,
                "priority": pipe["final_priority"],
                "category": getattr(r, "category", ""),
                "ai_score": pipe["ai_score"],
                "created": r.created_at.isoformat() if r.created_at else "",
                "description": (r.description or "")[:240],
            }
        )

    if lats and lons:
        center = (sum(lats) / len(lats), sum(lons) / len(lons))
        zoom = 11
    else:
        center = DEFAULT_MAP_CENTER
        zoom = 8

    return {
        "center": {"lat": center[0], "lng": center[1], "zoom": zoom},
        "markers": markers,
        "heatmap": heat,
    }


def priority_sort_rank(pipeline: dict[str, Any]) -> int:
    """1=Critical/High, 2=Medium, 3=Low, 4=unknown (for Recent Cases ordering)."""
    final = str(pipeline.get("final_priority") or "").strip().lower()
    score = float(pipeline.get("ai_score") or 0)

    if final in ("critical", "high") or score >= 70:
        return 1
    if final == "medium" or score >= 40:
        return 2
    if final == "low" or score < 40:
        return 3
    return 4


def sort_enriched_by_priority(enriched_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by priority rank ASC, then created_at DESC within each group."""

    def _key(row: dict[str, Any]) -> tuple[int, float]:
        pipeline = row.get("pipeline") or {}
        rank = priority_sort_rank(pipeline)
        created = getattr(row.get("rec"), "created_at", None)
        ts = created.timestamp() if created else 0.0
        return (rank, -ts)

    enriched_rows.sort(key=_key)
    return enriched_rows


def compute_kpis(enriched_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from django.utils import timezone

    today = timezone.localdate()
    total = len(enriched_rows)
    critical = sum(
        1
        for row in enriched_rows
        if row["pipeline"]["final_priority"] in ("Critical", "High")
    )  # « Critiques » au sens dashboard (Critical + High)
    scores = [row["pipeline"]["ai_score"] for row in enriched_rows]
    avg_ai = round(sum(scores) / len(scores), 1) if scores else 0.0
    today_n = sum(
        1
        for row in enriched_rows
        if getattr(row["rec"], "created_at", None)
        and timezone.localtime(row["rec"].created_at).date() == today
    )
    return {
        "total_reports": total,
        "critical_reports": critical,
        "avg_ai_score": avg_ai,
        "reports_today": today_n,
    }
