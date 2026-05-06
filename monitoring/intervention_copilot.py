"""
Municipality AI Copilot — plans d'intervention dynamiques (Groq + règles).
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from typing import Any

from django.conf import settings

# Gabès / Tunisie — zones indicatives
TUNISIA_LAT_RANGE = (30.2, 37.8)
TUNISIA_LON_RANGE = (7.5, 11.8)
GABES_LAT_RANGE = (33.45, 34.35)
GABES_LON_RANGE = (9.65, 10.75)

URGENCY_WORDS = (
    "grand",
    "grande",
    "profond",
    "profonde",
    "danger",
    "accident",
    "bloqué",
    "bloque",
    "route principale",
    "urgent",
    "critique",
)


def _norm(s: str) -> str:
    return (s or "").lower()


def _cost_jitter(lo: int, hi: int, seed: str) -> int:
    """Coût pseudo-aléatoire mais stable pour un même signalement."""
    if hi <= lo:
        return lo
    span = hi - lo + 1
    n = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12], 16)
    return lo + (n % span)


def _parse_json_obj(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _gps_context(lat: float | None, lon: float | None) -> tuple[str, bool]:
    """Retourne (label, coordonnées_plausibles)."""
    if lat is None or lon is None:
        return "missing", False
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return "invalid", False
    if not (TUNISIA_LAT_RANGE[0] <= la <= TUNISIA_LAT_RANGE[1]) or not (
        TUNISIA_LON_RANGE[0] <= lo <= TUNISIA_LON_RANGE[1]
    ):
        return "outside_tunisia", False
    in_gabes = GABES_LAT_RANGE[0] <= la <= GABES_LAT_RANGE[1] and GABES_LON_RANGE[
        0
    ] <= lo <= GABES_LON_RANGE[1]
    if in_gabes:
        return "gabes_area", True
    return "tunisia_other", True


def _image_dimensions(reclamation) -> tuple[int, int] | None:
    try:
        import cv2

        if not getattr(reclamation, "media", None):
            return None
        p = reclamation.media.path
        im = cv2.imread(str(p))
        if im is None:
            return None
        h, w = im.shape[:2]
        return int(w), int(h)
    except Exception:
        return None


def _max_bbox_area_ratio(
    vision: dict[str, Any], reclamation, media_type: str
) -> float | None:
    """Ratio aire max boîte / aire image (0–1). Vidéo : agrégation sur frames."""
    boxes: list[dict[str, Any]] = []
    if media_type == "video":
        for fr in vision.get("frame_results") or []:
            for b in fr.get("boxes") or []:
                boxes.append(b)
    else:
        boxes = list(vision.get("boxes") or [])

    if not boxes:
        return None

    dims = _image_dimensions(reclamation)
    if not dims:
        # Fallback taille type stream YOLO
        iw, ih = 640, 480
    else:
        iw, ih = dims

    if iw <= 0 or ih <= 0:
        return None
    img_area = float(iw * ih)
    best = 0.0
    for b in boxes:
        xy = b.get("xyxy")
        if not xy or len(xy) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in xy[:4]]
        except (TypeError, ValueError):
            continue
        a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        best = max(best, a / img_area)
    return round(best, 4)


def _size_label(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    if ratio < 0.05:
        return "small"
    if ratio <= 0.15:
        return "medium"
    return "large"


def _damage_type_label(
    cat: str,
    ratio: float | None,
    max_conf: float,
    vision_ok: bool,
    pothole_detected: bool,
) -> str:
    if not vision_ok or not pothole_detected:
        return "uncertain"
    if cat != "pothole":
        return "non_pothole_category"
    if ratio is None:
        return "uncertain"
    if ratio < 0.05:
        return "small pothole"
    if ratio <= 0.15:
        return "medium pothole"
    if max_conf >= 0.75:
        return "severe road damage"
    return "large pothole"


def _cluster_boost(nearby: int) -> tuple[int, str]:
    if nearby >= 5:
        return 2, "Zone à signalements répétés — prioriser inspection coordonnée."
    if nearby >= 3:
        return 1, "Plusieurs signalements citoyens à proximité (~1,3 km)."
    return 0, ""


def _rule_based_plan(
    reclamation,
    pipeline: dict[str, Any],
    vision: dict[str, Any],
    xai: dict[str, Any],
    cluster_context: dict[str, Any] | None,
) -> dict[str, Any]:
    cluster_context = cluster_context or {}
    nearby = int(cluster_context.get("nearby_count", pipeline.get("nearby_count", 0)))
    cat = getattr(reclamation, "category", "") or ""
    desc = _norm(getattr(reclamation, "description", "") or "")
    media_type = getattr(reclamation, "media_type", "") or "image"
    created = getattr(reclamation, "created_at", None)
    lat = getattr(reclamation, "latitude", None)
    lon = getattr(reclamation, "longitude", None)

    gps_label, gps_ok = _gps_context(
        float(lat) if lat is not None else None,
        float(lon) if lon is not None else None,
    )

    vision_ok = bool(vision and vision.get("ok") and not vision.get("stub"))
    max_conf = float(vision.get("max_confidence") or 0)
    conf_pct = float(pipeline.get("confidence_score") or 0)
    yolo_used = bool(pipeline.get("yolo_used"))
    verdict = pipeline.get("vision_verdict")
    pothole_detected = verdict == "pothole" or (
        yolo_used and pipeline.get("model_detection_label") == "pothole"
    )
    no_damage = verdict == "no_damage" or (
        yolo_used and pipeline.get("model_detection_label") == "none"
    )

    ratio = _max_bbox_area_ratio(vision, reclamation, media_type)
    size_lbl = _size_label(ratio)
    dmg_type = _damage_type_label(cat, ratio, max_conf, vision_ok, pothole_detected)

    urgency_boost = any(w in desc for w in URGENCY_WORDS)
    empty_desc = not (desc and desc.strip())
    cb_delta, cluster_note = _cluster_boost(nearby)

    # Risque & urgence de base (fonction des scores réels)
    prio = str(pipeline.get("final_priority") or "Medium")
    ai_score = float(pipeline.get("ai_score") or 0)
    sev = float(pipeline.get("severity_score") or 0)

    if no_damage or (yolo_used and conf_pct < 35 and cat == "pothole"):
        risk = "Low"
        urgency = "1 week"
        reason = (
            "Aucune classe dégât chaussée fiable détectée par YOLO, ou confiance visuelle faible — "
            "plan limité à la vérification terrain."
        )
    elif max_conf < 0.35 and vision_ok and cat == "pothole":
        risk = "Low"
        urgency = "1 week"
        reason = (
            f"Confiance YOLO maximale faible ({max_conf:.2f}) — prévoir constat avant réparation."
        )
    elif prio == "Critical" or ai_score >= 85:
        risk = "Critical"
        urgency = "Immediate"
        reason = (
            f"Score IA {ai_score:.0f}, priorité {prio}. "
            + ("Signalement textuel grave. " if urgency_boost else "")
            + (cluster_note + " " if cluster_note else "")
        )
    elif prio == "High" or ai_score >= 70:
        risk = "High"
        urgency = "48h"
        reason = (
            f"Priorité élevée (score {ai_score:.0f}, sévérité {sev:.0f}). "
            + (cluster_note if cluster_note else "Intervention rapide recommandée.")
        )
    elif prio == "Medium" or ai_score >= 45:
        risk = "Medium"
        urgency = "72h"
        reason = (
            f"Cas intermédiaire — score {ai_score:.0f}. "
            + ("Description insuffisante — valider sur place. " if empty_desc else "")
        )
    else:
        risk = "Low"
        urgency = "1 week"
        reason = f"Score modéré ({ai_score:.0f}) — planning standard."

    if empty_desc:
        reason += " Description absente : compléter après visite."

    if not gps_ok and gps_label == "outside_tunisia":
        reason += " GPS hors Tunisie — vérifier coordonnées ou erreur de saisie."
    elif gps_label == "tunisia_other":
        reason += " Hors périmètre Gabès indicatif — confirmer secteur d’intervention."

    if media_type == "video":
        reason += (
            " Média vidéo : décision basée sur l’agrégation des frames échantillonnées "
            f"(max conf {max_conf:.2f})."
        )

    visual_evidence = (
        f"{vision.get('detection_box_count', 0)} détection(s), conf max {max_conf:.2f}, "
        f"labels {', '.join(vision.get('labels') or []) or '—'}."
    )
    if ratio is not None:
        visual_evidence += f" Ratio boîte/image ~ {ratio*100:.1f}% ({size_lbl})."

    xai_txt = str((xai or {}).get("explanation") or "")[:400]
    damage_conf = int(round(max(conf_pct, max_conf * 100)))

    # Coûts dynamiques (TND)
    tier_cost = {
        "Critical": (800, 1500),
        "High": (350, 800),
        "Medium": (150, 350),
        "Low": (50, 120),
    }
    lo, hi = tier_cost.get(risk, (150, 350))
    if urgency_boost:
        hi = min(1800, hi + 150)
        lo += 40
    if cb_delta:
        hi = min(1800, hi + 80 * cb_delta)
        lo += 30 * cb_delta
    if size_lbl == "large":
        hi = min(1800, hi + 200)
    elif size_lbl == "small":
        hi = max(lo, hi - 80)

    base_workers = {"Critical": 5, "High": 3, "Medium": 2, "Low": 1}.get(risk, 2)
    workers = base_workers + cb_delta

    equip_low = ["kit d’inspection", "check-list photo"]
    equip_med = ["cônes", "balisage", "enrobé à froid"]
    equip_high = ["compacteur", "enrobé à chaud", "découpeuse", "diable"]
    equip_crit = ["contrôle trafic", "véhicule balisage", "équipe complète"]

    if risk == "Low" and (no_damage or max_conf < 0.4):
        phases = [
            {
                "phase": "Vérification terrain",
                "deadline": "72h",
                "actions": [
                    "Constat photo géolocalisé et mesure réelle du défaut.",
                    "Valider si le signalement correspond à la même zone GPS.",
                ],
                "responsible_team": "Équipe inspection municipale",
                "materials": ["check-list", "mètre ruban", "tablette terrain"],
                "estimated_cost_tnd": _cost_jitter(lo, min(hi, lo + 80), f"vrf-{risk}-{nearby}"),
            }
        ]
    elif risk in ("Critical", "High"):
        phases = [
            {
                "phase": "Sécurisation immédiate",
                "deadline": "0-6h",
                "actions": [
                    "Baliser et sécuriser la zone si danger pour usagers.",
                    "Informer circulation / police municipale si voie principale.",
                ],
                "responsible_team": "Équipe sécurité & signalisation",
                "materials": ["cônes", "panneaux temporaires", "rubalise"],
                "estimated_cost_tnd": _cost_jitter(60, 140, f"safe-{risk}-{nearby}"),
            },
            {
                "phase": "Inspection technique",
                "deadline": "24h",
                "actions": [
                    "Diagnostic structure chaussée et drainage.",
                    "Mesurer profondeur / surface impactée.",
                ],
                "responsible_team": "Maintenance routes",
                "materials": ["appareil photo", "rapport terrain"],
                "estimated_cost_tnd": _cost_jitter(90, 180, f"insp-{risk}-{nearby}"),
            },
            {
                "phase": "Réparation",
                "deadline": "48h",
                "actions": [
                    "Scalpage / découpe zone dégradée, nettoyage, tack coat.",
                    "Pose enrobé à chaud ou à froid selon disponibilité.",
                ],
                "responsible_team": "Crew enrobé municipal",
                "materials": equip_high + ["bitume", "tack coat"],
                "estimated_cost_tnd": _cost_jitter(max(lo, 200), hi, f"rep-{risk}-{nearby}"),
            },
        ]
    else:
        phases = [
            {
                "phase": "Inspection planifiée",
                "deadline": "48h",
                "actions": [
                    "Visite terrain et arbitrage réparation vs simple rebouchage.",
                ],
                "responsible_team": "Maintenance routes",
                "materials": equip_med,
                "estimated_cost_tnd": _cost_jitter(lo, min(hi, lo + 120), f"ip-{risk}-{nearby}"),
            },
            {
                "phase": "Intervention ciblée",
                "deadline": "72h",
                "actions": [
                    "Réparation selon dimension réelle constatée.",
                    "Contrôle qualité compactage.",
                ],
                "responsible_team": "Équipe réparation voirie",
                "materials": equip_med + ["compacteur léger"],
                "estimated_cost_tnd": _cost_jitter(lo + 40, hi, f"iv-{risk}-{nearby}"),
            },
        ]

    total_cost = sum(int(p.get("estimated_cost_tnd") or 0) for p in phases)

    exec_summary = (
        f"Signalement « {cat or '—'} » ({media_type}) du "
        f"{created.strftime('%d/%m/%Y %H:%M') if created else '—'}. "
        f"Score IA {ai_score:.0f}, risque {risk}. "
    )
    exec_summary += cluster_note or ""
    exec_summary += (
        f" Synthèse XAI : {(xai_txt[:180] + '…') if len(xai_txt) > 180 else xai_txt}"
        if xai_txt
        else ""
    )

    citizen_msg = (
        "Évitez la zone si danger apparent ; suivez les consignes municipales et la signalisation temporaire."
    )
    if risk == "Low" and no_damage:
        citizen_msg = (
            "Le traitement automatique n’a pas confirmé un défaut majeur sur ce média — "
            "signalez toute évolution sur place."
        )

    muni_msg = (
        "Prioriser selon ressources disponibles ; synchroniser avec la permanence voirie "
        "et documenter les photos avant/après dans le système."
    )

    follow_days = 7 if risk == "Low" else (3 if risk in ("High", "Critical") else 5)
    needs_second = risk in ("Medium", "High", "Critical") or empty_desc

    plan = {
        "executive_summary": exec_summary.strip(),
        "risk_level": risk,
        "urgency": urgency,
        "priority_reason": reason.strip(),
        "damage_assessment": {
            "estimated_damage_type": dmg_type,
            "estimated_size": size_lbl,
            "confidence": damage_conf,
            "visual_evidence": visual_evidence[:1200],
        },
        "intervention_plan": phases,
        "total_estimated_cost_tnd": total_cost,
        "resource_plan": {
            "teams_needed": list(
                {p.get("responsible_team") for p in phases if p.get("responsible_team")}
            )
            or ["Maintenance routes"],
            "workers_estimate": workers,
            "equipment": (equip_crit if risk == "Critical" else equip_high)
            if risk in ("Critical", "High")
            else equip_med,
            "materials": ["enrobé", "tack coat", "gravats evacuation"],
        },
        "citizen_message": citizen_msg,
        "municipality_message": muni_msg,
        "follow_up": {
            "needs_second_inspection": needs_second,
            "suggested_status": (
                "Intervention urgente planifiée"
                if risk in ("Critical", "High")
                else "Inspection planifiée"
            ),
            "verification_after_days": follow_days,
        },
        "source": "rules",
        "_meta": {
            "gps_context": gps_label,
            "bbox_area_ratio": ratio,
            "cluster_note": cluster_note,
        },
    }
    return plan


def _groq_plan(
    reclamation,
    pipeline: dict[str, Any],
    vision: dict[str, Any],
    xai: dict[str, Any],
    cluster_context: dict[str, Any] | None,
    rule_fallback: dict[str, Any],
) -> dict[str, Any] | None:
    groq_key = (getattr(settings, "GROQ_API_KEY", None) or "").strip()
    openai_key = (getattr(settings, "OPENAI_API_KEY", None) or "").strip()
    if not groq_key and not openai_key:
        return None

    try:
        cat_label_g = reclamation.get_category_display()
    except Exception:
        cat_label_g = ""

    payload = {
        "category": getattr(reclamation, "category", ""),
        "category_label": cat_label_g,
        "description": getattr(reclamation, "description", "") or "",
        "latitude": getattr(reclamation, "latitude", None),
        "longitude": getattr(reclamation, "longitude", None),
        "created_at": getattr(reclamation, "created_at", None).isoformat()
        if getattr(reclamation, "created_at", None)
        else "",
        "media_type": getattr(reclamation, "media_type", ""),
        "pipeline": {
            k: pipeline.get(k)
            for k in (
                "ai_score",
                "final_priority",
                "severity_score",
                "location_score",
                "confidence_score",
                "nearby_count",
                "yolo_used",
                "vision_verdict",
                "model_detection_label",
            )
        },
        "vision": {
            "ok": vision.get("ok"),
            "stub": vision.get("stub"),
            "max_confidence": vision.get("max_confidence"),
            "labels": vision.get("labels"),
            "detection_box_count": vision.get("detection_box_count"),
            "boxes_sample": (vision.get("boxes") or [])[:5],
        },
        "xai_excerpt": str((xai or {}).get("explanation", ""))[:900],
        "cluster": cluster_context or {},
        "bbox_area_ratio_hint": _max_bbox_area_ratio(
            vision,
            reclamation,
            getattr(reclamation, "media_type", "") or "image",
        ),
        "gps_context": _gps_context(
            float(getattr(reclamation, "latitude", None))
            if getattr(reclamation, "latitude", None) is not None
            else None,
            float(getattr(reclamation, "longitude", None))
            if getattr(reclamation, "longitude", None) is not None
            else None,
        )[0],
    }

    schema_hint = """
Réponds UNIQUEMENT avec un JSON valide (pas de markdown), clés obligatoires :
{
  "executive_summary": "string",
  "risk_level": "Critical|High|Medium|Low",
  "urgency": "Immediate|24h|48h|72h|1 week",
  "priority_reason": "string",
  "damage_assessment": {
    "estimated_damage_type": "string",
    "estimated_size": "small|medium|large|unknown",
    "confidence": nombre 0-100,
    "visual_evidence": "string"
  },
  "intervention_plan": [
    {
      "phase": "string",
      "deadline": "string",
      "actions": ["string"],
      "responsible_team": "string",
      "materials": ["string"],
      "estimated_cost_tnd": nombre
    }
  ],
  "total_estimated_cost_tnd": nombre,
  "resource_plan": {
    "teams_needed": ["string"],
    "workers_estimate": nombre,
    "equipment": ["string"],
    "materials": ["string"]
  },
  "citizen_message": "string",
  "municipality_message": "string",
  "follow_up": {
    "needs_second_inspection": bool,
    "suggested_status": "string",
    "verification_after_days": nombre
  }
}
Varie les formulations et les actions selon les preuves concrètes (pas de texte générique répété).
"""

    sys_prompt = (
        "Tu es un expert municipal tunisien en maintenance routière et sécurité urbaine (Gabès / Smart City). "
        "Tu produis des plans d'intervention opérationnels en français. "
        + schema_hint
    )

    provider = "openai" if openai_key else "groq"
    model = (
        getattr(settings, "OPENAI_MODEL", "gpt-4o-mini")
        if provider == "openai"
        else getattr(settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
    )
    endpoint = (
        "https://api.openai.com/v1/chat/completions"
        if provider == "openai"
        else "https://api.groq.com/openai/v1/chat/completions"
    )
    api_key = openai_key if provider == "openai" else groq_key

    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            "temperature": 0.4,
            "max_tokens": 3500,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None

    parsed = _parse_json_obj(content)
    if not isinstance(parsed, dict):
        return None
    merged = _normalize_plan(parsed, rule_fallback)
    merged["source"] = provider
    return merged


def _normalize_plan(raw: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Complète / corrige le JSON Groq pour correspondre au schéma attendu."""
    out = dict(fallback)
    out.update({k: v for k, v in raw.items() if k != "_meta"})

    for key in (
        "executive_summary",
        "risk_level",
        "urgency",
        "priority_reason",
        "citizen_message",
        "municipality_message",
    ):
        if key in raw and raw[key]:
            out[key] = raw[key]

    if isinstance(raw.get("damage_assessment"), dict):
        da = out.setdefault("damage_assessment", {})
        da.update(raw["damage_assessment"])

    if isinstance(raw.get("intervention_plan"), list) and raw["intervention_plan"]:
        out["intervention_plan"] = raw["intervention_plan"]

    if raw.get("total_estimated_cost_tnd") is not None:
        try:
            out["total_estimated_cost_tnd"] = int(raw["total_estimated_cost_tnd"])
        except (TypeError, ValueError):
            pass

    if isinstance(raw.get("resource_plan"), dict):
        rp = out.setdefault("resource_plan", {})
        rp.update(raw["resource_plan"])

    if isinstance(raw.get("follow_up"), dict):
        fu = out.setdefault("follow_up", {})
        fu.update(raw["follow_up"])

    # Recalc total si phases présentes
    phases = out.get("intervention_plan") or []
    if phases:
        try:
            out["total_estimated_cost_tnd"] = sum(
                int(p.get("estimated_cost_tnd") or 0) for p in phases
            )
        except (TypeError, ValueError):
            pass

    return out


def generate_intervention_plan(
    reclamation,
    pipeline: dict[str, Any],
    vision: dict[str, Any],
    xai: dict[str, Any],
    cluster_context: dict[str, Any] | None = None,
    *,
    rules_only: bool = False,
) -> dict[str, Any]:
    """
    Génère le plan JSON Copilot (Groq si clé, sinon règles dynamiques).
    Si rules_only=True (ex. rafraîchissement ESP rapide), pas d'appel Groq.
    """
    rule = _rule_based_plan(reclamation, pipeline, vision, xai, cluster_context)
    if rules_only:
        return rule
    groq = _groq_plan(reclamation, pipeline, vision, xai, cluster_context, rule)
    if groq:
        return groq
    return rule


def copilot_to_legacy_groq(plan: dict[str, Any]) -> dict[str, Any]:
    """Compatibilité avec l’ancien bloc groq_json (résumé, actions, équipe, etc.)."""
    phases = plan.get("intervention_plan") or []
    actions: list[str] = []
    for ph in phases:
        for a in ph.get("actions") or []:
            actions.append(f"[{ph.get('phase', '')}] {a}")
    if not actions:
        actions = ["Voir plan d’intervention détaillé (Copilot)."]

    team = ""
    if phases:
        team = str(phases[0].get("responsible_team") or "")
    if not team:
        team = ", ".join(plan.get("resource_plan", {}).get("teams_needed") or []) or "Municipality"

    maint_parts = []
    for ph in phases:
        maint_parts.append(f"{ph.get('phase', '')} ({ph.get('deadline', '')})")

    urg = str(plan.get("urgency") or "48h")
    allowed_legacy = ("Immediate", "24h", "48h", "1 week")
    if urg == "72h":
        legacy_urgency = "48h"
    elif urg in allowed_legacy:
        legacy_urgency = urg
    else:
        legacy_urgency = "48h"

    return {
        "summary": plan.get("executive_summary", ""),
        "risk_level": plan.get("risk_level", "Medium"),
        "recommended_actions": actions[:24],
        "responsible_team": team,
        "citizen_safety_message": plan.get("citizen_message", ""),
        "maintenance_plan": " · ".join(maint_parts) if maint_parts else (plan.get("municipality_message") or ""),
        "estimated_urgency": legacy_urgency,
        "source": plan.get("source", "rules"),
    }
