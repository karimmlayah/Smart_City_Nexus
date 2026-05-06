"""
Structured mock / demo payloads for the AI analytics dashboard.

Model rows are derived from ``django.conf.settings`` (paths & APIs actually wired in
this project). Request counts / trends remain illustrative until you plug real metrics.
"""

from __future__ import annotations

import hashlib
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from django.conf import settings


def _stable_seed(parts: tuple[str, ...]) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def _abs_path(raw: str | None) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    if not p.is_absolute():
        p = Path(getattr(settings, "BASE_DIR", ".")).resolve() / p
    try:
        return p.resolve()
    except OSError:
        return None


def _is_hub_shorthand(raw: str) -> bool:
    """e.g. ``yolov8n.pt`` téléchargé via Ultralytics — pas de dossier dans le chemin."""
    s = raw.strip().replace("\\", "/")
    return "/" not in s


def _local_model_active(raw: str | None) -> bool:
    """Fichier présent sur disque, ou nom hub .pt/.onnx sans chemin."""
    if not raw:
        return False
    s = str(raw).strip()
    if not s:
        return False
    if _is_hub_shorthand(s):
        suf = Path(s).suffix.lower()
        return suf in (".pt", ".onnx", ".pth", ".h5", ".keras")
    ap = _abs_path(s)
    return bool(ap and ap.is_file())


def _groq_active() -> bool:
    return bool(str(getattr(settings, "GROQ_API_KEY", "") or "").strip())


def _sightengine_active() -> bool:
    u = getattr(settings, "SIGHTENGINE_API_USER", "") or ""
    k = getattr(settings, "SIGHTENGINE_API_KEY", "") or ""
    return bool(str(u).strip() and str(k).strip())


def _demo_metrics(model_id: str, rng: random.Random) -> tuple[int, int, float]:
    """Volumes / latences démo stables par id (pas des métriques réelles)."""
    h = int(hashlib.md5(model_id.encode()).hexdigest()[:8], 16)
    requests = 1200 + (h % 98000)
    requests = int(requests * (0.82 + rng.random() * 0.28))
    avg_ms = 28 + (h % 420)
    err = round((h % 180) / 1000.0, 2)
    return requests, avg_ms, err


def collect_project_ai_models(rng: random.Random) -> list[dict[str, Any]]:
    """
    Une ligne par rôle configuré dans Django (même fichier physique peut apparaître plusieurs fois).
    """
    rows: list[dict[str, Any]] = []

    def append_row(
        mid: str,
        title: str,
        provider: str,
        path_or_hub: str | None,
        *,
        status_override: str | None = None,
    ) -> None:
        ref = (path_or_hub or "").strip()
        if not ref and status_override is None:
            return
        if any(r["id"] == mid for r in rows):
            return

        if status_override:
            status = status_override
        elif provider == "Groq":
            status = "active" if _groq_active() else "down"
        elif provider == "Sightengine":
            status = "active" if _sightengine_active() else "down"
        elif provider == "DeepFace":
            status = "active"
        else:
            status = "active" if _local_model_active(ref) else "down"

        req, avg_ms, err = _demo_metrics(mid, rng)
        if status == "down" and provider not in ("Groq", "Sightengine"):
            err = min(5.0, err + 2.5)

        rows.append(
            {
                "id": mid,
                "name": title,
                "provider": provider,
                "requests": req,
                "avg_ms": avg_ms,
                "error_rate": err,
                "status": status,
            }
        )

    # --- Feu / fumée ONNX ---
    append_row(
        "fire-onnx",
        f"Feu / fumée — {Path(str(getattr(settings, 'FIRE_ONNX_MODEL_PATH', '') or '')).name or 'ONNX'}",
        "Local (ONNX)",
        str(getattr(settings, "FIRE_ONNX_MODEL_PATH", "") or ""),
    )

    # --- MJPEG / foule ---
    append_row(
        "yolo-surveillance",
        f"YOLO foule / sources vidéo — {Path(str(getattr(settings, 'YOLO_MODEL_PATH', '') or '')).name}",
        "Local (Ultralytics)",
        str(getattr(settings, "YOLO_MODEL_PATH", "") or ""),
    )

    # --- Combat ---
    append_row(
        "fight-classifier",
        f"Classif. combat — {Path(str(getattr(settings, 'FIGHT_CLASSIFIER_PATH', '') or '')).name}",
        "Local (Ultralytics)",
        str(getattr(settings, "FIGHT_CLASSIFIER_PATH", "") or ""),
    )
    append_row(
        "fight-person-yolo",
        f"YOLO personnes (fusion) — {getattr(settings, 'FIGHT_PERSON_MODEL_PATH', '')}",
        "Local (Ultralytics / hub)",
        str(getattr(settings, "FIGHT_PERSON_MODEL_PATH", "") or ""),
    )

    # --- Armes ---
    append_row(
        "weapon-detector",
        f"Détection armes (fusion) — {Path(str(getattr(settings, 'WEAPON_DETECTOR_PATH', '') or '')).name}",
        "Local (Ultralytics)",
        str(getattr(settings, "WEAPON_DETECTOR_PATH", "") or ""),
    )
    append_row(
        "weapon-test",
        f"Weapon test — {getattr(settings, 'WEAPON_TEST_MODEL_PATH', '')}",
        "Local (Ultralytics / hub)",
        str(getattr(settings, "WEAPON_TEST_MODEL_PATH", "") or ""),
    )
    append_row(
        "weapon-gun",
        f"Gun test — {Path(str(getattr(settings, 'WEAPON_GUN_TEST_MODEL_PATH', '') or '')).name}",
        "Local (Ultralytics)",
        str(getattr(settings, "WEAPON_GUN_TEST_MODEL_PATH", "") or ""),
    )

    # --- Route : segmentation + classification ---
    append_row(
        "road-damage-seg",
        f"Route (segmentation) — {Path(str(getattr(settings, 'ROAD_DAMAGE_MODEL_PATH', '') or '')).name}",
        "Local (Ultralytics)",
        str(getattr(settings, "ROAD_DAMAGE_MODEL_PATH", "") or ""),
    )
    rvm = getattr(settings, "ROAD_VISION_MODELS", {}) or {}
    rvl = getattr(settings, "ROAD_VISION_MODEL_LABELS", {}) or {}
    for key in getattr(settings, "ROAD_VISION_CLASSIFICATION_UI_ORDER", tuple(rvm.keys())):
        if key not in rvm:
            continue
        path_str = rvm[key]
        label = rvl.get(key, key)
        append_row(
            f"road-cls-{key}",
            str(label)[:80],
            "Local",
            str(path_str),
        )

    # --- Déchets ---
    wmp = getattr(settings, "WASTE_MODEL_PATH", "") or ""
    if str(wmp).strip():
        append_row(
            "waste-yolo",
            f"Détection déchets — {Path(str(wmp)).name}",
            "Local (Ultralytics)",
            str(wmp),
        )

    # --- UAV Keras ---
    uav = getattr(settings, "UAV_MODEL_PATH", "") or ""
    if str(uav).strip():
        append_row(
            "uav-cnn-keras",
            f"UAV structure — {Path(str(uav)).name}",
            "Local (Keras)",
            str(uav),
        )

    # --- Traffic Nexus ---
    tn = getattr(settings, "TRAFFIC_NEXUS_MODEL_PATHS", {}) or {}
    for t_key, t_path in tn.items():
        ps = (t_path or "").strip()
        if not ps:
            continue
        append_row(
            f"traffic-{t_key}",
            f"Traffic Nexus — {t_key} ({Path(ps).name})",
            "Local (Ultralytics)",
            ps,
        )

    # --- Groq (rapports route) ---
    groq_model = getattr(settings, "GROQ_MODEL", "") or "llama-3.3-70b-versatile"
    append_row(
        "groq-llm",
        str(groq_model),
        "Groq",
        groq_model,
        status_override=("active" if _groq_active() else "down"),
    )

    # --- Sightengine ---
    append_row(
        "sightengine-api",
        "Sightengine (vidéo / modération)",
        "Sightengine",
        "sightengine",
        status_override=("active" if _sightengine_active() else "down"),
    )

    # --- DeepFace / Facenet (fusion biométrie) ---
    face_name = getattr(settings, "FACE_EMBED_MODEL_NAME", "") or "Facenet"
    append_row(
        "face-embedding",
        f"Empreinte faciale — {face_name}",
        "DeepFace",
        f"deepface:{face_name}",
        status_override="active",
    )

    return rows


def _activity_from_project() -> list[dict[str, Any]]:
    groq = getattr(settings, "GROQ_MODEL", "") or "Groq LLM"
    fire_name = Path(str(getattr(settings, "FIRE_ONNX_MODEL_PATH", "") or "best.onnx")).name
    fight_name = Path(str(getattr(settings, "FIGHT_CLASSIFIER_PATH", "") or "fight.pt")).name
    return [
        {
            "time": (datetime.now() - timedelta(minutes=3)).strftime("%H:%M"),
            "type": "api",
            "severity": "info",
            "message": f"Pipeline feu/caméra — inférence ONNX ({fire_name})",
        },
        {
            "time": (datetime.now() - timedelta(minutes=8)).strftime("%H:%M"),
            "type": "api",
            "severity": "warn" if not _sightengine_active() else "info",
            "message": (
                "Sightengine — identifiants manquants ou quota"
                if not _sightengine_active()
                else "Sightengine — analyse vidéo terminée"
            ),
        },
        {
            "time": (datetime.now() - timedelta(minutes=15)).strftime("%H:%M"),
            "type": "api",
            "severity": "info",
            "message": f"Fusion hub — classif. combat ({fight_name})",
        },
        {
            "time": (datetime.now() - timedelta(minutes=24)).strftime("%H:%M"),
            "type": "user",
            "severity": "info",
            "message": "Réclamation IA — scoring vision route + agents Groq",
        },
        {
            "time": (datetime.now() - timedelta(minutes=36)).strftime("%H:%M"),
            "type": "api",
            "severity": "error" if not _groq_active() else "info",
            "message": (
                f"Groq indisponible — clé API absente ({groq})"
                if not _groq_active()
                else f"Groq — complétion modèle {groq}"
            ),
        },
        {
            "time": (datetime.now() - timedelta(minutes=50)).strftime("%H:%M"),
            "type": "api",
            "severity": "info",
            "message": "Traffic Nexus — inférence YOLO sélectionnée",
        },
    ]


def build_ai_dashboard_summary(
    *,
    model_filter: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Return KPIs, model rows, chart series, and activity — demo load; models from settings."""
    seed = _stable_seed(
        (
            (model_filter or "all"),
            (date_from or ""),
            (date_to or ""),
            date.today().isoformat(),
        )
    )
    rng = random.Random(seed)

    fleet_models = collect_project_ai_models(rng)

    models_raw = fleet_models
    if model_filter and model_filter != "all":
        models_raw = [m for m in fleet_models if m["id"] == model_filter]
        if not models_raw:
            models_raw = [
                {
                    "id": model_filter,
                    "name": model_filter,
                    "provider": "—",
                    "requests": 0,
                    "avg_ms": 0,
                    "error_rate": 0.0,
                    "status": "active",
                }
            ]

    filter_options = [{"id": m["id"], "name": m["name"]} for m in fleet_models]

    def jitter_pct(base: float, spread: float = 4.0) -> float:
        return round(base + rng.uniform(-spread, spread), 2)

    total_requests = sum(m["requests"] for m in models_raw)
    avg_ms = (
        sum(m["avg_ms"] * m["requests"] for m in models_raw) / total_requests
        if total_requests
        else 0
    )
    err_w = (
        sum(m["error_rate"] * m["requests"] for m in models_raw) / total_requests
        if total_requests
        else 0
    )
    success_rate = max(0.0, min(100.0, 100.0 - err_w))

    kpis = {
        "total_models": len(fleet_models),
        "total_requests": total_requests,
        "success_rate": round(success_rate, 2),
        "avg_response_ms": int(round(avg_ms)),
        "active_users": 280 + seed % 180,
        "revenue_usd": 8420 + (seed % 4200),
    }

    trends = {
        "total_models": {"delta_pct": jitter_pct(2.1, 3), "up": True},
        "total_requests": {"delta_pct": jitter_pct(8.4, 2), "up": True},
        "success_rate": {"delta_pct": jitter_pct(0.3, 0.8), "up": rng.random() > 0.3},
        "avg_response_ms": {"delta_pct": jitter_pct(-4.2, 2), "up": False},
        "active_users": {"delta_pct": jitter_pct(5.1, 4), "up": True},
        "revenue_usd": {"delta_pct": jitter_pct(12.0, 5), "up": True},
    }

    labels_day: list[str] = []
    vals_day: list[int] = []
    today = date.today()
    for i in range(13, -1, -1):
        d = today - timedelta(days=i)
        labels_day.append(d.strftime("%d %b"))
        base = 3200 + (seed % 800) + i * 120
        vals_day.append(int(base * (0.85 + rng.random() * 0.35)))

    bar_labels = [m["name"][:22] for m in models_raw]
    bar_vals = [m["requests"] for m in models_raw]

    pie_slices = [
        {"label": m["name"][:28], "value": m["requests"]}
        for m in models_raw
        if m["requests"] > 0
    ]

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filters": {
            "model": model_filter or "all",
            "date_from": date_from,
            "date_to": date_to,
        },
        "kpis": kpis,
        "kpi_trends": trends,
        "models": models_raw,
        "charts": {
            "requests_over_time": {"labels": labels_day, "values": vals_day},
            "usage_per_model": {"labels": bar_labels, "values": bar_vals},
            "distribution": pie_slices,
        },
        "activity": _activity_from_project(),
        "filter_options": filter_options,
    }
