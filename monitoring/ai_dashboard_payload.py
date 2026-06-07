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


_TRAFFIC_DISPLAY: dict[str, str] = {
    "yolov8n": "Traffic Monitoring",
    "yolov8m": "Traffic Monitoring Pro",
    "yolo_congestion": "Congestion Detection",
    "best": "Traffic Analysis",
}

_ROAD_VISION_DISPLAY: dict[str, str] = {
    "yolov8n_100ep": "Road Damage Detection",
    "rtdetr": "Road Damage Detection",
    "model_road_damage": "Road Damage Detection",
}


def _traffic_service_name(key: str, path: str) -> str:
    return _TRAFFIC_DISPLAY.get(key, f"Traffic Analysis ({key})")


def _road_vision_display(key: str, fallback: str) -> str:
    if key in _ROAD_VISION_DISPLAY:
        return _ROAD_VISION_DISPLAY[key]
    low = (fallback or key).lower()
    if "road" in low or "damage" in low or "rtdetr" in low:
        return "Road Damage Detection"
    return fallback or "Road Analysis"


def collect_project_ai_models(rng: random.Random) -> list[dict[str, Any]]:
    """
    One row per configured AI role in Django settings.
    Each row includes ``domain`` for semantic chart colors.
    """
    rows: list[dict[str, Any]] = []

    def append_row(
        mid: str,
        title: str,
        provider: str,
        path_or_hub: str | None,
        *,
        domain: str = "vision",
        status_override: str | None = None,
    ) -> None:
        ref = (path_or_hub or "").strip()
        if not ref and status_override is None:
            return
        if any(r["id"] == mid for r in rows):
            return

        if status_override:
            status = status_override
        elif provider in ("Cloud AI", "Groq"):
            status = "active" if _groq_active() else "down"
        elif provider in ("Cloud Service", "Sightengine"):
            status = "active" if _sightengine_active() else "down"
        else:
            status = "active" if _local_model_active(ref) else "down"

        req, avg_ms, err = _demo_metrics(mid, rng)
        if status == "down" and provider not in ("Cloud AI", "Cloud Service", "Groq", "Sightengine"):
            err = min(5.0, err + 2.5)

        rows.append(
            {
                "id": mid,
                "name": title,
                "provider": provider,
                "domain": domain,
                "requests": req,
                "avg_ms": avg_ms,
                "error_rate": err,
                "status": status,
            }
        )

    # --- Fire & smoke ---
    append_row(
        "fire-onnx",
        "Fire and Smoke Detection",
        "On-device",
        str(getattr(settings, "FIRE_ONNX_MODEL_PATH", "") or ""),
        domain="safety",
    )

    # --- Crowd / video monitoring ---
    append_row(
        "yolo-surveillance",
        "City Monitoring",
        "On-device",
        str(getattr(settings, "YOLO_MODEL_PATH", "") or ""),
        domain="safety",
    )

    # --- Violence detection ---
    append_row(
        "fight-classifier",
        "Violence Detection",
        "On-device",
        str(getattr(settings, "FIGHT_CLASSIFIER_PATH", "") or ""),
        domain="safety",
    )
    append_row(
        "fight-person-yolo",
        "People Detection",
        "On-device",
        str(getattr(settings, "FIGHT_PERSON_MODEL_PATH", "") or ""),
        domain="safety",
    )

    # --- Security ---
    append_row(
        "weapon-detector",
        "Weapon Detection",
        "On-device",
        str(getattr(settings, "WEAPON_DETECTOR_PATH", "") or ""),
        domain="safety",
    )
    append_row(
        "weapon-test",
        "Security Object Detection",
        "On-device",
        str(getattr(settings, "WEAPON_TEST_MODEL_PATH", "") or ""),
        domain="safety",
    )
    append_row(
        "weapon-gun",
        "Gun Detection",
        "On-device",
        str(getattr(settings, "WEAPON_GUN_TEST_MODEL_PATH", "") or ""),
        domain="safety",
    )

    # --- Road analysis ---
    append_row(
        "road-damage-seg",
        "Road Damage Detection",
        "On-device",
        str(getattr(settings, "ROAD_DAMAGE_MODEL_PATH", "") or ""),
        domain="infra",
    )
    rvm = getattr(settings, "ROAD_VISION_MODELS", {}) or {}
    rvl = getattr(settings, "ROAD_VISION_MODEL_LABELS", {}) or {}
    for key in getattr(settings, "ROAD_VISION_CLASSIFICATION_UI_ORDER", tuple(rvm.keys())):
        if key not in rvm:
            continue
        path_str = rvm[key]
        label = _road_vision_display(key, rvl.get(key, key))
        append_row(
            f"road-cls-{key}",
            str(label)[:80],
            "On-device",
            str(path_str),
            domain="infra",
        )

    # --- Waste ---
    wmp = getattr(settings, "WASTE_MODEL_PATH", "") or ""
    if str(wmp).strip():
        append_row(
            "waste-yolo",
            "Waste Detection",
            "On-device",
            str(wmp),
            domain="environment",
        )

    # --- Drone ---
    uav = getattr(settings, "UAV_MODEL_PATH", "") or ""
    if str(uav).strip():
        append_row(
            "uav-cnn-keras",
            "Drone Structure Analysis",
            "On-device",
            str(uav),
            domain="aerial",
        )

    # --- Traffic ---
    tn = getattr(settings, "TRAFFIC_NEXUS_MODEL_PATHS", {}) or {}
    for t_key, t_path in tn.items():
        ps = (t_path or "").strip()
        if not ps:
            continue
        append_row(
            f"traffic-{t_key}",
            _traffic_service_name(t_key, ps),
            "On-device",
            ps,
            domain="traffic",
        )

    # --- AI assistant (Groq) ---
    groq_model = getattr(settings, "GROQ_MODEL", "") or "llama-3.3-70b-versatile"
    append_row(
        "groq-llm",
        "AI Assistant",
        "Cloud AI",
        groq_model,
        domain="cloud_api",
        status_override=("active" if _groq_active() else "down"),
    )

    # --- Video safety ---
    append_row(
        "sightengine-api",
        "Video Safety Analysis",
        "Cloud Service",
        "sightengine",
        domain="cloud_api",
        status_override=("active" if _sightengine_active() else "down"),
    )

    # --- Face recognition ---
    face_name = getattr(settings, "FACE_EMBED_MODEL_NAME", "") or "Facenet"
    append_row(
        "face-embedding",
        "Face Recognition",
        "On-device",
        f"deepface:{face_name}",
        domain="biometric",
        status_override="active",
    )

    return rows


def semantic_palette() -> dict[str, dict[str, str]]:
    """
    Couleurs sémantiques Smart City (domaine métier → teinte stable dans les graphes).
    """
    return {
        "safety": {
            "label": "Safety & surveillance",
            "fill": "rgba(6, 182, 212, 0.78)",
            "stroke": "#06b6d4",
        },
        "infra": {
            "label": "Civil infrastructure",
            "fill": "rgba(99, 102, 241, 0.78)",
            "stroke": "#6366f1",
        },
        "environment": {
            "label": "Environment & waste",
            "fill": "rgba(16, 185, 129, 0.78)",
            "stroke": "#10b981",
        },
        "aerial": {
            "label": "Aerial / UAV",
            "fill": "rgba(59, 130, 246, 0.78)",
            "stroke": "#3b82f6",
        },
        "traffic": {
            "label": "Traffic intelligence",
            "fill": "rgba(245, 158, 11, 0.78)",
            "stroke": "#f59e0b",
        },
        "cloud_api": {
            "label": "Cloud APIs",
            "fill": "rgba(168, 85, 247, 0.78)",
            "stroke": "#a855f7",
        },
        "biometric": {
            "label": "Biometrics",
            "fill": "rgba(244, 114, 182, 0.78)",
            "stroke": "#f472b6",
        },
        "vision": {
            "label": "Computer vision",
            "fill": "rgba(56, 189, 248, 0.78)",
            "stroke": "#38bdf8",
        },
    }


def _parse_date_arg(raw: str | None) -> date | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _requests_over_time_series(
    seed: int,
    rng: random.Random,
    date_from: date | None,
    date_to: date | None,
) -> tuple[list[str], list[int]]:
    """Série temporelle alignée sur la fenêtre de filtres (ou 14 derniers jours)."""
    today = date.today()
    if date_from and date_to and date_from <= date_to:
        span = (date_to - date_from).days + 1
        span = max(1, min(span, 90))
        labels: list[str] = []
        vals: list[int] = []
        for i in range(span):
            d = date_from + timedelta(days=i)
            labels.append(d.strftime("%d %b"))
            base = 2600 + (seed % 700) + i * 88
            vals.append(int(base * (0.82 + rng.random() * 0.38)))
        return labels, vals

    labels_day: list[str] = []
    vals_day: list[int] = []
    for i in range(13, -1, -1):
        d = today - timedelta(days=i)
        labels_day.append(d.strftime("%d %b"))
        base = 3200 + (seed % 800) + i * 120
        vals_day.append(int(base * (0.85 + rng.random() * 0.35)))
    return labels_day, vals_day


def _activity_from_project() -> list[dict[str, Any]]:
    return [
        {
            "time": (datetime.now() - timedelta(minutes=3)).strftime("%H:%M"),
            "type": "system",
            "severity": "info",
            "message": "Fire and smoke detection updated",
        },
        {
            "time": (datetime.now() - timedelta(minutes=8)).strftime("%H:%M"),
            "type": "system",
            "severity": "warn" if not _sightengine_active() else "info",
            "message": (
                "Video safety service needs configuration"
                if not _sightengine_active()
                else "Video analysis completed"
            ),
        },
        {
            "time": (datetime.now() - timedelta(minutes=15)).strftime("%H:%M"),
            "type": "system",
            "severity": "info",
            "message": "Violence detection model checked",
        },
        {
            "time": (datetime.now() - timedelta(minutes=24)).strftime("%H:%M"),
            "type": "user",
            "severity": "info",
            "message": "Road report processed",
        },
        {
            "time": (datetime.now() - timedelta(minutes=36)).strftime("%H:%M"),
            "type": "system",
            "severity": "error" if not _groq_active() else "info",
            "message": (
                "AI assistant unavailable — check configuration"
                if not _groq_active()
                else "AI assistant response generated"
            ),
        },
        {
            "time": (datetime.now() - timedelta(minutes=50)).strftime("%H:%M"),
            "type": "system",
            "severity": "info",
            "message": "Traffic analysis selected",
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
                    "domain": "vision",
                    "requests": 0,
                    "avg_ms": 0,
                    "error_rate": 0.0,
                    "status": "active",
                }
            ]

    filter_options = [{"id": m["id"], "name": m["name"]} for m in fleet_models]

    def jitter_pct(base: float, spread: float = 4.0) -> float:
        return round(base + rng.uniform(-spread, spread), 2)

    df = _parse_date_arg(date_from)
    dt = _parse_date_arg(date_to)

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

    online_n = sum(1 for m in fleet_models if m.get("status") == "active")
    offline_n = max(0, len(fleet_models) - online_n)

    kpis = {
        "total_models": len(fleet_models),
        "total_requests": total_requests,
        "success_rate": round(success_rate, 2),
        "avg_response_ms": int(round(avg_ms)),
        "online_services": online_n,
        "offline_services": offline_n,
    }

    trends = {
        "total_models": {"delta_pct": jitter_pct(0.4, 1.2), "up": rng.random() > 0.45},
        "total_requests": {"delta_pct": jitter_pct(8.4, 2), "up": True},
        "success_rate": {"delta_pct": jitter_pct(0.3, 0.8), "up": rng.random() > 0.3},
        "avg_response_ms": {"delta_pct": jitter_pct(-4.2, 2), "up": False},
        "online_services": {"delta_pct": jitter_pct(1.8, 2.5), "up": rng.random() > 0.35},
        "offline_services": {"delta_pct": jitter_pct(-2.5, 3.0), "up": rng.random() > 0.55},
    }

    labels_day, vals_day = _requests_over_time_series(seed, rng, df, dt)

    bar_labels = [m["name"][:22] for m in models_raw]
    bar_vals = [m["requests"] for m in models_raw]
    bar_domains = [str(m.get("domain") or "vision") for m in models_raw]

    pie_slices = [
        {
            "label": m["name"][:28],
            "value": m["requests"],
            "domain": str(m.get("domain") or "vision"),
        }
        for m in models_raw
        if m["requests"] > 0
    ]

    auto_refresh = int(getattr(settings, "AI_DASHBOARD_AUTO_REFRESH_SECONDS", 45) or 45)

    ui = {
        "hero_title": "MedinaMind Control Center",
        "hero_subtitle": "Monitor city safety, traffic, roads, waste, and alerts from one intelligent dashboard.",
        "nav_tag": "Urban Intelligence · Live Monitoring",
        "generated_label": "Last updated",
        "panels": {
            "line": {
                "kicker": "Activity",
                "title": "AI Activity Over Time",
                "hint": "Total platform activity for the selected period.",
            },
            "bar": {
                "kicker": "Usage",
                "title": "Activity by Service",
                "hint": "Compare activity across Smart City services.",
            },
            "pie": {
                "kicker": "Distribution",
                "title": "Service Usage Distribution",
                "hint": "Share of activity by service type.",
            },
            "table": {
                "kicker": "Services",
                "title": "AI Services",
                "hint": "All AI services connected to the MedinaMind platform.",
            },
            "feed": {
                "kicker": "Live Activity",
                "title": "Latest Platform Events",
                "hint": "Recent events across the platform.",
            },
        },
        "toolbar": {
            "model_label": "Service",
            "model_all": "All services",
            "from_label": "From",
            "to_label": "To",
            "auto_refresh": auto_refresh,
            "theme": "Theme",
            "refresh": "Refresh",
        },
        "table_headers": {
            "name": "Service",
            "provider": "Source",
            "requests": "Activity",
            "avg_ms": "Speed",
            "error_rate": "Errors",
            "status": "Status",
        },
        "status_labels": {"active": "Online", "down": "Offline"},
    }

    kpi_cards = [
        {
            "kpi": "total_models",
            "label": "Connected Services",
            "hint": "Services configured in the platform",
            "icon": "◎",
            "format": "int",
            "trend_key": "total_models",
            "trend_good": "neutral",
        },
        {
            "kpi": "total_requests",
            "label": "AI Activity",
            "hint": "Total AI activity during the selected period",
            "icon": "📡",
            "format": "int",
            "trend_key": "total_requests",
            "trend_good": "up",
        },
        {
            "kpi": "success_rate",
            "label": "System Reliability",
            "hint": "Overall system stability",
            "icon": "✓",
            "format": "pct",
            "trend_key": "success_rate",
            "trend_good": "up",
        },
        {
            "kpi": "avg_response_ms",
            "label": "Response Time",
            "hint": "Average response speed",
            "icon": "⏱",
            "format": "ms",
            "trend_key": "avg_response_ms",
            "trend_good": "down",
        },
        {
            "kpi": "online_services",
            "label": "Active Services",
            "hint": "Services currently available",
            "icon": "●",
            "format": "int",
            "trend_key": "online_services",
            "trend_good": "up",
        },
        {
            "kpi": "offline_services",
            "label": "Alerts",
            "hint": "Issues that require action",
            "icon": "⚠",
            "format": "int",
            "trend_key": "offline_services",
            "trend_good": "down",
        },
    ]

    chart_theme = {
        "requests_line": {
            "label": "Activity",
            "borderColor": "rgba(0, 213, 255, 0.92)",
            "backgroundColor": "rgba(0, 213, 255, 0.12)",
            "meaning": "Platform activity over time",
        },
    }

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "filters": {
            "model": model_filter or "all",
            "date_from": date_from,
            "date_to": date_to,
            "default_date_from": (df or (date.today() - timedelta(days=13))).isoformat(),
            "default_date_to": (dt or date.today()).isoformat(),
        },
        "ui": ui,
        "kpi_cards": kpi_cards,
        "semantic_palette": semantic_palette(),
        "chart_theme": chart_theme,
        "kpis": kpis,
        "kpi_trends": trends,
        "models": models_raw,
        "charts": {
            "requests_over_time": {"labels": labels_day, "values": vals_day},
            "usage_per_model": {
                "labels": bar_labels,
                "values": bar_vals,
                "domains": bar_domains,
            },
            "distribution": pie_slices,
        },
        "activity": _activity_from_project(),
        "filter_options": filter_options,
    }
