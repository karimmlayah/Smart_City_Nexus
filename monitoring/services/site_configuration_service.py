"""SiteConfiguration singleton helpers, defaults, health checks."""
from __future__ import annotations

import logging
from pathlib import Path

import requests
from django.conf import settings
from django.core.mail import get_connection
from django.db import connection

logger = logging.getLogger(__name__)


def _first_traffic_model() -> str:
    paths = getattr(settings, "TRAFFIC_NEXUS_MODEL_PATHS", {}) or {}
    for key in ("yolo", "vehicle", "default"):
        p = paths.get(key, "")
        if p and Path(p).is_file():
            return p
    for p in paths.values():
        if p and Path(p).is_file():
            return p
    return ""


def _first_road_model() -> str:
    models_map = getattr(settings, "ROAD_VISION_CLASSIFICATION_MODELS", {}) or {}
    default_key = getattr(settings, "ROAD_VISION_MODEL_DEFAULT_KEY", "")
    if default_key and models_map.get(default_key):
        return models_map[default_key]
    for p in models_map.values():
        if p:
            return p
    return ""


def apply_django_defaults(cfg) -> None:
    """Populate model fields from Django settings / env (does not save)."""
    cfg.platform_name = "MedinaMind"
    cfg.organization_name = "Smart City Operations"
    cfg.default_language = "en"
    cfg.theme_mode = "blue-dark"
    cfg.primary_accent_color = "#00d5ff"

    cfg.traffic_model_path = _first_traffic_model()
    cfg.road_damage_model_path = _first_road_model()
    cfg.waste_model_path = getattr(settings, "WASTE_MODEL_PATH", "") or ""
    cfg.fire_smoke_model_path = getattr(settings, "FIRE_ONNX_MODEL_PATH", "") or ""
    cfg.surveillance_model_path = getattr(settings, "FIGHT_CLASSIFIER_PATH", "") or ""
    cfg.uav_model_path = getattr(settings, "UAV_MODEL_PATH", "") or ""
    cfg.traffic_signal_model_path = getattr(settings, "TRAFFIC_SIGNAL_MODEL_PATH", "") or ""
    cfg.default_confidence_threshold = float(getattr(settings, "WASTE_YOLO_CONF", 0.25))
    cfg.default_iou_threshold = float(getattr(settings, "WASTE_YOLO_IOU", 0.45))

    try:
        from monitoring.services.fire_cam.store import get_fire_config

        fc = get_fire_config()
        cfg.esp32_cam_ip = (fc.esp_ip or "").strip()
    except Exception:
        cfg.esp32_cam_ip = ""

    cfg.stream_port = 81
    cfg.control_port = 80
    cfg.traffic_light_esp32_url = getattr(settings, "TRAFFIC_SIGNAL_ESP32_URL", "") or ""
    cfg.camera_refresh_interval = 5
    cfg.default_image_quality = 12
    cfg.live_camera_enabled = True
    cfg.auto_reconnect = True

    cfg.enable_email_alerts = bool(getattr(settings, "EMAIL_HOST", ""))
    cfg.enable_twilio_alerts = bool(getattr(settings, "TWILIO_ACCOUNT_SID", ""))
    cfg.municipality_email = getattr(settings, "WASTE_DEFAULT_MUNICIPALITY_EMAIL", "") or ""
    cfg.admin_alert_email = getattr(settings, "DEFAULT_FROM_EMAIL", "") or ""
    cfg.smtp_host = getattr(settings, "EMAIL_HOST", "") or ""
    cfg.smtp_port = int(getattr(settings, "EMAIL_PORT", 587) or 587)
    cfg.smtp_username = getattr(settings, "EMAIL_HOST_USER", "") or ""
    if not cfg.smtp_password:
        cfg.smtp_password = getattr(settings, "EMAIL_HOST_PASSWORD", "") or ""
    cfg.twilio_sid = getattr(settings, "TWILIO_ACCOUNT_SID", "") or ""
    if not cfg.twilio_auth_token:
        cfg.twilio_auth_token = getattr(settings, "TWILIO_AUTH_TOKEN", "") or ""
    cfg.twilio_phone_number = getattr(settings, "TWILIO_PHONE_NUMBER", "") or ""
    cfg.alert_delay_seconds = 2
    cfg.criticality_threshold = 0.75

    cfg.default_city = "Medina"
    cfg.default_latitude = 36.8065
    cfg.default_longitude = 10.1815
    cfg.default_zoom = 13


def ensure_site_configuration():
    from monitoring.models import SiteConfiguration

    cfg, created = SiteConfiguration.objects.get_or_create(pk=1)
    if created:
        apply_django_defaults(cfg)
        cfg.save()
    return cfg


def sync_fire_camera_ip(cfg) -> None:
    """Keep FireCameraConfig in sync when ESP32 IP changes."""
    ip = (cfg.esp32_cam_ip or "").strip()
    if not ip:
        return
    try:
        from monitoring.services.fire_cam.store import get_fire_config

        fc = get_fire_config()
        if fc.esp_ip != ip:
            fc.esp_ip = ip
            fc.save(update_fields=["esp_ip", "updated_at"])
    except Exception as exc:
        logger.warning("sync_fire_camera_ip failed: %s", exc)


def _model_file_ok(path: str) -> bool:
    return bool(path and Path(path).is_file())


def model_statuses(cfg) -> dict[str, str]:
    mapping = {
        "Traffic": cfg.traffic_model_path,
        "Road Damage": cfg.road_damage_model_path,
        "Waste": cfg.waste_model_path,
        "Fire & Smoke": cfg.fire_smoke_model_path,
        "Surveillance": cfg.surveillance_model_path,
        "UAV": cfg.uav_model_path,
        "Traffic Signal": cfg.traffic_signal_model_path,
    }
    out = {}
    for name, path in mapping.items():
        if not path:
            out[name] = "not configured"
        elif _model_file_ok(path):
            out[name] = "ready"
        else:
            out[name] = "missing"
    return out


def check_database() -> dict:
    try:
        connection.ensure_connection()
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
        return {"ok": True, "label": "Online", "detail": "Database reachable"}
    except Exception as exc:
        return {"ok": False, "label": "Offline", "detail": str(exc)}


def check_storage() -> dict:
    media = Path(getattr(settings, "MEDIA_ROOT", ""))
    try:
        media.mkdir(parents=True, exist_ok=True)
        test = media / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink(missing_ok=True)
        return {"ok": True, "label": "Ready", "detail": str(media)}
    except Exception as exc:
        return {"ok": False, "label": "Error", "detail": str(exc)}


def check_esp32(cfg) -> dict:
    ip = (cfg.esp32_cam_ip or "").strip()
    if not ip:
        return {"ok": False, "label": "Not set", "detail": "ESP32-CAM IP not configured"}
    url = f"http://{ip}:{cfg.control_port or 80}/"
    try:
        r = requests.get(url, timeout=4)
        return {"ok": r.status_code < 500, "label": "Reachable" if r.status_code < 500 else "Error", "detail": f"HTTP {r.status_code}"}
    except requests.RequestException as exc:
        return {"ok": False, "label": "Unreachable", "detail": str(exc)}


def check_email(cfg) -> dict:
    if not cfg.smtp_host:
        env_host = getattr(settings, "EMAIL_HOST", "")
        if not env_host:
            return {"ok": False, "label": "Not configured", "detail": "SMTP host missing"}
    try:
        conn = get_connection(
            host=cfg.smtp_host or getattr(settings, "EMAIL_HOST", ""),
            port=cfg.smtp_port or getattr(settings, "EMAIL_PORT", 587),
            username=cfg.smtp_username or getattr(settings, "EMAIL_HOST_USER", ""),
            password=cfg.smtp_password or getattr(settings, "EMAIL_HOST_PASSWORD", ""),
            use_tls=True,
        )
        conn.open()
        conn.close()
        return {"ok": True, "label": "Ready", "detail": "SMTP connection OK"}
    except Exception as exc:
        return {"ok": False, "label": "Failed", "detail": str(exc)}


def check_twilio(cfg) -> dict:
    sid = cfg.twilio_sid or getattr(settings, "TWILIO_ACCOUNT_SID", "")
    token = cfg.twilio_auth_token or getattr(settings, "TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        return {"ok": False, "label": "Not configured", "detail": "Twilio SID/token missing"}
    return {"ok": True, "label": "Configured", "detail": "Credentials present"}


def check_ai_models(cfg) -> dict:
    statuses = model_statuses(cfg)
    ready = sum(1 for v in statuses.values() if v == "ready")
    total = len(statuses)
    ok = ready > 0
    return {"ok": ok, "label": f"{ready}/{total} ready", "detail": ", ".join(f"{k}: {v}" for k, v in statuses.items())}


def platform_health(cfg) -> dict[str, dict]:
    return {
        "database": check_database(),
        "ai_models": check_ai_models(cfg),
        "esp32_camera": check_esp32(cfg),
        "email": check_email(cfg),
        "twilio": check_twilio(cfg),
        "storage": check_storage(),
        "api": {"ok": True, "label": "Online", "detail": "MedinaMind API responding"},
    }


def test_esp32_connection(cfg) -> dict:
    return check_esp32(cfg)


def test_email_connection(cfg) -> dict:
    return check_email(cfg)


def test_twilio_connection(cfg) -> dict:
    return check_twilio(cfg)


def test_ai_models(cfg) -> dict:
    return check_ai_models(cfg)
