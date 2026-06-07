"""MedinaMind Platform Settings — configuration center views."""
from __future__ import annotations

from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods, require_POST

from monitoring.platform_settings_forms import PlatformSettingsForm
from monitoring.services.site_configuration_service import (
    ensure_site_configuration,
    model_statuses,
    platform_health,
    sync_fire_camera_ip,
    test_ai_models,
    test_email_connection,
    test_esp32_connection,
    test_twilio_connection,
)

HEALTH_LABELS = {
    "database": "Database",
    "ai_models": "AI Models",
    "esp32_camera": "ESP32 Camera",
    "email": "Email Service",
    "twilio": "Twilio Service",
    "storage": "Storage",
    "api": "API Status",
}


def _theme_cookie_value(mode: str) -> str:
    m = (mode or "blue-dark").strip().lower()
    if m in {"light", "dark", "blue-dark", "system"}:
        return m
    return "blue-dark"


def _redirect_with_theme(request, url: str, config):
    resp = redirect(url)
    resp.set_cookie(
        "medinamind-platform-theme",
        _theme_cookie_value(config.theme_mode),
        max_age=365 * 24 * 3600,
        samesite="Lax",
    )
    return resp


SETTINGS_SECTIONS = (
    {"id": "general", "label": "General Platform", "icon": "⚙"},
    {"id": "ai", "label": "AI Modules", "icon": "🧠"},
    {"id": "camera", "label": "Camera & ESP32", "icon": "📷"},
    {"id": "alerts", "label": "Alerts & Notifications", "icon": "🔔"},
    {"id": "reports", "label": "Reports", "icon": "📄"},
    {"id": "maps", "label": "Maps & Location", "icon": "🗺"},
    {"id": "dashboard", "label": "Dashboard Display", "icon": "📊"},
    {"id": "access", "label": "Users & Access", "icon": "👥"},
    {"id": "retention", "label": "Data Retention", "icon": "💾"},
    {"id": "health", "label": "System Health", "icon": "❤"},
)


@require_http_methods(["GET", "POST"])
def platform_settings_page(request):
    config = ensure_site_configuration()
    active_section = request.GET.get("section") or request.POST.get("active_section") or "general"

    if request.method == "POST":
        action = request.POST.get("action", "save")
        if action == "reset":
            config.reset_to_defaults()
            messages.success(request, "Settings restored to platform defaults.")
            return _redirect_with_theme(request, "monitoring:platform_settings", config)
        form = PlatformSettingsForm(request.POST, request.FILES, instance=config)
        if form.is_valid():
            form.save()
            sync_fire_camera_ip(config)
            messages.success(request, "Platform settings saved successfully.")
            return _redirect_with_theme(request, f"{request.path}?section={active_section}", config)
        messages.error(request, "Please correct the errors below.")
    else:
        form = PlatformSettingsForm(instance=config)

    health = platform_health(config)
    model_status = model_statuses(config)
    health_items = [(HEALTH_LABELS.get(k, k.replace("_", " ").title()), v) for k, v in health.items()]

    return render(
        request,
        "monitoring/platform_settings.html",
        {
            "form": form,
            "config": config,
            "sections": SETTINGS_SECTIONS,
            "active_section": active_section,
            "health": health,
            "health_items": health_items,
            "model_status": model_status,
        },
    )


def _json_test(result: dict) -> JsonResponse:
    return JsonResponse({"ok": result.get("ok", False), "message": result.get("detail") or result.get("label", "")})


@require_POST
def platform_settings_test_esp32(request):
    cfg = ensure_site_configuration()
    return _json_test(test_esp32_connection(cfg))


@require_POST
def platform_settings_test_email(request):
    cfg = ensure_site_configuration()
    return _json_test(test_email_connection(cfg))


@require_POST
def platform_settings_test_twilio(request):
    cfg = ensure_site_configuration()
    return _json_test(test_twilio_connection(cfg))


@require_POST
def platform_settings_test_models(request):
    cfg = ensure_site_configuration()
    return _json_test(test_ai_models(cfg))
