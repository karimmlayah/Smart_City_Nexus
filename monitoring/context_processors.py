"""Template context for MedinaMind platform."""
from __future__ import annotations


def _resolve_theme_mode(raw: str) -> str:
    mode = (raw or "blue-dark").strip().lower()
    if mode == "light":
        return "light"
    if mode == "dark":
        return "dark"
    if mode == "system":
        return "system"
    return "blue-dark"


def platform_theme(request):
    """Expose platform theme from SiteConfiguration (singleton)."""
    try:
        from monitoring.services.site_configuration_service import ensure_site_configuration

        cfg = ensure_site_configuration()
        mode = _resolve_theme_mode(cfg.theme_mode)
    except Exception:
        mode = "blue-dark"

    return {
        "platform_theme_mode": mode,
        "platform_theme_default": "blue-dark" if mode == "system" else mode,
    }
