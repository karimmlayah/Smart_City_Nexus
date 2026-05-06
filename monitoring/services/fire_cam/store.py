from __future__ import annotations

from monitoring.models import FireCameraConfig


def get_fire_config() -> FireCameraConfig:
    obj, _ = FireCameraConfig.objects.get_or_create(
        pk=1,
        defaults={"esp_ip": "", "ui_prefs": {}},
    )
    return obj
