"""Applique les préférences caméra « qualité max » (UXGA + JPEG fin, etc.)."""

from django.db import migrations


def apply_quality_prefs(apps, schema_editor):
    FireCameraConfig = apps.get_model("monitoring", "FireCameraConfig")
    prefs = {
        "framesize": 13,
        "quality": 4,
        "brightness": 0,
        "contrast": 1,
        "saturation": 1,
        "sharpness": 2,
        "awb": True,
        "awb_gain": True,
        "wb_mode": 0,
        "aec": True,
        "aec2": True,
        "ae_level": 0,
        "agc": True,
        "gainceiling": 2,
        "bpc": True,
        "wpc": True,
        "raw_gma": True,
        "lenc": True,
        "dcw": True,
        "hmirror": False,
        "vflip": False,
        "special_effect": 0,
        "led_intensity": 0,
    }
    for row in FireCameraConfig.objects.all():
        row.ui_prefs = prefs
        row.save(update_fields=["ui_prefs"])


def noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):
    dependencies = [
        ("monitoring", "0011_fire_camera_config"),
    ]

    operations = [
        migrations.RunPython(apply_quality_prefs, noop_reverse),
    ]
