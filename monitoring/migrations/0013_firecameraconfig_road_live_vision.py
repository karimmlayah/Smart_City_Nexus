# Generated migration for drone live YOLO defaults

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("monitoring", "0012_fire_camera_quality_defaults"),
    ]

    operations = [
        migrations.AddField(
            model_name="firecameraconfig",
            name="road_live_conf",
            field=models.FloatField(
                blank=True,
                help_text="Seuil 0–1 pour esp-analyze live ; vide = valeur Django RECLAMATION_VISION_CONF_DEFAULT.",
                null=True,
                verbose_name="Confiance seuil inference live ",
            ),
        ),
        migrations.AddField(
            model_name="firecameraconfig",
            name="road_live_model_key",
            field=models.CharField(
                blank=True,
                default="",
                help_text="Pour /reclamations/…/ai/?live=1 — clé registre ROAD_VISION_MODELS (ex. yolov8n_100ep).",
                max_length=96,
                verbose_name="Clé modèle road damage live (YOLO drone/ESP)",
            ),
        ),
    ]
