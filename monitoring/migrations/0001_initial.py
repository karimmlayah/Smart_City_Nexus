import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="VideoSource",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(max_length=120, verbose_name="Nom")),
                (
                    "source_type",
                    models.CharField(
                        choices=[("webcam", "Webcam"), ("rtsp", "Flux RTSP / IP"), ("file", "Fichier vidéo")],
                        default="webcam",
                        max_length=20,
                        verbose_name="Type",
                    ),
                ),
                ("url", models.CharField(blank=True, max_length=500, verbose_name="URL / chemin")),
                ("webcam_index", models.PositiveSmallIntegerField(default=0, verbose_name="Index webcam")),
                (
                    "crowd_threshold",
                    models.PositiveSmallIntegerField(
                        default=12,
                        help_text="Nombre de personnes détectées déclenchant une alerte foule.",
                        verbose_name="Seuil alerte foule (nb personnes)",
                    ),
                ),
                (
                    "fast_movement_threshold",
                    models.FloatField(
                        default=25.0,
                        help_text="Vitesse moyenne des boîtes suivies au-delà de laquelle une alerte est émise.",
                        verbose_name="Seuil mouvement rapide (px/frame)",
                    ),
                ),
                ("is_active", models.BooleanField(default=True, verbose_name="Actif")),
                ("notes", models.TextField(blank=True, verbose_name="Notes")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={
                "verbose_name": "Source vidéo",
                "verbose_name_plural": "Sources vidéo",
                "ordering": ["-is_active", "name"],
            },
        ),
        migrations.CreateModel(
            name="Alert",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "alert_type",
                    models.CharField(
                        choices=[
                            ("crowd", "Foule dense"),
                            ("fast_movement", "Mouvement rapide / dispersion"),
                            ("high_overlap", "Regroupement serré (piste bagarre)"),
                            ("system", "Système"),
                        ],
                        max_length=32,
                        verbose_name="Type",
                    ),
                ),
                ("message", models.CharField(max_length=500, verbose_name="Message")),
                ("person_count", models.PositiveIntegerField(blank=True, null=True, verbose_name="Personnes")),
                ("confidence_max", models.FloatField(blank=True, null=True, verbose_name="Confiance max")),
                ("metadata", models.JSONField(blank=True, default=dict, verbose_name="Métadonnées")),
                ("thumbnail", models.ImageField(blank=True, null=True, upload_to="alerts/%Y/%m/", verbose_name="Aperçu")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "source",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="alerts",
                        to="monitoring.videosource",
                        verbose_name="Source",
                    ),
                ),
            ],
            options={
                "verbose_name": "Alerte",
                "verbose_name_plural": "Alertes",
                "ordering": ["-created_at"],
            },
        ),
    ]
