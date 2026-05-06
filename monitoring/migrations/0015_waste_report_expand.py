# Migration : conserve les lignes existantes (rename table + nouvelles colonnes).

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("monitoring", "0014_waste_street_report"),
    ]

    operations = [
        migrations.AddField(
            model_name="wastestreetreport",
            name="city",
            field=models.CharField(blank=True, default="", max_length=160, verbose_name="Ville"),
        ),
        migrations.AddField(
            model_name="wastestreetreport",
            name="address",
            field=models.CharField(blank=True, default="", max_length=400, verbose_name="Adresse / zone"),
        ),
        migrations.AddField(
            model_name="wastestreetreport",
            name="latitude",
            field=models.FloatField(blank=True, null=True, verbose_name="Latitude"),
        ),
        migrations.AddField(
            model_name="wastestreetreport",
            name="longitude",
            field=models.FloatField(blank=True, null=True, verbose_name="Longitude"),
        ),
        migrations.AddField(
            model_name="wastestreetreport",
            name="severity",
            field=models.CharField(
                choices=[("Low", "Low"), ("Medium", "Medium"), ("High", "High")],
                default="Medium",
                max_length=16,
                verbose_name="Gravité",
            ),
        ),
        migrations.AddField(
            model_name="wastestreetreport",
            name="recommended_action",
            field=models.TextField(blank=True, default="", verbose_name="Action recommandée"),
        ),
        migrations.AddField(
            model_name="wastestreetreport",
            name="sensitive_area_boost",
            field=models.BooleanField(default=False, verbose_name="Zone sensible (+1 niveau)"),
        ),
        migrations.AddField(
            model_name="wastestreetreport",
            name="email_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("mock_sent", "Mock sent"),
                    ("sent", "Sent"),
                    ("failed", "Failed"),
                ],
                default="pending",
                max_length=24,
                verbose_name="Statut email",
            ),
        ),
        migrations.AddField(
            model_name="wastestreetreport",
            name="email_detail",
            field=models.TextField(blank=True, default="", verbose_name="Détail email / erreur"),
        ),
        migrations.AlterField(
            model_name="wastestreetreport",
            name="location",
            field=models.CharField(
                blank=True,
                default="",
                help_text="Souvent « ville, adresse » pour affichage rapide.",
                max_length=400,
                verbose_name="Localisation (résumé)",
            ),
        ),
        migrations.AlterField(
            model_name="wastestreetreport",
            name="max_confidence",
            field=models.FloatField(default=0.0, verbose_name="Confiance max (0–1)"),
        ),
        migrations.AlterField(
            model_name="wastestreetreport",
            name="sms_status",
            field=models.CharField(
                choices=[
                    ("pending", "Pending"),
                    ("mock_sent", "Mock sent"),
                    ("sent", "Sent"),
                    ("failed", "Failed"),
                ],
                default="pending",
                max_length=24,
                verbose_name="Statut SMS",
            ),
        ),
        migrations.RenameModel(
            old_name="WasteStreetReport",
            new_name="WasteReport",
        ),
        migrations.AlterModelOptions(
            name="wastereport",
            options={
                "ordering": ["-created_at"],
                "verbose_name": "Rapport déchets (Waste Report)",
                "verbose_name_plural": "Rapports déchets (Waste)",
            },
        ),
    ]
