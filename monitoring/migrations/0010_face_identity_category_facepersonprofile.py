# Manually authored for FaceIdentity.category + FacePersonProfile

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("monitoring", "0009_face_identity_registry"),
    ]

    operations = [
        migrations.AddField(
            model_name="faceidentity",
            name="category",
            field=models.CharField(
                blank=True,
                default="",
                help_text="Ex. Boxer, Personnel, Visiteur.",
                max_length=120,
                verbose_name="Catégorie",
            ),
        ),
        migrations.AlterField(
            model_name="faceidentity",
            name="display_name",
            field=models.CharField(
                help_text="Pour l’UI : équivalent au « full name » métier.",
                max_length=160,
                verbose_name="Nom affiché (nom complet)",
            ),
        ),
        migrations.AlterField(
            model_name="faceidentity",
            name="person_code",
            field=models.CharField(
                blank=True,
                db_index=True,
                default="",
                help_text="Identifiant métier unique (ex. BOXER-001) — utilisé pour le profil étendu.",
                max_length=64,
                verbose_name="Code unique / matricule",
            ),
        ),
        migrations.CreateModel(
            name="FacePersonProfile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "identity_code",
                    models.CharField(
                        db_index=True,
                        help_text="Identique à FaceIdentity.person_code pour la jointure après match.",
                        max_length=64,
                        unique=True,
                        verbose_name="Code identité (unique)",
                    ),
                ),
                ("full_name", models.CharField(blank=True, default="", max_length=200, verbose_name="Nom complet")),
                ("age", models.PositiveSmallIntegerField(blank=True, null=True, verbose_name="Âge")),
                ("role", models.CharField(blank=True, default="", max_length=160, verbose_name="Rôle / fonction")),
                ("risk_level", models.CharField(blank=True, default="", max_length=64, verbose_name="Niveau de risque")),
                ("notes", models.TextField(blank=True, default="", verbose_name="Notes profil")),
                ("last_seen", models.DateTimeField(blank=True, null=True, verbose_name="Dernière vue")),
                ("extra_info", models.JSONField(blank=True, null=True, verbose_name="Métadonnées")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "face_identity",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="person_profiles",
                        to="monitoring.faceidentity",
                        verbose_name="Identité liée (optionnel)",
                    ),
                ),
            ],
            options={
                "verbose_name": "Profil personne (visage)",
                "verbose_name_plural": "Profils personnes (visage)",
                "ordering": ["identity_code"],
            },
        ),
    ]
