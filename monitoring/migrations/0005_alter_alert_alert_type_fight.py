from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("monitoring", "0004_alter_videosource_fast_movement_threshold"),
    ]

    operations = [
        migrations.AlterField(
            model_name="alert",
            name="alert_type",
            field=models.CharField(
                choices=[
                    ("crowd", "Foule dense"),
                    ("fast_movement", "Mouvement rapide / dispersion"),
                    ("high_overlap", "Regroupement serré (piste bagarre)"),
                    (
                        "theft_suspicious",
                        "Piste vol (sac / objet — changement de proximité)",
                    ),
                    ("fight", "Combat détecté (analyse vidéo)"),
                    ("system", "Système"),
                ],
                max_length=32,
                verbose_name="Type",
            ),
        ),
    ]
