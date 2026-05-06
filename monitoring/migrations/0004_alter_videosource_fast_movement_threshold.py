from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("monitoring", "0003_alter_alert_alert_type"),
    ]

    operations = [
        migrations.AlterField(
            model_name="videosource",
            name="fast_movement_threshold",
            field=models.FloatField(
                default=18.0,
                help_text="Vitesse max (centre de la boîte) d’au moins une personne suivie ; au-delà → alerte course / mouvement brusque. Plus bas = plus sensible.",
                verbose_name="Seuil mouvement rapide (px/frame)",
            ),
        ),
    ]
