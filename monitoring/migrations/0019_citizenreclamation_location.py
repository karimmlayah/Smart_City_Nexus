from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("monitoring", "0018_site_configuration"),
    ]

    operations = [
        migrations.AddField(
            model_name="citizenreclamation",
            name="address",
            field=models.TextField(blank=True, default="", verbose_name="Adresse / zone"),
        ),
        migrations.AddField(
            model_name="citizenreclamation",
            name="location_source",
            field=models.CharField(
                choices=[("gps", "GPS"), ("manual", "Manual"), ("map", "Map")],
                default="gps",
                max_length=10,
                verbose_name="Source localisation",
            ),
        ),
        migrations.AlterField(
            model_name="citizenreclamation",
            name="latitude",
            field=models.FloatField(blank=True, null=True, verbose_name="Latitude"),
        ),
        migrations.AlterField(
            model_name="citizenreclamation",
            name="longitude",
            field=models.FloatField(blank=True, null=True, verbose_name="Longitude"),
        ),
    ]
