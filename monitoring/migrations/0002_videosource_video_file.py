import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("monitoring", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="videosource",
            name="video_file",
            field=models.FileField(
                blank=True,
                help_text="Téléversez un MP4 (ou MOV, AVI, WebM). Prioritaire sur le champ chemin pour le type « Fichier vidéo ».",
                null=True,
                upload_to="sources/videos/%Y/%m/",
                validators=[
                    django.core.validators.FileExtensionValidator(
                        allowed_extensions=["mp4", "mov", "avi", "webm", "mkv"]
                    )
                ],
                verbose_name="Fichier vidéo (PC)",
            ),
        ),
    ]
