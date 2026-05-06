"""
Seed identities BOXER-001 / BOXER-002, profils étendus, références depuis media/face_registry/*/.

Usage :
  python manage.py seed_boxers_faces
  python manage.py seed_boxers_faces --purge
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand
from django.utils.text import slugify

from monitoring.models import FaceIdentity, FacePersonProfile, FaceReferenceImage

BOXERS: tuple[dict[str, object], ...] = (
    {
        "code": "BOXER-001",
        "full_name": "Gervonta Davis",
        "slug_folder": "gervonta_davis",
        "category": "Boxer",
        "notes": "Test identity for face recognition demo",
        "profile_notes": "Test identity for demo — Fusion Hub boxer gallery.",
        "risk": "low",
    },
    {
        "code": "BOXER-002",
        "full_name": "Hector Garcia",
        "slug_folder": "hector_garcia",
        "category": "Boxer",
        "notes": "Test identity for face recognition demo",
        "profile_notes": "Test identity for demo — Fusion Hub boxer gallery.",
        "risk": "low",
    },
)


class Command(BaseCommand):
    help = (
        "Crée FaceIdentity + FacePersonProfile pour les boxeurs démo et charge les JPEG "
        "depuis media/face_registry/<dossier>/."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--purge",
            action="store_true",
            help="Supprime uniquement les données démo BOXER-001 et BOXER-002.",
        )

    def handle(self, *args, **options):
        media_root = Path(settings.MEDIA_ROOT)

        if options["purge"]:
            codes = ["BOXER-001", "BOXER-002"]
            np = FacePersonProfile.objects.filter(identity_code__in=codes).delete()[0]
            fis = FaceIdentity.objects.filter(person_code__in=codes)
            nr = FaceReferenceImage.objects.filter(face_identity__in=fis).count()
            ni = fis.delete()[0]
            self.stdout.write(
                self.style.WARNING(f"Purge BOXER demo : profils={np}, refs={nr}, identités={ni}")
            )
            return

        total_refs = 0

        for row in BOXERS:
            code = str(row["code"])
            name = str(row["full_name"])
            slug_base = slugify(str(row["slug_folder"]))[:88] or slugify(name)[:88]

            fi, _created = FaceIdentity.objects.update_or_create(
                person_code=code,
                defaults={
                    "slug": slug_base,
                    "display_name": name,
                    "category": str(row["category"]),
                    "notes": str(row["notes"]),
                    "is_active": True,
                },
            )

            FacePersonProfile.objects.update_or_create(
                identity_code=code,
                defaults={
                    "full_name": name,
                    "role": str(row["category"]),
                    "risk_level": str(row["risk"]),
                    "notes": str(row["profile_notes"]),
                    "face_identity": fi,
                    "extra_info": {"demo": True, "source": "seed_boxers_faces"},
                },
            )

            reg_dir = media_root / "face_registry" / slug_base
            reg_dir.mkdir(parents=True, exist_ok=True)

            existing_names = set(
                Path(r.image.name).name for r in FaceReferenceImage.objects.filter(face_identity=fi)
            )

            imgs = sorted(
                p
                for p in reg_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )

            if not imgs:
                self.stdout.write(
                    self.style.WARNING(
                        f"Aucune image dans {reg_dir} — ajoutez des JPG pour {code}, puis relancez la commande."
                    )
                )
                continue

            primary_done = FaceReferenceImage.objects.filter(face_identity=fi, is_primary=True).exists()

            for pth in imgs:
                if pth.name in existing_names:
                    continue
                with pth.open("rb") as fh:
                    ref = FaceReferenceImage(face_identity=fi, is_primary=not primary_done)
                    ref.image.save(pth.name, File(fh), save=False)
                    ref.save()
                primary_done = True
                existing_names.add(pth.name)
                total_refs += 1
                self.stdout.write(f"  + référence {code} ← {pth.name}")

        self.stdout.write(
            self.style.SUCCESS(
                f"Seed boxeurs terminé — {total_refs} nouvelle(s) référence(s) fichier (doublons ignorés)."
            )
        )
