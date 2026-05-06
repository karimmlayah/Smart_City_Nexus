"""
Données fictives pour le registre visages (JPEG générés + embeddings DeepFace).
Usage : python manage.py seed_face_registry_demo [--purge]
"""

from __future__ import annotations

import random
from io import BytesIO

from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand

from monitoring.models import FaceIdentity, FaceReferenceImage


def slugify_fallback(s: str) -> str:
    from django.utils.text import slugify

    base = slugify(s)[:80]
    return base or "demo"


def _synthetic_rgb(seed: int) -> bytes:
    from PIL import Image, ImageDraw

    rng = random.Random(seed + 7919)
    w, h = 220, 220
    bg = tuple(int(rng.randint(170, 255)) for _ in range(3))
    skin = tuple(int(min(255, max(110, rng.randint(155, 240) + c // 40))) for c in bg)

    img = Image.new("RGB", (w, h), bg)
    draw = ImageDraw.Draw(img)
    oval = [int(w * 0.12), int(h * 0.08), int(w * 0.88), int(h * 0.92)]
    draw.ellipse(oval, fill=skin)
    lx, ly = int(w * 0.28), int(h * 0.42)
    draw.ellipse([lx, ly, lx + int(w * 0.09), ly + int(h * 0.08)], fill=(42, 44, 50))
    rx = int(w * 0.63)
    draw.ellipse([rx, ly, rx + int(w * 0.09), ly + int(h * 0.08)], fill=(42, 44, 50))
    draw.arc(
        [int(w * 0.38), int(h * 0.58), int(w * 0.62), int(h * 0.74)],
        0,
        180,
        fill=(140, 60, 60),
        width=3,
    )
    hx = rng.randint(42, min(148, max(52, lx - 8)))
    hy = rng.randint(118, min(146, ly + 72))
    draw.rectangle([hx, hy, hx + 14, hy + rng.randint(8, 26)], outline=(210, 30, 30), width=3)

    buff = BytesIO()
    img.save(buff, format="JPEG", quality=92)
    return buff.getvalue()


DEMO_ROWS: tuple[tuple[str, str, str], ...] = (
    ("EMP-DEMO-1001", "Camille Nguyen", "Registre test — aucune valeur légale"),
    ("EMP-DEMO-1002", "Mehdi Benali", ""),
    ("EMP-DEMO-1003", "Sonia Petrov", ""),
    ("EMP-DEMO-1004", "Lucas Martins", ""),
)


class Command(BaseCommand):
    help = (
        "Crée des identités faciales fictives sous media/face_registry/ avec visages JPEG synthétiques."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--purge",
            action="store_true",
            help="Supprime les identités dont person_code commence par EMP-DEMO-",
        )

    def handle(self, *args, **options):
        if options["purge"]:
            qs = FaceIdentity.objects.filter(person_code__startswith="EMP-DEMO")
            n = qs.count()
            qs.delete()
            self.stdout.write(self.style.WARNING(f"Purge EMP-DEMO : {n} identité(s)."))

        for i, (code, name, note) in enumerate(DEMO_ROWS):
            slug_seed = slugify_fallback(name)
            fi, created = FaceIdentity.objects.get_or_create(
                slug=slug_seed,
                defaults={
                    "display_name": name,
                    "person_code": code,
                    "notes": note or "(Données de démonstration Smart City)",
                },
            )
            if not created:
                fi.display_name = name
                fi.person_code = code
                if note:
                    fi.notes = note
                fi.is_active = True
                fi.save()

            jpeg = _synthetic_rgb(seed=9100 + i * 997)
            fn = f"demo_ref_{fi.slug}.jpg"
            ref, _ = FaceReferenceImage.objects.get_or_create(
                face_identity=fi,
                is_primary=True,
                defaults={},
            )
            ref.image.save(fn, ContentFile(jpeg), save=True)

        self.stdout.write(
            self.style.SUCCESS(
                f"OK — {len(DEMO_ROWS)} identité(s) démo (+ photos). DeepFace peut prendre quelques "
                "secondes au premier enregistrement (téléchargement des poids Facenet)."
            )
        )
