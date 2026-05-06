"""
Print FaceIdentity / FaceReferenceImage counts and per-identity embedding stats.

Usage:
  python manage.py debug_face_registry
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from monitoring.models import FaceIdentity, FaceReferenceImage


def _embedding_ok(emb) -> bool:
    return bool(emb and isinstance(emb, list) and len(emb) >= 16)


class Command(BaseCommand):
    help = "Debug: counts, active identities, reference rows, embeddings per identity."

    def handle(self, *args, **options):
        media_root = Path(settings.MEDIA_ROOT)
        reg_root = media_root / "face_registry"

        n_id = FaceIdentity.objects.count()
        n_id_active = FaceIdentity.objects.filter(is_active=True).count()
        n_ref = FaceReferenceImage.objects.count()

        refs = list(FaceReferenceImage.objects.iterator(chunk_size=500))
        n_ref_emb = sum(1 for r in refs if _embedding_ok(r.embedding))

        self.stdout.write(self.style.NOTICE("=== Face registry debug ==="))
        self.stdout.write(f"MEDIA_ROOT: {media_root}")
        self.stdout.write(f"face_registry folder exists: {reg_root.is_dir()} → {reg_root}")
        self.stdout.write("")
        self.stdout.write(f"FaceIdentity total: {n_id}")
        self.stdout.write(f"FaceIdentity active: {n_id_active}")
        self.stdout.write(f"FaceReferenceImage total: {n_ref}")
        self.stdout.write(f"FaceReferenceImage with valid embedding (len>=16): {n_ref_emb}")
        self.stdout.write("")

        for fi in FaceIdentity.objects.order_by("pk"):
            imgs = list(fi.reference_images.all())
            n_img = len(imgs)
            n_emb = sum(1 for r in imgs if _embedding_ok(r.embedding))
            self.stdout.write(
                f"· pk={fi.pk} slug={fi.slug!r} name={fi.display_name!r} "
                f"code={fi.person_code!r} category={fi.category!r} active={fi.is_active} "
                f"images={n_img} embeddings_ok={n_emb}"
            )

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                "Tip: disk-only folders under media/face_registry/ are ignored until you run "
                "`python manage.py seed_face_registry_from_folders`."
            )
        )
