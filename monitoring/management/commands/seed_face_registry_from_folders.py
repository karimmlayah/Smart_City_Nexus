"""
Importe chaque sous-dossier de media/face_registry/ en FaceIdentity + FaceReferenceImage
avec embeddings (DeepFace). Les images sur disque hors DB n’apparaissent pas à la galerie
tant que cette commande (ou l’admin) n’a pas créé les lignes Django.

Usage:
  python manage.py seed_face_registry_from_folders
  python manage.py seed_face_registry_from_folders --dry-run
  python manage.py seed_face_registry_from_folders --repair-embeddings
"""

from __future__ import annotations

import re
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.text import slugify

from monitoring.models import FaceIdentity, FaceReferenceImage
from monitoring.services.face_recognition_gallery import compute_embedding_and_store


IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def display_name_from_folder(folder_name: str) -> str:
    """« Naoya Inoue », « naoya_inoue », « naoya-inoue » → nom affiché lisible."""
    s = (folder_name or "").strip()
    if not s:
        return "Unknown"
    if " " in s:
        return " ".join(s.split())
    if "_" in s or "-" in s:
        return s.replace("_", " ").replace("-", " ").strip().title()
    return s


def slug_from_folder_name(folder_name: str) -> str:
    return slugify(folder_name.strip())[:88] or "person"


def next_person_code() -> str:
    highest = 0
    for code in FaceIdentity.objects.values_list("person_code", flat=True):
        if not code:
            continue
        m = re.match(r"^PERSON-(\d{1,6})$", str(code).strip(), re.I)
        if m:
            highest = max(highest, int(m.group(1)))
    return f"PERSON-{highest + 1:03d}"


class Command(BaseCommand):
    help = "Scan media/face_registry/<dossier>/ → FaceIdentity + FaceReferenceImage + embeddings."

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Affiche ce qui serait fait sans écrire en base.",
        )
        parser.add_argument(
            "--repair-embeddings",
            action="store_true",
            help="Recalcule les embeddings manquants pour toutes les FaceReferenceImage.",
        )

    def handle(self, *args, **options):
        media_root = Path(settings.MEDIA_ROOT)
        reg_root = media_root / "face_registry"
        dry = options["dry_run"]
        repair = options["repair_embeddings"]

        if repair:
            self._repair_embeddings(dry)
            return

        if not reg_root.is_dir():
            self.stdout.write(self.style.ERROR(f"Missing directory: {reg_root}"))
            return

        subdirs = sorted(
            p
            for p in reg_root.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name not in ("__pycache__",)
        )

        if not subdirs:
            self.stdout.write(self.style.WARNING(f"No subfolders under {reg_root}"))
            return

        self.stdout.write(self.style.NOTICE(f"Registry root: {reg_root}"))
        total_new_refs = 0

        for folder in subdirs:
            folder_name = folder.name
            display_name = display_name_from_folder(folder_name)
            slug = slug_from_folder_name(folder_name)

            imgs = sorted(
                p
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXT
            )

            self.stdout.write("")
            self.stdout.write(self.style.NOTICE(f"Folder: {folder_name!r} → slug={slug!r} display={display_name!r}"))
            self.stdout.write(f"  images found: {len(imgs)} {[p.name for p in imgs[:8]]}{'…' if len(imgs) > 8 else ''}")

            if not imgs:
                self.stdout.write(self.style.WARNING("  skip (no images)"))
                continue

            if dry:
                self.stdout.write(self.style.WARNING("  dry-run: no DB writes"))
                continue

            with transaction.atomic():
                fi, created = FaceIdentity.objects.get_or_create(
                    slug=slug,
                    defaults={
                        "display_name": display_name,
                        "person_code": next_person_code(),
                        "category": "Demo",
                        "notes": f"Imported from media/face_registry/{folder_name}",
                        "is_active": True,
                    },
                )
                if not created:
                    FaceIdentity.objects.filter(pk=fi.pk).update(
                        is_active=True,
                    )
                    if not (fi.display_name or "").strip():
                        FaceIdentity.objects.filter(pk=fi.pk).update(display_name=display_name)

                existing = set(
                    Path(r.image.name).name.split("/")[-1]
                    for r in FaceReferenceImage.objects.filter(face_identity=fi)
                )

                primary_done = FaceReferenceImage.objects.filter(
                    face_identity=fi, is_primary=True
                ).exists()

                for pth in imgs:
                    if pth.name in existing:
                        self.stdout.write(f"  skip duplicate filename: {pth.name}")
                        continue
                    with pth.open("rb") as fh:
                        ref = FaceReferenceImage(
                            face_identity=fi,
                            is_primary=not primary_done,
                        )
                        ref.image.save(pth.name, File(fh), save=True)
                    primary_done = True
                    existing.add(pth.name)
                    total_new_refs += 1

                    ok = compute_embedding_and_store(int(ref.pk))
                    ref.refresh_from_db(fields=["embedding", "embedding_error", "embedding_model"])
                    if ok and ref.embedding:
                        self.stdout.write(self.style.SUCCESS(f"  embedding OK: {pth.name} (pk={ref.pk})"))
                    else:
                        err = (ref.embedding_error or "unknown").strip()
                        self.stdout.write(
                            self.style.ERROR(f"  embedding FAILED: {pth.name} pk={ref.pk} err={err}")
                        )

        if dry:
            return

        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Done. New reference rows created this run: {total_new_refs}. "
                f"Run: python manage.py debug_face_registry"
            )
        )

    def _repair_embeddings(self, dry: bool) -> None:
        missing = [
            r.pk
            for r in FaceReferenceImage.objects.iterator(chunk_size=200)
            if not r.embedding or (isinstance(r.embedding, list) and len(r.embedding) < 16)
        ]
        self.stdout.write(self.style.NOTICE(f"Repair embeddings: {len(missing)} candidate(s)"))
        if dry:
            return
        for pk in missing:
            ok = compute_embedding_and_store(int(pk))
            ref = FaceReferenceImage.objects.filter(pk=pk).first()
            err = (ref.embedding_error if ref else "") or ""
            if ok:
                self.stdout.write(self.style.SUCCESS(f"  pk={pk} OK"))
            else:
                self.stdout.write(self.style.ERROR(f"  pk={pk} FAIL {err[:200]}"))
