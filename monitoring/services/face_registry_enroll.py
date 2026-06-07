"""Enroll a detected face crop into the Django face registry (identity + reference + embedding)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.files import File
from django.db import transaction
from django.utils.text import slugify

from monitoring.models import FaceIdentity, FacePersonProfile, FaceReferenceImage
from monitoring.services.face_biometric import _reference_image_media_url
from monitoring.services.face_recognition_gallery import compute_embedding_and_store


class FaceRegistryEnrollError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


def _media_path_from_public_url(url: str) -> Path | None:
    raw = (url or "").strip()
    if not raw:
        return None
    media_url = settings.MEDIA_URL.rstrip("/")
    if raw.startswith(media_url + "/") or raw == media_url:
        rel = raw[len(media_url) :].lstrip("/")
    elif raw.startswith("/media/"):
        rel = raw[len("/media/") :].lstrip("/")
    elif raw.startswith("media/"):
        rel = raw[len("media/") :].lstrip("/")
    else:
        return None
    base = Path(settings.MEDIA_ROOT).resolve()
    candidate = (base / rel.replace("\\", "/")).resolve()
    try:
        if not candidate.is_relative_to(base):
            return None
    except AttributeError:
        if base not in candidate.parents and candidate != base:
            return None
    return candidate if candidate.is_file() else None


def _unique_slug(display_name: str) -> str:
    base = slugify(display_name)[:80] or "person"
    slug = base
    n = 1
    while FaceIdentity.objects.filter(slug=slug).exists():
        slug = f"{base}-{n}"[:88]
        n += 1
    return slug


def _save_reference_image(
    face_identity: FaceIdentity,
    *,
    upload,
    disk_path: Path | None,
    is_primary: bool,
) -> FaceReferenceImage:
    ref = FaceReferenceImage(face_identity=face_identity, is_primary=is_primary)
    if upload is not None:
        name = getattr(upload, "name", "") or f"enrolled_{uuid.uuid4().hex[:8]}.jpg"
        ref.image.save(Path(name).name, upload, save=True)
    elif disk_path is not None:
        with disk_path.open("rb") as fh:
            ref.image.save(disk_path.name, File(fh), save=True)
    else:
        raise FaceRegistryEnrollError("No reference image provided.", 400)
    compute_embedding_and_store(int(ref.pk))
    ref.refresh_from_db(fields=["embedding", "embedding_error", "embedding_model"])
    if not ref.embedding:
        err = (ref.embedding_error or "embedding failed").strip()
        raise FaceRegistryEnrollError(f"Reference saved but embedding failed: {err}", 422)
    return ref


def enroll_face_to_registry(
    *,
    display_name: str,
    person_code: str,
    category: str = "",
    cropped_upload=None,
    reference_upload=None,
    crop_media_url: str = "",
) -> dict[str, Any]:
    name = (display_name or "").strip()
    code = (person_code or "").strip()
    cat = (category or "").strip()

    if not name:
        raise FaceRegistryEnrollError("Full name is required.", 400)
    if not code:
        raise FaceRegistryEnrollError("Unique ID is required.", 400)
    if len(code) > 64:
        raise FaceRegistryEnrollError("Unique ID must be 64 characters or fewer.", 400)

    if FaceIdentity.objects.filter(person_code__iexact=code).exists():
        raise FaceRegistryEnrollError("Unique ID already exists in the face registry.", 409)
    if FacePersonProfile.objects.filter(identity_code__iexact=code).exists():
        raise FaceRegistryEnrollError("Unique ID already exists in person profiles.", 409)

    crop_path = _media_path_from_public_url(crop_media_url)
    if cropped_upload is None and reference_upload is None and crop_path is None:
        raise FaceRegistryEnrollError(
            "Provide a cropped face image, reference upload, or valid crop URL.",
            400,
        )

    with transaction.atomic():
        fi = FaceIdentity.objects.create(
            slug=_unique_slug(name),
            display_name=name,
            person_code=code,
            category=cat or "General",
            notes="Enrolled from Surveillance / City Monitoring",
            is_active=True,
        )

        primary_ref = None
        if cropped_upload is not None:
            primary_ref = _save_reference_image(
                fi, upload=cropped_upload, disk_path=None, is_primary=True
            )
        elif crop_path is not None:
            primary_ref = _save_reference_image(
                fi, upload=None, disk_path=crop_path, is_primary=True
            )
        elif reference_upload is not None:
            primary_ref = _save_reference_image(
                fi, upload=reference_upload, disk_path=None, is_primary=True
            )

        if reference_upload is not None and cropped_upload is not None:
            _save_reference_image(
                fi, upload=reference_upload, disk_path=None, is_primary=False
            )

        profile, _ = FacePersonProfile.objects.update_or_create(
            identity_code=code,
            defaults={
                "full_name": name,
                "role": cat or "",
                "face_identity": fi,
            },
        )

    ref_pk = int(primary_ref.pk) if primary_ref else None
    ref_url = _reference_image_media_url(ref_pk) if ref_pk else ""
    photo_fields = {
        "reference_photo_url": ref_url,
        "reference_image_url": ref_url,
        "detected_crop_url": (crop_media_url or "").strip(),
    }

    match_out = {
        "identity_id": fi.pk,
        "reference_id": ref_pk,
        "display_name": name,
        "person_code": code,
        "category": cat or fi.category,
        "distance": 0.0,
        "confidence_pct": 100,
        **photo_fields,
        "card_display_name": name,
        "admin_add_facepersonprofile_url": "",
    }

    return {
        "success": True,
        "message": "Identity enrolled successfully",
        "identity": {
            "identity_id": fi.pk,
            "reference_id": ref_pk,
            "display_name": name,
            "person_code": code,
            "category": fi.category,
            **photo_fields,
        },
        "face": {
            "matched": True,
            "match": match_out,
            "crop_url": crop_media_url,
            "detected_crop_url": (crop_media_url or "").strip(),
            "full_name": name,
            "identity_code": code,
            "category": fi.category,
            "match_confidence": 100.0,
            "distance": 0.0,
            "profile_extended": True,
            "profile": {
                "identity_code": code,
                "full_name": name,
                "role": profile.role or "",
            },
            **photo_fields,
            "card_display_name": name,
        },
    }
