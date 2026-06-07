"""
Extraction multi-visages depuis une frame BGR, sauvegarde des crops et correspondance registre Django.
Complète le pipeline combat/armes sans le remplacer.

Les crops sont tirés uniquement du visage (DeepFace facial_area ou Haar), pas des boîtes « person ».
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


def _face_reference_image_url(ref) -> str | None:
    """Build public media URL for a FaceReferenceImage row."""
    if not ref or not ref.image or not ref.image.name:
        return None
    try:
        url = ref.image.url
        if not url:
            return None
        if not url.startswith(("http://", "https://", "/")):
            url = f"{settings.MEDIA_URL.rstrip('/')}/{ref.image.name.lstrip('/')}"
        return url
    except Exception:
        return None


def _reference_image_media_url(ref_pk: int) -> str | None:
    """URL publique de la photo de référence Django (par pk FaceReferenceImage)."""
    from monitoring.models import FaceReferenceImage

    ref = FaceReferenceImage.objects.filter(pk=int(ref_pk)).only("image").first()
    return _face_reference_image_url(ref) if ref else None


def _registry_reference_photo_url(
    identity_id: int,
    *,
    matched_reference_id: int | None = None,
) -> str | None:
    """
    Official registry/dossier photo for the Digital Identity Card.
    Prefers is_primary reference, then matched gallery row, then any reference.
    """
    from monitoring.models import FaceReferenceImage

    qs = FaceReferenceImage.objects.filter(face_identity_id=int(identity_id)).only(
        "image", "is_primary", "pk"
    )
    primary = qs.filter(is_primary=True).order_by("-pk").first()
    if primary:
        url = _face_reference_image_url(primary)
        if url:
            return url
    if matched_reference_id:
        url = _reference_image_media_url(int(matched_reference_id))
        if url:
            return url
    any_ref = qs.order_by("-is_primary", "-pk").first()
    return _face_reference_image_url(any_ref) if any_ref else None


def _identity_card_photo_fields(
    identity_id: int,
    matched_reference_id: int,
    detected_crop_url: str,
) -> dict[str, str]:
    """Separate registry photo vs scene crop for identity card rendering."""
    ref_url = (
        _registry_reference_photo_url(
            identity_id,
            matched_reference_id=matched_reference_id,
        )
        or ""
    )
    crop = (detected_crop_url or "").strip()
    return {
        "reference_photo_url": ref_url,
        "reference_image_url": ref_url,
        "detected_crop_url": crop,
    }


def _admin_add_facepersonprofile_url(person_code: str) -> str:
    """Lien admin « ajouter profil » ; `identity_code` en query (pré-remplissage si ModelAdmin le gère)."""
    try:
        from django.urls import reverse

        base = reverse("admin:monitoring_facepersonprofile_add")
    except Exception:
        return "/admin/monitoring/facepersonprofile/add/"
    code = (person_code or "").strip()
    if not code:
        return base
    return f"{base}?{urlencode({'identity_code': code})}"


def expand_square_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
    padding: float = 0.20,
) -> tuple[int, int, int, int]:
    """
    Convertit une boîte visage ``(x, y, w, h)`` en ROI carré autour du centre,
    agrandie de ``padding`` (marge relative : côté ≈ ``max(w,h) * (1 + 2*padding)``),
    puis recadrée en carré par la dimension minimale si le clamp aux bords déforme le rectangle.
    Tout reste dans l’image ; retour ``(x1, y1, x2, y2)`` en coordonnées image (pour slices numpy).
    """
    w_i = max(1, int(w))
    h_i = max(1, int(h))
    x_i = max(0, min(int(x), img_w - 1))
    y_i = max(0, min(int(y), img_h - 1))
    if x_i + w_i > img_w:
        w_i = img_w - x_i
    if y_i + h_i > img_h:
        h_i = img_h - y_i

    cx = x_i + w_i / 2.0
    cy = y_i + h_i / 2.0
    base_side = float(max(w_i, h_i))
    side = base_side * (1.0 + 2.0 * float(padding))
    half = max(1.0, side / 2.0)
    half_fit = float(min(half, cx, float(img_w) - cx, cy, float(img_h) - cy))
    half_fit = max(half_fit, 1.0)

    x1f = cx - half_fit
    y1f = cy - half_fit
    x2f = cx + half_fit
    y2f = cy + half_fit

    x1 = max(0, int(np.floor(x1f)))
    y1 = max(0, int(np.floor(y1f)))
    x2 = min(img_w, int(np.ceil(x2f)))
    y2 = min(img_h, int(np.ceil(y2f)))
    x2 = max(x1 + 1, min(x2, img_w))
    y2 = max(y1 + 1, min(y2, img_h))

    rw = x2 - x1
    rh = y2 - y1
    sz = min(rw, rh)
    if sz < 1:
        sz = 1

    xc = max(0, min(int(round(cx)), img_w - 1))
    yc = max(0, min(int(round(cy)), img_h - 1))
    sx = max(0, min(xc - sz // 2, img_w - sz))
    sy = max(0, min(yc - sz // 2, img_h - sz))
    sx = min(sx, img_w - sz)
    sy = min(sy, img_h - sz)
    return int(sx), int(sy), int(sx + sz), int(sy + sz)


def _resize_max_width_jpeg(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    sc = float(max_w) / float(max(w, 1))
    nw = max(1, int(round(w * sc)))
    nh = max(1, int(round(h * sc)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _encode_jpeg_b64_local(img_bgr: np.ndarray, quality: int = 82) -> str | None:
    if img_bgr is None or img_bgr.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    import base64

    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _laplacian_quality_label(crop_bgr: np.ndarray) -> str:
    if crop_bgr is None or crop_bgr.size < 256:
        return "low_resolution"
    try:
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        v = float(cv2.Laplacian(g, cv2.CV_64F).var())
        if v < 50:
            return "blurry_low"
        if v < 120:
            return "acceptable"
        return "good"
    except Exception:
        return "unknown"


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    sa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    sb = max(1.0, (bx2 - bx1) * (by2 - by1))
    uni = sa + sb - inter
    return inter / uni if uni > 0 else 0.0


def _nms_faces(boxes: list[list[float]], iou_thresh: float = 0.45) -> list[list[float]]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept: list[list[float]] = []
    for b in boxes:
        if any(_iou_xyxy(b, k) > iou_thresh for k in kept):
            continue
        kept.append(b)
    return kept


def _gather_raw_face_xywh(frame_bgr: np.ndarray, fh: int, fw: int) -> list[tuple[int, int, int, int]]:
    """
    Boîtes visage `(x,y,w,h)` depuis DeepFace `facial_area` uniquement, sinon Haar.
    """
    out: list[tuple[int, int, int, int]] = []

    if getattr(settings, "FACE_RECOGNITION_ENABLED", True):
        try:
            from deepface import DeepFace

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            det_backend = str(
                getattr(settings, "FACE_EXTRACT_DETECTOR_BACKEND", "opencv") or "opencv"
            )
            objs = DeepFace.extract_faces(
                img_path=rgb,
                detector_backend=det_backend,
                enforce_detection=False,
                align=True,
            )
            for ob in objs or []:
                fa = ob.get("facial_area") if isinstance(ob, dict) else None
                if not isinstance(fa, dict):
                    continue
                x = int(fa.get("x", 0))
                y = int(fa.get("y", 0))
                w = int(fa.get("w", 0))
                h = int(fa.get("h", 0))
                if w < 8 or h < 8:
                    continue
                x = max(0, min(x, fw - 1))
                y = max(0, min(y, fh - 1))
                w = min(w, fw - x)
                h = min(h, fh - y)
                if w >= 8 and h >= 8:
                    out.append((x, y, w, h))
        except Exception as exc:
            logger.warning("DeepFace extract_faces indisponible ou échec: %s", exc)

    if not out:
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            for (x, y, w, h) in rects:
                x_i = max(0, min(int(x), fw - 1))
                y_i = max(0, min(int(y), fh - 1))
                w_i = min(int(w), fw - x_i)
                h_i = min(int(h), fh - y_i)
                if w_i >= 8 and h_i >= 8:
                    out.append((x_i, y_i, w_i, h_i))
        except Exception as exc:
            logger.warning("Repli Haar échec: %s", exc)

    boxes_xyxy = [[float(x), float(y), float(x + w), float(y + h)] for x, y, w, h in out]
    picked_xyxy = _nms_faces(boxes_xyxy, 0.42)
    picked_xyxy.sort(key=lambda b: b[0])

    xywh: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in picked_xyxy:
        xi = int(round(x1))
        yi = int(round(y1))
        wi = int(round(x2 - x1))
        hi = int(round(y2 - y1))
        if wi >= 8 and hi >= 8:
            xywh.append((xi, yi, wi, hi))

    return xywh


def _resize_square_output(patch: np.ndarray, out_size: int) -> np.ndarray:
    """Redimensionnement carré conservé pour embedding / sauvegarde."""
    if patch is None or patch.size == 0:
        raise ValueError("empty patch")
    o = max(96, min(int(out_size), 640))
    h, w = patch.shape[:2]
    interp = cv2.INTER_AREA if max(h, w) > o else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(patch), (o, o), interpolation=interp)


def iterate_tight_square_face_crops(
    frame_bgr: np.ndarray,
    *,
    verbose_log: bool = False,
) -> list[tuple[list[float], list[float], np.ndarray]]:
    """
    Pour chaque visage : bbox serrée (xyxy), bbox étendue carrée (xyxy pour overlay),
    crop final BGR ``out_size × out_size``.

    verbose_log=True : ``logger.info`` par visage original / étendu / taille crop (sans chemin fichier).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []
    fh, fw = frame_bgr.shape[:2]
    mx = max(1, min(int(getattr(settings, "FUSION_FACE_DETECT_MAX", 8)), 16))
    pad_frac = float(getattr(settings, "FACE_BIOMETRIC_BBOX_PADDING", 0.20))
    out_sz = int(getattr(settings, "FACE_BIOMETRIC_CROP_SIZE", 256))

    raw_xywh = _gather_raw_face_xywh(frame_bgr, fh, fw)[:mx]
    triples: list[tuple[list[float], list[float], np.ndarray]] = []

    for xt, yt, wt, ht in raw_xywh:
        tight_xyxy = [float(xt), float(yt), float(xt + wt), float(yt + ht)]
        ex1, ey1, ex2, ey2 = expand_square_bbox(xt, yt, wt, ht, fw, fh, pad_frac)
        expanded_xyxy = [float(ex1), float(ey1), float(ex2), float(ey2)]

        patch = frame_bgr[ey1:ey2, ex1:ex2]
        if patch is None or patch.size == 0:
            continue
        crop_out = _resize_square_output(patch, out_sz)

        triples.append((expanded_xyxy, tight_xyxy, crop_out))

        if verbose_log:
            hr, wc = crop_out.shape[:2]
            zx1, zy1, zx2, zy2 = tight_xyxy
            logger.info(
                "Face crop biometric: tight_xyxy=%s tight_xywh=(%s,%s,%s,%s) expanded_xyxy=%s patch_hw=%sx%s resized=%sx%s",
                [round(v, 1) for v in tight_xyxy],
                int(zx1),
                int(zy1),
                max(1, int(round(zx2 - zx1))),
                max(1, int(round(zy2 - zy1))),
                [round(v, 1) for v in expanded_xyxy],
                patch.shape[0],
                patch.shape[1],
                hr,
                wc,
            )

    return triples


def extract_face_boxes_bgr(frame_bgr: np.ndarray) -> list[tuple[list[float], np.ndarray]]:
    """(bbox_xyxy étendue pour overlay live, crop carré taille ``FACE_BIOMETRIC_CROP_SIZE``)."""
    out: list[tuple[list[float], np.ndarray]] = []
    dbg = getattr(settings, "FACE_BIOMETRIC_DEBUG_LIVE_CROPS", False) or logger.isEnabledFor(
        logging.DEBUG
    )
    for expanded_xyxy, tight_xyxy, crop in iterate_tight_square_face_crops(
        frame_bgr, verbose_log=False
    ):
        out.append((expanded_xyxy, crop))
        if dbg:
            logger.debug(
                "Biometric live crop: tight_xyxy=%s expanded_xyxy=%s crop_square=%sx%s path=n/a",
                [round(v, 1) for v in tight_xyxy],
                [round(v, 1) for v in expanded_xyxy],
                crop.shape[0],
                crop.shape[1],
            )

    return out


def _serialize_face_person_profile(person_code: str) -> tuple[dict[str, Any] | None, bool]:
    """(profil JSON ou None, trouvé étendu)."""
    code = (person_code or "").strip()
    if not code:
        return None, False
    from monitoring.models import FacePersonProfile

    prof = FacePersonProfile.objects.filter(identity_code=code).first()
    if not prof:
        return None, False
    ei = prof.extra_info
    extra_disp = ""
    if ei is not None:
        if isinstance(ei, (dict, list)):
            try:
                extra_disp = json.dumps(ei, ensure_ascii=False, indent=2)[:1200]
            except Exception:
                extra_disp = str(ei)[:1200]
        else:
            extra_disp = str(ei)[:1200]
    payload: dict[str, Any] = {
        "identity_code": prof.identity_code,
        "full_name": prof.full_name or "",
        "age": prof.age,
        "role": prof.role or "",
        "risk_level": prof.risk_level or "",
        "notes": prof.notes or "",
        "last_seen": prof.last_seen.isoformat() if prof.last_seen else None,
        "extra_info": prof.extra_info,
        "extra_info_display": extra_disp,
    }
    return payload, True


def build_face_biometric_snapshot(
    frame_bgr: np.ndarray,
    run_slug: str,
    *,
    subdir: str = "face_crops",
) -> dict[str, Any]:
    """
    Retour JSON-safe :
    face_matches, faces_detected, face_overlay_boxes, face_scan_gallery (alias de face_matches).
    """
    faces_detected: list[dict[str, Any]] = []
    face_matches: list[dict[str, Any]] = []
    overlay_boxes: list[dict[str, Any]] = []

    mu = settings.MEDIA_URL.rstrip("/")
    mr = Path(settings.MEDIA_ROOT)
    crop_root = mr / subdir / run_slug
    crop_root.mkdir(parents=True, exist_ok=True)

    if not getattr(settings, "FACE_RECOGNITION_ENABLED", True):
        return {
            "faces_detected": [],
            "face_matches": [],
            "face_overlay_boxes": [],
            "face_scan_gallery": [],
            "matching_disabled": True,
        }

    try:
        from monitoring.services.face_recognition_gallery import (
            active_reference_rows,
            best_identity_match_for_bgr_crop,
            diagnose_face_gallery_match,
            match_confidence_pct_from_distance,
        )
    except Exception:
        return {
            "faces_detected": [],
            "face_matches": [],
            "face_overlay_boxes": [],
            "face_scan_gallery": [],
            "matching_disabled": True,
        }

    gallery_empty = len(active_reference_rows()) == 0

    pairs = iterate_tight_square_face_crops(frame_bgr, verbose_log=False)

    try:
        for i, (bbox, tight_xy, crop) in enumerate(pairs):
            face_id = uuid.uuid4().hex[:16]
            fn = f"face_{i:02d}_{face_id[:8]}.jpg"
            fp = crop_root / fn
            cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            rel_url = f"{mu}/{subdir}/{run_slug}/{fn}"
            q = _laplacian_quality_label(crop)

            tx1, ty1, tx2, ty2 = tight_xy
            logger.info(
                "Face crop saved: original_bbox(xyxy)=%s original_bbox(xywh)=(%s,%s,%s,%s) expanded_bbox(xyxy)=%s crop_square=%sx%s saved_path=%s",
                [round(float(v), 1) for v in tight_xy],
                int(tx1),
                int(ty1),
                max(1, int(round(tx2 - tx1))),
                max(1, int(round(ty2 - ty1))),
                [round(float(v), 1) for v in bbox],
                crop.shape[0],
                crop.shape[1],
                str(fp.resolve()),
            )

            overlay_boxes.append({"bbox": [round(float(x), 1) for x in bbox], "label": "face"})

            faces_detected.append(
                {
                    "face_id": face_id,
                    "crop_url": rel_url,
                    "bbox": [round(float(x), 1) for x in bbox],
                    "bbox_tight": [round(float(x), 1) for x in tight_xy],
                    "quality": q,
                    "status": "scanning",
                }
            )

            m = None
            crop_abs = str(fp.resolve())
            try:
                m = best_identity_match_for_bgr_crop(crop, debug_crop_path=crop_abs)
            except Exception as exc:
                logger.debug("Match visage échec: %s", exc)

            match_out = None
            prof_payload = None
            has_extended = False

            if m:
                pcode = str(m.get("person_code", "") or "").strip()
                dist = float(m["distance"])
                conf_pct = float(match_confidence_pct_from_distance(dist))
                cat = (m.get("category") or "").strip()

                prof_payload, has_extended = _serialize_face_person_profile(pcode)

                identity_id = int(m["identity_id"])
                ref_id = int(m["reference_id"])
                photo_fields = _identity_card_photo_fields(identity_id, ref_id, rel_url)
                ref_img_url = photo_fields["reference_photo_url"]
                admin_profile_url = _admin_add_facepersonprofile_url(pcode)
                card_name = str(m["display_name"])
                if prof_payload and has_extended:
                    fn = (prof_payload.get("full_name") or "").strip()
                    if fn:
                        card_name = fn

                match_out = {
                    "identity_id": identity_id,
                    "reference_id": ref_id,
                    "display_name": str(m["display_name"]),
                    "person_code": pcode,
                    "category": cat,
                    "distance": dist,
                    "confidence_pct": int(round(conf_pct)),
                    **photo_fields,
                    "card_display_name": card_name,
                    "admin_add_facepersonprofile_url": admin_profile_url,
                }

                face_matches.append(
                    {
                        "face_id": face_id,
                        "crop_url": rel_url,
                        "detected_crop_url": rel_url,
                        "bbox": [round(float(x), 1) for x in bbox],
                        "bbox_tight": [round(float(x), 1) for x in tight_xy],
                        "quality": q,
                        "matched": True,
                        "identity_code": pcode or str(m["identity_id"]),
                        "full_name": str(m["display_name"]),
                        "category": cat,
                        "distance": round(dist, 4),
                        "match_confidence": round(conf_pct, 1),
                        "status": "match_found",
                        "profile": prof_payload,
                        "profile_extended": has_extended,
                        "no_extended_profile_message": ""
                        if has_extended
                        else "No extended profile found",
                        "match": match_out,
                        "gallery_empty": gallery_empty,
                        **photo_fields,
                        "card_display_name": card_name,
                        "admin_add_facepersonprofile_url": admin_profile_url,
                    }
                )
            else:
                diag = diagnose_face_gallery_match(crop)
                na = diag.get("nearest_any") if isinstance(diag.get("nearest_any"), dict) else None
                thr = float(diag.get("threshold", 0.0) or 0.0)
                hint_parts: list[str] = []
                if na:
                    hint_parts.append(
                        f"Nearest: {na.get('display_name', '')} · distance {na.get('distance')} "
                        f"(threshold {thr})"
                    )
                elif not diag.get("query_embedding_ok"):
                    hint_parts.append("Query embedding failed — check DeepFace / crop quality.")
                elif int(diag.get("gallery_size") or 0) == 0:
                    hint_parts.append("Gallery empty — run seed_face_registry_from_folders or add references in admin.")
                match_hint = " ".join(hint_parts).strip()
                face_matches.append(
                    {
                        "face_id": face_id,
                        "crop_url": rel_url,
                        "bbox": [round(float(x), 1) for x in bbox],
                        "bbox_tight": [round(float(x), 1) for x in tight_xy],
                        "quality": q,
                        "matched": False,
                        "identity_code": "",
                        "full_name": "Unknown Individual",
                        "category": "",
                        "distance": None,
                        "match_confidence": 0.0,
                        "status": "unknown",
                        "profile": None,
                        "profile_extended": False,
                        "no_extended_profile_message": "",
                        "match": None,
                        "gallery_empty": gallery_empty,
                        "match_diagnostic": diag,
                        "match_hint": match_hint,
                        "nearest_distance": na.get("distance") if na else None,
                        "nearest_display_name": (na.get("display_name") or "") if na else "",
                        "match_threshold": round(thr, 4),
                    }
                )
    except Exception as exc:
        logger.exception("Pipeline biométrique: %s", exc)

    for fd in faces_detected:
        fd["status"] = "complete"

    return {
        "faces_detected": faces_detected,
        "face_matches": face_matches,
        "face_overlay_boxes": overlay_boxes,
        "face_scan_gallery": face_matches,
        "matching_disabled": False,
        "gallery_empty": gallery_empty,
    }


def live_face_row_from_crop(
    crop_bgr: np.ndarray,
    bbox_xyxy: list[float],
    *,
    person_conf: float = 1.0,
) -> dict[str, Any]:
    """Une entrée pour `live_faces` (caméra) alignée avec le panneau serveur."""
    from monitoring.services.face_recognition_gallery import (
        best_identity_match_for_bgr_crop,
        match_confidence_pct_from_distance,
    )

    if not getattr(settings, "FACE_RECOGNITION_ENABLED", True):
        live_preview = max(96, int(getattr(settings, "FACE_BIOMETRIC_LIVE_PREVIEW", 140)))

        def _enc(c: np.ndarray) -> str:
            crs = _resize_max_width_jpeg(np.ascontiguousarray(c), live_preview)
            enc = _encode_jpeg_b64_local(crs, quality=82)
            return f"data:image/jpeg;base64,{enc}" if enc else ""

        return {
            "bbox": [round(float(v), 1) for v in bbox_xyxy[:4]],
            "person_conf": round(float(person_conf), 4),
            "crop_data_url": _enc(crop_bgr),
            "match": None,
            "matched": False,
            "status": "unavailable",
            "history_id": uuid.uuid4().hex[:18],
        }

    live_preview = max(96, int(getattr(settings, "FACE_BIOMETRIC_LIVE_PREVIEW", 140)))
    crs = _resize_max_width_jpeg(np.ascontiguousarray(crop_bgr), live_preview)
    enc = _encode_jpeg_b64_local(crs, quality=82)
    data_url = f"data:image/jpeg;base64,{enc}" if enc else ""

    m = None
    try:
        m = best_identity_match_for_bgr_crop(np.ascontiguousarray(crop_bgr))
    except Exception:
        pass

    match_js = None
    prof = None
    has_ext = False
    matched = False
    if m:
        matched = True
        pcode = str(m.get("person_code", "") or "").strip()
        dist = float(m["distance"])
        conf_pct = match_confidence_pct_from_distance(dist)
        cat = (m.get("category") or "").strip()
        prof, has_ext = _serialize_face_person_profile(pcode)
        identity_id = int(m["identity_id"])
        ref_id = int(m["reference_id"])
        photo_fields = _identity_card_photo_fields(identity_id, ref_id, "")
        admin_profile_url = _admin_add_facepersonprofile_url(pcode)
        card_name = str(m["display_name"])
        if prof and has_ext:
            fn = (prof.get("full_name") or "").strip()
            if fn:
                card_name = fn
        match_js = {
            "identity_id": identity_id,
            "reference_id": ref_id,
            "display_name": str(m["display_name"]),
            "person_code": pcode,
            "category": cat,
            "distance": dist,
            "confidence_pct": int(conf_pct),
            "profile": prof,
            "profile_extended": has_ext,
            "full_name": str(m["display_name"]),
            "matched": True,
            "match_confidence": float(conf_pct),
            "identity_code": pcode,
            "status": "match_found",
            "no_extended_profile_message": "" if has_ext else "No extended profile found",
            **photo_fields,
            "card_display_name": card_name,
            "admin_add_facepersonprofile_url": admin_profile_url,
        }

    disp = (match_js.get("display_name") if match_js else "") or ""
    row: dict[str, Any] = {
        "bbox": [round(float(v), 1) for v in bbox_xyxy[:4]],
        "person_conf": round(float(person_conf), 4),
        "crop_data_url": data_url,
        "detected_crop_url": data_url,
        "match": match_js,
        "matched": matched,
        "full_name": disp if matched else "Unknown Individual",
        "status": ("match_found" if matched else "unknown"),
        "profile": prof,
        "profile_extended": has_ext if matched else False,
        "history_id": uuid.uuid4().hex[:18],
    }
    if matched and match_js:
        row["reference_photo_url"] = match_js.get("reference_photo_url") or ""
        row["reference_image_url"] = match_js.get("reference_image_url") or ""

    return row