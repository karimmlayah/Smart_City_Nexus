"""
Registre visages Django : embeddings (DeepFace) + correspondance sur recadrages BGR.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)


def compute_embedding_and_store(ref_pk: int) -> bool:
    """Lit l’image disque et met à jour JSON embedding (sans re-déclencher save())."""
    from monitoring.models import FaceReferenceImage

    ref = FaceReferenceImage.objects.select_related("face_identity").filter(pk=ref_pk).first()
    if ref is None:
        return False
    f = getattr(ref, "image", None)
    if not f:
        return False
    path = getattr(f, "path", "") or ""
    if not path:
        return False
    err_msg = ""
    vec_list = None
    model_used = str(getattr(settings, "FACE_EMBED_MODEL_NAME", "Facenet") or "Facenet")
    arr = cv2.imread(path, cv2.IMREAD_COLOR)
    if arr is None:
        err_msg = "ouverture_fichier"
    else:
        emb = embedding_from_bgr(arr)
        if emb is None:
            err_msg = "embedding_impossible"
        else:
            vec_list = emb.astype(float).tolist()
    FaceReferenceImage.objects.filter(pk=ref_pk).update(
        embedding=vec_list,
        embedding_model=model_used if vec_list else "",
        embedding_error="" if vec_list else err_msg[:500],
    )
    return vec_list is not None


def _represent_rgb_numpy(rgb_u8: np.ndarray) -> np.ndarray | None:
    if not getattr(settings, "FACE_RECOGNITION_ENABLED", True):
        return None
    try:
        from deepface import DeepFace  # type: ignore[import-untyped]

        DeepFace_module = DeepFace  # pylint: disable=no-member
    except Exception:
        return None
    model = str(getattr(settings, "FACE_EMBED_MODEL_NAME", "Facenet") or "Facenet")
    det = str(getattr(settings, "FACE_EMBED_DETECTOR_BACKEND", "opencv") or "opencv")
    fb_skip = bool(getattr(settings, "FACE_EMBED_FALLBACK_SKIP", True))

    sequences: list[tuple[str, bool]] = [(det, True)]
    if fb_skip:
        sequences.append(("skip", False))

    for db, enf in sequences:
        try:
            reps = DeepFace_module.represent(
                img_path=rgb_u8,
                model_name=model,
                detector_backend=db,
                enforce_detection=enf,
            )
            vec = reps[0].get("embedding")
            if vec is None:
                continue
            return np.asarray(vec, dtype=np.float64).ravel()
        except Exception:
            continue
    return None


def embedding_from_bgr(bgr: np.ndarray | None) -> np.ndarray | None:
    if bgr is None or bgr.size == 0:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return _represent_rgb_numpy(rgb)


def active_reference_rows() -> list[dict[str, Any]]:
    from monitoring.models import FaceReferenceImage

    qs = (
        FaceReferenceImage.objects.filter(face_identity__is_active=True)
        .select_related("face_identity")
        .order_by("-is_primary", "pk")
    )
    out = []
    for r in qs:
        emb_list = getattr(r, "embedding", None)
        if emb_list is None:
            continue
        if isinstance(emb_list, list) and len(emb_list) < 16:
            continue
        fid = r.face_identity
        arr = np.asarray(emb_list, dtype=np.float64).ravel()
        out.append(
            {
                "ref_pk": r.pk,
                "identity_id": fid.pk,
                "display_name": fid.display_name,
                "person_code": (fid.person_code or "").strip(),
                "category": (getattr(fid, "category", "") or "").strip(),
                "vector": arr,
            }
        )
    return out


def _rank_gallery_cosine_distances(
    qvec: np.ndarray,
    refs: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    """Liste triée (distance croissante, ligne galerie)."""
    try:
        from deepface.modules.verification import find_cosine_distance  # type: ignore[import-untyped]
    except Exception:
        return []
    ranked: list[tuple[float, dict[str, Any]]] = []
    for row in refs:
        d = float(find_cosine_distance(qvec, row["vector"]))
        ranked.append((d, row))
    ranked.sort(key=lambda x: x[0])
    return ranked


def diagnose_face_gallery_match(crop_bgr: np.ndarray | None) -> dict[str, Any]:
    """
    Diagnostic JSON-safe : taille galerie, seuil, meilleur sous seuil, plus proche absolu, top-5 distances.
    Utile quand l’UI affiche UNKNOWN mais une identité est « proche » du seuil.
    """
    threshold = float(getattr(settings, "FACE_MATCH_MAX_COSINE_DISTANCE", 0.4))
    out: dict[str, Any] = {
        "gallery_size": 0,
        "threshold": round(threshold, 4),
        "query_embedding_ok": False,
        "best_under_threshold": None,
        "nearest_any": None,
        "top5": [],
    }
    if crop_bgr is None or crop_bgr.size == 0:
        return out
    if not getattr(settings, "FACE_RECOGNITION_ENABLED", True):
        return out
    refs = active_reference_rows()
    out["gallery_size"] = len(refs)
    if not refs:
        return out
    qvec = embedding_from_bgr(crop_bgr)
    if qvec is None:
        return out
    out["query_embedding_ok"] = True
    ranked = _rank_gallery_cosine_distances(qvec, refs)
    for d, row in ranked[:5]:
        out["top5"].append(
            {
                "display_name": str(row.get("display_name", "")),
                "person_code": str(row.get("person_code", "") or ""),
                "distance": round(float(d), 4),
            }
        )
    if ranked:
        d0, row0 = ranked[0]
        out["nearest_any"] = {
            "display_name": str(row0.get("display_name", "")),
            "person_code": str(row0.get("person_code", "") or ""),
            "distance": round(float(d0), 4),
        }
    for d, row in ranked:
        if float(d) <= threshold:
            out["best_under_threshold"] = {
                "identity_id": int(row["identity_id"]),
                "reference_id": int(row["ref_pk"]),
                "display_name": str(row["display_name"]),
                "person_code": str(row.get("person_code", "") or ""),
                "category": (row.get("category") or "").strip(),
                "distance": round(float(d), 4),
            }
            break
    return out


def best_identity_match_for_bgr_crop(
    crop_bgr: np.ndarray | None,
    *,
    debug_crop_path: str | None = None,
) -> dict | None:
    """
    Meilleure identité sous le seuil pour un seul recadrage BGR (affichage par visage).
    Retourne None si pas d’embedding, pas de galerie, ou aucun match sous seuil.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if not getattr(settings, "FACE_RECOGNITION_ENABLED", True):
        return None
    refs = active_reference_rows()
    if not refs:
        if getattr(settings, "FACE_MATCH_DEBUG_LOG", False):
            logger.info("Face match: gallery empty (no active references with embeddings)")
        return None
    qvec = embedding_from_bgr(crop_bgr)
    if qvec is None:
        if getattr(settings, "FACE_MATCH_DEBUG_LOG", False):
            logger.info("Face match: query embedding failed (DeepFace / image)")
        return None
    threshold = float(getattr(settings, "FACE_MATCH_MAX_COSINE_DISTANCE", 0.4))
    ranked = _rank_gallery_cosine_distances(qvec, refs)
    if not ranked:
        return None
    best: dict | None = None
    for d, row in ranked:
        if float(d) > threshold:
            continue
        best = {
            "identity_id": row["identity_id"],
            "reference_id": row["ref_pk"],
            "display_name": row["display_name"],
            "person_code": row["person_code"],
            "category": (row.get("category") or "").strip(),
            "distance": round(float(d), 4),
        }
        break
    if getattr(settings, "FACE_MATCH_DEBUG_LOG", False) and ranked:
        d0, r0 = ranked[0]
        crop_tag = debug_crop_path or "n/a"
        logger.info(
            "Face match: crop=%s gallery=%s threshold=%s best_match_candidate=%s best_distance=%.4f accepted_under_threshold=%s",
            crop_tag,
            len(refs),
            threshold,
            r0.get("display_name"),
            float(d0),
            best is not None,
        )
        for d, row in ranked[: min(6, len(ranked))]:
            logger.info(
                "  compare identity=%s code=%s cosine_distance=%.4f",
                row.get("display_name"),
                row.get("person_code") or "",
                float(d),
            )
    return best


def match_confidence_pct_from_distance(distance: float, *, span: float = 0.55) -> int:
    """Score d’affichage 0–100 (cosine distance plus bas = meilleur)."""
    d = max(0.0, float(distance))
    s = float(span) if float(span) > 1e-6 else 0.55
    pct = (1.0 - min(d / s, 1.0)) * 100.0
    return int(round(max(0.0, min(100.0, pct))))


def match_faces_in_bgr_crops(bgr_images: list[np.ndarray]) -> list[dict]:
    """Sur une liste de recadrages OpenCV BGR ; retourne les meilleurs rapprochements par identité."""
    if not getattr(settings, "FACE_RECOGNITION_ENABLED", True):
        return []

    refs = active_reference_rows()
    if not refs or not bgr_images:
        return []

    try:
        from deepface.modules.verification import find_cosine_distance  # type: ignore[import-untyped]
    except Exception:
        return []

    threshold = float(getattr(settings, "FACE_MATCH_MAX_COSINE_DISTANCE", 0.4))
    pooled: dict[int, dict] = {}

    for crop in bgr_images:
        qvec = embedding_from_bgr(crop)
        if qvec is None:
            continue
        for row in refs:
            d = float(find_cosine_distance(qvec, row["vector"]))
            if d > threshold:
                continue
            payload = {
                "identity_id": row["identity_id"],
                "reference_id": row["ref_pk"],
                "display_name": row["display_name"],
                "person_code": row["person_code"],
                "category": (row.get("category") or "").strip(),
                "distance": round(d, 4),
            }
            iid = int(row["identity_id"])
            old = pooled.get(iid)
            if old is None or d < float(old["distance"]):
                pooled[iid] = payload

    mx = max(1, int(getattr(settings, "FACE_MATCH_MAX_RESULTS", 5)))
    return sorted(pooled.values(), key=lambda x: float(x["distance"]))[:mx]


def face_matches_roundtrip_json(matches: list[dict]) -> list[dict]:
    """Réponse HTTP / JSON-safe."""
    cleaned = []
    for m in matches:
        cleaned.append(
            {
                "identity_id": int(m["identity_id"]),
                "reference_id": int(m["reference_id"]),
                "display_name": str(m["display_name"]),
                "person_code": str(m.get("person_code", "")),
                "category": str(m.get("category", "")),
                "distance": float(m["distance"]),
            }
        )
    return cleaned
