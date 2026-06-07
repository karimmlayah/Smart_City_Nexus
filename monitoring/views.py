import json
import logging
import os
import base64
from pathlib import Path

import requests
from django.conf import settings as django_settings
from django.contrib import messages
from django.db.models import Count
from django.http import FileResponse, JsonResponse, StreamingHttpResponse
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.cache import never_cache
from django.views.decorators.clickjacking import xframe_options_exempt

from .forms import (
    FightAnalyzeForm,
    FusionHubForm,
    QuickVideoEditForm,
    QuickVideoForm,
    RoadDamageAnalyzeForm,
    SightengineWeaponForm,
    SmartCrowdSafetyForm,
    WeaponAnalyzeForm,
)
from .models import Alert, CitizenReclamation, VideoSource
from .ai_dashboard_payload import build_ai_dashboard_summary
from .openai_client import create_openai_client, get_openai_api_key, log_openai_key_status

logger = logging.getLogger(__name__)


def _active_source_for_request(request):
    sources = list(VideoSource.objects.all().order_by("-is_active", "name"))
    raw = request.GET.get("source")
    selected = None
    if raw:
        try:
            selected = VideoSource.objects.filter(pk=int(raw)).first()
        except (TypeError, ValueError):
            selected = None
    if selected is None:
        selected = VideoSource.objects.filter(is_active=True).order_by("-updated_at").first()
    return sources, selected


def _fusion_live_faces_from_scan_gallery(gallery: list | None) -> list[dict]:
    """Adapte ``face_scan_gallery`` (URLs média) pour le panneau live qui lit aussi ``crop_data_url``."""
    from uuid import uuid4

    out: list[dict] = []
    for row in gallery or []:
        if not isinstance(row, dict):
            continue
        u = (row.get("crop_url") or "").strip()
        merged = dict(row)
        if u and not merged.get("crop_data_url"):
            merged["crop_data_url"] = u
        if not merged.get("history_id"):
            merged["history_id"] = (merged.get("face_id") or uuid4().hex)[:32]
        out.append(merged)
    return out


def _fusion_hub_pick_thumb(primary: str, fb_http: str | None, fb_face: str | None) -> str:
    """URL miniature affichée sur les cartes (évite vignettes vides sans inventer false positives)."""
    for cand in (primary, fb_http, fb_face):
        s = (cand or "").strip()
        if s:
            return s
    return ""


def _fusion_hub_threat_cards_from_result(
    result,
    *,
    fallback_thumb_url: str | None = None,
    fallback_face_thumb: str | None = None,
) -> list[dict]:
    """
    Jalons légers pour cartes UI (pas de données inventées hors pipeline existant).
    """
    if not isinstance(result, dict) or result.get("error"):
        return []
    cards: list[dict] = []
    fb_h = fallback_thumb_url
    fb_f = fallback_face_thumb
    for i, ev in enumerate(result.get("dual_evidence_gallery") or []):
        thumbs = ev.get("crop_urls") or []
        cards.append(
            {
                "id": f"dual-{i}-{ev.get('t', i)}",
                "category": "suspicious",
                "title": "Suspicious activity",
                "thumb_url": _fusion_hub_pick_thumb(
                    thumbs[0] if thumbs else "",
                    fb_h,
                    fb_f,
                ),
                "time_label": f"t≈{ev.get('t', 0)}s",
                "confidence_display": round(
                    max(
                        float(ev.get("p_fight") or 0.0),
                        float(ev.get("weapon_max") or 0.0),
                    )
                    * 100.0,
                    1,
                ),
                "t_sec": ev.get("t"),
                "face_matches": ev.get("face_matches") or [],
                "crop_urls": thumbs,
                "weapon_max": ev.get("weapon_max"),
                "p_fight": ev.get("p_fight"),
                "labels": [],
            }
        )

    tl = result.get("timeline") or []
    occupied_ts = {
        round(float(e.get("t_sec", -1)), 2) for e in cards if e.get("t_sec") is not None
    }

    idx = 0
    for row in tl[-400:]:
        t = round(float(row.get("t") or 0), 4)
        if t in occupied_ts:
            continue
        label = str(row.get("label") or "")
        wm = float(row.get("weapon_max") or 0.0)
        pf = float(row.get("p_fight") or 0.0)
        is_fight = label == "fight" or pf >= 0.55
        is_weapon = wm >= 0.38

        cat = ""
        title = ""
        conf_pct = 0.0
        if is_weapon and wm >= pf:
            cat = "weapon"
            title = "Gun Detected"
            conf_pct = round(wm * 100.0, 1)
        elif is_fight:
            cat = "fight"
            title = "Fight Detected"
            conf_pct = round(pf * 100.0, 1)
        elif is_weapon:
            cat = "weapon"
            title = "Gun Detected"
            conf_pct = round(wm * 100.0, 1)
        else:
            continue

        cat_key = (cat or "suspicious").strip().lower()
        if cat_key not in ("fight", "weapon", "suspicious"):
            cat_key = "suspicious"

        cards.append(
            {
                "id": f"{cat_key}-{idx}-{t}",
                "category": cat_key,
                "title": title,
                "thumb_url": _fusion_hub_pick_thumb("", fb_h, fb_f),
                "time_label": f"t≈{t}s",
                "confidence_display": conf_pct,
                "t_sec": t,
                "face_matches": [],
                "crop_urls": [],
                "weapon_max": row.get("weapon_max"),
                "p_fight": row.get("p_fight"),
                "labels": [label] if label else [],
            }
        )
        occupied_ts.add(round(t, 2))
        idx += 1
        if len(cards) >= 18:
            break

    return cards[:18]


def _fusion_hub_report_payload(result, threat_cards):
    """Données légères pour l’aperçu rapport (sans PDF serveur fusion)."""
    from django.utils import timezone

    if not isinstance(result, dict) or result.get("error"):
        return {
            "title": "Fusion Hub",
            "ts": timezone.now().isoformat(),
            "verdict": "",
            "avg_p_fight": None,
            "highlights": list(threat_cards or [])[:12],
            "faces_summary": "",
        }
    face_bits: list[str] = []
    for ev in result.get("dual_evidence_gallery") or []:
        for fm in ev.get("face_matches") or []:
            nm = fm.get("display_name") or ""
            if nm and nm not in face_bits:
                face_bits.append(str(nm))
    scan_rows = result.get("face_matches") or result.get("face_scan_gallery") or []
    for row in scan_rows:
        if row.get("matched") and row.get("full_name"):
            nm = str(row["full_name"])
            if nm and nm not in face_bits:
                face_bits.append(nm)
                continue
        m = row.get("match")
        if m and m.get("display_name"):
            nm = str(m["display_name"])
            if nm not in face_bits:
                face_bits.append(nm)
    return {
        "title": "Fusion Hub analysis run",
        "ts": timezone.now().isoformat(),
        "verdict": str(result.get("verdict_label") or ""),
        "avg_p_fight": result.get("avg_p_fight"),
        "highlights": list(threat_cards or [])[:12],
        "faces_summary": "; ".join(face_bits[:24]),
    }


def _fusion_openai_threat_assessment_from_bgr(
    frame_bgr,
    *,
    source_label: str,
    timestamp_label: str,
) -> dict | None:
    """Vision threat check (OpenAI API), optional and best-effort."""
    import cv2

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None

    model = (os.environ.get("OPENAI_SURVEILLANCE_MODEL") or "gpt-4.1-mini").strip()
    if not model:
        model = "gpt-4.1-mini"

    try:
        h, w = frame_bgr.shape[:2]
    except Exception:
        return None

    max_side = 768
    if max(h, w) > max_side:
        scale = float(max_side) / float(max(h, w))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        frame_bgr = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
    if not ok:
        return None
    img_b64 = base64.b64encode(enc.tobytes()).decode("ascii")

    prompt = (
        "You are a CCTV threat detector. Return STRICT JSON only with keys: "
        "is_threat (bool), threat_type (string), risk_score (number 0..1), "
        "confidence (number 0..1), summary (string), evidence (array of short strings). "
        "Allowed threat_type values: none, violence, gun, knife, fight, panic, suspicious_object, other. "
        "Context source="
        + source_label
        + ", timestamp="
        + timestamp_label
        + "."
    )

    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{img_b64}"},
                ],
            }
        ],
        "temperature": 0,
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=16,
        )
        resp.raise_for_status()
        raw = resp.json()
    except Exception as exc:
        logger.warning("[fusion_openai] threat assessment failed: %s", str(exc))
        return None

    text = str(raw.get("output_text") or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    is_threat = bool(parsed.get("is_threat"))
    threat_type = str(parsed.get("threat_type") or "none").strip().lower()
    if threat_type not in {
        "none",
        "violence",
        "gun",
        "knife",
        "fight",
        "panic",
        "suspicious_object",
        "other",
    }:
        threat_type = "other"
    try:
        risk_score = max(0.0, min(1.0, float(parsed.get("risk_score") or 0.0)))
    except Exception:
        risk_score = 0.0
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence") or 0.0)))
    except Exception:
        confidence = 0.0
    summary = str(parsed.get("summary") or "").strip()[:280]
    evidence = parsed.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = []

    return {
        "enabled": True,
        "is_threat": is_threat and threat_type != "none" and risk_score >= 0.25,
        "threat_type": threat_type,
        "risk_score": round(risk_score, 4),
        "confidence": round(confidence, 4),
        "summary": summary,
        "evidence": [str(x)[:120] for x in evidence[:5]],
        "model": model,
    }


def _fusion_hub_ai_threat_card(ai_assessment: dict, *, thumb_url: str, t_sec: float | None) -> dict | None:
    if not isinstance(ai_assessment, dict):
        return None
    is_threat = bool(ai_assessment.get("is_threat"))
    ttype = str(ai_assessment.get("threat_type") or "none").strip().lower()
    if is_threat and ttype in {"gun", "knife"}:
        cat = "weapon"
        title = "AI anomaly: weapon"
    elif is_threat and ttype in {"violence", "fight"}:
        cat = "fight"
        title = "AI anomaly: violence"
    elif is_threat:
        cat = "suspicious"
        title = "AI anomaly detected"
    else:
        cat = "suspicious"
        title = "AI: no anomaly"
    conf = round(float(ai_assessment.get("risk_score") or 0.0) * 100.0, 1)
    tlabel = f"t≈{round(float(t_sec), 1)}s" if t_sec is not None else "live"
    return {
        "id": f"ai-{ttype}-{round(float(t_sec), 2) if t_sec is not None else 'live'}",
        "category": cat,
        "title": title,
        "thumb_url": (thumb_url or "").strip(),
        "time_label": tlabel,
        "confidence_display": conf,
        "t_sec": t_sec,
        "face_matches": [],
        "crop_urls": [],
        "weapon_max": None,
        "p_fight": None,
        "labels": [ttype] if ttype else [],
        "ai_summary": str(ai_assessment.get("summary") or "")[:180],
    }


def fight_predict(request):
    """Analyse combat : URL YouTube ou fichier vidéo uploadé."""
    from uuid import uuid4

    from .services.fight_classifier import (
        analyze_local_video_path,
        analyze_youtube_url,
        extract_youtube_video_id,
    )
    from .services.model_inventory import (
        coerce_fight_weights_pick,
        default_relative_fight_weights,
        default_relative_weapon_weights,
        fight_model_choices,
        resolve_weights_for_ultralytics,
        weapon_model_choices,
        with_default_choice,
    )

    result = None
    run_error = None
    youtube_video_id = None
    local_video_url = None
    annotated_video_url = None

    fc = fight_model_choices()
    wc = with_default_choice(
        weapon_model_choices(),
        default_relative_weapon_weights(),
    )

    fight_pick = coerce_fight_weights_pick(default_relative_fight_weights())
    weapon_pick = default_relative_weapon_weights()

    if request.method == "POST":
        form = FightAnalyzeForm(
            request.POST,
            request.FILES,
            fight_choices=fc,
            weapon_choices=wc,
        )
        if form.is_valid():
            fight_pick = coerce_fight_weights_pick(form.cleaned_data["fight_weights"])
            weapon_pick = form.cleaned_data["weapon_weights"]
            fy = resolve_weights_for_ultralytics(
                fight_pick,
                django_settings.FIGHT_CLASSIFIER_PATH,
            )
            wy = resolve_weights_for_ultralytics(
                weapon_pick,
                django_settings.WEAPON_DETECTOR_PATH,
            )
            vf = form.cleaned_data.get("video_file")
            has_upload = bool(
                vf
                and getattr(vf, "name", "")
                and getattr(vf, "size", 0) > 0
            )
            if has_upload:
                ext = Path(vf.name).suffix.lower()
                if ext not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                    ext = ".mp4"
                rel = Path("fight_uploads") / f"{uuid4().hex}{ext}"
                full = Path(django_settings.MEDIA_ROOT) / rel
                full.parent.mkdir(parents=True, exist_ok=True)
                with open(full, "wb+") as out:
                    for chunk in vf.chunks():
                        out.write(chunk)
                local_video_url = (
                    f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                )
                annotated_rel = Path("fight_annotated") / f"{uuid4().hex}.mp4"
                annotated_full = Path(django_settings.MEDIA_ROOT) / annotated_rel
                try:
                    result = analyze_local_video_path(
                        full,
                        annotated_output_path=annotated_full,
                        fight_yolo_weights=fy,
                        weapon_yolo_weights=wy,
                    )
                except Exception as exc:
                    run_error = str(exc)
            else:
                cleaned_url = form.cleaned_data["youtube_url"]
                youtube_video_id = extract_youtube_video_id(cleaned_url)
                annotated_rel = Path("fight_annotated") / f"{uuid4().hex}.mp4"
                annotated_full = Path(django_settings.MEDIA_ROOT) / annotated_rel
                try:
                    result = analyze_youtube_url(
                        cleaned_url,
                        annotated_output_path=annotated_full,
                        fight_yolo_weights=fy,
                        weapon_yolo_weights=wy,
                    )
                except Exception as exc:
                    run_error = str(exc)
            ann_path = (result or {}).get("annotated_video_path")
            if ann_path:
                ann_full = Path(ann_path)
                if ann_full.is_file() and ann_full.stat().st_size > 0:
                    rel_ann = ann_full.relative_to(Path(django_settings.MEDIA_ROOT))
                    annotated_video_url = (
                        f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_ann.as_posix()}"
                    )
    else:
        form = FightAnalyzeForm(
            fight_choices=fc,
            weapon_choices=wc,
            initial={
                "fight_weights": fight_pick,
                "weapon_weights": weapon_pick,
            },
        )

    fw_show = (
        (request.POST.get("fight_weights") or fight_pick)
        if request.method == "POST"
        else fight_pick
    )
    ww_show = (
        (request.POST.get("weapon_weights") or weapon_pick)
        if request.method == "POST"
        else weapon_pick
    )
    fight_model_name = Path(fw_show.replace("\\", "/")).name.split("/")[-1]
    weapon_model_name = Path(ww_show.replace("\\", "/")).name.split("/")[-1]

    fight_upload_max_mb = int(getattr(django_settings, "FIGHT_UPLOAD_MAX_MB", 200))
    fight_alert_pct = int(
        float(getattr(django_settings, "FIGHT_ALERT_MIN_CONFIDENCE", 0.99)) * 100
    )
    weapon_alert_pct = int(
        float(getattr(django_settings, "WEAPON_ALERT_MIN_CONF", 0.82)) * 100
    )
    timeline = None
    if result and not result.get("error") and result.get("timeline"):
        timeline = result["timeline"]

    return render(
        request,
        "monitoring/fight_predict.html",
        {
            "form": form,
            "result": result,
            "run_error": run_error,
            "fight_model_name": fight_model_name,
            "youtube_video_id": youtube_video_id,
            "local_video_url": local_video_url,
            "annotated_video_url": annotated_video_url,
            "timeline": timeline,
            "fight_upload_max_mb": fight_upload_max_mb,
            "fight_alert_pct": fight_alert_pct,
            "weapon_alert_pct": weapon_alert_pct,
            "weapon_model_name": weapon_model_name,
        },
    )


def fusion_analyze_frame(request):
    """Analyse fusion d’un photogramme vidéo (session progressive Fusion Hub)."""
    from django.core.cache import cache
    from django.http import HttpResponseNotAllowed

    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])

    from .services.fight_classifier import analyze_uploaded_image_fusion, read_video_frame_bgr_at_time
    from .services.model_inventory import (
        coerce_fight_weights_pick,
        default_relative_fight_weights,
        default_relative_weapon_weights,
        resolve_weights_for_ultralytics,
    )

    pv = (request.POST.get("video_id") or request.POST.get("pv") or "").strip()
    ts_raw = (request.POST.get("timestamp_seconds") or "0").strip()
    try:
        t_req = float(ts_raw)
    except ValueError:
        return JsonResponse({"status": "error", "detail": "Invalid timestamp_seconds"}, status=400)
    if not pv:
        return JsonResponse({"status": "error", "detail": "video_id/pv required"}, status=400)
    use_openai_threat = str(request.POST.get("use_openai_threat", "0")).lower() in ("1", "true", "yes")

    interval = float(getattr(django_settings, "FUSION_VIDEO_ANALYSIS_INTERVAL_SECONDS", 1.0) or 1.0)
    if interval <= 0:
        interval = 1.0
    bucket = round(t_req / interval) * interval
    frame_ttl = int(getattr(django_settings, "FUSION_PROGRESSIVE_FRAME_CACHE_SECONDS", 86400))
    ck = f"fusion_pv_frame:{pv}:{int(round(bucket * 1000))}"
    hit = cache.get(ck)
    if hit is not None:
        return JsonResponse(hit)

    meta = cache.get(f"fusion_pv:{pv}")
    if not meta or not meta.get("rel_media"):
        return JsonResponse({"status": "error", "detail": "Unknown progressive session"}, status=404)

    full = Path(django_settings.MEDIA_ROOT) / meta["rel_media"]
    if not full.is_file():
        return JsonResponse({"status": "error", "detail": "Video file missing"}, status=404)

    frame_bgr, vmeta = read_video_frame_bgr_at_time(full, bucket)
    if frame_bgr is None:
        err = {"status": "error", "detail": "Could not decode frame", "timestamp": bucket}
        cache.set(ck, err, 60)
        return JsonResponse(err, status=422)

    ann_dir = Path(django_settings.MEDIA_ROOT) / "fight_progress" / pv
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_path = ann_dir / f"t{int(round(bucket * 1000))}.jpg"

    use_fight = bool(meta.get("use_fight", True))
    use_weapon = bool(meta.get("use_weapon", True))
    fy = meta.get("fight_resolve") or resolve_weights_for_ultralytics(
        coerce_fight_weights_pick(meta.get("fight_weights")),
        django_settings.FIGHT_CLASSIFIER_PATH,
    )
    wy = meta.get("weapon_resolve") or resolve_weights_for_ultralytics(
        meta.get("weapon_weights") or default_relative_weapon_weights(),
        django_settings.WEAPON_DETECTOR_PATH,
    )
    lbl = meta.get("label") or full.name

    fusion_result = analyze_uploaded_image_fusion(
        annotated_output_path=ann_path,
        frame_bgr=frame_bgr,
        t_round=float(bucket),
        media_label=str(lbl),
        suppress_site_alerts=True,
        enable_fight=use_fight,
        enable_weapon=use_weapon,
        fight_yolo_weights=fy,
        weapon_yolo_weights=wy,
    )

    mu = django_settings.MEDIA_URL.rstrip("/")
    annotated_url = ""
    ai_path = fusion_result.get("annotated_image_path")
    if ai_path:
        p = Path(ai_path)
        if p.is_file():
            annotated_url = f"{mu}/{p.relative_to(Path(django_settings.MEDIA_ROOT)).as_posix()}"

    if fusion_result.get("error"):
        err = {
            "status": "error",
            "detail": fusion_result["error"],
            "timestamp": bucket,
        }
        cache.set(ck, err, 120)
        return JsonResponse(err, status=422)

    fb_thumb = annotated_url.strip() or None
    fb_face = None
    fm_list = fusion_result.get("face_matches") or fusion_result.get("face_scan_gallery") or []
    if fm_list and isinstance(fm_list[0], dict):
        u = fm_list[0].get("crop_url")
        fb_face = str(u).strip() if u else None
        if fb_face == "":
            fb_face = None
    threats = _fusion_hub_threat_cards_from_result(
        fusion_result,
        fallback_thumb_url=fb_thumb,
        fallback_face_thumb=fb_face,
    )
    ai_assessment = None
    should_call_ai = use_openai_threat
    if should_call_ai:
        ai_assessment = _fusion_openai_threat_assessment_from_bgr(
            frame_bgr,
            source_label=str(lbl),
            timestamp_label=f"{bucket:.2f}s",
        )
        ai_card = _fusion_hub_ai_threat_card(
            ai_assessment or {},
            thumb_url=fb_thumb or "",
            t_sec=float(bucket),
        )
        if ai_card:
            threats = [ai_card, *threats][:18]

    weapon_dets_serial: list[dict] = []
    det_rows = fusion_result.get("details") or []
    row0 = det_rows[0] if det_rows else {}
    if use_fight and row0:
        weapon_dets_serial.append(
            {
                "type": "fight",
                "label": str(row0.get("label", "")),
                "p_fight": round(float(row0.get("p_fight", 0.0)), 4),
                "top1conf": round(float(row0.get("top1conf", 0.0)), 4),
            }
        )
    for det in fusion_result.get("weapon_detection_boxes") or []:
        weapon_dets_serial.append(
            {
                "type": "weapon",
                "label": str(det.get("label", "")),
                "confidence": float(det.get("confidence", 0.0)),
                "bbox": det.get("bbox") or [],
            }
        )

    timeline_pt = {}
    tls = fusion_result.get("timeline") or []
    if tls:
        timeline_pt = tls[0]

    out = {
        "status": "ok",
        "timestamp": float(bucket),
        "annotated_frame_url": annotated_url,
        "detections": weapon_dets_serial,
        "face_matches": fm_list,
        "face_scan_gallery": fusion_result.get("face_scan_gallery") or [],
        "threats": threats,
        "timeline_point": timeline_pt,
        "verdict": fusion_result.get("verdict"),
        "verdict_label": fusion_result.get("verdict_label"),
        "avg_p_fight": fusion_result.get("avg_p_fight"),
        "video_meta": vmeta,
        "dual_evidence_gallery": fusion_result.get("dual_evidence_gallery") or [],
        "ai_threat_assessment": ai_assessment,
    }
    cache.set(ck, out, frame_ttl)
    return JsonResponse(out)


def _fusion_hub_page(request, *, list_url_name: str):
    """Studio unifié : YouTube / fichier / caméra PC, modèles combat + armes activables.

    ``list_url_name`` : nom d'URL Django pour la redirection progressive (upload vidéo).
    """
    from uuid import uuid4

    from django.core.cache import cache
    from django.urls import reverse

    from .services.fight_classifier import (
        analyze_local_video_path,
        analyze_uploaded_image_fusion,
        analyze_youtube_url,
        extract_youtube_video_id,
    )
    from .services.model_inventory import (
        coerce_fight_weights_pick,
        default_relative_fight_weights,
        default_relative_weapon_weights,
        fight_model_choices,
        resolve_weights_for_ultralytics,
        weapon_model_choices,
        with_default_choice,
    )

    result = None
    run_error = None
    youtube_video_id = None
    local_video_url = None
    annotated_video_url = None
    annotated_image_url = None
    fusion_threat_cards: list = []
    fusion_source_is_image = False
    fusion_report_payload: dict = {}
    fusion_progressive_mode = False
    fusion_progressive_pv: str | None = None
    progressive_timeline: list = []

    prog_enabled = bool(getattr(django_settings, "FUSION_PROGRESSIVE_VIDEO_ANALYSIS", True))
    upload_ttl = int(getattr(django_settings, "FUSION_PROGRESSIVE_UPLOAD_CACHE_SECONDS", 7200))

    fc = fight_model_choices()
    wc = with_default_choice(
        weapon_model_choices(),
        default_relative_weapon_weights(),
    )

    fight_pick = coerce_fight_weights_pick(default_relative_fight_weights())
    weapon_pick = default_relative_weapon_weights()

    meta_get: dict | None = None
    replay_meta: dict | None = None
    progressive_get = False

    if request.method == "GET":
        pv_in = (request.GET.get("pv") or "").strip()
        if pv_in and prog_enabled:
            mg = cache.get(f"fusion_pv:{pv_in}")
            if mg and mg.get("rel_media"):
                progressive_get = True
                fusion_progressive_mode = True
                fusion_progressive_pv = pv_in
                meta_get = mg
                local_video_url = (
                    f"{django_settings.MEDIA_URL.rstrip('/')}/"
                    f"{Path(mg['rel_media']).as_posix()}"
                )
                fight_pick = coerce_fight_weights_pick(mg.get("fight_weights") or fight_pick)
                weapon_pick = mg.get("weapon_weights") or weapon_pick
                progressive_timeline = []

    replay_done = False
    if request.method == "POST":
        rep_pv = (request.POST.get("fusion_replay_pv") or "").strip()
        if rep_pv and request.POST.get("full_video_analysis") == "1":
            replay_meta = cache.get(f"fusion_pv:{rep_pv}")
            if not replay_meta:
                run_error = "Progressive session expired or unknown — upload the video again."
                replay_done = True
            else:
                fight_pick = coerce_fight_weights_pick(
                    request.POST.get("fight_weights")
                    or replay_meta.get("fight_weights")
                    or fight_pick
                )
                weapon_pick = request.POST.get("weapon_weights") or replay_meta.get("weapon_weights") or weapon_pick
                use_fight_r = bool(replay_meta.get("use_fight", True))
                use_weapon_r = bool(replay_meta.get("use_weapon", True))
                fy = resolve_weights_for_ultralytics(
                    fight_pick,
                    django_settings.FIGHT_CLASSIFIER_PATH,
                )
                wy = resolve_weights_for_ultralytics(
                    weapon_pick,
                    django_settings.WEAPON_DETECTOR_PATH,
                )
                full_vp = Path(django_settings.MEDIA_ROOT) / replay_meta["rel_media"]
                local_video_url = (
                    f"{django_settings.MEDIA_URL.rstrip('/')}/"
                    f"{Path(replay_meta['rel_media']).as_posix()}"
                )
                if not full_vp.is_file():
                    run_error = "Video file is no longer available on disk."
                    replay_done = True
                else:
                    annotated_rel = Path("fight_annotated") / f"{uuid4().hex}.mp4"
                    annotated_full = Path(django_settings.MEDIA_ROOT) / annotated_rel
                    try:
                        result = analyze_local_video_path(
                            full_vp,
                            annotated_output_path=annotated_full,
                            enable_fight=use_fight_r,
                            enable_weapon=use_weapon_r,
                            fight_yolo_weights=fy,
                            weapon_yolo_weights=wy,
                        )
                    except Exception as exc:
                        run_error = str(exc)
                    replay_done = True

    form = None
    if progressive_get and meta_get is not None:
        form = FusionHubForm(
            fight_choices=fc,
            weapon_choices=wc,
            initial={
                "fight_weights": coerce_fight_weights_pick(
                    meta_get.get("fight_weights") or fight_pick
                ),
                "weapon_weights": meta_get.get("weapon_weights") or weapon_pick,
                "use_fight": meta_get.get("use_fight", True),
                "use_weapon": meta_get.get("use_weapon", True),
                "use_openai_threat": meta_get.get("use_openai_threat", False),
            },
        )
    elif request.method == "POST" and replay_done:
        base_meta = replay_meta or {}
        form = FusionHubForm(
            fight_choices=fc,
            weapon_choices=wc,
            initial={
                "fight_weights": fight_pick,
                "weapon_weights": weapon_pick,
                "use_fight": base_meta.get("use_fight", True),
                "use_weapon": base_meta.get("use_weapon", True),
                "use_openai_threat": base_meta.get("use_openai_threat", False),
            },
        )
    elif request.method == "POST" and not replay_done:
        form = FusionHubForm(
            request.POST,
            request.FILES,
            fight_choices=fc,
            weapon_choices=wc,
        )
        if form.is_valid():
            use_fight = form.cleaned_data["use_fight"]
            use_weapon = form.cleaned_data["use_weapon"]
            use_openai_threat = form.cleaned_data.get("use_openai_threat", False)
            fight_pick = coerce_fight_weights_pick(form.cleaned_data["fight_weights"])
            weapon_pick = form.cleaned_data["weapon_weights"]
            fy = resolve_weights_for_ultralytics(
                fight_pick,
                django_settings.FIGHT_CLASSIFIER_PATH,
            )
            wy = resolve_weights_for_ultralytics(
                weapon_pick,
                django_settings.WEAPON_DETECTOR_PATH,
            )
            vf = form.cleaned_data.get("video_file")
            imgf = form.cleaned_data.get("fusion_image_file")
            has_video_upload = bool(
                vf
                and getattr(vf, "name", "")
                and getattr(vf, "size", 0) > 0
            )
            has_image_upload = bool(
                imgf
                and getattr(imgf, "name", "")
                and getattr(imgf, "size", 0) > 0
            )
            progressive_upload = (
                prog_enabled
                and has_video_upload
                and request.POST.get("full_video_analysis") != "1"
            )
            if progressive_upload:
                ext = Path(vf.name).suffix.lower()
                if ext not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                    ext = ".mp4"
                rel_up = Path("fight_uploads") / f"{uuid4().hex}{ext}"
                full_up = Path(django_settings.MEDIA_ROOT) / rel_up
                full_up.parent.mkdir(parents=True, exist_ok=True)
                with open(full_up, "wb+") as out:
                    for chunk in vf.chunks():
                        out.write(chunk)
                pv = uuid4().hex
                cache.set(
                    f"fusion_pv:{pv}",
                    {
                        "rel_media": rel_up.as_posix(),
                        "use_fight": use_fight,
                        "use_weapon": use_weapon,
                        "use_openai_threat": use_openai_threat,
                        "fight_weights": fight_pick,
                        "weapon_weights": weapon_pick,
                        "fight_resolve": fy,
                        "weapon_resolve": wy,
                        "label": rel_up.name,
                    },
                    upload_ttl,
                )
                url = reverse(list_url_name) + f"?pv={pv}"
                return redirect(url)
            if has_video_upload:
                ext = Path(vf.name).suffix.lower()
                if ext not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                    ext = ".mp4"
                rel = Path("fight_uploads") / f"{uuid4().hex}{ext}"
                full = Path(django_settings.MEDIA_ROOT) / rel
                full.parent.mkdir(parents=True, exist_ok=True)
                with open(full, "wb+") as out:
                    for chunk in vf.chunks():
                        out.write(chunk)
                local_video_url = (
                    f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                )
                annotated_rel = Path("fight_annotated") / f"{uuid4().hex}.mp4"
                annotated_full = Path(django_settings.MEDIA_ROOT) / annotated_rel
                try:
                    result = analyze_local_video_path(
                        full,
                        annotated_output_path=annotated_full,
                        enable_fight=use_fight,
                        enable_weapon=use_weapon,
                        fight_yolo_weights=fy,
                        weapon_yolo_weights=wy,
                    )
                except Exception as exc:
                    run_error = str(exc)
            elif has_image_upload:
                fusion_source_is_image = True
                raw_ext = Path(imgf.name).suffix.lower()
                ext = raw_ext if raw_ext in (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                ) else ".jpg"
                rel = Path("fight_uploads") / f"{uuid4().hex}{ext}"
                full = Path(django_settings.MEDIA_ROOT) / rel
                full.parent.mkdir(parents=True, exist_ok=True)
                with open(full, "wb+") as out:
                    for chunk in imgf.chunks():
                        out.write(chunk)
                local_image_url_prefix = (
                    f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                )
                annotated_rel_img = Path("fight_annotated") / f"{uuid4().hex}.jpg"
                annotated_full_img = Path(django_settings.MEDIA_ROOT) / annotated_rel_img
                local_video_url = local_image_url_prefix
                try:
                    if use_fight or use_weapon:
                        result = analyze_uploaded_image_fusion(
                            full,
                            annotated_output_path=annotated_full_img,
                            enable_fight=use_fight,
                            enable_weapon=use_weapon,
                            fight_yolo_weights=fy,
                            weapon_yolo_weights=wy,
                        )
                    else:
                        # OpenAI-only mode: keep source image and let AI decide anomaly/no anomaly.
                        result = {
                            "error": None,
                            "timeline": [],
                            "face_matches": [],
                            "face_scan_gallery": [],
                            "dual_evidence_gallery": [],
                            "verdict": "ai_only",
                            "verdict_label": "AI only",
                            "avg_p_fight": 0.0,
                        }
                except Exception as exc:
                    run_error = str(exc)
            else:
                cleaned_url = form.cleaned_data["youtube_url"]
                youtube_video_id = extract_youtube_video_id(cleaned_url)
                annotated_rel = Path("fight_annotated") / f"{uuid4().hex}.mp4"
                annotated_full = Path(django_settings.MEDIA_ROOT) / annotated_rel
                try:
                    result = analyze_youtube_url(
                        cleaned_url,
                        annotated_output_path=annotated_full,
                        enable_fight=use_fight,
                        enable_weapon=use_weapon,
                        fight_yolo_weights=fy,
                        weapon_yolo_weights=wy,
                    )
                except Exception as exc:
                    run_error = str(exc)
            ann_path = (result or {}).get("annotated_video_path")
            if ann_path:
                ann_full = Path(ann_path)
                if ann_full.is_file() and ann_full.stat().st_size > 0:
                    rel_ann = ann_full.relative_to(Path(django_settings.MEDIA_ROOT))
                    annotated_video_url = (
                        f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_ann.as_posix()}"
                    )
            ann_img = (result or {}).get("annotated_image_path")
            if ann_img:
                ai_full = Path(ann_img)
                if ai_full.is_file() and ai_full.stat().st_size > 0:
                    rel_ai = ai_full.relative_to(Path(django_settings.MEDIA_ROOT))
                    annotated_image_url = (
                        f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_ai.as_posix()}"
                    )
    elif request.method == "GET" and not progressive_get:
        form = FusionHubForm(
            fight_choices=fc,
            weapon_choices=wc,
            initial={
                "fight_weights": fight_pick,
                "weapon_weights": weapon_pick,
            },
        )

    if form is None:
        form = FusionHubForm(
            fight_choices=fc,
            weapon_choices=wc,
            initial={
                "fight_weights": fight_pick,
                "weapon_weights": weapon_pick,
            },
        )

    fw_show = (
        (request.POST.get("fight_weights") or fight_pick)
        if request.method == "POST"
        else fight_pick
    )
    ww_show = (
        (request.POST.get("weapon_weights") or weapon_pick)
        if request.method == "POST"
        else weapon_pick
    )
    fight_model_name = Path(fw_show.replace("\\", "/")).name.split("/")[-1]
    weapon_model_name = Path(ww_show.replace("\\", "/")).name.split("/")[-1]

    fight_upload_max_mb = int(getattr(django_settings, "FIGHT_UPLOAD_MAX_MB", 200))
    fight_alert_pct = int(
        float(getattr(django_settings, "FIGHT_ALERT_MIN_CONFIDENCE", 0.99)) * 100
    )
    weapon_alert_pct = int(
        float(getattr(django_settings, "WEAPON_ALERT_MIN_CONF", 0.82)) * 100
    )
    timeline = None
    if fusion_progressive_mode:
        timeline = progressive_timeline
    elif result and not result.get("error") and result.get("timeline"):
        timeline = result["timeline"]
    fusion_face_cards: list = []
    fusion_match_disabled = False
    fusion_gallery_empty = False
    if fusion_progressive_mode:
        fusion_face_cards = []
        fusion_report_payload = _fusion_hub_report_payload(None, [])
    elif result and not result.get("error"):
        fusion_face_cards = result.get("face_matches") or result.get("face_scan_gallery") or []
        fusion_match_disabled = bool(result.get("face_matching_disabled"))
        fusion_gallery_empty = bool(result.get("face_gallery_empty"))
        fb_thumb = (annotated_image_url or "").strip() or None
        fb_face = None
        if fusion_face_cards and isinstance(fusion_face_cards[0], dict):
            u = fusion_face_cards[0].get("crop_url")
            fb_face = (str(u).strip()) if u else None
            if fb_face == "":
                fb_face = None
        fusion_threat_cards = _fusion_hub_threat_cards_from_result(
            result,
            fallback_thumb_url=fb_thumb,
            fallback_face_thumb=fb_face,
        )
        use_openai_threat_final = False
        if request.method == "POST":
            use_openai_threat_final = str(request.POST.get("use_openai_threat", "0")).lower() in (
                "1",
                "true",
                "yes",
            )
        if use_openai_threat_final and fusion_source_is_image:
            import cv2

            ai_eval = None
            img_source = None
            media_prefix = f"{str(django_settings.MEDIA_URL).rstrip('/')}/"
            if annotated_image_url:
                ann_url = str(annotated_image_url).strip()
                if ann_url.startswith(media_prefix):
                    rel_ann = ann_url[len(media_prefix) :].replace("\\", "/")
                    img_source = Path(django_settings.MEDIA_ROOT) / Path(rel_ann)
            if img_source is None or not img_source.is_file():
                loc_url = (local_video_url or "").strip()
                if loc_url.startswith(media_prefix):
                    rel_loc = loc_url[len(media_prefix) :].replace("\\", "/")
                    img_source = Path(django_settings.MEDIA_ROOT) / Path(rel_loc)
            if img_source and img_source.is_file():
                bgr = cv2.imread(str(img_source))
                if bgr is not None:
                    ai_eval = _fusion_openai_threat_assessment_from_bgr(
                        bgr,
                        source_label=str(img_source.name),
                        timestamp_label="0.00s",
                    )
            if ai_eval:
                ai_card = _fusion_hub_ai_threat_card(
                    ai_eval,
                    thumb_url=(annotated_image_url or local_video_url or ""),
                    t_sec=0.0,
                )
                if ai_card:
                    fusion_threat_cards = [ai_card, *fusion_threat_cards][:18]
        fusion_report_payload = _fusion_hub_report_payload(result, fusion_threat_cards)

    face_recognition_enabled = getattr(django_settings, "FACE_RECOGNITION_ENABLED", True)
    face_registry_admin_url = reverse("admin:monitoring_faceidentity_changelist")
    face_registry_enroll_url = reverse("api_face_registry_enroll")

    return render(
        request,
        "monitoring/fusion_hub.html",
        {
            "form": form,
            "result": result,
            "run_error": run_error,
            "fight_model_name": fight_model_name,
            "youtube_video_id": youtube_video_id,
            "local_video_url": local_video_url,
            "annotated_video_url": annotated_video_url,
            "annotated_image_url": annotated_image_url,
            "fusion_source_is_image": fusion_source_is_image,
            "fusion_threat_cards": fusion_threat_cards,
            "timeline": timeline,
            "fight_upload_max_mb": fight_upload_max_mb,
            "fight_alert_pct": fight_alert_pct,
            "weapon_alert_pct": weapon_alert_pct,
            "weapon_model_name": weapon_model_name,
            "fusion_live_frame_url": reverse("monitoring:fusion_live_frame"),
            "fusion_report_payload": fusion_report_payload,
            "fusion_face_cards": fusion_face_cards,
            "fusion_match_disabled": fusion_match_disabled,
            "fusion_gallery_empty": fusion_gallery_empty,
            "face_recognition_enabled": face_recognition_enabled,
            "face_registry_admin_url": face_registry_admin_url,
            "face_registry_enroll_url": face_registry_enroll_url,
            "face_registry_seed_command": "python manage.py seed_face_registry_from_folders",
            "fusion_progressive_mode": fusion_progressive_mode,
            "fusion_progressive_pv": fusion_progressive_pv,
            "fusion_analysis_interval_seconds": float(
                getattr(django_settings, "FUSION_VIDEO_ANALYSIS_INTERVAL_SECONDS", 1.0) or 1.0
            ),
            "fusion_analyze_frame_url": reverse("monitoring:fusion_analyze_frame"),
            "fusion_live_fight_display_min_p": float(
                getattr(django_settings, "FUSION_LIVE_FIGHT_DISPLAY_MIN_P", 0.52) or 0.52
            ),
        },
    )


def fusion_hub(request):
    """Ancienne URL ``/fusion/`` → ``/surveillance/`` (conserve la query, ex. ``?pv=…``)."""
    from django.http import HttpResponseRedirect
    from django.urls import reverse

    dest = reverse("monitoring:surveillance_dashboard")
    if request.GET:
        dest = f"{dest}?{request.GET.urlencode()}"
    return HttpResponseRedirect(dest)


def surveillance_dashboard(request):
    """Route /surveillance/ — même fonctionnalités que /fusion/ (upload, YouTube, analyse, live)."""
    return _fusion_hub_page(request, list_url_name="monitoring:surveillance_dashboard")


@require_POST
def fusion_live_frame(request):
    """Inference sur une image JPEG (caméra navigateur)."""
    import cv2
    import numpy as np
    from uuid import uuid4

    from django.core.cache import cache

    from .services.fight_classifier import analyze_live_frame_bgr, analyze_uploaded_image_fusion
    from .services.model_inventory import coerce_fight_weights_pick, resolve_weights_for_ultralytics

    viz_pb = str(request.POST.get("viz_persons", "0")).lower() in ("1", "true", "yes")
    viz_fm = str(request.POST.get("viz_faces", "0")).lower() in ("1", "true", "yes")
    use_openai_threat = str(request.POST.get("use_openai_threat", "0")).lower() in (
        "1",
        "true",
        "yes",
    )
    uf = request.POST.get("use_fight", "1") not in ("0", "false", "False")
    uw = request.POST.get("use_weapon", "1") not in ("0", "false", "False")
    fy = resolve_weights_for_ultralytics(
        coerce_fight_weights_pick((request.POST.get("fight_weights") or "").strip() or None),
        django_settings.FIGHT_CLASSIFIER_PATH,
    )
    wy = resolve_weights_for_ultralytics(
        (request.POST.get("weapon_weights") or "").strip() or None,
        django_settings.WEAPON_DETECTOR_PATH,
    )
    up = request.FILES.get("frame")
    if not up:
        return JsonResponse({"ok": False, "error": "no_frame"}, status=400)
    raw = up.read()
    if len(raw) > 5 * 1024 * 1024:
        return JsonResponse({"ok": False, "error": "too_large"}, status=400)
    arr = np.frombuffer(raw, dtype=np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        return JsonResponse({"ok": False, "error": "decode"}, status=400)
    try:
        payload = analyze_live_frame_bgr(
            im,
            enable_fight=uf,
            enable_weapon=uw,
            fight_yolo_weights=fy,
            weapon_yolo_weights=wy,
            viz_person_boxes=viz_pb or viz_fm,
            viz_face_match=viz_fm,
        )
    except Exception as exc:
        return JsonResponse({"ok": False, "error": str(exc)[:500]}, status=500)

    if not payload.get("ok"):
        return JsonResponse(payload)

    fight = payload.get("fight") or {}
    weapon = payload.get("weapon") or {}
    pf = float(fight.get("p_fight") or 0.0)
    top1 = int(fight.get("top1", 1))
    mx = float(weapon.get("max_conf") or 0.0)

    fp_thr = float(getattr(django_settings, "FUSION_LIVE_IMAGE_FUSION_FIGHT_P", 0.38))
    wm_thr = float(getattr(django_settings, "FUSION_LIVE_IMAGE_FUSION_WEAPON_MAX", 0.30))
    want_fight = uf and (top1 == 0 or pf >= fp_thr)
    want_weapon = uw and mx >= wm_thr
    want_full = want_fight or want_weapon

    cooldown = float(getattr(django_settings, "FUSION_LIVE_IMAGE_FUSION_COOLDOWN_SEC", 0) or 0)
    skip_full = False
    if want_full and cooldown > 0:
        ck = f"fusion_live_imgfus:{request.META.get('REMOTE_ADDR', 'anon')}"
        if cache.get(ck):
            skip_full = True
        else:
            cache.set(ck, 1, timeout=max(0.05, cooldown))

    if want_full and not skip_full:
        slug = uuid4().hex[:20]
        ann_dir = Path(django_settings.MEDIA_ROOT) / "fight_live_fusion" / slug
        ann_dir.mkdir(parents=True, exist_ok=True)
        ann_path = ann_dir / "frame.jpg"
        try:
            fr = analyze_uploaded_image_fusion(
                annotated_output_path=ann_path,
                frame_bgr=im,
                t_round=0.0,
                media_label="live_camera",
                suppress_site_alerts=True,
                enable_fight=uf,
                enable_weapon=uw,
                fight_yolo_weights=fy,
                weapon_yolo_weights=wy,
            )
        except Exception as exc:
            fr = {"error": str(exc)[:400]}
        if isinstance(fr, dict) and not fr.get("error"):
            mu = django_settings.MEDIA_URL.rstrip("/")
            ai_path = fr.get("annotated_image_path")
            ann_url = ""
            if ai_path:
                pth = Path(ai_path)
                if pth.is_file():
                    ann_url = f"{mu}/{pth.relative_to(Path(django_settings.MEDIA_ROOT)).as_posix()}"
            if ann_url:
                payload["annotated_frame_url"] = ann_url
            payload["fusion_live_pipeline"] = "image_fusion"

            d0 = (fr.get("details") or [{}])[0] if isinstance(fr.get("details"), list) else {}
            if isinstance(d0, dict) and d0:
                payload["fight"] = {
                    "p_fight": round(float(d0.get("p_fight", pf)), 4),
                    "label": str(d0.get("label", fight.get("label", ""))),
                    "top1": int(d0.get("top1", top1)),
                    "top1conf": round(float(d0.get("top1conf", fight.get("top1conf", 0.0))), 4),
                }

            wboxes = fr.get("weapon_detection_boxes") or []
            ser: list[dict] = []
            for det in wboxes:
                if not isinstance(det, dict):
                    continue
                bb = det.get("bbox") or [0, 0, 0, 0]
                ser.append(
                    {
                        "label": str(det.get("label", "")),
                        "conf": round(float(det.get("confidence", 0.0)), 4),
                        "bbox": [round(float(x), 2) for x in bb[:4]],
                    }
                )
            mx2 = max((float(x["conf"]) for x in ser), default=0.0)
            payload["weapon"] = {"detections": ser, "max_conf": round(float(mx2), 4)}

            dual = fr.get("dual_evidence_gallery") or []
            if dual and isinstance(dual[0], dict):
                ev0 = dual[0]
                curls = ev0.get("crop_urls") or []
                if curls:
                    payload["dual_view"] = {
                        "show": True,
                        "persons": [],
                        "crop_data_urls": curls,
                        "face_matches": ev0.get("face_matches") or [],
                    }

            fsg = fr.get("face_scan_gallery") or []
            lf_rows = _fusion_live_faces_from_scan_gallery(fsg if isinstance(fsg, list) else [])
            if lf_rows:
                payload["live_faces"] = lf_rows

            fb_thumb = ann_url.strip() or None
            fb_face = None
            fm_list = fr.get("face_matches") or fr.get("face_scan_gallery") or []
            if fm_list and isinstance(fm_list[0], dict):
                uu = fm_list[0].get("crop_url")
                fb_face = str(uu).strip() if uu else None
                if fb_face == "":
                    fb_face = None
            threats = _fusion_hub_threat_cards_from_result(
                fr,
                fallback_thumb_url=fb_thumb,
                fallback_face_thumb=fb_face,
            )
            ai_assessment = None
            should_call_ai = use_openai_threat
            if should_call_ai:
                ai_assessment = _fusion_openai_threat_assessment_from_bgr(
                    im,
                    source_label="live_camera",
                    timestamp_label="live",
                )
                ai_card = _fusion_hub_ai_threat_card(
                    ai_assessment or {},
                    thumb_url=fb_thumb or "",
                    t_sec=None,
                )
                if ai_card:
                    threats = [ai_card, *threats][:18]
            if threats:
                payload["threat_cards"] = threats
            if ai_assessment:
                payload["ai_threat_assessment"] = ai_assessment

    return JsonResponse(payload)


def _weapon_test_page(
    request,
    *,
    template_name: str,
    configured_model_path: str,
    fallback_model_name: str,
    page_title: str,
    target_labels: list[str] | tuple[str, ...] | None = None,
    analyzer_options: dict | None = None,
):
    """Analyse arme : URL YouTube ou fichier uploadé, avec modèle configurable."""
    from uuid import uuid4

    from .services.fight_classifier import extract_youtube_video_id
    from .services.weapon_tester import (
        analyze_image_path,
        analyze_local_video_path,
        analyze_youtube_url,
    )

    result = None
    run_error = None
    youtube_video_id = None
    local_video_url = None
    annotated_video_url = None
    local_image_url = None
    annotated_image_url = None
    knife_crop_items = []
    analyzer_options = analyzer_options or {}

    if request.method == "POST":
        form = WeaponAnalyzeForm(request.POST, request.FILES)
        if form.is_valid():
            vf = form.cleaned_data.get("video_file")
            imgf = form.cleaned_data.get("image_file")
            has_upload = bool(
                vf
                and getattr(vf, "name", "")
                and getattr(vf, "size", 0) > 0
            )
            has_image_upload = bool(
                imgf
                and getattr(imgf, "name", "")
                and getattr(imgf, "size", 0) > 0
            )
            annotated_rel = Path("weapon_annotated") / f"{uuid4().hex}.mp4"
            annotated_full = Path(django_settings.MEDIA_ROOT) / annotated_rel
            annotated_img_rel = Path("weapon_annotated") / f"{uuid4().hex}.jpg"
            annotated_img_full = Path(django_settings.MEDIA_ROOT) / annotated_img_rel
            crops_rel = Path("weapon_knife_crops") / uuid4().hex
            crops_full = Path(django_settings.MEDIA_ROOT) / crops_rel
            crops_full.mkdir(parents=True, exist_ok=True)
            if has_upload:
                ext = Path(vf.name).suffix.lower()
                if ext not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                    ext = ".mp4"
                rel = Path("weapon_uploads") / f"{uuid4().hex}{ext}"
                full = Path(django_settings.MEDIA_ROOT) / rel
                full.parent.mkdir(parents=True, exist_ok=True)
                with open(full, "wb+") as out:
                    for chunk in vf.chunks():
                        out.write(chunk)
                local_video_url = (
                    f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                )
                try:
                    result = analyze_local_video_path(
                        full,
                        annotated_output_path=annotated_full,
                        crops_output_dir=crops_full,
                        model_path=configured_model_path,
                        target_labels=target_labels,
                        **analyzer_options,
                    )
                except Exception as exc:
                    run_error = str(exc)
            elif has_image_upload:
                ext = Path(imgf.name).suffix.lower()
                if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                    ext = ".jpg"
                rel = Path("weapon_images") / f"{uuid4().hex}{ext}"
                full = Path(django_settings.MEDIA_ROOT) / rel
                full.parent.mkdir(parents=True, exist_ok=True)
                with open(full, "wb+") as out:
                    for chunk in imgf.chunks():
                        out.write(chunk)
                local_image_url = (
                    f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                )
                try:
                    result = analyze_image_path(
                        full,
                        annotated_output_path=annotated_img_full,
                        crops_output_dir=crops_full,
                        model_path=configured_model_path,
                        target_labels=target_labels,
                    )
                except Exception as exc:
                    run_error = str(exc)
            else:
                cleaned_url = form.cleaned_data["youtube_url"]
                youtube_video_id = extract_youtube_video_id(cleaned_url)
                try:
                    result = analyze_youtube_url(
                        cleaned_url,
                        annotated_output_path=annotated_full,
                        crops_output_dir=crops_full,
                        model_path=configured_model_path,
                        target_labels=target_labels,
                        **analyzer_options,
                    )
                except Exception as exc:
                    run_error = str(exc)
            ann_path = (result or {}).get("annotated_video_path")
            if ann_path:
                ann_full = Path(ann_path)
                if ann_full.is_file() and ann_full.stat().st_size > 0:
                    rel_ann = ann_full.relative_to(Path(django_settings.MEDIA_ROOT))
                    annotated_video_url = (
                        f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_ann.as_posix()}"
                    )
            ann_img_path = (result or {}).get("annotated_image_path")
            if ann_img_path:
                ann_img_full = Path(ann_img_path)
                if ann_img_full.is_file() and ann_img_full.stat().st_size > 0:
                    rel_ann_img = ann_img_full.relative_to(Path(django_settings.MEDIA_ROOT))
                    annotated_image_url = (
                        f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_ann_img.as_posix()}"
                    )
            for row in (result or {}).get("knife_crops") or []:
                rel = row.get("rel_path")
                if not rel:
                    continue
                p = Path(django_settings.MEDIA_ROOT) / rel
                if not p.is_file():
                    continue
                cf = float(row.get("conf") or 0)
                knife_crop_items.append(
                    {
                        "url": f"{django_settings.MEDIA_URL.rstrip('/')}/{rel}",
                        "t": row.get("t"),
                        "conf": row.get("conf"),
                        "conf_pct": int(round(cf * 100)),
                        "label": row.get("label", "knife"),
                    }
                )
    else:
        form = WeaponAnalyzeForm()

    weapon_model_name = Path(configured_model_path or "").name
    if not weapon_model_name:
        weapon_model_name = fallback_model_name
    fight_upload_max_mb = int(getattr(django_settings, "FIGHT_UPLOAD_MAX_MB", 200))
    weapon_alert_pct = int(
        float(getattr(django_settings, "WEAPON_ALERT_MIN_CONF", 0.82)) * 100
    )

    return render(
        request,
        template_name,
        {
            "form": form,
            "result": result,
            "run_error": run_error,
            "weapon_model_name": weapon_model_name,
            "page_title": page_title,
            "youtube_video_id": youtube_video_id,
            "local_video_url": local_video_url,
            "annotated_video_url": annotated_video_url,
            "local_image_url": local_image_url,
            "annotated_image_url": annotated_image_url,
            "fight_upload_max_mb": fight_upload_max_mb,
            "weapon_alert_pct": weapon_alert_pct,
            "knife_crop_items": knife_crop_items,
        },
    )


def weapon_test(request):
    """Analyse arme : URL YouTube ou fichier vidéo uploadé (modèle principal)."""
    return _weapon_test_page(
        request,
        template_name="monitoring/weapon_test.html",
        configured_model_path=(getattr(django_settings, "WEAPON_TEST_MODEL_PATH", "") or "").strip(),
        fallback_model_name="model.pt",
        page_title="Weapon Test",
        target_labels=list(getattr(django_settings, "WEAPON_TEST_TARGET_LABELS", ["knife"])),
        analyzer_options={},
    )


def weapon_gun_test(request):
    """Page dédiée au modèle model_gun.pt (ou config équivalente)."""
    return _weapon_test_page(
        request,
        template_name="monitoring/weapon_gun_test.html",
        configured_model_path=(getattr(django_settings, "WEAPON_GUN_TEST_MODEL_PATH", "") or "").strip(),
        fallback_model_name="model_gun.pt",
        page_title="Gun Model Test",
        target_labels=list(
            getattr(
                django_settings,
                "WEAPON_GUN_TEST_TARGET_LABELS",
                ["gun", "pistol", "rifle", "weapon"],
            )
        ),
        analyzer_options={
            "scan_stride": int(getattr(django_settings, "WEAPON_GUN_SCAN_STRIDE", 4)),
            "max_frames": int(getattr(django_settings, "WEAPON_GUN_MAX_FRAMES", 900)),
            "timeline_max_points": int(
                getattr(django_settings, "WEAPON_GUN_TIMELINE_MAX_POINTS", 240)
            ),
            "alert_min_conf": float(
                getattr(django_settings, "WEAPON_GUN_ALERT_MIN_CONF", 0.55)
            ),
            "draw_min_conf": float(
                getattr(django_settings, "WEAPON_GUN_DRAW_MIN_CONF", 0.4)
            ),
            "transcode_output": bool(
                getattr(django_settings, "WEAPON_GUN_TRANSCODE_OUTPUT", True)
            ),
        },
    )


def smart_crowd_safety_ai(request):
    """Smart Crowd Safety AI : infos + interface de test (upload vidéo)."""
    import sys
    import time
    from uuid import uuid4

    project_dir = Path(django_settings.BASE_DIR) / "smart_crowd_safety_ai"
    readme_exists = (project_dir / "README.md").is_file()

    result = None
    run_error = None
    local_video_url = None
    annotated_video_url = None
    elapsed_sec = None
    frames_used = None

    if request.method == "POST":
        form = SmartCrowdSafetyForm(request.POST, request.FILES)
        if form.is_valid():
            if not project_dir.is_dir():
                run_error = (
                    "Dossier smart_crowd_safety_ai introuvable à la racine du projet. "
                    "Vérifiez l’installation."
                )
            else:
                vf = form.cleaned_data["video_file"]
                max_f = int(form.cleaned_data["max_frames"])
                frames_used = max_f
                ext = Path(vf.name).suffix.lower()
                if ext not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                    ext = ".mp4"
                rel_in = Path("smart_crowd_uploads") / f"{uuid4().hex}{ext}"
                full_in = Path(django_settings.MEDIA_ROOT) / rel_in
                full_in.parent.mkdir(parents=True, exist_ok=True)
                with open(full_in, "wb+") as out:
                    for chunk in vf.chunks():
                        out.write(chunk)
                local_video_url = (
                    f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_in.as_posix()}"
                )

                rel_out = Path("smart_crowd_output") / f"{uuid4().hex}.mp4"
                full_out = Path(django_settings.MEDIA_ROOT) / rel_out
                full_out.parent.mkdir(parents=True, exist_ok=True)

                try:
                    if str(project_dir) not in sys.path:
                        sys.path.insert(0, str(project_dir))
                    from src.pipeline import run_video_pipeline

                    t0 = time.perf_counter()
                    result = run_video_pipeline(
                        full_in,
                        output_path=full_out,
                        show_preview=False,
                        max_frames=max_f,
                    )
                    elapsed_sec = round(time.perf_counter() - t0, 2)
                    if full_out.is_file() and full_out.stat().st_size > 0:
                        annotated_video_url = (
                            f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_out.as_posix()}"
                        )
                except Exception as exc:
                    run_error = str(exc)
    else:
        form = SmartCrowdSafetyForm()

    return render(
        request,
        "monitoring/smart_crowd_safety_ai.html",
        {
            "form": form,
            "project_dir": project_dir,
            "readme_exists": readme_exists,
            "result": result,
            "run_error": run_error,
            "local_video_url": local_video_url,
            "annotated_video_url": annotated_video_url,
            "elapsed_sec": elapsed_sec,
            "frames_used": frames_used,
            "smartcrowd_upload_max_mb": int(
                getattr(django_settings, "SMARTCROWD_UPLOAD_MAX_MB", 150)
            ),
            "smartcrowd_default_frames": int(
                getattr(django_settings, "SMARTCROWD_WEB_MAX_FRAMES_DEFAULT", 400)
            ),
            "smartcrowd_cap_frames": int(
                getattr(django_settings, "SMARTCROWD_WEB_MAX_FRAMES_CAP", 2500)
            ),
        },
    )


def weapon_test_sightengine(request):
    """Interface de test arme via Sightengine (image ou vidéo)."""
    from uuid import uuid4

    from .services.sightengine_client import (
        SightengineError,
        analyze_image_file,
        analyze_image_url,
        weapon_verdict_from_prob,
    )
    from .services.youtube_downloader import download_youtube_to_mp4
    from .services.sightengine_video_frames import (
        analyze_video_with_sightengine_as_images,
    )

    result = None
    run_error = None
    local_image_url = None
    local_video_url = None
    api_raw = None

    if request.method == "POST":
        form = SightengineWeaponForm(request.POST, request.FILES)
        if form.is_valid():
            imgf = form.cleaned_data.get("image_file")
            image_url = (form.cleaned_data.get("image_url") or "").strip()
            vf = form.cleaned_data.get("video_file")
            video_url = (form.cleaned_data.get("video_url") or "").strip()
            youtube_url = (form.cleaned_data.get("youtube_url") or "").strip()
            has_upload_video = bool(
                vf
                and getattr(vf, "name", "")
                and getattr(vf, "size", 0) > 0
            )
            try:
                if imgf:
                    ext = Path(imgf.name).suffix.lower() or ".jpg"
                    rel = Path("sightengine_images") / f"{uuid4().hex}{ext}"
                    full = Path(django_settings.MEDIA_ROOT) / rel
                    full.parent.mkdir(parents=True, exist_ok=True)
                    with open(full, "wb+") as out:
                        for chunk in imgf.chunks():
                            out.write(chunk)
                    local_image_url = (
                        f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                    )
                    api_raw = analyze_image_file(full)
                elif image_url:
                    api_raw = analyze_image_url(image_url)
                    local_image_url = image_url
                elif has_upload_video:
                    ext = Path(vf.name).suffix.lower()
                    if ext not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                        ext = ".mp4"
                    rel = Path("sightengine_videos") / f"{uuid4().hex}{ext}"
                    full = Path(django_settings.MEDIA_ROOT) / rel
                    full.parent.mkdir(parents=True, exist_ok=True)
                    with open(full, "wb+") as out:
                        for chunk in vf.chunks():
                            out.write(chunk)
                    local_video_url = (
                        f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                    )
                    # Vidéo → frames → API images Sightengine (compatible plan gratuit).
                    frames_subdir = f"sightengine_frames/{uuid4().hex}"
                    summary = analyze_video_with_sightengine_as_images(
                        full,
                        frames_subdir=frames_subdir,
                        frame_stride=int(getattr(django_settings, "SIGHTENGINE_FRAME_STRIDE", 30)),
                        max_frames=int(getattr(django_settings, "SIGHTENGINE_MAX_FRAMES", 60)),
                    )
                    best = summary.get("best") or {}
                    prob = float(best.get("prob", 0.0))
                    label = best.get("label", "weapon")
                    result = {
                        "kind": "video",
                        "prob": prob,
                        "label": label,
                        "verdict": weapon_verdict_from_prob(prob),
                        "frames": summary.get("frames") or [],
                    }
                elif video_url:
                    # Télécharge la vidéo distante en local puis applique la même logique frames+images.
                    from urllib.parse import urlparse
                    import requests

                    parsed = urlparse(video_url)
                    if not parsed.scheme:
                        video_url = f"https://{video_url}"
                    rel = Path("sightengine_videos_url") / f"{uuid4().hex}.mp4"
                    full = Path(django_settings.MEDIA_ROOT) / rel
                    full.parent.mkdir(parents=True, exist_ok=True)
                    with requests.get(video_url, stream=True, timeout=40) as resp:
                        resp.raise_for_status()
                        with open(full, "wb") as out:
                            for chunk in resp.iter_content(chunk_size=8192):
                                if not chunk:
                                    continue
                                out.write(chunk)
                    local_video_url = (
                        f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                    )
                    frames_subdir = f"sightengine_frames/{uuid4().hex}"
                    summary = analyze_video_with_sightengine_as_images(
                        full,
                        frames_subdir=frames_subdir,
                        frame_stride=int(getattr(django_settings, "SIGHTENGINE_FRAME_STRIDE", 30)),
                        max_frames=int(getattr(django_settings, "SIGHTENGINE_MAX_FRAMES", 60)),
                    )
                    best = summary.get("best") or {}
                    prob = float(best.get("prob", 0.0))
                    label = best.get("label", "weapon")
                    result = {
                        "kind": "video",
                        "prob": prob,
                        "label": label,
                        "verdict": weapon_verdict_from_prob(prob),
                        "frames": summary.get("frames") or [],
                    }
                elif youtube_url:
                    # Télécharge la vidéo YouTube en MP4 puis applique la logique frames+images.
                    yt_path = download_youtube_to_mp4(youtube_url, subdir="sightengine_youtube")
                    rel = yt_path.relative_to(Path(django_settings.MEDIA_ROOT))
                    local_video_url = f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                    frames_subdir = f"sightengine_frames/{uuid4().hex}"
                    summary = analyze_video_with_sightengine_as_images(
                        yt_path,
                        frames_subdir=frames_subdir,
                        frame_stride=int(getattr(django_settings, "SIGHTENGINE_FRAME_STRIDE", 30)),
                        max_frames=int(getattr(django_settings, "SIGHTENGINE_MAX_FRAMES", 60)),
                    )
                    best = summary.get("best") or {}
                    prob = float(best.get("prob", 0.0))
                    label = best.get("label", "weapon")
                    result = {
                        "kind": "video",
                        "prob": prob,
                        "label": label,
                        "verdict": weapon_verdict_from_prob(prob),
                        "frames": summary.get("frames") or [],
                    }
            except SightengineError as exc:
                run_error = str(exc)
            except Exception as exc:  # pragma: no cover
                run_error = str(exc)

            if api_raw is not None and not run_error and result is None:
                weapon_data = api_raw.get("weapon") or {}
                prob = float(weapon_data.get("prob", 0.0))
                label = weapon_data.get("type", "weapon")
                result = {
                    "kind": "image",
                    "raw": api_raw,
                    "prob": prob,
                    "label": label,
                    "verdict": weapon_verdict_from_prob(prob),
                }
    else:
        form = SightengineWeaponForm()

    return render(
        request,
        "monitoring/weapon_sightengine.html",
        {
            "form": form,
            "result": result,
            "run_error": run_error,
            "local_image_url": local_image_url,
            "local_video_url": local_video_url,
            "sightengine_poll_sec": int(
                getattr(django_settings, "SIGHTENGINE_VIDEO_POLL_MAX_SEC", 120)
            ),
        },
    )


def road_damage_test(request):
    """Dashboard signalements citoyens + IA multi-agents + carte ; lab YOLO road damage conservé."""
    from uuid import uuid4

    from django.urls import reverse

    from . import ai_agents as citizen_ai
    from . import vision_analysis as citizen_vision

    _citizen_recs = list(CitizenReclamation.objects.all().order_by("-created_at"))
    citizen_enriched = []
    for r in _citizen_recs:
        try:
            vis = citizen_vision.analyze_reclamation_media(r)
        except Exception:
            vis = {"stub": True, "ok": False}
        citizen_enriched.append(
            {
                "rec": r,
                "pipeline": citizen_ai.compute_ai_pipeline(
                    r, _citizen_recs, vision=vis
                ),
                "vision": vis,
            }
        )
    citizen_enriched = citizen_ai.sort_enriched_by_priority(citizen_enriched)
    citizen_kpis = citizen_ai.compute_kpis(citizen_enriched)
    citizen_map_payload = citizen_ai.build_map_payload(citizen_enriched)
    citizen_priority_queue = [
        row
        for row in citizen_enriched
        if row["pipeline"]["final_priority"] in ("Critical", "High")
    ][:16]

    from .services.road_damage_tester import (
        analyze_image_path,
        analyze_image_url,
        analyze_local_video_path,
        analyze_video_url,
    )

    result = None
    run_error = None
    local_image_url = None
    local_video_url = None
    annotated_video_url = None
    annotated_image_url = None
    xai_heatmap_url = None
    xai_baseline_conf = None
    xai_top_regions = []
    xai_report_markdown = None
    xai_report_error = None
    risk_level_label = None
    risk_level_tone = None
    risk_score_pct = 0
    priority_label = None
    budget_range_label = None
    action_plan = []
    ticket_data = None
    quality_data = None
    geo_url = ""
    pdf_report_url = ""
    alerting_data = None
    anti_fp_data = None
    route_history = []

    if request.method == "POST":
        form = RoadDamageAnalyzeForm(request.POST, request.FILES)
        if form.is_valid():
            imgf = form.cleaned_data.get("image_file")
            image_url = (form.cleaned_data.get("image_url") or "").strip()
            vf = form.cleaned_data.get("video_file")
            video_url = (form.cleaned_data.get("video_url") or "").strip()
            route_name = (form.cleaned_data.get("route_name") or "").strip()
            latitude = form.cleaned_data.get("latitude")
            longitude = form.cleaned_data.get("longitude")
            has_upload_video = bool(vf and getattr(vf, "name", "") and getattr(vf, "size", 0) > 0)
            try:
                annotated_video_rel = Path("road_damage_annotated") / f"{uuid4().hex}.mp4"
                annotated_video_full = Path(django_settings.MEDIA_ROOT) / annotated_video_rel
                annotated_img_rel = Path("road_damage_annotated") / f"{uuid4().hex}.jpg"
                annotated_img_full = Path(django_settings.MEDIA_ROOT) / annotated_img_rel
                if imgf:
                    ext = Path(imgf.name).suffix.lower() or ".jpg"
                    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
                        ext = ".jpg"
                    rel = Path("road_damage_images") / f"{uuid4().hex}{ext}"
                    full = Path(django_settings.MEDIA_ROOT) / rel
                    full.parent.mkdir(parents=True, exist_ok=True)
                    with open(full, "wb+") as out:
                        for chunk in imgf.chunks():
                            out.write(chunk)
                    from .services.road_damage_agents import assess_image_quality

                    quality_data = assess_image_quality(full)
                    if not quality_data.get("ok"):
                        run_error = quality_data.get("message") or "Qualite image insuffisante."
                        result = None
                        local_image_url = f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                    else:
                        local_image_url = f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                        result = analyze_image_path(full, annotated_output_path=annotated_img_full)
                elif image_url:
                    local_image_url = image_url
                    result = analyze_image_url(image_url, annotated_output_path=annotated_img_full)
                elif has_upload_video:
                    ext = Path(vf.name).suffix.lower()
                    if ext not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                        ext = ".mp4"
                    rel = Path("road_damage_videos") / f"{uuid4().hex}{ext}"
                    full = Path(django_settings.MEDIA_ROOT) / rel
                    full.parent.mkdir(parents=True, exist_ok=True)
                    with open(full, "wb+") as out:
                        for chunk in vf.chunks():
                            out.write(chunk)
                    local_video_url = f"{django_settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"
                    result = analyze_local_video_path(full, annotated_output_path=annotated_video_full)
                elif video_url:
                    local_video_url = video_url
                    result = analyze_video_url(video_url, annotated_output_path=annotated_video_full)
            except Exception as exc:  # pragma: no cover
                run_error = str(exc)

            ann_path = (result or {}).get("annotated_video_path")
            if ann_path:
                ann_full = Path(ann_path)
                if ann_full.is_file() and ann_full.stat().st_size > 0:
                    rel_ann = ann_full.relative_to(Path(django_settings.MEDIA_ROOT))
                    annotated_video_url = f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_ann.as_posix()}"
            ann_img_path = (result or {}).get("annotated_image_path")
            if ann_img_path:
                ann_img_full = Path(ann_img_path)
                if ann_img_full.is_file() and ann_img_full.stat().st_size > 0:
                    rel_ann_img = ann_img_full.relative_to(Path(django_settings.MEDIA_ROOT))
                    annotated_image_url = f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_ann_img.as_posix()}"

            # XAI occlusion + recommandations LLM (Groq) sur image.
            if result and not (result or {}).get("error") and bool((result or {}).get("is_image")):
                from .services.road_damage_xai import (
                    build_occlusion_heatmap,
                    generate_road_damage_report_with_groq,
                )

                xai_source_path = (result or {}).get("annotated_image_path")
                if xai_source_path:
                    xai_data = build_occlusion_heatmap(xai_source_path)
                    if xai_data.get("error"):
                        xai_report_error = xai_data["error"]
                    else:
                        heatmap_path = Path(xai_data["heatmap_path"])
                        if heatmap_path.is_file() and heatmap_path.stat().st_size > 0:
                            rel_hm = heatmap_path.relative_to(Path(django_settings.MEDIA_ROOT))
                            xai_heatmap_url = f"{django_settings.MEDIA_URL.rstrip('/')}/{rel_hm.as_posix()}"
                        xai_baseline_conf = xai_data.get("baseline_conf")
                        xai_top_regions = xai_data.get("top_regions") or []

                        report_data = generate_road_damage_report_with_groq(result=result, xai_data=xai_data)
                        if report_data.get("error"):
                            xai_report_error = report_data["error"]
                        else:
                            xai_report_markdown = report_data.get("report_markdown")

            if result and not (result or {}).get("error"):
                from .models import RoadDamageAnalysis
                from .services.road_damage_agents import (
                    anti_false_positive_adjust,
                    build_ticket,
                    compute_priority_and_plan,
                    export_pdf_report,
                    geo_link,
                    send_risk_alerts,
                )

                anti_fp_data = anti_false_positive_adjust(result)
                result = anti_fp_data["result"]
                geo_url = geo_link(latitude, longitude)
    else:
        form = RoadDamageAnalyzeForm()

    if result and not (result or {}).get("error"):
        max_conf = float((result or {}).get("max_conf") or 0.0)
        damage_ratio = float((result or {}).get("damage_ratio") or 0.0)
        verdict = str((result or {}).get("verdict") or "")
        risk_score = (0.7 * max_conf) + (0.3 * damage_ratio)
        if verdict != "damage":
            risk_score *= 0.6
        risk_score = max(0.0, min(1.0, risk_score))
        risk_score_pct = int(round(risk_score * 100))
        if risk_score >= 0.75:
            risk_level_label = "Risque eleve"
            risk_level_tone = "high"
        elif risk_score >= 0.45:
            risk_level_label = "Risque moyen"
            risk_level_tone = "medium"
        else:
            risk_level_label = "Risque faible"
            risk_level_tone = "low"

        from .services.road_damage_agents import (
            build_ticket,
            compute_priority_and_plan,
            export_pdf_report,
            send_risk_alerts,
        )

        plan_data = compute_priority_and_plan(
            risk_score=risk_score,
            confidence=max_conf,
            damage_ratio=damage_ratio,
        )
        priority_label = plan_data["priority"]
        budget_range_label = f"{plan_data['budget_min']} - {plan_data['budget_max']} TND"
        action_plan = plan_data["plan"]
        ticket_data = build_ticket(priority_label, (form.cleaned_data.get("route_name") or "").strip() if request.method == "POST" else "")
        alerting_data = send_risk_alerts(
            risk_score=risk_score,
            title=f"[RoadDamage] {risk_level_label}",
            message=f"Risque {risk_score_pct}% | Priorite {priority_label} | Ticket {ticket_data['ticket_id']}",
        )

        annotated_fs = ""
        if annotated_image_url:
            annotated_fs = str(Path(django_settings.MEDIA_ROOT) / annotated_image_url.replace(django_settings.MEDIA_URL, "").lstrip("/"))
        heatmap_fs = ""
        if xai_heatmap_url:
            heatmap_fs = str(Path(django_settings.MEDIA_ROOT) / xai_heatmap_url.replace(django_settings.MEDIA_URL, "").lstrip("/"))
        try:
            pdf_path = export_pdf_report(
                route_name=(form.cleaned_data.get("route_name") or "").strip() if request.method == "POST" else "",
                risk_label=risk_level_label,
                risk_pct=risk_score_pct,
                priority=priority_label,
                ticket_id=ticket_data["ticket_id"],
                report_text=xai_report_markdown or "",
                annotated_image_path=annotated_fs,
                heatmap_image_path=heatmap_fs,
            )
            pdf_rel = Path(pdf_path).relative_to(Path(django_settings.MEDIA_ROOT))
            pdf_report_url = f"{django_settings.MEDIA_URL.rstrip('/')}/{pdf_rel.as_posix()}"
        except Exception:
            pdf_report_url = ""

        if request.method == "POST":
            from .models import RoadDamageAnalysis

            route_name = (form.cleaned_data.get("route_name") or "").strip()
            latitude = form.cleaned_data.get("latitude")
            longitude = form.cleaned_data.get("longitude")
            source_ref = local_image_url or local_video_url or ""
            obj = RoadDamageAnalysis.objects.create(
                route_name=route_name,
                source_type="image" if bool((result or {}).get("is_image")) else "video",
                source_ref=source_ref[:580],
                annotated_image=annotated_image_url or "",
                heatmap_image=xai_heatmap_url or "",
                verdict=str(result.get("verdict") or ""),
                risk_score=risk_score,
                risk_level=risk_level_tone or "low",
                confidence_max=max_conf,
                damage_ratio=damage_ratio,
                priority=priority_label,
                estimated_budget_min=int(plan_data["budget_min"]),
                estimated_budget_max=int(plan_data["budget_max"]),
                action_plan=action_plan,
                quality=quality_data or {},
                xai_summary={"baseline_conf": xai_baseline_conf, "top_regions": xai_top_regions},
                llm_report=xai_report_markdown or "",
                ticket_id=ticket_data["ticket_id"],
                ticket_deadline_days=int(ticket_data["deadline_days"]),
                latitude=latitude,
                longitude=longitude,
                anti_fp_flag=bool((anti_fp_data or {}).get("corrected")),
                pdf_report_path=pdf_report_url or "",
            )

        route_filter = ""
        if request.method == "POST":
            route_filter = (form.cleaned_data.get("route_name") or "").strip()
        from .models import RoadDamageAnalysis
        qs = RoadDamageAnalysis.objects.all()
        if route_filter:
            qs = qs.filter(route_name__iexact=route_filter)
        route_history = list(qs[:8])

    _live_u = reverse("monitoring:reclamation_ai_live", kwargs={"pk": 987654321})
    reclamation_ai_live_url_template = _live_u.replace("987654321", "__PK__")
    _mission_u = reverse("monitoring:reclamation_mission_done", kwargs={"pk": 987654321})
    reclamation_mission_done_url_template = _mission_u.replace("987654321", "__PK__")

    return render(
        request,
        "monitoring/road_damage_test.html",
        {
            "form": form,
            "result": result,
            "run_error": run_error,
            "local_image_url": local_image_url,
            "local_video_url": local_video_url,
            "annotated_video_url": annotated_video_url,
            "annotated_image_url": annotated_image_url,
            "road_model_name": Path(getattr(django_settings, "ROAD_DAMAGE_MODEL_PATH", "")).name or "best (2).pt",
            "fight_upload_max_mb": int(getattr(django_settings, "FIGHT_UPLOAD_MAX_MB", 200)),
            "road_alert_pct": int(float(getattr(django_settings, "ROAD_DAMAGE_ALERT_MIN_CONF", 0.5)) * 100),
            "xai_heatmap_url": xai_heatmap_url,
            "xai_baseline_conf": xai_baseline_conf,
            "xai_top_regions": xai_top_regions,
            "xai_report_markdown": xai_report_markdown,
            "xai_report_error": xai_report_error,
            "risk_level_label": risk_level_label,
            "risk_level_tone": risk_level_tone,
            "risk_score_pct": risk_score_pct,
            "priority_label": priority_label,
            "budget_range_label": budget_range_label,
            "action_plan": action_plan,
            "ticket_data": ticket_data,
            "quality_data": quality_data,
            "geo_url": geo_url,
            "pdf_report_url": pdf_report_url,
            "alerting_data": alerting_data,
            "anti_fp_data": anti_fp_data,
            "route_history": route_history,
            "citizen_enriched": citizen_enriched,
            "citizen_kpis": citizen_kpis,
            "citizen_map_payload": citizen_map_payload,
            "citizen_priority_queue": citizen_priority_queue,
            "reclamation_ai_live_url_template": reclamation_ai_live_url_template,
            "reclamation_mission_done_url_template": reclamation_mission_done_url_template,
        },
    )


@require_POST
def reclamation_mission_done(request, pk):
    """Mark a citizen road damage case as completed and remove it from active cases."""
    rec = CitizenReclamation.objects.filter(pk=pk).first()
    if not rec:
        return JsonResponse({"success": False, "message": "Case not found"}, status=404)
    rec.delete()
    return JsonResponse({"success": True, "message": "Mission completed"})


def reclamation_ai_detail(request, pk):
    """Détail signalement citoyen : YOLO road damage, XAI, recommandation Groq / règles."""
    from collections import Counter
    from urllib.parse import urlencode

    from django.conf import settings as dj_settings
    from django.urls import reverse

    from . import ai_agents, intervention_copilot, vision_analysis, xai

    rec = get_object_or_404(CitizenReclamation, pk=pk)

    _reg = getattr(dj_settings, "ROAD_VISION_MODELS", {})
    _dk = getattr(dj_settings, "ROAD_VISION_MODEL_DEFAULT_KEY", "best2")
    _conf_default = float(
        getattr(
            dj_settings,
            "RECLAMATION_VISION_CONF_DEFAULT",
            dj_settings.ROAD_DAMAGE_YOLO_CONF,
        )
    )

    def _parse_conf(raw):
        try:
            if raw is None or str(raw).strip() == "":
                return None
            return float(str(raw).strip().replace(",", "."))
        except (TypeError, ValueError):
            return None

    def _is_live_request(req) -> bool:
        if (req.GET.get("live") or "").strip() == "1":
            return True
        esp_g = (req.GET.get("esp") or "").strip().lower()
        if esp_g in ("1", "true", "yes", "on"):
            return True
        if req.method == "POST" and req.POST.get("live_mode_carry") == "1":
            return True
        return False

    is_live_mode = _is_live_request(request)

    fc_live = None
    if is_live_mode:
        from monitoring.services.fire_cam.store import get_fire_config

        fc_live = get_fire_config()

    _gk = request.GET.get("model")
    _gk = (_gk or "").strip() or None
    _raw = None
    if _gk:
        _raw = _gk
    elif is_live_mode and fc_live is not None:
        dmk = (getattr(fc_live, "road_live_model_key", "") or "").strip()
        if dmk and dmk in _reg:
            _raw = dmk
    if not _raw:
        sk = request.session.get("vision_model_key")
        _raw = (sk or "").strip() or None
    if not _raw:
        _raw = _dk
    selected_model_key = _raw if _raw in _reg else _dk

    _gq = request.GET.get("conf")
    if _gq is not None and str(_gq).strip() != "":
        conf_threshold = _parse_conf(_gq)
    elif is_live_mode and fc_live is not None:
        lc = getattr(fc_live, "road_live_conf", None)
        conf_threshold = _parse_conf(lc) if lc is not None else None
        if conf_threshold is None:
            conf_threshold = _parse_conf(request.session.get("vision_conf"))
    else:
        conf_threshold = _parse_conf(request.session.get("vision_conf"))
    if conf_threshold is None:
        conf_threshold = _conf_default
    conf_threshold = max(0.01, min(1.0, float(conf_threshold)))

    if request.method == "POST" and request.POST.get("action") == "reanalyze":
        mk = (request.POST.get("model_key") or selected_model_key).strip()
        if mk not in _reg:
            mk = _dk
        pc = _parse_conf(request.POST.get("conf"))
        if pc is not None:
            conf_threshold = max(0.01, min(1.0, pc))
        request.session["vision_model_key"] = mk
        request.session["vision_conf"] = str(conf_threshold)
        messages.success(request, "Analyse actualisée.")
        base = reverse("monitoring:reclamation_ai_detail", kwargs={"pk": pk})
        qparams = [("model", mk), ("conf", str(conf_threshold))]
        if request.POST.get("live_mode_carry") == "1":
            qparams.append(("live", "1"))
        if request.POST.get("esp_mode_carry") == "1":
            qparams.append(("esp", "1"))
        rsc = (request.POST.get("refresh_sec_carry") or "").strip()
        if rsc:
            try:
                rsv = int(float(rsc))
                rsv = max(3, min(120, rsv))
                qparams.append(("refresh_sec", str(rsv)))
            except (TypeError, ValueError):
                pass
        return redirect(f"{base}?{urlencode(qparams)}")

    request.session["vision_model_key"] = selected_model_key
    request.session["vision_conf"] = str(conf_threshold)

    all_recs = list(CitizenReclamation.objects.all())
    vision = vision_analysis.analyze_reclamation_media(
        rec, model_key=selected_model_key, conf=conf_threshold
    )
    vision["detection_box_count"] = vision_analysis.count_detection_boxes(vision)
    pipeline = ai_agents.compute_ai_pipeline(rec, all_recs, vision=vision)

    if rec.media_type == "image":
        try:
            xai_result = xai.explain_reclamation_image(
                rec.media.path,
                model_key=selected_model_key,
                inference_conf=conf_threshold,
                yolo_used=bool(pipeline.get("yolo_used")),
                max_confidence=float(vision.get("max_confidence") or 0),
                boxes=list(vision.get("boxes") or []),
                pipeline_confidence=float(pipeline.get("confidence_score") or 0),
            )
        except Exception as exc:
            xai_result = {
                "ok": False,
                "fallback": True,
                "visual_overlay_only": False,
                "heatmap_url": None,
                "overlay_url": None,
                "explanation": (
                    f"Échec du calcul XAI : {exc}. La priorité du pipeline reste utilisable."
                ),
                "method": "error",
                "xai_quality_badge": "unavailable",
                "xai_quality_label_fr": "Indisponible",
                "model_confidence_pct": round(
                    float(vision.get("max_confidence") or 0) * 100, 1
                ),
                "legend_fr": "",
                "cached": False,
                "skip_reason_fr": str(exc),
                "detection_label_fr": "—",
            }
    else:
        xai_result = {
            "ok": False,
            "fallback": False,
            "visual_overlay_only": False,
            "heatmap_url": None,
            "overlay_url": None,
            "explanation": (
                "L’analyse XAI pixel n’est pas disponible pour les vidéos dans cette version — "
                "voir les aperçus YOLO ci-dessus."
            ),
            "method": "video_skip",
            "xai_quality_badge": "unavailable",
            "xai_quality_label_fr": "Indisponible",
            "model_confidence_pct": round(
                float(vision.get("max_confidence") or 0) * 100, 1
            ),
            "legend_fr": "",
            "cached": False,
            "detection_label_fr": "—",
        }

    cluster_ctx = {"nearby_count": int(pipeline.get("nearby_count") or 0)}
    copilot_plan = intervention_copilot.generate_intervention_plan(
        rec,
        pipeline,
        vision,
        xai_result or {},
        cluster_ctx,
    )
    groq_json = intervention_copilot.copilot_to_legacy_groq(copilot_plan)

    vision_filter_query = urlencode(
        {"model": selected_model_key, "conf": conf_threshold}
    )

    if request.GET.get("download") == "pdf":
        from .services.reclamation_pdf import export_reclamation_ai_pdf

        try:
            category_distribution = Counter(
                (getattr(r, "category", "") or "") for r in all_recs
            )
            pdf_path = export_reclamation_ai_pdf(
                report_pk=rec.pk,
                created_label=rec.created_at.strftime("%d/%m/%Y %H:%M"),
                category_label=rec.get_category_display(),
                description=rec.description or "",
                latitude=rec.latitude,
                longitude=rec.longitude,
                pipeline=pipeline,
                vision=vision,
                xai_result=xai_result or {},
                groq_json=groq_json,
                selected_model_key=selected_model_key,
                conf_threshold=conf_threshold,
                media_type=getattr(rec, "media_type", "image") or "image",
                original_media_url=(
                    rec.media.url if getattr(rec, "media", None) else None
                ),
                category_distribution=dict(category_distribution),
                total_system_reports=len(all_recs),
                copilot_plan=copilot_plan,
            )
            resp = FileResponse(
                open(pdf_path, "rb"),
                as_attachment=True,
                filename=f"signalement_{rec.pk}_rapport_ia.pdf",
            )
            resp["Content-Type"] = "application/pdf"
            return resp
        except Exception as exc:
            messages.error(request, f"Échec génération PDF : {exc}")
            return redirect(
                f"{reverse('monitoring:reclamation_ai_detail', kwargs={'pk': pk})}?{vision_filter_query}"
            )

    if request.GET.get("download") == "json":

        def _json_safe(o):
            if isinstance(o, dict):
                return {k: _json_safe(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_json_safe(v) for v in o]
            if hasattr(o, "item"):
                try:
                    return o.item()
                except Exception:
                    return float(o)
            if isinstance(o, (str, int, float, bool)) or o is None:
                return o
            return str(o)

        return JsonResponse(
            _json_safe(
                {
                    "report_id": rec.pk,
                    "pipeline": pipeline,
                    "vision": vision,
                    "xai": xai_result,
                    "groq": groq_json,
                    "copilot": {
                        k: v
                        for k, v in (copilot_plan or {}).items()
                        if k != "_meta"
                    },
                }
            ),
            json_dumps_params={"ensure_ascii": False, "indent": 2},
        )

    tone = pipeline.get("priority_tone") or "medium"
    map_color = (
        "#e53935"
        if tone in ("critical", "high")
        else ("#fb8c00" if tone == "medium" else "#43a047")
    )
    mini_map = None
    if rec.latitude is not None and rec.longitude is not None:
        mini_map = {
            "lat": float(rec.latitude),
            "lng": float(rec.longitude),
            "zoom": 15,
            "color": map_color,
            "id": rec.pk,
        }

    try:
        rs_raw = request.GET.get("refresh_sec")
        if (rs_raw is None or str(rs_raw).strip() == "") and request.method == "POST":
            rs_raw = request.POST.get("refresh_sec_carry")
        refresh_sec = int(float(rs_raw or 10))
    except (TypeError, ValueError):
        refresh_sec = 10
    refresh_sec = max(3, min(120, refresh_sec))

    esp_stream_src = ""
    esp_host = ""
    if is_live_mode:
        import time
        from urllib.parse import urlencode

        from monitoring.services.fire_cam.esp_net import resolve_esp_host
        from monitoring.services.fire_cam.store import get_fire_config

        cfg = get_fire_config()
        esp_host = resolve_esp_host(cfg) or ""
        stream_proxy_path = reverse("monitoring:esp_stream_proxy")
        if esp_host:
            esp_stream_src = f"{stream_proxy_path}?{urlencode({'esp_ip': esp_host, '_': str(int(time.time() * 1000))})}"

    esp_settings_url = reverse("monitoring:camera_settings")

    live_model_from_cam_db = False
    live_conf_from_cam_db = False
    if is_live_mode and fc_live is not None:
        if not (_gk):
            dm = (getattr(fc_live, "road_live_model_key", "") or "").strip()
            live_model_from_cam_db = bool(
                dm and dm in _reg and selected_model_key == dm
            )
        if (_gq is None or str(_gq).strip() == "") and getattr(
            fc_live, "road_live_conf", None
        ) is not None:
            try:
                dcf = float(fc_live.road_live_conf)
                live_conf_from_cam_db = (
                    abs(float(conf_threshold) - max(0.01, min(1.0, dcf))) < 1e-6
                )
            except (TypeError, ValueError):
                live_conf_from_cam_db = False

    return render(
        request,
        "monitoring/reclamation_ai_detail.html",
        {
            "rec": rec,
            "pipeline": pipeline,
            "vision": vision,
            "xai_result": xai_result,
            "groq_json": groq_json,
            "copilot_plan": copilot_plan,
            "dashboard_url": reverse("monitoring:road_damage_test"),
            "mini_map": mini_map,
            "vision_models": vision_analysis.list_models_for_ui(),
            "selected_model_key": selected_model_key,
            "conf_threshold": conf_threshold,
            "vision_filter_query": vision_filter_query,
            "segmentation_info": getattr(
                dj_settings, "ROAD_VISION_SEGMENTATION_INFO", {}
            ),
            "is_live_mode": is_live_mode,
            "refresh_sec": refresh_sec,
            "esp_ip": esp_host,
            "esp_host": esp_host,
            "esp_stream_src": esp_stream_src,
            "esp_settings_url": esp_settings_url,
            "esp_analyze_url": reverse(
                "monitoring:reclamation_ai_esp_analyze", kwargs={"pk": pk}
            ),
            "live_model_from_cam_db": live_model_from_cam_db,
            "live_conf_from_cam_db": live_conf_from_cam_db,
        },
    )


def reclamation_ai_live(request, pk):
    """Redirige vers l’analyse IA avec rechargement automatique (mode drone / temps réel)."""
    from urllib.parse import urlencode

    from django.shortcuts import redirect

    rs = (request.GET.get("refresh_sec") or "10").strip()
    try:
        rsv = int(float(rs))
        rsv = max(3, min(120, rsv))
    except (TypeError, ValueError):
        rsv = 10
    base = reverse("monitoring:reclamation_ai_detail", kwargs={"pk": pk})
    qd = {"live": "1", "refresh_sec": str(rsv)}
    if (request.GET.get("esp") or "").strip().lower() in ("1", "true", "yes", "on"):
        qd["esp"] = "1"
    q = urlencode(qd)
    return redirect(f"{base}?{q}")


def _json_safe_api(o):
    if isinstance(o, dict):
        return {k: _json_safe_api(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe_api(v) for v in o]
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            return float(o)
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


@require_POST
def reclamation_ai_esp_analyze(request, pk):
    """
    Un cliché ESP32 (/capture) + pipeline road damage (YOLO, agents, XAI optionnelle, copilot).
    Même logique métier que la page /reclamations/<id>/ai/ mais sur l’image temps réel.
    """
    import base64
    import json
    import uuid

    import cv2

    from django.conf import settings as dj_settings

    from monitoring.services.fire_cam.capture import fetch_jpeg_capture
    from monitoring.services.fire_cam.esp_net import base_url, resolve_esp_host
    from monitoring.services.fire_cam.store import get_fire_config

    from . import ai_agents, intervention_copilot, vision_analysis, xai

    rec = get_object_or_404(CitizenReclamation, pk=pk)

    try:
        body = json.loads(request.body.decode() or "{}")
    except json.JSONDecodeError:
        body = {}

    _reg = getattr(dj_settings, "ROAD_VISION_MODELS", {})
    _dk = getattr(dj_settings, "ROAD_VISION_MODEL_DEFAULT_KEY", "best2")
    mk = (body.get("model_key") or request.session.get("vision_model_key") or _dk).strip()
    if mk not in _reg:
        mk = _dk

    def _parse_conf(raw):
        try:
            if raw is None or str(raw).strip() == "":
                return None
            return float(str(raw).strip().replace(",", "."))
        except (TypeError, ValueError):
            return None

    _conf_default = float(
        getattr(
            dj_settings,
            "RECLAMATION_VISION_CONF_DEFAULT",
            dj_settings.ROAD_DAMAGE_YOLO_CONF,
        )
    )
    conf_threshold = _parse_conf(body.get("conf"))
    if conf_threshold is None:
        conf_threshold = _parse_conf(request.session.get("vision_conf"))
    if conf_threshold is None:
        conf_threshold = _conf_default
    conf_threshold = max(0.01, min(1.0, conf_threshold))

    full_xai = bool(body.get("full_xai"))
    if "rules_only" in body:
        rules_only = bool(body.get("rules_only"))
    else:
        rules_only = True

    manual_trigger = bool(body.get("manual_trigger"))
    logger.info(
        "[esp_analyze] POST pk=%s manual_trigger=%s model_key=%s conf=%.3f",
        pk,
        manual_trigger,
        mk,
        conf_threshold,
    )
    if manual_trigger:
        logger.info(
            "[esp_analyze] >>> clic bouton « Capturer et analyser » (signalement #%s)",
            pk,
        )

    cfg = get_fire_config()
    host = resolve_esp_host(cfg)
    bu = base_url(host)
    if not bu:
        logger.warning("[esp_analyze] no_ip pk=%s", pk)
        return JsonResponse(
            {
                "ok": False,
                "error": "no_ip",
                "message": (
                    "Aucune IP ESP : enregistrez-la dans Réglages caméra (module feu), "
                    "ou définissez DRONE_HOST dans .env."
                ),
            },
            status=400,
        )

    frame = fetch_jpeg_capture(bu)
    if frame is None:
        logger.warning(
            "[esp_analyze] no_frame pk=%s host=%s url=%s/capture",
            pk,
            host,
            bu.rstrip("/"),
        )
        return JsonResponse(
            {"ok": False, "error": "no_frame", "message": "Impossible de lire /capture sur l’ESP."},
            status=502,
        )

    fh, fw = frame.shape[:2]
    logger.info(
        "[esp_analyze] capture OK pk=%s frame=%sx%s manual=%s",
        pk,
        fw,
        fh,
        manual_trigger,
    )

    vision = vision_analysis.analyze_bgr_frame(frame, model_key=mk, conf=conf_threshold)
    vision["detection_box_count"] = vision_analysis.count_detection_boxes(vision)

    all_recs = list(CitizenReclamation.objects.all())
    pipeline = ai_agents.compute_ai_pipeline(rec, all_recs, vision=vision)

    if full_xai:
        analysis_dir = Path(dj_settings.MEDIA_ROOT) / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        xai_path = analysis_dir / f"esp_xai_{uuid.uuid4().hex}.jpg"
        try:
            cv2.imwrite(str(xai_path), frame)
            xai_result = xai.explain_reclamation_image(
                str(xai_path),
                model_key=mk,
                inference_conf=conf_threshold,
                yolo_used=bool(pipeline.get("yolo_used")),
                max_confidence=float(vision.get("max_confidence") or 0),
                boxes=list(vision.get("boxes") or []),
                pipeline_confidence=float(pipeline.get("confidence_score") or 0),
            )
        except Exception as exc:
            xai_result = {
                "ok": False,
                "fallback": True,
                "explanation": f"Échec XAI : {exc}",
                "method": "error",
            }
    else:
        nbox = int(vision.get("detection_box_count") or 0)
        labels = vision.get("labels") or []
        xai_result = {
            "ok": True,
            "fallback": True,
            "visual_overlay_only": False,
            "heatmap_url": None,
            "overlay_url": None,
            "explanation": (
                f"Image live ESP32 ({host}) — {nbox} détection(s). "
                f"Classes : {', '.join(labels) if labels else 'aucune'}. "
                "Activez « XAI complète » dans la requête pour heatmap occlusion (plus lent)."
            ),
            "method": "esp_live_lite",
            "xai_quality_badge": "lite",
            "xai_quality_label_fr": "Temps réel (léger)",
            "model_confidence_pct": round(float(vision.get("max_confidence") or 0) * 100, 1),
            "legend_fr": "",
            "cached": False,
            "detection_label_fr": "—",
        }

    cluster_ctx = {"nearby_count": int(pipeline.get("nearby_count") or 0)}
    copilot_plan = intervention_copilot.generate_intervention_plan(
        rec,
        pipeline,
        vision,
        xai_result or {},
        cluster_ctx,
        rules_only=rules_only,
    )
    groq_json = intervention_copilot.copilot_to_legacy_groq(copilot_plan)

    annotated_url_resp = vision.get("annotated_image_url") or ""
    seg_ann_url = vision.get("annotated_segmentation_image_url") or ""

    analysis_root = Path(dj_settings.MEDIA_ROOT) / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    capture_raw_image_url = None
    capture_raw_b64 = ""
    try:
        raw_disk = analysis_root / f"esp_capture_{uuid.uuid4().hex}.jpg"
        if cv2.imwrite(str(raw_disk), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]):
            rel = raw_disk.relative_to(Path(dj_settings.MEDIA_ROOT))
            capture_raw_image_url = (
                f"{dj_settings.MEDIA_URL.rstrip('/')}/"
                f"{rel.as_posix().replace(chr(92), '/')}"
            )
    except Exception:
        capture_raw_image_url = None
    if not capture_raw_image_url:
        ok_cap, buf_cap = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok_cap:
            capture_raw_b64 = base64.standard_b64encode(buf_cap.tobytes()).decode("ascii")

    # Réponses JSON légères : préférer les URLs média (parse navigateur / taille).
    jpeg_b64 = ""
    if not annotated_url_resp:
        ok_j, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok_j:
            jpeg_b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")

    logger.info(
        "[esp_analyze] DONE pk=%s manual=%s raw_url=%s raw_fallback_b64=%s nbox=%s cls_ann=%s seg_ann=%s",
        pk,
        manual_trigger,
        bool(capture_raw_image_url),
        bool(capture_raw_b64),
        vision.get("detection_box_count"),
        bool(annotated_url_resp),
        bool(seg_ann_url),
    )

    out = {
        "ok": True,
        "esp_host": host,
        "manual_trigger_echo": manual_trigger,
        "capture_raw_image_url": capture_raw_image_url or None,
        "capture_raw_base64": (capture_raw_b64 or None) if not capture_raw_image_url else None,
        "jpeg_b64": jpeg_b64 or None,
        "annotated_image_base64": jpeg_b64 or None,
        "annotated_segmentation_base64": None,
        "annotated_segmentation_image_url": seg_ann_url or None,
        "annotated_image_url": annotated_url_resp or None,
        "vision": vision,
        "pipeline": pipeline,
        "xai": xai_result,
        "copilot": {k: v for k, v in (copilot_plan or {}).items() if k != "_meta"},
        "groq": groq_json,
    }
    return JsonResponse(_json_safe_api(out), json_dumps_params={"ensure_ascii": False})


@csrf_exempt
@require_POST
def sightengine_callback(request):
    """Callback Sightengine (vidéo). Réponse minimale 200."""
    return JsonResponse({"status": "ok"})


def live_realtime(request):
    """Page dédiée flux MJPEG + stats (même pipeline YOLO que le dashboard)."""
    sources, active_source = _active_source_for_request(request)
    yolo_model_name = Path(django_settings.YOLO_MODEL_PATH).name
    return render(
        request,
        "monitoring/live_realtime.html",
        {
            "sources": sources,
            "active_source": active_source,
            "yolo_model_name": yolo_model_name,
        },
    )


def dashboard(request):
    sources, active_source = _active_source_for_request(request)
    alerts = list(Alert.objects.select_related("source")[:50])
    alert_totals = {
        row["alert_type"]: row["n"]
        for row in Alert.objects.values("alert_type").annotate(n=Count("id"))
    }
    total_alerts = sum(alert_totals.values())
    active_count = sum(1 for s in sources if s.is_active)
    crowd_n = int(alert_totals.get("crowd", 0))
    move_n = int(alert_totals.get("fast_movement", 0))
    overlap_n = int(alert_totals.get("high_overlap", 0))
    theft_n = int(alert_totals.get("theft_suspicious", 0))
    fight_n = int(alert_totals.get("fight", 0))
    weapon_n = int(alert_totals.get("weapon", 0))
    if total_alerts <= 0:
        donut_gradient = "conic-gradient(rgba(204,255,0,0.15) 0deg 360deg)"
    else:
        a = 360.0 * crowd_n / total_alerts
        b = a + 360.0 * move_n / total_alerts
        c = b + 360.0 * overlap_n / total_alerts
        d = c + 360.0 * theft_n / total_alerts
        e = d + 360.0 * fight_n / total_alerts
        f = e + 360.0 * weapon_n / total_alerts
        donut_gradient = (
            f"conic-gradient(var(--neon) 0deg {a}deg, "
            f"var(--chart-blue) {a}deg {b}deg, "
            f"var(--chart-orange) {b}deg {c}deg, "
            f"var(--chart-magenta) {c}deg {d}deg, "
            f"var(--danger) {d}deg {e}deg, "
            f"#c9a227 {e}deg {f}deg, "
            f"rgba(255,255,255,0.1) {f}deg 360deg)"
        )
    trend_bars = [38, 62, 45, 78, 52, 88]
    if total_alerts:
        base = min(95, 25 + total_alerts * 3)
        trend_bars = [min(95, max(18, base - i * 7)) for i in range(6)]
    yolo_model_name = Path(django_settings.YOLO_MODEL_PATH).name
    return render(
        request,
        "monitoring/dashboard.html",
        {
            "sources": sources,
            "active_source": active_source,
            "alerts": alerts,
            "alert_totals": alert_totals,
            "total_alerts": total_alerts,
            "active_count": active_count,
            "donut_gradient": donut_gradient,
            "trend_bars": trend_bars,
            "yolo_model_name": yolo_model_name,
        },
    )


def source_create(request):
    if request.method == "POST":
        form = QuickVideoForm(request.POST, request.FILES)
        if form.is_valid():
            cleaned = form.cleaned_data
            f = cleaned["video_file"]
            label = (cleaned.get("name") or "").strip()
            if not label:
                label = Path(f.name).stem or "Vidéo"
            obj = VideoSource(
                name=label[:120],
                source_type=VideoSource.SourceType.FILE,
                is_active=True,
            )
            obj.video_file = f
            obj.save()
            messages.success(request, "Vidéo ajoutée.")
            return redirect("monitoring:surveillance_dashboard")
    else:
        form = QuickVideoForm()
    return render(request, "monitoring/video_upload.html", {"form": form})


def source_edit(request, pk):
    source = get_object_or_404(VideoSource, pk=pk)
    if request.method == "POST":
        form = QuickVideoEditForm(request.POST, request.FILES, instance=source)
        if form.is_valid():
            obj = form.save()
            messages.success(request, "Enregistré.")
            return redirect("monitoring:surveillance_dashboard")
    else:
        form = QuickVideoEditForm(instance=source)
    return render(
        request,
        "monitoring/video_edit.html",
        {"form": form, "source": source},
    )


def source_delete(request, pk):
    source = get_object_or_404(VideoSource, pk=pk)
    if request.method == "POST":
        source.delete()
        messages.success(request, "Source supprimée.")
        return redirect("monitoring:surveillance_dashboard")
    return render(request, "monitoring/source_confirm_delete.html", {"source": source})


@xframe_options_exempt
def video_stream(request, pk):
    from monitoring.runtime import ml_unavailable_http, ml_deps_available

    if not ml_deps_available():
        return ml_unavailable_http(request, feature="Live video stream (YOLO)")
    from .services.detection import stream_mjpeg_frames

    source = get_object_or_404(VideoSource, pk=pk, is_active=True)
    return StreamingHttpResponse(
        stream_mjpeg_frames(source),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


@never_cache
def stream_stats(request, pk):
    from monitoring.runtime import ml_unavailable_json, ml_deps_available

    if not ml_deps_available():
        return ml_unavailable_json("Live stream stats (YOLO)")
    from .services.detection import get_live_stats

    get_object_or_404(VideoSource, pk=pk)
    return JsonResponse(get_live_stats(pk))


def stream_page(request, pk):
    source = get_object_or_404(VideoSource, pk=pk)
    if not source.is_active:
        messages.warning(request, "Source inactive — activez-la dans les réglages.")
    return redirect("monitoring:surveillance_dashboard")


def homepage(request):
    """MedinaMind landing page — Smart City platform introduction."""
    from .team_data import TEAM_MEMBERS

    return render(
        request,
        "monitoring/homepage.html",
        {"team_members": TEAM_MEMBERS},
    )


@ensure_csrf_cookie
def ai_dashboard(request):
    """Central analytics dashboard for AI model usage (demo data via JSON API)."""
    bootstrap = build_ai_dashboard_summary()
    return render(
        request,
        "monitoring/ai_dashboard.html",
        {"dashboard_bootstrap_json": json.dumps(bootstrap)},
    )


@never_cache
@require_GET
def ai_dashboard_api_summary(request):
    """JSON summary for charts, KPIs, and tables. Swap `build_ai_dashboard_summary` for DB-backed stats."""
    payload = build_ai_dashboard_summary(
        model_filter=(request.GET.get("model") or "").strip() or None,
        date_from=(request.GET.get("date_from") or "").strip() or None,
        date_to=(request.GET.get("date_to") or "").strip() or None,
    )
    if payload["filters"].get("model") in (None, "", "all"):
        payload["filters"]["model"] = "all"
    return JsonResponse(payload)


_AI_DASH_CHAT_SYSTEM = (
    "You are the MedinaMind Smart City dashboard assistant. "
    "Help users understand city monitoring, safety, traffic, roads, waste, drones, and alerts. "
    "Reply in clear, professional English. "
    "You do not have access to live sensor feeds or internal databases — if asked for live data, "
    "explain that and offer general guidance or best practices."
)

_AI_ASSISTANT_NOT_CONFIGURED = (
    "AI assistant is not configured. Please contact the administrator."
)
_AI_ASSISTANT_MISSING_KEY = (
    "AI assistant is not configured. Missing OPENAI_API_KEY on the server."
)
_AI_ASSISTANT_UNAVAILABLE = (
    "The AI assistant is temporarily unavailable. Please try again later."
)


def _ai_assistant_client_error(status_code: int, provider_message: str = "") -> tuple[str, str, int]:
    """Map provider failures to safe client messages (never expose API keys)."""
    raw = (provider_message or "").lower()
    if status_code in (401, 403) or "api key" in raw or "incorrect" in raw:
        return _AI_ASSISTANT_UNAVAILABLE, "assistant_unavailable", 502
    if status_code == 503:
        return _AI_ASSISTANT_NOT_CONFIGURED, "assistant_not_configured", 503
    return _AI_ASSISTANT_UNAVAILABLE, "assistant_unavailable", 502


@never_cache
@require_POST
def ai_dashboard_chatbot(request):
    """Proxy chat OpenAI — OPENAI_API_KEY from environment only (never sent to the client)."""
    log_openai_key_status("AI dashboard chat")
    api_key = get_openai_api_key()
    if not api_key:
        logger.warning("AI dashboard chat: OPENAI_API_KEY is missing after env load")
        return JsonResponse(
            {
                "error": _AI_ASSISTANT_MISSING_KEY,
                "code": "assistant_not_configured",
            },
            status=503,
        )
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)
    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return JsonResponse({"error": "The messages field (list) is required."}, status=400)

    cleaned: list[dict[str, str]] = []
    for item in raw_messages[-32:]:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").strip()
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        cleaned.append({"role": role, "content": content[:12000]})
    if not cleaned:
        return JsonResponse({"error": "No valid user or assistant message."}, status=400)

    model = (os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()
    api_messages = [{"role": "system", "content": _AI_DASH_CHAT_SYSTEM}] + cleaned

    client = create_openai_client()
    if client is None:
        logger.warning("AI dashboard chat: OpenAI client could not be created")
        return JsonResponse(
            {
                "error": _AI_ASSISTANT_MISSING_KEY,
                "code": "assistant_not_configured",
            },
            status=503,
        )

    try:
        from openai import APIConnectionError, AuthenticationError, OpenAIError, RateLimitError

        completion = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=0.55,
            max_tokens=1200,
        )
    except AuthenticationError as exc:
        logger.warning("OpenAI chat authentication failed: %s", exc)
        user_msg, code, http_status = _ai_assistant_client_error(401, str(exc))
        return JsonResponse({"error": user_msg, "code": code}, status=http_status)
    except (APIConnectionError, RateLimitError) as exc:
        logger.warning("OpenAI chat request failed: %s", exc)
        return JsonResponse(
            {
                "error": _AI_ASSISTANT_UNAVAILABLE,
                "code": "assistant_unavailable",
            },
            status=502,
        )
    except OpenAIError as exc:
        logger.warning("OpenAI chat error: %s", exc)
        user_msg, code, http_status = _ai_assistant_client_error(
            getattr(exc, "status_code", None) or 502,
            str(exc),
        )
        return JsonResponse({"error": user_msg, "code": code}, status=http_status)
    except Exception as exc:
        logger.warning("OpenAI chat unexpected error: %s", exc)
        return JsonResponse(
            {
                "error": _AI_ASSISTANT_UNAVAILABLE,
                "code": "assistant_unavailable",
            },
            status=502,
        )

    reply = ""
    if completion.choices:
        reply = (completion.choices[0].message.content or "").strip()
    if not reply:
        logger.warning("OpenAI chat returned an empty or unexpected response")
        return JsonResponse(
            {
                "error": _AI_ASSISTANT_UNAVAILABLE,
                "code": "assistant_unavailable",
            },
            status=502,
        )
    return JsonResponse({"reply": reply})


def introduction_page(request):
    """MedinaMind interactive introduction page with webcam + hologram city."""
    return render(request, "monitoring/introduction.html", {})


def ar_live_stats(request):
    """GET /api/ar/live-stats/ — city stats for AR left panel."""
    from monitoring.ar_api import build_ar_live_stats

    return JsonResponse(build_ar_live_stats())


def ar_live_info(request):
    """GET /api/ar/live-info/ — time, date, weather for AR panel."""
    from monitoring.ar_api import build_ar_live_info

    return JsonResponse(build_ar_live_info())


def ar_modules(request):
    """GET /api/ar/modules/ — smart city module list."""
    from monitoring.ar_api import build_ar_modules

    return JsonResponse(build_ar_modules())
