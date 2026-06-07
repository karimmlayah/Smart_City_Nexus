"""Mission Mayor: Save the City — page and game API views."""
from __future__ import annotations

import json
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt, ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from monitoring.runtime import ML_UNAVAILABLE_MESSAGE, ml_deps_available

logger = logging.getLogger(__name__)

TOTAL_ROUNDS = 5
ROUND_SECONDS = 55


def _demo():
    from monitoring import demo_mayor_service as demo

    return demo


def _ml_guard_json():
    return JsonResponse(
        {"success": False, "error": ML_UNAVAILABLE_MESSAGE},
        status=503,
    )


@ensure_csrf_cookie
@never_cache
def mayor_mission_page(request):
    return render(
        request,
        "monitoring/mayor_mission.html",
        {
            "challenges_url": reverse("monitoring:mayor_mission_challenges"),
            "classic_url": reverse("monitoring:mayor_mission_run_classic"),
            "medinamind_url": reverse("monitoring:mayor_mission_run_medinamind"),
            "finish_url": reverse("monitoring:mayor_mission_finish"),
            "certificate_url": reverse("monitoring:mayor_mission_certificate"),
            "total_rounds": TOTAL_ROUNDS,
            "round_seconds": ROUND_SECONDS,
        },
    )


@never_cache
@require_GET
def mayor_mission_challenges(request):
    if not ml_deps_available():
        return _ml_guard_json()
    demo = _demo()
    return JsonResponse({"challenges": demo.get_challenges(), "total_rounds": TOTAL_ROUNDS})


@never_cache
@require_GET
def mayor_mission_scenarios(request):
    if not ml_deps_available():
        return _ml_guard_json()
    demo = _demo()
    return JsonResponse({"scenarios": demo.get_scenarios(), "total_rounds": TOTAL_ROUNDS})


@csrf_exempt
@never_cache
@require_POST
def mayor_mission_run_classic(request):
    if not ml_deps_available():
        return _ml_guard_json()
    demo = _demo()
    body = _parse_body(request)
    challenge_id = (body.get("challenge_id") or body.get("scenario_id") or "").strip()
    if not challenge_id:
        return JsonResponse({"success": False, "error": "challenge_id required."}, status=400)
    timed_out = bool(body.get("timed_out"))
    result = demo.run_classic_round(challenge_id, timed_out=timed_out)
    status = 200 if result.get("success") else 404
    return JsonResponse(result, status=status)


@csrf_exempt
@never_cache
@require_POST
def mayor_mission_run_medinamind(request):
    if not ml_deps_available():
        return _ml_guard_json()
    demo = _demo()
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON."}, status=400)
    challenge_id = (body.get("challenge_id") or body.get("scenario_id") or "").strip()
    if not challenge_id:
        return JsonResponse({"success": False, "error": "challenge_id required."}, status=400)
    inputs = body.get("inputs") if isinstance(body.get("inputs"), dict) else {}
    result = demo.run_medinamind_round(challenge_id, inputs)
    status = 200 if result.get("success") else 404
    return JsonResponse(result, status=status)


@csrf_exempt
@never_cache
@require_POST
def mayor_mission_finish(request):
    if not ml_deps_available():
        return _ml_guard_json()
    demo = _demo()
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON."}, status=400)
    return JsonResponse(demo.finish_mission(body))


@csrf_exempt
@never_cache
@require_POST
def mayor_mission_run(request):
    if not ml_deps_available():
        return _ml_guard_json()
    demo = _demo()
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON."}, status=400)
    cid = (body.get("scenario_id") or body.get("challenge_id") or "").strip()
    mode = (body.get("mode") or "classic").strip().lower()
    inputs = body.get("inputs") if isinstance(body.get("inputs"), dict) else {}
    if not cid:
        return JsonResponse({"ok": False, "error": "challenge_id required."}, status=400)
    if mode == "medinamind":
        result = demo.run_medinamind_simulation(cid, inputs)
    else:
        result = demo.run_classic_simulation(cid)
    status = 200 if result.get("ok") or result.get("success") else 404
    return JsonResponse(result, status=status)


@csrf_exempt
@never_cache
@require_POST
def mayor_mission_certificate(request):
    if not ml_deps_available():
        return _ml_guard_json()
    demo = _demo()
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Invalid JSON."}, status=400)
    state = body.get("game_state") or body.get("result") or body
    mayor_name = (body.get("mayor_name") or "Mayor").strip()[:80]
    cert = demo.build_certificate_payload(state, mayor_name=mayor_name)
    return JsonResponse({"ok": True, "certificate": cert})


def _parse_body(request) -> dict:
    try:
        return json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return {}


def _parse_challenge_id(request) -> str:
    body = _parse_body(request)
    return (body.get("challenge_id") or body.get("scenario_id") or "").strip()
