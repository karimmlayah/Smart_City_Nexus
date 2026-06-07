"""Mission Mayor: Save the City — interactive game orchestration + real AI models."""
from __future__ import annotations

import base64
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

DEFAULT_ROI = [(40, 40), (600, 40), (600, 440), (40, 440)]
TOTAL_ROUNDS = 5
ROUND_SECONDS = 55
MEDIA_SUBDIR = "mayor_mission"

# Local incident images — source folder + static URL prefix for the browser
MISSION_ASSETS_DIR = Path(settings.BASE_DIR) / "images" / "game"
MISSION_STATIC_PREFIX = "images/game"

# 4,311,489 MAD ≈ 1,364,026 TND
MAD_TO_TND_RATIO = 1_364_026 / 4_311_489

MODULE_LABELS = {
    "traffic": "Traffic Nexus",
    "road_damage": "Road Analysis",
    "waste": "Street Waste",
    "fire_smoke": "Fire & Smoke",
    "surveillance": "City Monitoring",
    "uav": "Drone Monitoring",
    "traffic_signal": "Smart Traffic Lights",
}

CHALLENGES: list[dict[str, Any]] = [
    {
        "id": "traffic_crisis",
        "title": "Traffic Accident Blocking Emergency Access",
        "story": "A crash has blocked the main corridor. Ambulances cannot reach the scene in time.",
        "zone": "Central Corridor",
        "severity": "critical",
        "module": "traffic",
        "asset_file": "ambulance.jpg",
        "classic_penalty": 25,
        "medinamind_reward": 35,
        "classic_health_delta": 22,
        "medinamind_health_delta": 18,
        "map_markers": ["traffic", "emergency"],
    },
    {
        "id": "road_damage_crisis",
        "title": "Road Damage Emergency",
        "story": "Dangerous cracks are spreading on a key route. Delay means more risk for everyone.",
        "zone": "North Emergency Spine",
        "severity": "high",
        "module": "road_damage",
        "asset_file": "cracks.jpg",
        "classic_penalty": 22,
        "medinamind_reward": 32,
        "classic_health_delta": 20,
        "medinamind_health_delta": 16,
        "map_markers": ["road", "traffic"],
    },
    {
        "id": "waste_crisis",
        "title": "Waste Hotspot Alert",
        "story": "Waste is piling up near a public area. Health and safety risks are rising fast.",
        "zone": "Civic Square",
        "severity": "high",
        "module": "waste",
        "asset_file": "garbage.jpg",
        "classic_penalty": 20,
        "medinamind_reward": 30,
        "classic_health_delta": 18,
        "medinamind_health_delta": 14,
        "map_markers": ["waste", "citizens"],
    },
    {
        "id": "fire_smoke_crisis",
        "title": "Urban Fire Risk",
        "story": "Smoke is rising near dense housing. Every minute of delay increases fire spread.",
        "zone": "Old Quarter",
        "severity": "critical",
        "module": "fire_smoke",
        "asset_file": "fire.jpg",
        "classic_penalty": 28,
        "medinamind_reward": 38,
        "classic_health_delta": 25,
        "medinamind_health_delta": 20,
        "map_markers": ["fire", "alert"],
    },
    {
        "id": "uav_crisis",
        "title": "Building Damage / Infrastructure Risk",
        "story": "Structural damage threatens a critical zone. Ground teams need instant AI prioritization.",
        "zone": "River Bridge Sector",
        "severity": "critical",
        "module": "uav",
        "asset_file": "batiment.png",
        "classic_penalty": 26,
        "medinamind_reward": 36,
        "classic_health_delta": 24,
        "medinamind_health_delta": 19,
        "map_markers": ["uav", "road", "alert"],
    },
]

CLASSIC_BASELINES: dict[str, dict[str, Any]] = {
    "traffic_crisis": {"response_delay_min": 32, "citizens_impacted": 1240, "damage_increase_tnd": 474_526, "cost_tnd": 151_872},
    "road_damage_crisis": {"response_delay_min": 28, "citizens_impacted": 890, "damage_increase_tnd": 379_621, "cost_tnd": 123_398},
    "waste_crisis": {"response_delay_min": 24, "citizens_impacted": 620, "damage_increase_tnd": 215_118, "cost_tnd": 66_444},
    "fire_smoke_crisis": {"response_delay_min": 19, "citizens_impacted": 980, "damage_increase_tnd": 569_431, "cost_tnd": 101_248},
    "uav_crisis": {"response_delay_min": 35, "citizens_impacted": 1100, "damage_increase_tnd": 664_337, "cost_tnd": 164_528},
}


def mad_to_tnd(mad: int | float) -> int:
    return int(round(float(mad) * MAD_TO_TND_RATIO))


def _asset_path(filename: str) -> Path | None:
    if not filename:
        return None
    local = MISSION_ASSETS_DIR / filename
    if local.is_file():
        return local
    static = Path(settings.BASE_DIR) / "static" / MISSION_STATIC_PREFIX / filename
    if static.is_file():
        return static
    return None


def _asset_url(filename: str) -> str:
    static_base = settings.STATIC_URL.rstrip("/")
    return f"{static_base}/{MISSION_STATIC_PREFIX}/{filename}"


def _load_challenge_frame(challenge_id: str) -> np.ndarray:
    challenge = get_challenge(challenge_id)
    if challenge:
        path = _asset_path(challenge.get("asset_file", ""))
        if path:
            frame = cv2.imread(str(path))
            if frame is not None:
                return frame
    return _scenario_frame(challenge_id)


def get_challenges() -> list[dict[str, Any]]:
    return [_public_challenge(c) for c in CHALLENGES]


def get_challenge(challenge_id: str) -> dict[str, Any] | None:
    for c in CHALLENGES:
        if c["id"] == challenge_id:
            return dict(c)
    return None


def get_scenarios() -> list[dict[str, Any]]:
    return get_challenges()


def get_scenario(scenario_id: str) -> dict[str, Any] | None:
    return get_challenge(scenario_id)


def _public_challenge(c: dict[str, Any]) -> dict[str, Any]:
    out = dict(c)
    asset = c.get("asset_file") or ""
    url = _asset_url(asset) if asset and _asset_path(asset) else ""
    out["problem_image_url"] = url
    out["worse_image_url"] = url  # same image — frontend applies damage effects
    out["modules"] = [c["module"]]
    out.pop("asset_file", None)
    return out


def _media_dir() -> Path:
    d = Path(settings.MEDIA_ROOT) / MEDIA_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _media_url(rel: Path) -> str:
    return f"{settings.MEDIA_URL.rstrip('/')}/{rel.as_posix()}"


def _scenario_frame(challenge_id: str) -> np.ndarray:
    w, h = 640, 480
    rng = np.random.default_rng(abs(hash(challenge_id)) % 2**32)
    palettes = {
        "traffic_crisis": (35, 42, 58),
        "road_damage_crisis": (40, 38, 52),
        "waste_crisis": (28, 48, 38),
        "fire_smoke_crisis": (48, 32, 28),
        "uav_crisis": (22, 40, 58),
    }
    base = np.full((h, w, 3), palettes.get(challenge_id, (30, 38, 50)), dtype=np.uint8)
    for _ in range(100):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(base, (x, y), int(rng.integers(3, 16)), (int(rng.integers(70, 220)),) * 3, -1)
    cv2.rectangle(base, (0, h // 2 - 24), (w, h // 2 + 24), (75, 78, 88), -1)
    label = challenge_id.replace("_", " ")[:28].upper()
    cv2.putText(base, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 230, 255), 2)
    return base


def _decode_upload_image(image_b64: str | None) -> np.ndarray | None:
    if not image_b64:
        return None
    try:
        raw = image_b64.split(",", 1)[-1]
        buf = np.frombuffer(base64.b64decode(raw), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning("mayor image decode failed: %s", exc)
        return None


def _bgr_to_data_url(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return ""
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _module_result(module: str, status: str, data: dict | None = None, error: str | None = None) -> dict[str, Any]:
    return {"module": module, "status": status, "data": data or {}, "error": error}


def _load_fire_class_names(model_path: Path) -> list[str]:
    candidates = [
        model_path.parent / "classes.txt",
        Path(settings.BASE_DIR) / "assets" / "models" / "classes.txt",
        Path(settings.BASE_DIR) / "smart_crowd_safety_ai" / "assets" / "models" / "classes.txt",
    ]
    for txt in candidates:
        if txt.is_file():
            names = [ln.strip() for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
            if names:
                return names
    return []


def _is_fire_smoke_class(name: str) -> bool:
    n = name.lower().replace("é", "e").replace("è", "e")
    return any(k in n for k in ("fire", "smoke", "feu", "fumee", "fumée", "fumee"))


def _run_traffic(frame: np.ndarray) -> dict[str, Any]:
    """Traffic / ambulance — direct YOLO (évite l'import yt_dlp de traffic_nexus_inference)."""
    try:
        from ultralytics import YOLO

        from monitoring.traffic_nexus_helpers import VEHICLE_LABELS
        from monitoring.traffic_tdss.vehicle_detection import detect_in_zones_multi

        h, w = frame.shape[:2]
        roi = [[(0, 0), (w, 0), (w, h), (0, h)]]
        defaults = getattr(settings, "TRAFFIC_NEXUS_MODEL_PATHS", {}) or {}
        p_best = defaults.get("best") or defaults.get("yolo11n")
        p_yolo = defaults.get("yolov8n") or defaults.get("yolov8m")
        if not p_best or not Path(p_best).is_file():
            return _module_result("traffic", "fallback", error="Ambulance model (best.pt) not found")
        mb = YOLO(str(p_best))
        mn = YOLO(str(p_yolo)) if p_yolo and Path(p_yolo).is_file() else mb
        det = detect_in_zones_multi(
            models=[mb, mn],
            frame=frame,
            roi_polygons=roi,
            allowed_labels=set(VEHICLE_LABELS),
            imgsz=640,
            max_det=80,
            draw_boxes=True,
        )
        vd = det.vehicle_detections or []
        amb = [d for d in vd if "ambul" in str(d.get("label", "")).lower()]
        cars = [d for d in vd if str(d.get("label", "")).lower().startswith("car")]
        total = int(det.detections_total or len(vd))
        amb_count = len(amb)
        max_amb_conf = max((float(d.get("confidence") or 0.85) for d in amb), default=0.0)
        conf = max(0.55, max_amb_conf) if amb_count else min(0.85, 0.45 + total * 0.02)
        severity = "Critical" if amb_count and total >= 8 else "High" if total >= 5 else "Moderate"
        if amb_count:
            issue = f"{amb_count} ambulance(s) detected — {total} vehicles blocking corridor"
        else:
            issue = f"{total} vehicles detected — emergency route congested"
        ann = _bgr_to_data_url(det.processed_frame) if det.processed_frame is not None else ""
        return _module_result("traffic", "live", {
            "vehicle_count": total,
            "ambulance_count": amb_count,
            "car_count": len(cars),
            "congestion_level": "high" if total >= 8 else "moderate",
            "confidence": round(conf, 3),
            "severity": severity,
            "detected_issue": issue,
            "recommendation": "AI reroutes traffic and clears the emergency corridor.",
            "annotated_image_url": ann,
        })
    except Exception as exc:
        logger.warning("mayor traffic: %s", exc)
        return _module_result("traffic", "fallback", error=str(exc))


def _run_road_damage(frame: np.ndarray) -> dict[str, Any]:
    try:
        from monitoring.services.road_damage_tester import analyze_image_path

        ann_path = _media_dir() / f"road_{uuid.uuid4().hex}.jpg"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            src = tmp.name
        try:
            out = analyze_image_path(src, annotated_output_path=ann_path)
        finally:
            Path(src).unlink(missing_ok=True)
        if out.get("error"):
            return _module_result("road_damage", "fallback", error=str(out.get("error")))
        ann_url = ""
        if ann_path.is_file():
            rel = ann_path.relative_to(Path(settings.MEDIA_ROOT))
            ann_url = _media_url(rel)
        conf = float(out.get("max_conf") or 0)
        dets_detail = (out.get("details") or [{}])[0].get("damage_compact") or ""
        issue = out.get("verdict_label") or "Road surface analyzed"
        if dets_detail:
            issue = f"Cracks / damage detected: {dets_detail}"
        elif out.get("verdict") == "damage":
            issue = "Road cracks and pavement damage detected"
        return _module_result("road_damage", "live", {
            "verdict": out.get("verdict"),
            "max_conf": conf,
            "confidence": conf,
            "severity": "High" if out.get("verdict") == "damage" else "Moderate",
            "detected_issue": issue,
            "recommendation": "AI detects the damage and recommends the fastest response.",
            "annotated_image_url": ann_url,
        })
    except Exception as exc:
        logger.warning("mayor road: %s", exc)
        return _module_result("road_damage", "fallback", error=str(exc))


def _run_waste(frame: np.ndarray) -> dict[str, Any]:
    try:
        from monitoring.services.waste_detection import detect_waste_bgr

        out = detect_waste_bgr(frame, location_hint="mayor-mission")
        if not out.get("success"):
            return _module_result("waste", "fallback", error=str(out.get("error") or "unavailable"))
        conf = float(out.get("max_confidence") or 0)
        count = int(out.get("count") or 0)
        labels = ", ".join(
            sorted({str(d.get("label", "")) for d in (out.get("detections") or []) if d.get("label")})[:4]
        )
        issue = f"{count} waste object(s) detected"
        if labels:
            issue = f"Garbage detected ({count}): {labels}"
        return _module_result("waste", "live", {
            "count": count,
            "severity": out.get("severity"),
            "confidence": conf,
            "detected_issue": issue,
            "recommendation": out.get("recommended_action") or "AI prioritizes cleanup teams to the hotspot.",
            "annotated_image_url": out.get("annotated_image_url") or "",
        })
    except Exception as exc:
        logger.warning("mayor waste: %s", exc)
        return _module_result("waste", "fallback", error=str(exc))


def _run_fire_smoke(frame: np.ndarray) -> dict[str, Any]:
    try:
        path = Path(str(getattr(settings, "FIRE_ONNX_MODEL_PATH", "") or ""))
        if not path.is_file():
            return _module_result("fire_smoke", "fallback", error=f"ONNX not found: {path}")
        from monitoring.services.fire_cam.onnx_yolo import YoloOnnx

        yolo = YoloOnnx(path, conf_thres=0.40, iou_thres=0.45)
        class_names = _load_fire_class_names(path)
        if class_names:
            yolo.class_names = class_names
        dets = yolo.predict_boxes(frame)
        fire_like = [d for d in dets if _is_fire_smoke_class(yolo.class_names[d[0]])]
        # Garder les détections les plus confiantes pour l'affichage
        fire_like.sort(key=lambda d: d[1], reverse=True)
        top = fire_like[:20]
        max_conf = float(top[0][1]) if top else 0.0
        drawn = yolo.draw_boxes(frame.copy(), top if top else dets[:15])
        classes = []
        for d in top[:5]:
            name = yolo.class_names[d[0]]
            if name not in classes:
                classes.append(name)
        if classes:
            issue = f"Fire / smoke detected: {', '.join(classes)}"
        elif fire_like:
            issue = f"Fire / smoke alert ({len(fire_like)} detection(s))"
        else:
            issue = "Scene analyzed — no fire confirmed"
        return _module_result("fire_smoke", "live", {
            "detection_count": len(fire_like),
            "confidence": round(max_conf, 4) if max_conf else 0.72,
            "severity": "Critical" if max_conf >= 0.45 else "Moderate",
            "detected_issue": issue,
            "recommendation": "AI confirms the alert and dispatches fire units." if fire_like else "Continue monitoring.",
            "annotated_image_url": _bgr_to_data_url(drawn),
        })
    except Exception as exc:
        logger.warning("mayor fire: %s", exc)
        return _module_result("fire_smoke", "fallback", error=str(exc))


def _run_uav(frame: np.ndarray) -> dict[str, Any]:
    try:
        from monitoring.services.uav_cnn_inference import predict_structural_state

        out = predict_structural_state(frame)
        if not out.get("success"):
            return _module_result("uav", "fallback", error=str(out.get("error") or out.get("message")))
        conf = float(out.get("confidence") or 0)
        label = out.get("predicted_label") or out.get("badge_key") or "Unknown"
        badge = str(out.get("badge_key") or "").capitalize() or label
        return _module_result("uav", "live", {
            "label": label,
            "confidence": conf,
            "severity": badge if badge.lower() in ("damaged", "collapsed", "intact") else label,
            "detected_issue": f"Building state: {label} ({int(conf * 100)}% confidence)",
            "recommendation": "AI flags structural risk and restricts access.",
            "annotated_image_url": out.get("gradcam_image_url") or out.get("original_image_url") or "",
        })
    except Exception as exc:
        logger.warning("mayor uav: %s", exc)
        return _module_result("uav", "fallback", error=str(exc))


def _run_surveillance(frame: np.ndarray) -> dict[str, Any]:
    try:
        from monitoring.services.fight_classifier import analyze_live_frame_bgr

        out = analyze_live_frame_bgr(frame, enable_fight=True, enable_weapon=True)
        if not out.get("ok", True) and out.get("error"):
            return _module_result("surveillance", "fallback", error=str(out.get("error")))
        fight = out.get("fight") or {}
        weapon = out.get("weapon") or {}
        conf = float(fight.get("top1conf") or 0)
        return _module_result("surveillance", "live", {
            "fight_label": fight.get("label"),
            "confidence": conf,
            "severity": "High" if conf > 0.5 else "Moderate",
            "detected_issue": f"Safety signal: {fight.get('label', 'monitoring')}",
            "recommendation": "Dispatch safety units and generate incident report.",
            "annotated_image_url": _bgr_to_data_url(frame),
        })
    except Exception as exc:
        logger.warning("mayor surveillance: %s", exc)
        return _module_result("surveillance", "fallback", error=str(exc))


MODULE_RUNNERS = {
    "traffic": _run_traffic,
    "road_damage": _run_road_damage,
    "waste": _run_waste,
    "fire_smoke": _run_fire_smoke,
    "uav": _run_uav,
    "surveillance": _run_surveillance,
}


def run_classic_round(challenge_id: str, timed_out: bool = False) -> dict[str, Any]:
    challenge = get_challenge(challenge_id)
    if not challenge:
        return {"success": False, "error": "Unknown challenge."}
    pub = _public_challenge(challenge)
    baseline = CLASSIC_BASELINES.get(challenge_id, CLASSIC_BASELINES["traffic_crisis"])
    penalty = int(challenge.get("classic_penalty") or 25)
    health_delta = -int(challenge.get("classic_health_delta") or 20)
    if timed_out:
        penalty += 10
        health_delta -= 5
    damage_tnd = baseline["damage_increase_tnd"]
    cost_tnd = baseline["cost_tnd"]
    headline = "Time Expired" if timed_out else "Delayed Response"
    sublines = (
        ["Auto Classic Response", "Damage Increased", f"City Safety {health_delta}"]
        if timed_out
        else ["Damage Increased", f"City Safety {health_delta}", f"+{baseline['response_delay_min']} min delay"]
    )
    return {
        "success": True,
        "method": "classic",
        "timed_out": timed_out,
        "challenge": pub,
        "real_model_used": False,
        "score_delta": -penalty,
        "city_health_delta": health_delta,
        "citizens_impacted": baseline["citizens_impacted"],
        "damage_increase_tnd": damage_tnd,
        "damage_cost_tnd": cost_tnd,
        "response_delay_min": baseline["response_delay_min"],
        "worse_image_url": pub["problem_image_url"],
        "headline": headline,
        "sublines": sublines,
        "overlay_type": "negative",
    }


def run_medinamind_round(challenge_id: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    challenge = get_challenge(challenge_id)
    if not challenge:
        return {"success": False, "error": "Unknown challenge."}
    inputs = inputs or {}
    pub = _public_challenge(challenge)
    module_key = challenge.get("module") or "traffic"
    frame = _decode_upload_image(inputs.get("image_b64")) or _load_challenge_frame(challenge_id)
    runner = MODULE_RUNNERS.get(module_key)
    if not runner:
        return {"success": False, "error": f"No runner for module {module_key}"}

    result = runner(frame)
    live = result.get("status") == "live"
    data = result.get("data") or {}
    reward = int(challenge.get("medinamind_reward") or 30)
    health_delta = int(challenge.get("medinamind_health_delta") or 15)
    if not live:
        reward = max(8, reward // 2)
        health_delta = max(6, health_delta // 2)

    conf = float(data.get("confidence") or data.get("max_confidence") or data.get("max_conf") or 0.72)
    baseline = CLASSIC_BASELINES.get(challenge_id, CLASSIC_BASELINES["traffic_crisis"])
    citizens = int(baseline["citizens_impacted"] * (0.45 + conf * 0.35))
    damage_avoided = int(baseline["damage_increase_tnd"] * (0.4 + conf * 0.45))
    time_saved = max(8, int(baseline["response_delay_min"] * (0.55 + conf * 0.25)))

    rec = data.get("recommendation") or data.get("recommended_action") or "AI recommends the fastest safe response."
    if module_key == "road_damage":
        rec = "AI detects the damage and recommends the fastest response."
    elif module_key == "traffic":
        rec = "AI reroutes traffic and clears the emergency corridor."
    elif module_key == "waste":
        rec = "AI prioritizes cleanup teams to the hotspot."
    elif module_key == "fire_smoke":
        rec = "AI confirms the alert and dispatches fire units."
    elif module_key == "uav":
        rec = "AI flags structural risk and restricts access."

    return {
        "success": True,
        "method": "medinamind",
        "challenge": pub,
        "module_used": module_key,
        "module_label": MODULE_LABELS.get(module_key, module_key),
        "real_model_used": live,
        "fallback_used": not live,
        "fallback_message": None if live else f"Real model unavailable — demo fallback used ({result.get('error') or 'model offline'})",
        "detected_issue": data.get("detected_issue") or data.get("verdict_label") or "AI analysis complete",
        "confidence": round(conf, 3),
        "severity": data.get("severity") or "Moderate",
        "recommendation": rec,
        "annotated_image_url": data.get("annotated_image_url") or "",
        "score_delta": reward,
        "city_health_delta": health_delta,
        "citizens_protected": citizens,
        "damage_avoided_tnd": damage_avoided,
        "response_time_saved_min": time_saved,
        "headline": "AI Intervention Successful",
        "sublines": ["Risk Reduced", f"City Safety +{health_delta}", "Emergency Response Optimized"],
        "overlay_type": "positive",
        "module_result": result,
    }


def finish_mission(payload: dict[str, Any]) -> dict[str, Any]:
    score = int(payload.get("score") or 0)
    city_health = int(payload.get("city_health") or 0)
    stars = _stars_from_score(score)
    win = score >= 70
    return {
        "success": True,
        "win": win,
        "title": "CITY SAVED!" if win else "CITY AT RISK",
        "subtitle": "Congratulations Mayor!" if win else "Try MedinaMind AI to improve city safety.",
        "score": score,
        "city_health": city_health,
        "stars": stars,
        "citizens_protected": int(payload.get("citizens_protected") or 0),
        "citizens_impacted": int(payload.get("citizens_impacted") or 0),
        "damage_avoided_tnd": int(payload.get("damage_avoided_tnd") or payload.get("damage_avoided") or 0),
        "damage_cost_tnd": int(payload.get("damage_cost_tnd") or 0),
        "response_time_saved_min": int(payload.get("response_time_saved") or 0),
        "modules_used": payload.get("modules_used") or [],
        "decision_history": payload.get("decision_history") or [],
        "rounds_played": int(payload.get("rounds_played") or 0),
    }


def _stars_from_score(score: int) -> int:
    if score >= 90:
        return 5
    if score >= 75:
        return 4
    if score >= 60:
        return 3
    if score >= 40:
        return 2
    return 1


def build_certificate_payload(state: dict[str, Any], mayor_name: str = "Mayor") -> dict[str, Any]:
    return {
        "certificate_id": uuid.uuid4().hex[:12].upper(),
        "title": "Mayor of MedinaMind — City Saved",
        "mayor_name": mayor_name,
        "final_score": state.get("score"),
        "city_health": state.get("city_health"),
        "stars": _stars_from_score(int(state.get("score") or 0)),
        "citizens_protected": state.get("citizens_protected"),
        "damage_avoided_tnd": state.get("damage_avoided_tnd") or state.get("damage_avoided"),
        "damage_cost_tnd": state.get("damage_cost_tnd"),
        "response_time_saved_min": state.get("response_time_saved"),
        "modules_used": state.get("modules_used") or [],
        "issued_at": "MedinaMind Mission Control",
    }


# Legacy wrappers
def run_classic_simulation(scenario_id: str, timed_out: bool = False) -> dict[str, Any]:
    r = run_classic_round(scenario_id, timed_out=timed_out)
    if not r.get("success"):
        return {"ok": False, "error": r.get("error")}
    return {"ok": True, "mode": "classic", **r}


def run_medinamind_simulation(scenario_id: str, inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    r = run_medinamind_round(scenario_id, inputs)
    if not r.get("success"):
        return {"ok": False, "error": r.get("error")}
    return {"ok": True, "mode": "medinamind", **r}
