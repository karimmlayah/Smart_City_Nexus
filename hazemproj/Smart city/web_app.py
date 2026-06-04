import os
import base64
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from datetime import datetime
import shutil
import subprocess
import tempfile
import importlib.util
import urllib.error
import urllib.parse
import urllib.request

import cv2
import imageio_ffmpeg
import numpy as np
import joblib
from flask import Flask, render_template, request, send_file, url_for, jsonify, redirect
from ultralytics import YOLO
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
OUTPUT_DIR = BASE_DIR / "web_outputs"

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

key = os.getenv("ROBOFLOW_API_KEY", "")
print("ROBOFLOW_API_KEY EXISTS:", bool(key))
print("ROBOFLOW_API_KEY START:", key[:4])
print("ROBOFLOW_API_KEY LENGTH:", len(key))
print("ROBOFLOW_WORKFLOW_ID:", os.getenv("ROBOFLOW_WORKFLOW_ID"))
print("USE WORKFLOW:", os.getenv("ROBOFLOW_USE_WORKFLOW", "0"))
print("MODEL ID:", os.getenv("ROBOFLOW_MODEL_ID", ""))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
print("OPENAI_API_KEY EXISTS:", bool(OPENAI_API_KEY))

OCR_API_KEY = os.getenv("OCR_API_KEY", "")
print("OCR_API_KEY EXISTS:", bool(OCR_API_KEY))

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
print("EMAIL configured:", bool(EMAIL_USER and EMAIL_PASSWORD))

# Check if AI agents are available
try:
    from ai_agents import ViolationAnalysisAgent, ViolationSummaryAgent, RecommendationAgent
    from pdf_generator import ViolationPDFGenerator
    AI_AGENTS_AVAILABLE = True
    print("AI AGENTS: Available")
except ImportError as e:
    AI_AGENTS_AVAILABLE = False
    print(f"AI AGENTS: Not available ({e})")

# Check if OCR module is available
try:
    from plate_ocr import PlateOCR, process_violation_plates
    OCR_AVAILABLE = True
    print("PLATE OCR: Available")
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"PLATE OCR: Not available ({e})")

# Check if database and email modules are available
try:
    from database import VehicleDatabase, db
    from email_notifier import ViolationEmailNotifier
    DB_EMAIL_AVAILABLE = True
    print("DATABASE & EMAIL: Available")
except ImportError as e:
    DB_EMAIL_AVAILABLE = False
    print(f"DATABASE & EMAIL: Not available ({e})")

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
STREET_SIGN_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Unused legacy paths (kept for compatibility with old functions)
STRAY_SCRIPT_PATH = BASE_DIR / "shoplifting_video_xai (1).py"
STRAY_BASELINE_PATH = BASE_DIR / "stray_health_baseline.npz"

_street_sign_model = None
_traffic_light_model = None
_new_traffic_sign_model = None
_tn_plate_detector = None
_tn_plate_detector_error: str | None = None
_tunisian_mlp_model = None
_tunisian_knn_model = None
_easyocr_reader = None
_easyocr_init_attempted = False

# Animal-Behaviour-and-Disease-Detection (YOLO + project models)
BEHAVIOR_LABELS = ["Standing", "Eating", "Resting"]
DISEASE_LABELS = ["cognitive", "Injured", "mange"]
ANIMAL_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_animal_keras_model = None
_keras_load_attempted = False
_yolo_animal_model = None
_animalpose_module = None
_animalpose_attempted = False
# COCO class names (YOLOv8) — quadrupeds + birds commonly used as “animal”
YOLO_ANIMAL_NAMES = frozenset(
    {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
)

# LLaVA v1.5 7B — text explanation for XAI (web page only, not burned into video)
# Prefer HF-native weights; liuhaotian checkpoint is tried first if present.
_llava_proc = None
_llava_model = None
_llava_load_error: str | None = None


def _llava_model_candidates() -> list[str]:
    env = os.environ.get("LLAVA_MODEL_ID", "").strip()
    if env:
        return [x.strip() for x in env.split(",") if x.strip()]
    # Use HF-converted checkpoint by default; it's packaged for AutoProcessor/LlavaForConditionalGeneration.
    return ["llava-hf/llava-1.5-7b-hf"]


def _ensure_llava(local_files_only: bool = False):
    """Lazy-load one LLaVA model; returns (processor, model) or raises RuntimeError with message."""
    global _llava_proc, _llava_model, _llava_load_error
    if _llava_load_error is not None:
        raise RuntimeError(_llava_load_error)
    if _llava_model is not None and _llava_proc is not None:
        return _llava_proc, _llava_model
    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    last_err = None
    for model_id in _llava_model_candidates():
        try:
            proc = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, local_files_only=local_files_only
            )
            kwargs = {"low_cpu_mem_usage": True, "trust_remote_code": True}
            if torch.cuda.is_available():
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    local_files_only=local_files_only,
                    **kwargs,
                )
            else:
                model = LlavaForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.float32, local_files_only=local_files_only, **kwargs
                )
                model = model.to("cpu")
            _llava_proc = proc
            _llava_model = model
            return proc, model
        except Exception as e:
            last_err = e
            continue
    _llava_load_error = f"Could not load LLaVA ({last_err})"
    raise RuntimeError(_llava_load_error)


def _bgr_to_pil_rgb(frame_bgr: np.ndarray):
    from PIL import Image

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _llava_explain_crop(
    roi_bgr: np.ndarray,
    species: str | None,
    beh: str,
    dis: str,
    summary: str,
    decision_meta: dict | None = None,
    local_files_only: bool = False,
) -> tuple[str | None, str | None]:
    """Return (explanation_text, error_message). Explanation is never drawn on video."""
    try:
        proc, model = _ensure_llava(local_files_only=local_files_only)
    except RuntimeError as e:
        return None, str(e)

    import torch

    pil = _bgr_to_pil_rgb(roi_bgr)

    meta_text = ""
    if decision_meta:
        parts = [f"{k}={v}" for k, v in decision_meta.items()]
        meta_text = "Decision signals: " + "; ".join(parts) + "\n\n"
    user_block = (
        "You are assisting a wildlife / stray-animal monitoring system.\n"
        f"Automatic outputs: YOLO species hint: {species or 'unknown'}; "
        f"behaviour label: {beh}; disease label: {dis}; fused summary: {summary}\n\n"
        f"{meta_text}"
        "Return a short XAI explanation with these headings:\n"
        "1) Visible cues\n2) Why this behavior/disease was selected\n3) Uncertainty\n"
        "Ground all points in the image and decision signals. Do not claim medical diagnosis."
    )
    prompt = f"USER: <image>\n{user_block}\nASSISTANT:"

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    try:
        inputs = proc(images=pil, text=prompt, return_tensors="pt")
    except Exception:
        try:
            inputs = proc(text=prompt, images=pil, return_tensors="pt")
        except Exception:
            inputs = proc(prompt, pil, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"].dtype != dtype:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    in_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    new_tokens = gen[0, in_len:]
    tokenizer = getattr(proc, "tokenizer", None)
    if tokenizer is not None:
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    else:
        text = proc.decode(new_tokens, skip_special_tokens=True).strip()
    return text, None


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _normalize_fps(raw_fps: float) -> float:
    # Some files report invalid FPS values (0, NaN, very high timebase-like values).
    if raw_fps is None:
        return 20.0
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        return 20.0
    if not np.isfinite(fps) or fps <= 1.0 or fps > 120.0:
        return 20.0
    return fps


def _encode_frames_to_h264_mp4(frames_dir: Path, fps: float, output_path: Path) -> None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    input_pattern = str(frames_dir / "frame_%06d.png")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-framerate",
        f"{fps:.4f}",
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg encode failed: {proc.stderr[-1200:]}")


def _validate_h264_output(output_path: Path) -> None:
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run([ffmpeg_exe, "-i", str(output_path)], capture_output=True, text=True)
    info = proc.stderr
    if "Video: h264" not in info:
        raise RuntimeError("Output validation failed: video stream is not H.264.")
    if "Duration: 00:00:00" in info:
        raise RuntimeError("Output validation failed: duration is zero seconds.")


def _load_stray_module():
    if not STRAY_SCRIPT_PATH.exists():
        raise RuntimeError(f"Stray script not found: {STRAY_SCRIPT_PATH}")
    module_name = "stray_animal_health_runtime"
    spec = importlib.util.spec_from_file_location(module_name, STRAY_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load stray animal script module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_stray_baseline_from_video(input_path: Path, max_frames: int = 320) -> dict:
    module = _load_stray_module()
    detector = module.AnimalDetector(conf=module.CFG.det_conf)
    scorer = module.ZeroShotHealthScorer()
    class_names = module.CLASS_NAMES

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    vectors = []
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if frame_count > max_frames:
            break
        crop, _, _ = detector.detect_main_animal(frame)
        if crop is None:
            continue
        res = scorer.score_crop(crop)
        vec = np.array([res["probs"][c] for c in class_names], dtype=np.float32)
        vectors.append(vec)

    cap.release()
    if len(vectors) < 8:
        raise RuntimeError("Not enough dog/cat detections to build baseline. Use a clearer healthy video.")

    arr = np.stack(vectors, axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-3
    np.savez(
        STRAY_BASELINE_PATH,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        classes=np.array(class_names, dtype=object),
    )
    return {
        "frames_processed": frame_count,
        "peak_score": float(mean[0]),
        "suspicious_frames": 0,
        "alert_frames": 0,
        "baseline_samples": int(arr.shape[0]),
    }


def _load_stray_baseline():
    if not STRAY_BASELINE_PATH.exists():
        return None
    data = np.load(STRAY_BASELINE_PATH, allow_pickle=True)
    return {
        "mean": data["mean"].astype(np.float32),
        "std": data["std"].astype(np.float32),
        "classes": [str(x) for x in data["classes"].tolist()],
    }


def _render_to_h264(
    cap: cv2.VideoCapture,
    output_path: Path,
    frame_processor,
    max_frames: int = 500,
    pass_frame_index: bool = False,
) -> dict:
    fps = _normalize_fps(cap.get(cv2.CAP_PROP_FPS))
    temp_dir = Path(tempfile.mkdtemp(prefix="shoplift_frames_", dir=str(OUTPUT_DIR)))
    frame_count = 0
    suspicious_frames = 0
    alert_frames = 0
    peak_score = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if frame_count > max_frames:
                break

            if pass_frame_index:
                show, score, status = frame_processor(frame, frame_count)
            else:
                show, score, status = frame_processor(frame)
            peak_score = max(peak_score, score)
            if "ALERT" in status:
                alert_frames += 1
            elif "Suspicious" in status:
                suspicious_frames += 1

            frame_path = temp_dir / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_path), show)

        if frame_count == 0:
            raise RuntimeError("Video had no readable frames.")
        _encode_frames_to_h264_mp4(temp_dir, fps, output_path)
        _validate_h264_output(output_path)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Processing finished but output file is empty.")
    finally:
        cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "frames_processed": frame_count,
        "peak_score": round(peak_score, 4),
        "suspicious_frames": suspicious_frames,
        "alert_frames": alert_frames,
    }


def analyze_video(input_path: Path, output_path: Path, max_frames: int = 500) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    scorer = ZeroShotShopliftingScorer()
    score_buffer = []
    
    def process_frame(frame: np.ndarray):
        score_info = scorer.score_frame(frame)
        score_buffer.append(score_info["suspicious_score"])
        if len(score_buffer) > CFG.frame_vote_window:
            score_buffer.pop(0)

        final_score = float(np.mean(score_buffer))
        if final_score >= CFG.alarm_threshold:
            status, color = "ALERT: suspicious stealing", (0, 0, 255)
        elif final_score >= CFG.suspicious_threshold:
            status, color = "Suspicious behavior", (0, 165, 255)
        else:
            status, color = "Normal", (0, 255, 0)

        lines = [
            f"Status: {status}",
            f"Score: {final_score:.3f}",
            f"Suspicious score: {score_info['suspicious_score']:.3f}",
            "Mode: Zero-shot vision scorer",
        ]
        show = overlay_text(frame.copy(), lines, color=color)
        return show, final_score, status

    return _render_to_h264(cap, output_path, process_frame, max_frames=max_frames)


def analyze_video_yolo(input_path: Path, output_path: Path, weights_path: Path, max_frames: int = 500) -> dict:
    if not weights_path.exists():
        raise RuntimeError(
            f"YOLO weights not found: {weights_path}. Add the .pt file or provide another path."
        )

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    model = YOLO(str(weights_path))
    class_names = model.names if hasattr(model, "names") else {}

    def process_frame(frame: np.ndarray):
        result = model.predict(frame, verbose=False)[0]
        status = "Loading..."
        frame_out = frame.copy()
        peak_cls_score = 0.0

        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), conf, cls_idx in zip(xyxy, confs, classes):
                label = str(class_names.get(int(cls_idx), cls_idx)).lower()
                is_shoplifting = ("shop" in label) or (int(cls_idx) == 1)
                color = (0, 0, 255) if is_shoplifting else (0, 255, 0)
                txt = f"{label} {conf:.2f}"
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_out, txt, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if is_shoplifting:
                    status = "ALERT: suspicious stealing"
                elif status == "Loading..." and conf >= 0.5:
                    status = "Not shoplifting"
                peak_cls_score = max(peak_cls_score, float(conf))

        if status == "Loading...":
            status = "No detection"

        lines = [
            f"Status: {status}",
            f"Score: {peak_cls_score:.3f}",
            f"Model: {weights_path.name}",
            "Mode: YOLO project detector",
        ]
        color = (0, 0, 255) if "ALERT" in status else (0, 165, 255) if "Suspicious" in status else (0, 255, 0)
        show = overlay_text(frame_out, lines, color=color)
        return show, peak_cls_score, status

    return _render_to_h264(cap, output_path, process_frame, max_frames=max_frames)


def analyze_video_stray(input_path: Path, output_path: Path, max_frames: int = 500) -> dict:
    module = _load_stray_module()
    detector = module.AnimalDetector(conf=module.CFG.det_conf)
    scorer = module.ZeroShotHealthScorer()
    class_names = module.CLASS_NAMES
    baseline = _load_stray_baseline()

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    health_buffer = []
    # Conservative calibration to reduce false red alerts in zero-shot mode.
    class_scale = np.array([1.25, 1.0, 0.95, 0.72], dtype=np.float32)
    severe_trigger = 0.50
    severe_margin = 0.10
    anomaly_buffer = []

    def process_frame(frame: np.ndarray):
        frame_out = frame.copy()
        crop, box, species = detector.detect_main_animal(frame)
        if crop is None:
            status = "No dog/cat detected"
            show = module.overlay_text(frame_out, [status, "Mode: Stray animal health"], color=(180, 180, 180))
            return show, 0.0, status

        result = scorer.score_crop(crop)
        vec = np.array([result["probs"][c] for c in class_names], dtype=np.float32)
        health_buffer.append(vec)
        if len(health_buffer) > module.CFG.frame_vote_window:
            health_buffer.pop(0)
        avg = np.mean(np.stack(health_buffer), axis=0)

        # Reweight classes to counter severe-class bias from prompt-based scoring.
        calibrated = avg * class_scale
        calibrated = calibrated / (calibrated.sum() + 1e-8)
        pred_idx = int(np.argmax(calibrated))
        pred_class = class_names[pred_idx]
        score = float(calibrated[pred_idx])

        severe_idx = class_names.index("severe_condition")
        healthy_idx = class_names.index("healthy")
        severe_score = float(calibrated[severe_idx])
        healthy_score = float(calibrated[healthy_idx])

        anomaly_score = 0.0
        if baseline is not None and baseline["classes"] == class_names:
            z = np.abs((calibrated - baseline["mean"]) / baseline["std"])
            anomaly_score = float(np.mean(z))
            anomaly_buffer.append(anomaly_score)
            if len(anomaly_buffer) > module.CFG.frame_vote_window:
                anomaly_buffer.pop(0)
            smooth_anomaly = float(np.mean(anomaly_buffer))
            if smooth_anomaly >= 2.3:
                status, color = "ALERT: abnormal condition", (0, 0, 255)
            elif smooth_anomaly >= 1.5:
                status, color = "Suspicious health condition", (0, 165, 255)
            else:
                status, color = "Healthy-like", (0, 255, 0)
        else:
            if severe_score >= severe_trigger and (severe_score - healthy_score) >= severe_margin:
                status, color = "ALERT: severe condition", (0, 0, 255)
                pred_class = "severe_condition"
                score = severe_score
            elif pred_class == "healthy":
                status, color = "Healthy", (0, 255, 0)
            elif pred_class in ["underweight_or_malnourished", "skin_disease_or_visible_lesion"]:
                status, color = "Suspicious health condition", (0, 165, 255)
            else:
                # Default to amber unless severe confidence is truly strong.
                status, color = "Suspicious health condition", (0, 165, 255)

        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame_out,
                f"{species or 'animal'}: {pred_class}",
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )

        lines = [
            f"Status: {status}",
            f"Predicted: {pred_class}",
            f"Score: {score:.3f}",
            f"Healthy: {calibrated[healthy_idx]:.3f} | Severe: {calibrated[severe_idx]:.3f}",
            f"Anomaly score: {anomaly_score:.2f}",
            f"Species: {species or 'unknown'}",
            "Mode: Stray animal health",
        ]
        show = module.overlay_text(frame_out, lines, color=color)
        return show, score, status

    return _render_to_h264(cap, output_path, process_frame, max_frames=max_frames)




def _get_street_sign_model():
    """Load street sign detection model (my_model.pt)"""
    global _street_sign_model
    if _street_sign_model is None:
        weights = BASE_DIR / "my_model.pt"
        if not weights.exists():
            raise RuntimeError(f"Street sign model not found: {weights}")
        _street_sign_model = YOLO(str(weights), task="detect")
    return _street_sign_model


def _get_car_detector_model():
    """YOLOv8n COCO model used exclusively for car detection."""
    global _traffic_light_model
    if _traffic_light_model is None:
        weights = BASE_DIR / "yolov8n.pt"
        if not weights.exists():
            raise RuntimeError(f"Car detector model not found: {weights}")
        _traffic_light_model = YOLO(str(weights))
    return _traffic_light_model


def _get_new_traffic_sign_model():
    """Load traffic sign/light detection model (traffic_sign_detector.pt)"""
    global _new_traffic_sign_model
    if _new_traffic_sign_model is None:
        weights = BASE_DIR / "traffic_sign_detector.pt"
        if not weights.exists():
            raise RuntimeError(f"Traffic sign model not found: {weights}")
        _new_traffic_sign_model = YOLO(str(weights), task="detect")
    return _new_traffic_sign_model


_plate_detector_model = None

def _get_plate_detector_model():
    """YOLOv11 license plate detector."""
    global _plate_detector_model
    if _plate_detector_model is None:
        try:
            plate_weights = BASE_DIR / "license-plate-finetune-v1x.pt"
            if not plate_weights.exists():
                print(f"Warning: Plate detector not found at {plate_weights}")
                return None
            _plate_detector_model = YOLO(str(plate_weights))
            print(f"Plate detector loaded from {plate_weights}")
        except Exception as e:
            print(f"Warning: Could not load plate detector: {e}")
            _plate_detector_model = None
    return _plate_detector_model


def suggest_stop_line_with_openai(frame_bgr: np.ndarray) -> dict:
    """
    Use OpenAI Vision to suggest a stop line position.
    Returns: {"stop_line_points": [[x1,y1],[x2,y2]], "confidence": 0.0-1.0, "explanation": "..."}
    """
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set in .env file"}
    
    try:
        # Encode frame as base64 JPEG
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Call OpenAI Vision API
        import requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        prompt = """You are a traffic-scene calibration assistant with expertise in traffic light systems.

Analyze this traffic scene carefully:
1. Identify ALL traffic lights in the image (including directional arrows for left/right turns)
2. Determine which traffic light controls the main lane vs turn lanes
3. Identify the intersection and road layout
4. Place the stop line where vehicles MUST stop when the relevant traffic light is RED

IMPORTANT RULES:
- If there's a LEFT TURN arrow light, the stop line should be for the left turn lane
- If there's a RIGHT TURN arrow light, the stop line should be for the right turn lane  
- If there's only a main traffic light (no arrows), place the line for the main through lane
- The stop line should be BEFORE the intersection, where vehicles wait at red lights
- Consider lane markings, crosswalks, and existing road paint if visible
- The line should span across the relevant lane(s)

Return ONLY valid JSON with no markdown formatting:
{
  "stop_line_points": [[x1, y1], [x2, y2]],
  "confidence": 0.0-1.0,
  "explanation": "Placed for [main/left turn/right turn] lane based on [traffic light type] at [location]",
  "traffic_light_type": "main|left_arrow|right_arrow|multiple",
  "lane_type": "through|left_turn|right_turn|multiple"
}

Use pixel coordinates relative to the image dimensions."""

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": f"OpenAI API error: {response.status_code}"}
        
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        
        # Remove markdown code blocks if present
        content = content.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        data = json.loads(content)
        return data
        
    except Exception as e:
        return {"error": f"Failed to get AI suggestion: {str(e)}"}


def _get_tunisian_plate_detector():
    global _tn_plate_detector, _tn_plate_detector_error
    if _tn_plate_detector is None:
        if _tn_plate_detector_error is not None:
            raise RuntimeError(_tn_plate_detector_error)
        try:
            # Ultralytics HUB model as requested by user.
            _tn_plate_detector = YOLO("Safe-Drive-TN/Tunisian-Licence-plate-Detection")
        except Exception as exc:
            _tn_plate_detector_error = f"{exc}"
            raise RuntimeError(_tn_plate_detector_error)
    return _tn_plate_detector


def _tunisian_recognition_dir() -> Path:
    base = BASE_DIR / "ANPR-master" / "Tunisian_anpr" / "Licence_plate_recognition" / "Tunisian_plates"
    if base.exists():
        return base
    alt = Path(
        r"C:\Users\USER\Desktop\Smart city\ANPR-master\Tunisian_anpr\Licence_plate_recognition\Tunisian_plates"
    )
    if alt.exists():
        return alt
    raise RuntimeError("Tunisian recognition folder not found.")


def _get_tunisian_models():
    global _tunisian_mlp_model, _tunisian_knn_model
    if _tunisian_mlp_model is None or _tunisian_knn_model is None:
        d = _tunisian_recognition_dir()
        mlp_path = d / "mlp.pkl"
        knn_path = d / "knn.pkl"
        if not mlp_path.exists() or not knn_path.exists():
            raise RuntimeError("Tunisian recognition models mlp.pkl/knn.pkl are missing.")
        _tunisian_mlp_model = joblib.load(str(mlp_path))
        _tunisian_knn_model = joblib.load(str(knn_path))
    return _tunisian_mlp_model, _tunisian_knn_model


def _segment_plate_characters(plate_bgr: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)
    bw = cv2.morphologyEx(
        bw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bw.shape[:2]
    chars = []
    boxes = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.002 * (h * w):
            continue
        if ch < 0.18 * h or ch > 0.98 * h:
            continue
        if cw < 0.01 * w or cw > 0.50 * w:
            continue
        boxes.append((x, y, cw, ch))
    boxes.sort(key=lambda b: b[0])
    for x, y, cw, ch in boxes[:10]:
        digit = bw[y : y + ch, x : x + cw]
        side = max(digit.shape[0], digit.shape[1])
        sq = np.zeros((side, side), dtype=np.uint8)
        yy = (side - digit.shape[0]) // 2
        xx = (side - digit.shape[1]) // 2
        sq[yy : yy + digit.shape[0], xx : xx + digit.shape[1]] = digit
        char_img = cv2.resize(sq, (20, 20), interpolation=cv2.INTER_AREA)
        chars.append(char_img)
    return chars


def _plate_ocr_variants(plate_bgr: np.ndarray) -> list[np.ndarray]:
    """Generate a few robust plate variants to improve classical ANPR OCR."""
    variants: list[np.ndarray] = []
    if plate_bgr is None or plate_bgr.size == 0:
        return variants
    variants.append(plate_bgr)

    # Upscale often helps contour-based character segmentation.
    h, w = plate_bgr.shape[:2]
    if h < 80 or w < 220:
        up = cv2.resize(
            plate_bgr,
            (max(1, int(w * 2.0)), max(1, int(h * 2.0))),
            interpolation=cv2.INTER_CUBIC,
        )
        variants.append(up)

    # CLAHE on luminance helps under/overexposed shots.
    try:
        ycrcb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y2 = clahe.apply(y)
        enh = cv2.cvtColor(cv2.merge((y2, cr, cb)), cv2.COLOR_YCrCb2BGR)
        variants.append(enh)
    except cv2.error:
        pass

    # Denoised + sharpened version for motion blur/noisy frames.
    try:
        den = cv2.bilateralFilter(plate_bgr, 5, 30, 30)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(den, -1, kernel)
        variants.append(sharp)
    except cv2.error:
        pass
    return variants


def _clean_plate_ocr_text(text: str) -> str:
    s = re.sub(r"[^0-9A-Za-z\u0600-\u06FF]", "", str(text or ""))
    return s.strip()


def _get_easyocr_reader():
    global _easyocr_reader, _easyocr_init_attempted
    if _easyocr_init_attempted:
        return _easyocr_reader
    _easyocr_init_attempted = True
    try:
        import easyocr

        langs = (os.environ.get("OCR_EASYOCR_LANGS", "en,ar") or "en,ar").split(",")
        langs = [x.strip() for x in langs if x.strip()]
        _easyocr_reader = easyocr.Reader(langs, gpu=False)
    except Exception:
        _easyocr_reader = None
    return _easyocr_reader


def _recognize_plate_text_easyocr(plate_bgr: np.ndarray) -> tuple[str | None, float]:
    reader = _get_easyocr_reader()
    if reader is None:
        return None, 0.0
    best_text: str | None = None
    best_conf = 0.0
    for candidate in _plate_ocr_variants(plate_bgr):
        try:
            rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb, detail=1, paragraph=False)
        except Exception:
            continue
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            txt = _clean_plate_ocr_text(str(item[1]))
            conf = float(item[2] or 0.0)
            if len(txt) < 3:
                continue
            if conf > best_conf or (abs(conf - best_conf) < 1e-6 and len(txt) > len(best_text or "")):
                best_text = txt
                best_conf = conf
    return best_text, best_conf


def _recognize_plate_text_tesseract(plate_bgr: np.ndarray) -> tuple[str | None, float]:
    try:
        import pytesseract
    except Exception:
        return None, 0.0
    # Force single text line and plate-like charset.
    cfg = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    best_text: str | None = None
    best_conf = 0.0
    for candidate in _plate_ocr_variants(plate_bgr):
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
        try:
            data = pytesseract.image_to_data(thr, output_type=pytesseract.Output.DICT, config=cfg)
        except Exception:
            continue
        tokens = []
        confs = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = _clean_plate_ocr_text(data["text"][i])
            if len(txt) < 1:
                continue
            try:
                c = float(data.get("conf", ["-1"])[i])
            except (ValueError, TypeError):
                c = -1.0
            if c < 0:
                continue
            tokens.append(txt)
            confs.append(c)
        if not tokens:
            continue
        joined = _clean_plate_ocr_text("".join(tokens))
        if len(joined) < 3:
            continue
        avg_conf = (sum(confs) / max(1, len(confs))) / 100.0
        if avg_conf > best_conf or (abs(avg_conf - best_conf) < 1e-6 and len(joined) > len(best_text or "")):
            best_text = joined
            best_conf = avg_conf
    return best_text, max(0.0, min(1.0, best_conf))


def _recognize_tunisian_plate_text(plate_bgr: np.ndarray) -> tuple[str | None, float]:
    backend = os.environ.get("OCR_BACKEND", "auto").strip().lower()
    best_text: str | None = None
    best_conf = 0.0
    if backend in ("auto", "anpr"):
        try:
            mlp_model, knn_model = _get_tunisian_models()
            best_len = 0
            for candidate in _plate_ocr_variants(plate_bgr):
                chars = _segment_plate_characters(candidate)
                if not chars:
                    continue
                preds = []
                agree = 0
                for ch in chars:
                    feat = ch.ravel().astype(np.float32)
                    p_mlp = str(mlp_model.predict([feat])[0])
                    p_knn = str(knn_model.predict([feat])[0])
                    if p_mlp == p_knn:
                        agree += 1
                    preds.append(p_mlp)
                text = _clean_plate_ocr_text("".join(preds))
                conf = float(agree / max(1, len(preds)))
                if text and (conf > best_conf or (abs(conf - best_conf) < 1e-6 and len(text) > best_len)):
                    best_text = text
                    best_conf = conf
                    best_len = len(text)
        except Exception:
            pass

    if not best_text and backend in ("auto", "easyocr"):
        t, c = _recognize_plate_text_easyocr(plate_bgr)
        if t:
            best_text, best_conf = t, c

    if not best_text and backend in ("auto", "tesseract"):
        t, c = _recognize_plate_text_tesseract(plate_bgr)
        if t:
            best_text, best_conf = t, c

    if os.environ.get("ROBOFLOW_DEBUG_OCR", "0").strip() == "1":
        print(
            "OCR RESULT:",
            best_text if best_text else "(none)",
            "conf=",
            round(best_conf, 3),
            "backend=",
            backend,
        )
    return best_text, best_conf


def _detect_plate_in_car_edges(car_bgr: np.ndarray):
    """Return best plate-like rectangle inside this crop (pixel coords relative to crop)."""
    gray = cv2.cvtColor(car_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_score = 0.0
    h, w = gray.shape[:2]
    if h < 8 or w < 8:
        return None
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 0.10 * w or ch < 0.04 * h:
            continue
        if cw > 0.98 * w or ch > 0.50 * h:
            continue
        ratio = cw / float(max(1, ch))
        if ratio < 1.8 or ratio > 8.5:
            continue
        area = cw * ch
        score = area * (1.0 - abs(ratio - 4.0) / 4.0)
        if score > best_score:
            best_score = score
            best = (x, y, x + cw, y + ch)
    if best is None:
        return None
    x1, y1, x2, y2 = best
    crop = car_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None
    return (x1, y1, x2, y2), crop


def _detect_plate_in_car(car_bgr: np.ndarray):
    """Prefer lower bumper band (plates sit low on front/rear)."""
    h, w = car_bgr.shape[:2]
    if h >= 72:
        y0 = int(h * 0.40)
        bumper = car_bgr[y0:, :, :]
        res = _detect_plate_in_car_edges(bumper)
        if res is not None:
            (x1, y1, x2, y2), crop = res
            return (x1, y1 + y0, x2, y2 + y0), crop
    return _detect_plate_in_car_edges(car_bgr)


def _roboflow_plate_env() -> tuple[str, str, str, str, str, int, str, str]:
    """Return (api_key, api_url, workspace, workflow_id, classes, stride, image_input_key, classes_param_key)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip().lstrip("\ufeff")
    api_url = os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com").rstrip("/")
    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "homerbot").strip()
    workflow_id = os.environ.get("ROBOFLOW_WORKFLOW_ID", "detect-and-classify-6").strip()
    # Only sent to Roboflow when non-empty (e.g. segmentation workflows); "detect-and-classify" has no classes param.
    classes = os.environ.get("ROBOFLOW_PLATE_CLASSES", "").strip()
    try:
        stride = max(1, int(os.environ.get("ROBOFLOW_PLATE_STRIDE", "10") or "10"))
    except ValueError:
        stride = 10
    image_key = (os.environ.get("ROBOFLOW_IMAGE_INPUT_NAME", "image") or "image").strip()
    classes_key = (os.environ.get("ROBOFLOW_CLASSES_PARAM", "classes") or "classes").strip()
    return api_key, api_url, workspace, workflow_id, classes, stride, image_key, classes_key


def _roboflow_use_workflow() -> bool:
    """Only when ROBOFLOW_USE_WORKFLOW is exactly 1: serverless workflows; otherwise hosted model REST."""
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass
    return os.getenv("ROBOFLOW_USE_WORKFLOW", "0").strip() == "1"


def _roboflow_prediction_class_ok(pr: dict) -> bool:
    """If ROBOFLOW_PLATE_OUTPUT_CLASSES is set (comma-separated), only keep those class names."""
    filt = os.environ.get("ROBOFLOW_PLATE_OUTPUT_CLASSES", "").strip()
    if not filt:
        return True
    allowed = {x.strip().lower() for x in filt.split(",") if x.strip()}
    c = str(pr.get("class", "")).strip().lower()
    return c in allowed


def _roboflow_normalize_workflow_response(data) -> dict:
    """HTTP/SDK may return a bare list (workflow step list) or {\"outputs\": ...}."""
    if isinstance(data, list):
        return {"outputs": data}
    if isinstance(data, dict):
        return data
    return {"outputs": [data]}


def _frame_jpeg_base64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Could not JPEG-encode frame for Roboflow.")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _roboflow_workflow_http_headers() -> dict[str, str]:
    """Cloudflare (error 1010) often blocks the default Python-urllib User-Agent."""
    try:
        from dotenv import load_dotenv

        load_dotenv(BASE_DIR / ".env")
    except ImportError:
        pass
    ua = os.environ.get("ROBOFLOW_HTTP_USER_AGENT", "").strip()
    if not ua:
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": ua,
    }


def _roboflow_model_http_headers() -> dict[str, str]:
    """Same browser-like User-Agent as workflows; body is raw base64 per hosted-model REST."""
    h = _roboflow_workflow_http_headers()
    h["Content-Type"] = "application/x-www-form-urlencoded"
    h.pop("Accept", None)
    return h


def _roboflow_run_model_http(frame_bgr: np.ndarray) -> dict:
    """POST base64-encoded JPEG to serverless `/{model_id}` (see Roboflow hosted inference REST)."""
    api_key, api_url, *_ = _roboflow_plate_env()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set.")
    model_id = (os.environ.get("ROBOFLOW_MODEL_ID", "tunisian-license-plate-9umhi-26wmt/1") or "").strip().strip("/")
    if not model_id:
        raise RuntimeError("ROBOFLOW_MODEL_ID is not set.")
    try:
        conf = float(os.environ.get("ROBOFLOW_CONFIDENCE", "0.4") or "0.4")
    except ValueError:
        conf = 0.4
    try:
        overlap = float(os.environ.get("ROBOFLOW_OVERLAP", "0.3") or "0.3")
    except ValueError:
        overlap = 0.3
    qs = urllib.parse.urlencode(
        {"api_key": api_key, "confidence": str(conf), "overlap": str(overlap)}
    )
    url = f"{api_url}/{model_id}?{qs}"
    body = _frame_jpeg_base64(frame_bgr).encode("ascii")
    req = urllib.request.Request(url, data=body, method="POST", headers=_roboflow_model_http_headers())
    print("DIRECT MODEL CALL URL:", url.split("?")[0])
    print("DIRECT MODEL KEY START:", api_key[:4])
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body_txt = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Roboflow HTTP {e.code}: {detail}") from e
    return json.loads(body_txt)


def _roboflow_run_workflow_http(frame_bgr: np.ndarray) -> dict:
    """POST to Roboflow serverless workflow (same JSON shape as inference-sdk)."""
    api_key, api_url, workspace, workflow_id, classes, _, image_key, classes_key = _roboflow_plate_env()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set.")
    url = f"{api_url}/{workspace}/workflows/{workflow_id}"
    inputs: dict = {image_key: {"type": "base64", "value": _frame_jpeg_base64(frame_bgr)}}
    if classes:
        inputs[classes_key] = classes
    payload = {
        "api_key": api_key,
        "use_cache": True,
        "enable_profiling": False,
        "inputs": inputs,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers=_roboflow_workflow_http_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(f"Roboflow HTTP {e.code}: {detail}") from e
    return _roboflow_normalize_workflow_response(json.loads(body))


def _roboflow_run_workflow_sdk(frame_bgr: np.ndarray) -> dict:
    """Same call as Roboflow docs (InferenceHTTPClient.run_workflow). Requires inference-sdk (Python <3.13)."""
    from inference_sdk import InferenceHTTPClient

    api_key, api_url, workspace, workflow_id, classes, _, image_key, classes_key = _roboflow_plate_env()
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set.")
    client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
    params = {classes_key: classes} if classes else {}
    wf_kw: dict = {
        "workspace_name": workspace,
        "workflow_id": workflow_id,
        "use_cache": True,
    }
    if params:
        wf_kw["parameters"] = params
    try:
        wf_kw["images"] = {image_key: frame_bgr}
        result = client.run_workflow(**wf_kw)
    except TypeError:
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            cv2.imwrite(path, frame_bgr)
            wf_kw["images"] = {image_key: path}
            result = client.run_workflow(**wf_kw)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    if isinstance(result, list):
        return {"outputs": result}
    if isinstance(result, dict):
        return result
    return {"outputs": [result]}


def _roboflow_run_workflow(frame_bgr: np.ndarray) -> dict:
    """Prefer inference-sdk when installed (ROBOFLOW_USE_SDK=auto|sdk); else JSON HTTP."""
    mode = os.environ.get("ROBOFLOW_USE_SDK", "auto").strip().lower()
    if mode in ("http", "0", "false", "no"):
        return _roboflow_run_workflow_http(frame_bgr)
    if mode in ("sdk", "1", "true", "yes"):
        return _roboflow_run_workflow_sdk(frame_bgr)
    try:
        import inference_sdk  # noqa: F401
    except ImportError:
        return _roboflow_run_workflow_http(frame_bgr)
    try:
        return _roboflow_run_workflow_sdk(frame_bgr)
    except Exception:
        return _roboflow_run_workflow_http(frame_bgr)


def _roboflow_plate_inference(frame_bgr: np.ndarray) -> dict:
    """Workflow JSON or direct model JSON with `predictions`; same downstream box collection."""
    if _roboflow_use_workflow():
        return _roboflow_run_workflow(frame_bgr)
    return _roboflow_run_model_http(frame_bgr)


def _prediction_dict_to_xyxy(
    p: dict, frame_w: int, frame_h: int
) -> tuple[int, int, int, int, float] | None:
    conf = float(p.get("confidence", p.get("score", 0.0)) or 0.0)
    if "xyxy" in p and isinstance(p["xyxy"], (list, tuple)) and len(p["xyxy"]) >= 4:
        x1, y1, x2, y2 = (float(p["xyxy"][i]) for i in range(4))
    elif all(k in p for k in ("x1", "y1", "x2", "y2")):
        x1, y1, x2, y2 = float(p["x1"]), float(p["y1"]), float(p["x2"]), float(p["y2"])
    elif all(k in p for k in ("x", "y", "width", "height")):
        xc, yc = float(p["x"]), float(p["y"])
        w, h = float(p["width"]), float(p["height"])
        if max(abs(xc), abs(yc), w, h) <= 1.5 and frame_w > 2 and frame_h > 2:
            xc *= frame_w
            yc *= frame_h
            w *= frame_w
            h *= frame_h
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0
    else:
        return None
    x1i = int(round(x1))
    y1i = int(round(y1))
    x2i = int(round(x2))
    y2i = int(round(y2))
    x1i = max(0, min(x1i, frame_w - 1))
    x2i = max(0, min(x2i, frame_w))
    y1i = max(0, min(y1i, frame_h - 1))
    y2i = max(0, min(y2i, frame_h))
    if x2i <= x1i or y2i <= y1i:
        return None
    return x1i, y1i, x2i, y2i, conf


def _polygon_points_to_xyxy(
    points, frame_w: int, frame_h: int
) -> tuple[int, int, int, int, float] | None:
    if not isinstance(points, list) or len(points) < 3:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for pt in points:
        if isinstance(pt, dict):
            xs.append(float(pt.get("x", 0)))
            ys.append(float(pt.get("y", 0)))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            xs.append(float(pt[0]))
            ys.append(float(pt[1]))
    if not xs:
        return None
    if max(abs(x) for x in xs + ys) <= 1.5 and frame_w > 2 and frame_h > 2:
        xs = [x * frame_w for x in xs]
        ys = [y * frame_h for y in ys]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    conf = 0.75
    return _prediction_dict_to_xyxy(
        {"x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0, "width": x2 - x1, "height": y2 - y1, "confidence": conf},
        frame_w,
        frame_h,
    )


def _bbox_list_to_xyxy(raw, frame_w: int, frame_h: int) -> tuple[int, int, int, int, float] | None:
    if isinstance(raw, dict):
        if all(k in raw for k in ("x_min", "y_min", "x_max", "y_max")):
            return _prediction_dict_to_xyxy(
                {
                    "x1": raw["x_min"],
                    "y1": raw["y_min"],
                    "x2": raw["x_max"],
                    "y2": raw["y_max"],
                    "confidence": raw.get("confidence", 0.5),
                },
                frame_w,
                frame_h,
            )
        return None
    if isinstance(raw, (list, tuple)) and len(raw) >= 4:
        x1, y1, x2, y2 = (float(raw[i]) for i in range(4))
        return _prediction_dict_to_xyxy(
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": 0.5},
            frame_w,
            frame_h,
        )
    return None


def _extract_boxes_from_prediction_dict(
    pr: dict, frame_w: int, frame_h: int
) -> list[tuple[int, int, int, int, float]]:
    found: list[tuple[int, int, int, int, float]] = []
    b = _prediction_dict_to_xyxy(pr, frame_w, frame_h)
    if b:
        found.append(b)
    for key in ("points", "polygon", "segmentation"):
        pts = pr.get(key)
        if isinstance(pts, list):
            pb = _polygon_points_to_xyxy(pts, frame_w, frame_h)
            if pb:
                found.append(pb)
    for key in ("bbox", "box", "xyxy"):
        if key in pr:
            bb = _bbox_list_to_xyxy(pr[key], frame_w, frame_h)
            if bb:
                found.append(bb)
    return found


# Do not recurse into server-rendered assets (huge base64) when hunting for boxes.
_ROBOFLOW_JSON_SKIP_KEYS = frozenset(
    {
        "bounding_box_visualization_output",
        "mask_visualization_output",
        "segmentation_visualization_output",
    }
)


def _roboflow_find_server_visualization_jpeg_b64(obj) -> str | None:
    """First Roboflow `bounding_box_visualization_output` JPEG base64 string, if any."""
    if isinstance(obj, dict):
        vis = obj.get("bounding_box_visualization_output")
        if isinstance(vis, dict) and str(vis.get("type", "")).lower() == "base64":
            val = vis.get("value")
            if isinstance(val, str) and len(val) > 200:
                return val
        for v in obj.values():
            hit = _roboflow_find_server_visualization_jpeg_b64(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for item in obj:
            hit = _roboflow_find_server_visualization_jpeg_b64(item)
            if hit:
                return hit
    return None


def _unwrap_workflow_step_payload(obj):
    """Many workflow steps return {\"type\": ..., \"value\": <nested json>}."""
    if not isinstance(obj, dict):
        return obj
    if "value" not in obj:
        return obj
    val = obj["value"]
    if isinstance(val, str) and len(val) > 800:
        return obj
    if isinstance(val, (dict, list)):
        return val
    return obj


def _collect_boxes_from_roboflow_payload(obj, frame_w: int, frame_h: int, out: list[tuple[int, int, int, int, float]]):
    if isinstance(obj, dict):
        obj = _unwrap_workflow_step_payload(obj)
    if isinstance(obj, dict):
        for list_key in ("predictions", "detections", "objects", "instances", "masks"):
            preds = obj.get(list_key)
            if not isinstance(preds, list):
                continue
            for pr in preds:
                if not isinstance(pr, dict):
                    continue
                if not _roboflow_prediction_class_ok(pr):
                    continue
                for b in _extract_boxes_from_prediction_dict(pr, frame_w, frame_h):
                    out.append(b)
        for k, v in obj.items():
            if k in ("predictions", "detections", "objects", "instances", "masks"):
                continue
            if k in _ROBOFLOW_JSON_SKIP_KEYS:
                continue
            if isinstance(v, str) and len(v) > 2000:
                continue
            _collect_boxes_from_roboflow_payload(v, frame_w, frame_h, out)
    elif isinstance(obj, list):
        for item in obj:
            _collect_boxes_from_roboflow_payload(item, frame_w, frame_h, out)


def _roboflow_plate_boxes(frame_bgr: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    fh, fw = frame_bgr.shape[:2]
    data = _roboflow_normalize_workflow_response(_roboflow_plate_inference(frame_bgr))
    if os.environ.get("ROBOFLOW_DEBUG_SAVE_LAST", "").strip() == "1":
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "roboflow_last_plate_response.json").write_text(
                json.dumps(data, indent=2)[:500000],
                encoding="utf-8",
            )
            vis_b64 = _roboflow_find_server_visualization_jpeg_b64(data.get("outputs", data))
            if vis_b64:
                (OUTPUT_DIR / "roboflow_last_visualization.jpg").write_bytes(
                    base64.b64decode(vis_b64, validate=False)
                )
        except (OSError, ValueError, TypeError):
            pass
    raw = data.get("outputs", data)
    boxes: list[tuple[int, int, int, int, float]] = []
    _collect_boxes_from_roboflow_payload(raw, fw, fh, boxes)
    dedup: list[tuple[int, int, int, int, float]] = []
    seen = set()
    for b in boxes:
        key = (b[0], b[1], b[2], b[3])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(b)
    return dedup


def _append_plate_detections_from_boxes(
    frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int, float]], out: list
):
    for px1, py1, px2, py2, pconf in boxes:
        plate_crop = frame_bgr[py1:py2, px1:px2]
        if plate_crop.size == 0:
            continue
        ptext, rconf = _recognize_tunisian_plate_text(plate_crop)
        label = f"plate {ptext}" if ptext else "plate"
        score = max(pconf, rconf)
        out.append(
            {
                "label": label,
                "confidence": max(0.25, min(0.99, score if score > 0 else 0.30)),
                "x1": int(px1),
                "y1": int(py1),
                "x2": int(px2),
                "y2": int(py2),
                "source": "plate",
            }
        )


def _detect_combined_street_objects(
    frame_bgr: np.ndarray,
    sign_conf: float = 0.5,
    traffic_conf: float = 0.35,
    stream_frame_index: int | None = None,
):
    print(f"DEBUG: Running _detect_combined_street_objects (v11 integration active)")
    out = []
    
    # 1. Street Sign Model (my_model.pt / YOLOv8)
    # Handles all traffic signs including Stop signs.
    # Skips any generic "traffic light" class — those come from the YOLOv11 model below.
    sign_model = _get_street_sign_model()
    sign_labels = sign_model.names if hasattr(sign_model, "names") else {}
    sign_result = sign_model.predict(frame_bgr, verbose=False)[0]
    sign_boxes = sign_result.boxes if sign_result is not None and sign_result.boxes is not None else []
    for det in sign_boxes:
        conf = float(det.conf.item())
        if conf < sign_conf:
            continue
        cls_idx = int(det.cls.item())
        name = str(sign_labels.get(cls_idx, cls_idx)).lower()

        # Skip generic "traffic light" — specific Red/Green Light come from the YOLOv11 model
        if "traffic light" in name:
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        out.append({"label": name, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "source": "street_sign"})

    # 2. YOLOv11 Traffic Light / Speed Limit Model (traffic_sign_detector.pt)
    # Classes: Green Light, Red Light, Speed Limit 20-120.
    # "Stop" class from this model is intentionally skipped — my_model.pt handles stop signs.
    new_tl_model = _get_new_traffic_sign_model()
    new_tl_labels = new_tl_model.names if hasattr(new_tl_model, "names") else {}
    new_tl_result = new_tl_model.predict(frame_bgr, verbose=False)[0]
    new_tl_boxes = new_tl_result.boxes if new_tl_result is not None and new_tl_result.boxes is not None else []
    for det in new_tl_boxes:
        conf = float(det.conf.item())
        if conf < traffic_conf:
            continue
        cls_idx = int(det.cls.item())
        name = str(new_tl_labels.get(cls_idx, cls_idx))

        # Skip "Stop" from this model — my_model.pt is the authoritative stop sign detector
        if name.lower() == "stop":
            continue

        # Skip any generic "traffic light" label (not Red/Green Light)
        if name.lower() == "traffic light":
            continue

        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        out.append(
            {
                "label": name,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "source": "new_traffic_sign",
            }
        )

    # 3. YOLOv8n COCO model — Cars only
    tl_model = _get_car_detector_model()
    tl_labels = tl_model.names if hasattr(tl_model, "names") else {}
    tl_result = tl_model.predict(frame_bgr, verbose=False)[0]
    tl_boxes = tl_result.boxes if tl_result is not None and tl_result.boxes is not None else []
    
    has_red_light = any(d["label"].lower() == "red light" for d in out if d["source"] == "new_traffic_sign")
    
    for det in tl_boxes:
        conf = float(det.conf.item())
        if conf < traffic_conf:
            continue
        cls_idx = int(det.cls.item())
        name = str(tl_labels.get(cls_idx, cls_idx)).lower()
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        
        # EXPLICITLY ONLY CARS FROM COCO
        if name == "car":
            car_det = {
                "label": "car",
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "source": "car",
            }
            
            # Simple Red Light Violation Heuristic: 
            # If red light is present and car is in the lower half of the frame (crossing imaginary line)
            if has_red_light and y2 > (frame_bgr.shape[0] * 0.65):
                car_det["violation"] = "red_light_violation"
                car_det["label"] = "car (VIOLATION)"
                
            out.append(car_det)

    # Plate detection and OCR stopped as requested
    plate_backend = "stopped"
    return out, None, plate_backend


def _best_conf_for_label(detections, labels: dict, target_label: str) -> float:
    best = 0.0
    if detections is None:
        return best
    for det in detections:
        cls_idx = int(det.cls.item())
        label = str(labels.get(cls_idx, cls_idx)).lower()
        if label != target_label:
            continue
        conf = float(det.conf.item())
        if conf > best:
            best = conf
    return best


def _compute_real_xai_cells(frame_bgr: np.ndarray, det: dict, grid: int = 4) -> dict:
    """Occlusion-based sensitivity map inside one detection box."""
    x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
    if x2 - x1 < 8 or y2 - y1 < 8:
        return {"cells": [], "faithfulness": 0.0, "baseline_conf": float(det.get("confidence", 0.0))}
    source = str(det.get("source", "street_sign"))
    target_label = str(det.get("label", "")).lower()
    if source == "traffic_light":
        # Legacy source label — map to car detector (COCO)
        model = _get_car_detector_model()
    elif source == "new_traffic_sign":
        model = _get_new_traffic_sign_model()
    elif source in {"car", "plate"}:
        return {"cells": [], "faithfulness": 0.0, "baseline_conf": float(det.get("confidence", 0.0))}
    else:
        model = _get_street_sign_model()
    labels = model.names if hasattr(model, "names") else {}
    baseline = float(det.get("confidence", 0.0))
    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return {"cells": [], "faithfulness": 0.0, "baseline_conf": float(det.get("confidence", 0.0))}
    fill_color = tuple(int(x) for x in np.mean(roi.reshape(-1, 3), axis=0))
    w = x2 - x1
    h = y2 - y1
    cells = []
    raw_drops = []
    for gy in range(grid):
        for gx in range(grid):
            cx1 = x1 + int((gx / grid) * w)
            cx2 = x1 + int(((gx + 1) / grid) * w)
            cy1 = y1 + int((gy / grid) * h)
            cy2 = y1 + int(((gy + 1) / grid) * h)
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            masked = frame_bgr.copy()
            masked[cy1:cy2, cx1:cx2] = fill_color
            result = model.predict(masked, verbose=False)[0]
            detections = result.boxes if result is not None and result.boxes is not None else None
            new_conf = _best_conf_for_label(detections, labels, target_label)
            drop = max(0.0, baseline - new_conf)
            raw_drops.append(drop)
            cells.append({"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2, "score": drop})
    if not cells:
        return {"cells": [], "faithfulness": 0.0, "baseline_conf": baseline}
    mx = max(c["score"] for c in cells)
    if mx <= 1e-8:
        return {"cells": [], "faithfulness": 0.0, "baseline_conf": baseline}
    for c in cells:
        c["score"] = float(c["score"] / mx)
    mean_drop = float(np.mean(raw_drops)) if raw_drops else 0.0
    faithfulness = float(mean_drop / max(1e-6, baseline))
    faithfulness = max(0.0, min(1.0, faithfulness))
    return {"cells": cells, "faithfulness": faithfulness, "baseline_conf": float(baseline)}


def _build_object_heatmap(roi_bgr: np.ndarray, conf: float) -> np.ndarray | None:
    """
    Build a CAM-like heatmap for one detection ROI.
    Fast proxy for Grad-CAM on exported detector pipelines:
    combines local gradient energy + center prior weighted by detector confidence.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None
    h, w = roi_bgr.shape[:2]
    if h < 6 or w < 6:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    grad_mag = cv2.GaussianBlur(grad_mag, (0, 0), sigmaX=1.2, sigmaY=1.2)
    grad_norm = cv2.normalize(grad_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    sx = max(1.0, w * 0.35)
    sy = max(1.0, h * 0.35)
    center_prior = np.exp(-(((xx - cx) ** 2) / (2 * sx * sx) + ((yy - cy) ** 2) / (2 * sy * sy)))

    conf_w = float(max(0.15, min(1.0, conf)))
    cam_like = (0.65 * grad_norm + 0.35 * center_prior) * conf_w
    cam_like = cv2.normalize(cam_like, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(cam_like, cv2.COLORMAP_JET)


def _overlay_detection_heatmaps(
    frame_bgr: np.ndarray,
    detections: list[dict],
    plate_boxes: list[tuple[int, int, int, int]] | None = None,
) -> np.ndarray:
    """
    Overlay heatmaps for traffic lights and vehicles.
    Plate boxes are excluded from heatmap blending to protect OCR readability.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    safe_plate_boxes = plate_boxes or []

    for det in detections:
        source = str(det.get("source", "")).lower()
        label = str(det.get("label", "")).lower()
        is_light = "light" in label and source in {"new_traffic_sign", "traffic_light"}
        is_car_like = source == "car" or label.startswith("car")
        if not (is_light or is_car_like):
            continue

        x1 = max(0, min(w - 1, int(det.get("x1", 0))))
        y1 = max(0, min(h - 1, int(det.get("y1", 0))))
        x2 = max(0, min(w, int(det.get("x2", 0))))
        y2 = max(0, min(h, int(det.get("y2", 0))))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        heat = _build_object_heatmap(roi, float(det.get("confidence", 0.0)))
        if heat is None:
            continue

        alpha = 0.30 if is_light else 0.25
        blended = cv2.addWeighted(roi, 1.0 - alpha, heat, alpha, 0.0)

        if safe_plate_boxes and is_car_like:
            mask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255
            for px1, py1, px2, py2 in safe_plate_boxes:
                ix1 = max(x1, int(px1))
                iy1 = max(y1, int(py1))
                ix2 = min(x2, int(px2))
                iy2 = min(y2, int(py2))
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                mask[iy1 - y1 : iy2 - y1, ix1 - x1 : ix2 - x1] = 0
            keep_orig = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            keep_heat = cv2.bitwise_and(blended, blended, mask=mask)
            out[y1:y2, x1:x2] = cv2.add(keep_orig, keep_heat)
        else:
            out[y1:y2, x1:x2] = blended

    return out


def _annotate_street_sign_frame(frame_bgr: np.ndarray, detections, road_zone: dict | None = None):
    out = _overlay_detection_heatmaps(frame_bgr, detections, plate_boxes=None)
    if road_zone:
        x1 = int(road_zone.get("x1", 0))
        x2 = int(road_zone.get("x2", out.shape[1] - 1))
        y1 = int(road_zone.get("y1", 0))
        cv2.rectangle(out, (x1, y1), (x2, out.shape[0] - 1), (40, 200, 255), 1)
        cv2.putText(
            out,
            "Road crossing zone",
            (x1 + 6, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 200, 255),
            1,
            cv2.LINE_AA,
        )
    hit_counts: dict[str, int] = {}
    max_conf = 0.0
    kept = 0
    for det in detections:
        conf = float(det["confidence"])
        name = str(det["label"])
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        source = str(det.get("source", "street_sign"))
        violation = det.get("violation")
        
        if violation == "red_light_violation":
            color = (0, 0, 255) # Bright Red for violations
        elif source == "street_sign":
            color = (96, 202, 231)
        elif source == "traffic_light":
            color = (80, 255, 120)
        elif source == "new_traffic_sign":
            color = (50, 255, 255)  # Cyan for new model
        elif source == "car":
            color = (255, 200, 80)
        elif source == "plate":
            color = (255, 120, 80)
        else:
            color = (180, 180, 180)
        thick = 4 if violation else (3 if source == "plate" else 2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = f"{name}: {int(conf * 100)}%"
        if violation:
            text = f"!!! {text} !!!"
        cv2.putText(out, text, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        hit_counts[name] = hit_counts.get(name, 0) + 1
        max_conf = max(max_conf, conf)
        kept += 1
    top_sign = max(hit_counts.items(), key=lambda kv: kv[1])[0] if hit_counts else "none"
    lines = [
        f"Mode: Signs (my_model.pt) + Lights/Speed (YOLOv11) + Cars (YOLOv8)",
        f"Top sign: {top_sign}",
        f"Detections: {kept}",
        f"Peak confidence: {max_conf:.3f}",
    ]
    color = (0, 255, 0) if kept == 0 else (0, 165, 255)
    return overlay_text(out, lines, color=color), hit_counts, max_conf, kept


def _street_sign_plate_hints(plate_backend: str) -> dict[str, str]:
    """User-facing hints when plate pipeline falls back or errors."""
    hints: dict[str, str] = {}
    if plate_backend == "stopped":
        hints["plate_hint"] = "Car plate detection and OCR are currently disabled by request."
    elif plate_backend == "fallback_contour":
        key, *_ = _roboflow_plate_env()
        if not key:
            hints["plate_hint"] = (
                "Roboflow is not active: ROBOFLOW_API_KEY is empty or was not read from .env next to web_app.py. "
                "Set the key there and restart the Flask server so the environment reloads."
            )
        else:
            hints["plate_hint"] = (
                "The Ultralytics Tunisian plate hub model failed to load; contour fallback is weak on real plates. "
                "Fix Roboflow env vars or network so the API path can run."
            )
    elif "roboflow_error" in plate_backend:
        extra = ""
        if "1010" in plate_backend or "error code: 1010" in plate_backend.lower():
            extra = (
                " HTTP 403 + error code 1010 is usually Cloudflare blocking Python’s default client. "
                "The app now sends a normal browser User-Agent; restart the server. "
                "You can still set ROBOFLOW_HTTP_USER_AGENT in .env if needed. "
                "Also confirm ROBOFLOW_API_KEY is your Private key (not the publishable inference.js key) for server inference."
            )
        mode = "workflow" if _roboflow_use_workflow() else "hosted model"
        hints["plate_hint"] = (
            f"Roboflow plate {mode} HTTP call failed. Check ROBOFLOW_* env (model id or workspace/workflow), API key, and billing."
            + extra
            + f" Detail: {plate_backend[:400]}"
        )
    elif "fallback_contour" in plate_backend and plate_backend.startswith("roboflow"):
        hints["plate_hint"] = (
            "Roboflow ran but returned no plate boxes this run (or the JSON shape did not match the parser). "
            "Set ROBOFLOW_DEBUG_SAVE_LAST=1 in .env to write web_outputs/roboflow_last_plate_response.json for inspection."
        )
    return hints


def _extract_plate_text_from_label(label: str) -> str:
    """Normalize plate label text from variants like 'plate ABC123' or 'plate: ABC123'."""
    s = str(label or "").strip()
    if not s:
        return ""
    low = s.lower()
    if low.startswith("plate"):
        tail = s[5:].strip()
        if tail.startswith(":"):
            tail = tail[1:].strip()
        return tail
    return ""


def analyze_street_sign_image(input_path: Path, output_path: Path, conf_threshold: float = 0.5) -> dict:
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError("Could not read image file.")
    detections, road_zone, plate_backend = _detect_combined_street_objects(
        frame, sign_conf=conf_threshold, traffic_conf=0.35
    )
    show, hit_counts, max_conf, kept = _annotate_street_sign_frame(frame, detections, road_zone=road_zone)
    cv2.imwrite(str(output_path), show)
    top_sign = max(hit_counts.items(), key=lambda kv: kv[1])[0] if hit_counts else "none"
    
    out = {
        "media_type": "image",
        "frames_processed": 1,
        "top_sign": top_sign,
        "detections": kept,
        "peak_score": round(max_conf, 4),
        "sign_counts": hit_counts,
        "plate_detector_backend": plate_backend,
    }
    out.update(_street_sign_plate_hints(plate_backend))
    return out


def analyze_street_sign_video(input_path: Path, output_path: Path, max_frames: int = 500, conf_threshold: float = 0.5) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    all_counts: Counter = Counter()

    def process_frame(frame: np.ndarray, frame_index: int):
        detections, road_zone, plate_backend_local = _detect_combined_street_objects(
            frame, sign_conf=conf_threshold, traffic_conf=0.35, stream_frame_index=frame_index
        )
        show, hit_counts, max_conf, kept = _annotate_street_sign_frame(frame, detections, road_zone=road_zone)
        for k, v in hit_counts.items():
            all_counts[k] += v
        status = "No sign detected" if kept == 0 else f"Detected {kept} object(s)"
        process_frame.last_plate_backend = plate_backend_local
        return show, max_conf, status

    process_frame.last_plate_backend = "unknown"
    stats = _render_to_h264(
        cap, output_path, process_frame, max_frames=max_frames, pass_frame_index=True
    )
    stats["top_sign"] = all_counts.most_common(1)[0][0] if all_counts else "none"
    stats["sign_counts"] = dict(all_counts.most_common(10))
    stats["total_detections"] = int(sum(all_counts.values()))
    stats["plate_detector_backend"] = process_frame.last_plate_backend
    stats.update(_street_sign_plate_hints(process_frame.last_plate_backend))
    return stats


# ---------------------------------------------------------------------------
# Stop-line crossing helpers
# ---------------------------------------------------------------------------

def _bottom_center(x1: int, y1: int, x2: int, y2: int) -> tuple[float, float]:
    """Return the bottom-center point of a bounding box."""
    return ((x1 + x2) / 2.0, float(y2))


def _segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    """Return True if segment p1→p2 crosses segment q1→q2 (2-D, robust cross-product test)."""
    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def _on_segment(o, a, b):
        return (
            min(o[0], b[0]) <= a[0] <= max(o[0], b[0])
            and min(o[1], b[1]) <= a[1] <= max(o[1], b[1])
        )

    d1 = _cross(q1, q2, p1)
    d2 = _cross(q1, q2, p2)
    d3 = _cross(p1, p2, q1)
    d4 = _cross(p1, p2, q2)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True

    # Collinear / endpoint cases
    if d1 == 0 and _on_segment(q1, p1, q2):
        return True
    if d2 == 0 and _on_segment(q1, p2, q2):
        return True
    if d3 == 0 and _on_segment(p1, q1, p2):
        return True
    if d4 == 0 and _on_segment(p1, q2, p2):
        return True
    return False


def _draw_stop_line(
    frame: np.ndarray,
    stop_line: list[tuple[int, int]],
    violation_count: int,
    normal_count: int,
) -> None:
    """Draw the stop line and counters onto frame in-place."""
    if len(stop_line) < 2:
        return
    p1, p2 = tuple(stop_line[0]), tuple(stop_line[1])
    # Magenta line
    cv2.line(frame, p1, p2, (255, 0, 255), 3, cv2.LINE_AA)
    # Endpoint dots
    cv2.circle(frame, p1, 6, (255, 0, 255), -1)
    cv2.circle(frame, p2, 6, (255, 0, 255), -1)
    # Label
    mid_x = (p1[0] + p2[0]) // 2
    mid_y = (p1[1] + p2[1]) // 2
    cv2.putText(
        frame, "STOP LINE",
        (mid_x - 40, max(16, mid_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA,
    )
    # Counters
    cv2.putText(
        frame, f"Violations: {violation_count}",
        (10, frame.shape[0] - 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"Crossings: {normal_count}",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )


def analyze_street_sign_video_tracked(
    input_path: Path,
    output_path: Path,
    max_frames: int = 500,
    conf_threshold: float = 0.5,
) -> dict:
    """
    Process a traffic video with YOLO tracking.  Saves per-frame tracking data
    (car positions + traffic-light state) to a JSON sidecar file so that
    violation checking can be done later without re-running YOLO.

    Returns stats dict that includes 'tracking_token' — the key used to look up
    the sidecar JSON via POST /street-sign/check-violations.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = _normalize_fps(cap.get(cv2.CAP_PROP_FPS))
    temp_dir = Path(tempfile.mkdtemp(prefix="tracked_frames_", dir=str(OUTPUT_DIR)))

    sign_model = _get_street_sign_model()
    tl_model   = _get_new_traffic_sign_model()
    car_model  = _get_car_detector_model()

    all_counts: Counter = Counter()
    frame_count = 0

    # Sidecar: list of per-frame records for later violation replay
    tracking_frames: list[dict] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            if frame_count > max_frames:
                break

            annotated = frame.copy()
            sign_dets: list[dict] = []
            tl_state = "unknown"

            # ── signs (my_model.pt) ───────────────────────────────────────────
            s_res = sign_model.predict(frame, verbose=False)[0]
            if s_res and s_res.boxes is not None:
                for det in s_res.boxes:
                    conf = float(det.conf.item())
                    if conf < conf_threshold:
                        continue
                    cls_idx = int(det.cls.item())
                    name = str(sign_model.names.get(cls_idx, cls_idx)).lower()
                    if "traffic light" in name:
                        continue
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    sign_dets.append({"label": name, "confidence": conf,
                                      "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                      "source": "street_sign"})
                    all_counts[name] += 1

            # ── traffic lights + speed limits (YOLOv11) ───────────────────────
            tl_res = tl_model.predict(frame, verbose=False)[0]
            if tl_res and tl_res.boxes is not None:
                for det in tl_res.boxes:
                    conf = float(det.conf.item())
                    if conf < 0.35:
                        continue
                    cls_idx = int(det.cls.item())
                    name = str(tl_model.names.get(cls_idx, cls_idx))
                    name_low = name.lower()
                    if name_low in ("stop", "traffic light"):
                        continue
                    if name_low == "red light":
                        tl_state = "red"
                    elif name_low == "green light":
                        tl_state = "green"
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    sign_dets.append({"label": name, "confidence": conf,
                                      "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                      "source": "new_traffic_sign"})
                    all_counts[name] += 1

            # ── car tracking (YOLOv8n COCO) ───────────────────────────────────
            try:
                track_results = car_model.track(frame, persist=True, verbose=False,
                                                 conf=0.35, classes=[2])
            except Exception:
                track_results = car_model.predict(frame, verbose=False, conf=0.35)

            frame_cars: list[dict] = []
            if track_results and len(track_results) > 0:
                res = track_results[0]
                for det in (res.boxes or []):
                    conf = float(det.conf.item())
                    cls_idx = int(det.cls.item())
                    name = str(car_model.names.get(cls_idx, cls_idx)).lower()
                    if name not in ("car", "truck", "bus", "motorcycle"):
                        continue
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    bc = _bottom_center(x1, y1, x2, y2)
                    track_id: int | None = None
                    if det.id is not None:
                        try:
                            track_id = int(det.id.item())
                        except Exception:
                            pass
                    frame_cars.append({
                        "track_id": track_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "bcx": bc[0], "bcy": bc[1],
                        "conf": conf,
                    })
                    all_counts["car"] += 1

            # save sidecar record for this frame
            tracking_frames.append({
                "frame": frame_count,
                "tl_state": tl_state,
                "cars": frame_cars,
            })

            # ── CAM-like heatmaps (traffic lights + cars) ────────────────────
            heatmap_dets = list(sign_dets)
            for car in frame_cars:
                heatmap_dets.append(
                    {
                        "label": "car",
                        "confidence": float(car.get("conf", 0.0)),
                        "x1": int(car["x1"]),
                        "y1": int(car["y1"]),
                        "x2": int(car["x2"]),
                        "y2": int(car["y2"]),
                        "source": "car",
                    }
                )
            annotated = _overlay_detection_heatmaps(annotated, heatmap_dets, plate_boxes=None)

            # ── annotate frame (no stop line yet) ────────────────────────────
            for det in sign_dets:
                src = det["source"]
                name_low = det["label"].lower()
                if src == "street_sign":
                    color = (96, 202, 231)
                elif name_low == "red light":
                    color = (0, 0, 255)
                elif name_low == "green light":
                    color = (0, 255, 0)
                else:
                    color = (50, 255, 255)
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{det['label']}: {int(det['confidence']*100)}%",
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            for car in frame_cars:
                x1, y1, x2, y2 = car["x1"], car["y1"], car["x2"], car["y2"]
                tid = car["track_id"]
                label = f"car ID:{tid}" if tid is not None else "car"
                color = (255, 200, 80)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                cv2.circle(annotated, (int(car["bcx"]), int(car["bcy"])), 4, color, -1)

            tl_color = (0, 0, 255) if tl_state == "red" else (0, 255, 0) if tl_state == "green" else (180, 180, 180)
            tl_text = {"red": "Red Light: ON", "green": "Green Light: ON"}.get(tl_state, "Light: unknown")
            cv2.putText(annotated, tl_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, tl_color, 2, cv2.LINE_AA)

            frame_path = temp_dir / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_path), annotated)

        if frame_count == 0:
            raise RuntimeError("Video had no readable frames.")

        _encode_frames_to_h264_mp4(temp_dir, fps, output_path)
        _validate_h264_output(output_path)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Processing finished but output file is empty.")

    finally:
        cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Save sidecar JSON — keyed by the output stem so the check-violations
    # endpoint can find it without any extra state.
    tracking_token = output_path.stem          # e.g. "abc12345_street_sign"
    sidecar_path = OUTPUT_DIR / f"{tracking_token}_tracking.json"
    sidecar_path.write_text(
        json.dumps({"fps": fps, "frames": tracking_frames}, separators=(",", ":")),
        encoding="utf-8",
    )

    top_sign = all_counts.most_common(1)[0][0] if all_counts else "none"
    return {
        "media_type": "video",
        "frames_processed": frame_count,
        "top_sign": top_sign,
        "sign_counts": dict(all_counts.most_common(10)),
        "total_detections": int(sum(all_counts.values())),
        "plate_detector_backend": "stopped",
        "plate_hint": "Car plate detection and OCR are currently disabled by request.",
        "tracking_token": tracking_token,   # sent to frontend for violation check
        "stop_line_used": False,
    }


def analyze_street_sign_video_with_violations(
    video_path: Path,
    output_path: Path,
    token: str,
    stop_line: list[tuple[int, int]],
) -> dict:
    """
    Process video with stop line already defined.
    Violations are marked with RED boxes in the output video.
    """
    sign_model = _get_street_sign_model()
    tl_model = _get_new_traffic_sign_model()
    car_model = _get_car_detector_model()
    plate_model = _get_plate_detector_model()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    temp_dir = UPLOAD_DIR / f"temp_{token}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory for plate crops
    plates_dir = OUTPUT_DIR / f"{token}_plates"
    plates_dir.mkdir(parents=True, exist_ok=True)

    all_counts = Counter()
    frame_count = 0
    tl_state = "unknown"  # Persistent traffic light state
    
    # Violation tracking
    violated_ids = set()
    normal_crossing_ids = set()
    previous_positions = {}
    
    # Plate tracking: map car_id -> relative plate position within car box
    # Run plate detection less frequently to speed up processing
    car_plate_relative = {}  # Store relative position: (rel_x1, rel_y1, rel_x2, rel_y2)
    car_plate_best_crop = {}  # Store best crop for each car: {car_id: (crop_image, confidence)}
    PLATE_DETECTION_INTERVAL = 15  # Detect every 15 frames (twice per second at 30fps)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            annotated = frame.copy()

            # ── street signs (YOLOv8) ────────────────────────────────────────
            sign_dets = []
            sign_res = sign_model.predict(frame, verbose=False)[0]
            if sign_res and sign_res.boxes is not None:
                for det in sign_res.boxes:
                    conf = float(det.conf.item())
                    if conf < 0.35:
                        continue
                    cls_idx = int(det.cls.item())
                    name = str(sign_model.names.get(cls_idx, cls_idx)).lower()
                    if "traffic light" in name:
                        continue
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    sign_dets.append({"label": name, "confidence": conf,
                                      "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                      "source": "street_sign"})
                    all_counts[name] += 1

            # ── traffic lights + speed limits (YOLOv11) ───────────────────────
            tl_res = tl_model.predict(frame, verbose=False)[0]
            if tl_res and tl_res.boxes is not None:
                for det in tl_res.boxes:
                    conf = float(det.conf.item())
                    if conf < 0.35:
                        continue
                    cls_idx = int(det.cls.item())
                    name = str(tl_model.names.get(cls_idx, cls_idx))
                    name_low = name.lower()
                    if name_low in ("stop", "traffic light"):
                        continue
                    if name_low == "red light":
                        tl_state = "red"
                    elif name_low == "green light":
                        tl_state = "green"
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    sign_dets.append({"label": name, "confidence": conf,
                                      "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                      "source": "new_traffic_sign"})
                    all_counts[name] += 1

            # ── car tracking (YOLOv8n COCO) ───────────────────────────────────
            try:
                # Use ByteTrack with AGGRESSIVE parameters for maximum ID persistence
                # Especially important for cars that get occluded by other vehicles
                track_results = car_model.track(
                    frame, 
                    persist=True, 
                    verbose=False,
                    conf=0.3,  # Lower confidence to detect partially occluded cars
                    classes=[2],
                    tracker="bytetrack.yaml",
                    # AGGRESSIVE ByteTrack parameters for occlusion handling
                    iou=0.3,  # Lower IOU - allows more flexible matching during occlusion
                    track_high_thresh=0.3,  # Lower high threshold - keep tracking weak detections
                    track_low_thresh=0.05,  # Very low threshold for re-identification after occlusion
                    new_track_thresh=0.6,   # Higher threshold - harder to create new IDs (prevents duplicates)
                    track_buffer=90,  # Keep lost tracks for 90 frames (3 seconds at 30fps) - longer memory
                    match_thresh=0.7  # Lower matching threshold - more lenient re-identification
                )
            except Exception:
                track_results = car_model.predict(frame, verbose=False, conf=0.35)

            frame_cars = []
            if track_results and len(track_results) > 0:
                res = track_results[0]
                for det in (res.boxes or []):
                    conf = float(det.conf.item())
                    cls_idx = int(det.cls.item())
                    name = str(car_model.names.get(cls_idx, cls_idx)).lower()
                    if name not in ("car", "truck", "bus", "motorcycle"):
                        continue
                    x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
                    bc = _bottom_center(x1, y1, x2, y2)
                    track_id = None
                    if det.id is not None:
                        try:
                            track_id = int(det.id.item())
                        except Exception:
                            pass
                    
                    # Detect plate inside car bounding box - every 15 frames
                    plate_text = None
                    plate_box = None
                    if plate_model is not None and track_id is not None:
                        car_width = x2 - x1
                        car_height = y2 - y1
                        
                        # Run detection on this interval to update position
                        if frame_count % PLATE_DETECTION_INTERVAL == 0:
                            # Crop car region
                            car_crop = frame[y1:y2, x1:x2]
                            if car_crop.size > 0 and car_width > 0 and car_height > 0:
                                try:
                                    plate_results = plate_model.predict(car_crop, verbose=False, conf=0.25)
                                    if plate_results and len(plate_results) > 0:
                                        plate_res = plate_results[0]
                                        if plate_res.boxes is not None and len(plate_res.boxes) > 0:
                                            # Get the plate with highest confidence
                                            best_plate = None
                                            best_conf = 0.0
                                            for plate_det in plate_res.boxes:
                                                plate_conf = float(plate_det.conf.item())
                                                if plate_conf > best_conf:
                                                    best_conf = plate_conf
                                                    best_plate = plate_det
                                            
                                            if best_plate is not None:
                                                px1, py1, px2, py2 = map(int, best_plate.xyxy[0].tolist())
                                                
                                                # Update relative position (0.0 to 1.0)
                                                rel_x1 = px1 / car_width
                                                rel_y1 = py1 / car_height
                                                rel_x2 = px2 / car_width
                                                rel_y2 = py2 / car_height
                                                car_plate_relative[track_id] = (rel_x1, rel_y1, rel_x2, rel_y2)
                                                
                                                # Calculate absolute position for this frame
                                                abs_px1 = x1 + px1
                                                abs_py1 = y1 + py1
                                                abs_px2 = x1 + px2
                                                abs_py2 = y1 + py2
                                                plate_box = (abs_px1, abs_py1, abs_px2, abs_py2)
                                                plate_text = f"Plate_{track_id}"
                                                
                                                # Save cropped plate image if it's better than previous
                                                plate_crop = car_crop[py1:py2, px1:px2]
                                                if plate_crop.size > 0:
                                                    # Only save if this is the best confidence for this car
                                                    if track_id not in car_plate_best_crop or best_conf > car_plate_best_crop[track_id][1]:
                                                        car_plate_best_crop[track_id] = (plate_crop.copy(), best_conf)
                                                        all_counts["license_plate"] = len(car_plate_best_crop)
                                except Exception as e:
                                    print(f"Plate detection error: {e}")
                        
                        # Use stored relative position if we have it (even on non-detection frames)
                        if track_id in car_plate_relative and plate_box is None:
                            # Calculate absolute position from relative position
                            rel_x1, rel_y1, rel_x2, rel_y2 = car_plate_relative[track_id]
                            abs_px1 = x1 + int(rel_x1 * car_width)
                            abs_py1 = y1 + int(rel_y1 * car_height)
                            abs_px2 = x1 + int(rel_x2 * car_width)
                            abs_py2 = y1 + int(rel_y2 * car_height)
                            plate_box = (abs_px1, abs_py1, abs_px2, abs_py2)
                            plate_text = f"Plate_{track_id}"
                    
                    # Check for line crossing
                    if track_id is not None and track_id in previous_positions:
                        prev_bc = previous_positions[track_id]
                        if _segments_intersect(prev_bc, bc, stop_line[0], stop_line[1]):
                            if tl_state == "red" and track_id not in violated_ids:
                                violated_ids.add(track_id)
                            elif tl_state != "red" and track_id not in normal_crossing_ids:
                                normal_crossing_ids.add(track_id)
                    
                    if track_id is not None:
                        previous_positions[track_id] = bc
                    
                    frame_cars.append({
                        "track_id": track_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "bcx": bc[0], "bcy": bc[1],
                        "conf": conf,
                        "plate": plate_text,
                        "plate_box": plate_box,
                    })
                    all_counts["car"] += 1

            # ── CAM-like heatmaps (traffic lights + cars, exclude plate areas) ──
            heatmap_dets = list(sign_dets)
            heatmap_plate_boxes = []
            for car in frame_cars:
                heatmap_dets.append(
                    {
                        "label": "car",
                        "confidence": float(car.get("conf", 0.0)),
                        "x1": int(car["x1"]),
                        "y1": int(car["y1"]),
                        "x2": int(car["x2"]),
                        "y2": int(car["y2"]),
                        "source": "car",
                    }
                )
                pb = car.get("plate_box")
                if pb:
                    heatmap_plate_boxes.append(pb)
            annotated = _overlay_detection_heatmaps(
                annotated,
                heatmap_dets,
                plate_boxes=heatmap_plate_boxes,
            )

            # ── annotate frame ────────────────────────────────────────────────
            for det in sign_dets:
                src = det["source"]
                name_low = det["label"].lower()
                if src == "street_sign":
                    color = (96, 202, 231)
                elif name_low == "red light":
                    color = (0, 0, 255)
                elif name_low == "green light":
                    color = (0, 255, 0)
                else:
                    color = (50, 255, 255)
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{det['label']}: {int(det['confidence']*100)}%",
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            for car in frame_cars:
                x1, y1, x2, y2 = car["x1"], car["y1"], car["x2"], car["y2"]
                tid = car["track_id"]
                plate = car.get("plate")
                plate_box = car.get("plate_box")
                
                # RED box for violations, normal color otherwise
                if tid is not None and tid in violated_ids:
                    color = (0, 0, 255)  # RED
                    label = f"VIOLATION ID:{tid}"
                    if plate:
                        label += f" {plate}"
                else:
                    color = (255, 200, 80)  # Normal cyan
                    label = f"car ID:{tid}" if tid is not None else "car"
                    if plate:
                        label += f" {plate}"
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                cv2.circle(annotated, (int(car["bcx"]), int(car["bcy"])), 4, color, -1)
                
                # Draw plate bounding box in yellow
                if plate_box:
                    px1, py1, px2, py2 = plate_box
                    cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 255, 255), 2)
                    cv2.putText(annotated, "PLATE", (px1, max(12, py1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw stop line and counters
            _draw_stop_line(annotated, stop_line, len(violated_ids), len(normal_crossing_ids))

            # Traffic light status
            tl_color = (0, 0, 255) if tl_state == "red" else (0, 255, 0) if tl_state == "green" else (180, 180, 180)
            tl_text = {"red": "Red Light: ON", "green": "Green Light: ON"}.get(tl_state, "Light: unknown")
            cv2.putText(annotated, tl_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, tl_color, 2, cv2.LINE_AA)

            frame_path = temp_dir / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_path), annotated)

        if frame_count == 0:
            raise RuntimeError("Video had no readable frames.")

        _encode_frames_to_h264_mp4(temp_dir, fps, output_path)
        _validate_h264_output(output_path)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Processing finished but output file is empty.")

    finally:
        cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Save best plate crops for each car
    plate_images = {}
    for car_id, (crop, conf) in car_plate_best_crop.items():
        plate_filename = plates_dir / f"plate_car{car_id}_conf{int(conf*100)}.jpg"
        cv2.imwrite(str(plate_filename), crop)
        # Store relative path for template
        plate_images[car_id] = {
            "filename": plate_filename.name,
            "confidence": conf,
            "path": f"{token}_plates/{plate_filename.name}"
        }

    top_sign = all_counts.most_common(1)[0][0] if all_counts else "none"
    return {
        "media_type": "video",
        "frames_processed": frame_count,
        "top_sign": top_sign,
        "sign_counts": dict(all_counts.most_common(10)),
        "total_detections": int(sum(all_counts.values())),
        "plate_detector_backend": "YOLOv11",
        "plate_hint": f"Detected {len(plate_images)} license plates",
        "stop_line_used": True,
        "violation_count": len(violated_ids),
        "violated_ids": sorted(violated_ids),
        "normal_crossing_count": len(normal_crossing_ids),
        "plate_images": plate_images,
    }


def check_violations_from_tracking(
    tracking_token: str,
    stop_line: list[tuple[int, int]],
) -> dict:
    """
    Pure geometry pass — no YOLO.  Replays saved tracking data against the
    user-drawn stop line and returns violation results instantly.
    """
    sidecar_path = OUTPUT_DIR / f"{tracking_token}_tracking.json"
    if not sidecar_path.exists():
        raise RuntimeError(f"Tracking data not found for token '{tracking_token}'. "
                           "Process the video first.")

    data = json.loads(sidecar_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])

    q1 = (float(stop_line[0][0]), float(stop_line[0][1]))
    q2 = (float(stop_line[1][0]), float(stop_line[1][1]))

    prev_positions: dict[int, tuple[float, float]] = {}
    violated_ids: set[int] = set()
    crossed_ids:  set[int] = set()
    violation_count = 0
    normal_crossing_count = 0

    for frec in frames:
        tl_state = frec.get("tl_state", "unknown")
        for car in frec.get("cars", []):
            tid = car.get("track_id")
            if tid is None:
                continue
            bc = (float(car["bcx"]), float(car["bcy"]))
            prev = prev_positions.get(tid)
            if prev is not None:
                if _segments_intersect(prev, bc, q1, q2):
                    if tl_state == "red" and tid not in violated_ids:
                        violated_ids.add(tid)
                        violation_count += 1
                    elif tl_state != "red" and tid not in crossed_ids:
                        crossed_ids.add(tid)
                        normal_crossing_count += 1
            prev_positions[tid] = bc

    return {
        "ok": True,
        "violation_count": violation_count,
        "normal_crossing_count": normal_crossing_count,
        "violated_ids": sorted(violated_ids),
    }


def _animal_project_dir() -> Path:
    candidates = [
        BASE_DIR / "Animal-Behaviour-and-Disease-Detection-main" / "Animal-Behaviour-and-Disease-Detection-main",
        Path(r"C:\Users\USER\Downloads\Animal-Behaviour-and-Disease-Detection-main\Animal-Behaviour-and-Disease-Detection-main"),
    ]
    for c in candidates:
        if (c / "mobilenetv2_image_classifier.h5").is_file():
            return c
    raise RuntimeError(
        "Animal Behaviour project not found. Copy the folder "
        "`Animal-Behaviour-and-Disease-Detection-main/Animal-Behaviour-and-Disease-Detection-main` "
        "into your Smart city directory (next to web_app.py), or keep it under Downloads."
    )


def _load_keras_mobilenet():
    """Load project .h5 if TensorFlow is available; otherwise return None."""
    global _animal_keras_model, _keras_load_attempted
    if _keras_load_attempted:
        return _animal_keras_model
    _keras_load_attempted = True
    try:
        from tensorflow.keras.models import load_model

        weights = _animal_project_dir() / "mobilenetv2_image_classifier.h5"
        _animal_keras_model = load_model(str(weights))
    except Exception:
        _animal_keras_model = None
    return _animal_keras_model


def _get_yolo_animal_detector():
    global _yolo_animal_model
    if _yolo_animal_model is None:
        _yolo_animal_model = YOLO("yolov8n.pt")
    return _yolo_animal_model


def _yolo_best_animal_crop(frame_bgr: np.ndarray, conf: float = 0.15):
    """Return crop + metadata for the largest COCO animal detection."""
    model = _get_yolo_animal_detector()
    results = model.predict(source=frame_bgr, conf=conf, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return None, None, None, 0.0, 0.0
    names = results.names
    best_area = 0
    best_box = None
    best_name = None
    best_conf = 0.0
    for b in results.boxes:
        cls_id = int(b.cls.item())
        name = str(names.get(cls_id, "")).lower()
        if name not in YOLO_ANIMAL_NAMES:
            continue
        b_conf = float(b.conf.item()) if hasattr(b, "conf") else 0.0
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best_box = (x1, y1, x2, y2)
            best_name = name
            best_conf = b_conf
    if best_box is None:
        return None, None, None, 0.0, 0.0
    x1, y1, x2, y2 = best_box
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame_bgr[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None, None, None, 0.0, 0.0
    area_ratio = best_area / float(max(1, w * h))
    return crop, best_box, best_name, best_conf, area_ratio


def _load_animalpose_module():
    """Load original animal project behavior recognizer, if available."""
    global _animalpose_module, _animalpose_attempted
    if _animalpose_attempted:
        return _animalpose_module
    _animalpose_attempted = True
    proj_dir = _animal_project_dir()
    module_path = proj_dir / "animalpose.py"
    if not module_path.exists():
        _animalpose_module = None
        return None
    try:
        spec = importlib.util.spec_from_file_location("animalpose_runtime", module_path)
        if spec is None or spec.loader is None:
            _animalpose_module = None
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _animalpose_module = module
    except Exception:
        _animalpose_module = None
    return _animalpose_module


def _predict_disease_bgr(frame_bgr: np.ndarray):
    """Returns (label, probs[3]|None, backend, confidence)."""
    model = _load_keras_mobilenet()
    if model is None:
        return "Unknown", None, "unavailable", 0.0
    from tensorflow.keras.preprocessing import image as keras_image

    r = cv2.resize(frame_bgr, (224, 224))
    rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
    arr = keras_image.img_to_array(rgb)
    arr = np.expand_dims(arr, axis=0) / 255.0
    preds = model.predict(arr, verbose=0)[0]
    cls = int(np.argmax(preds))
    return DISEASE_LABELS[cls], preds, "keras", float(preds[cls])


def _predict_behavior_bgr(frame_bgr: np.ndarray):
    """Returns (behavior_label, confidence, backend)."""
    module = _load_animalpose_module()
    if module is not None:
        try:
            idx = int(module.recognize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)))
            idx = max(0, min(idx, len(BEHAVIOR_LABELS) - 1))
            return BEHAVIOR_LABELS[idx], 1.0, "animalpose"
        except Exception:
            pass
    # Fallback without CLIP: simple visible-pose heuristic from crop geometry.
    h, w = frame_bgr.shape[:2]
    if h <= 1 or w <= 1:
        return "Standing", 0.34, "heuristic"
    ratio = w / float(h)
    if ratio >= 1.35:
        return "Resting", min(0.95, 0.55 + (ratio - 1.35) * 0.3), "heuristic"
    if ratio >= 1.05:
        return "Eating", min(0.90, 0.52 + (ratio - 1.05) * 0.5), "heuristic"
    return "Standing", min(0.90, 0.55 + (1.05 - ratio) * 0.5), "heuristic"


def _annotate_animal_frame(
    frame_bgr: np.ndarray,
    box: tuple | None,
    species: str | None,
    beh: str,
    dis: str,
    summary: str,
    color: tuple,
):
    """Draw YOLO box + labels on video/image. Natural-language XAI (LLaVA) is on the web page only."""
    out = frame_bgr.copy()
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{species or 'animal'}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    lines = [
        f"YOLO: {species or 'no animal box — classifiers use full frame'}",
        f"Behavior: {beh}",
        f"Disease: {dis}",
        f"Summary: {summary}",
        "LLaVA explanation: see web page (not on video)",
    ]
    return overlay_text(out, lines, color=color)


def _combined_summary(behavior: str, disease: str) -> str:
    if disease == "Unknown":
        return f"Behavior appears {behavior.lower()}; disease model is unavailable."
    if behavior == "Standing" and disease == "Injured":
        return "The animal is standing and appears to be injured."
    if behavior == "Eating" and disease == "Injured":
        return "The animal is eating and appears to be injured."
    if behavior == "Resting" and disease == "Injured":
        return "The animal is resting and appears to be injured."
    if behavior == "Standing" and disease == "mange":
        return "The animal is standing and might have mange."
    if behavior == "Eating" and disease == "mange":
        return "The animal is eating and might have mange."
    if behavior == "Resting" and disease == "mange":
        return "The animal is resting and might have mange."
    if behavior == "Standing" and disease == "cognitive":
        return "The animal is standing and might have cognitive issues."
    if behavior == "Eating" and disease == "cognitive":
        return "The animal is eating and might have cognitive issues."
    if behavior == "Resting" and disease == "cognitive":
        return "The animal is resting and might have cognitive issues."
    return f"Behavior: {behavior}, disease signal: {disease}."


def analyze_animal_behaviour_image(input_path: Path, output_path: Path) -> dict:
    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError("Could not read image file.")
    crop, box, species, yolo_conf, box_area_ratio = _yolo_best_animal_crop(frame)
    roi = crop if crop is not None else frame
    beh, beh_conf, beh_backend = _predict_behavior_bgr(roi)
    dis, dis_probs, dis_backend, dis_conf = _predict_disease_bgr(roi)
    summary = _combined_summary(beh, dis)
    color = (0, 0, 255) if dis == "Injured" else ((0, 165, 255) if dis == "mange" else (0, 255, 0))
    show = _annotate_animal_frame(frame, box, species, beh, dis, summary, color)
    cv2.imwrite(str(output_path), show)
    decision_meta = {
        "yolo_species": species or "unknown",
        "yolo_conf": f"{yolo_conf:.3f}",
        "crop_area_ratio": f"{box_area_ratio:.3f}",
        "behavior_backend": beh_backend,
        "behavior_conf": f"{beh_conf:.3f}",
        "disease_backend": dis_backend,
        "disease_conf": f"{dis_conf:.3f}",
    }
    if dis_backend == "unavailable":
        decision_meta["disease_note"] = "Install TensorFlow + add mobilenetv2_image_classifier.h5"
    llava_text, llava_err = _llava_explain_crop(roi, species, beh, dis, summary, decision_meta=decision_meta)
    return {
        "media_type": "image",
        "frames_processed": 1,
        "behavior": beh,
        "disease": dis,
        "summary": summary,
        "behavior_vote": beh,
        "disease_vote": dis,
        "decision_evidence": decision_meta,
        "llava_xai": llava_text,
        "llava_error": llava_err,
    }


def analyze_animal_behaviour_video(input_path: Path, output_path: Path, max_frames: int = 400, stride: int = 2) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    fps = _normalize_fps(cap.get(cv2.CAP_PROP_FPS))
    temp_dir = Path(tempfile.mkdtemp(prefix="animal_frames_", dir=str(OUTPUT_DIR)))
    behaviors = []
    diseases = []
    frame_count = 0
    last_roi_bgr = None
    last_species = None
    yolo_conf_values = []
    yolo_area_values = []
    beh_conf_values = []
    last_beh_backend = "unknown"
    dis_conf_values = []
    last_dis_backend = "unknown"
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride != 0:
                continue
            frame_count += 1
            if frame_count > max_frames:
                break
            crop, box, species, yolo_conf, box_area_ratio = _yolo_best_animal_crop(frame)
            roi = crop if crop is not None else frame
            beh, beh_conf, beh_backend = _predict_behavior_bgr(roi)
            dis, dis_probs, dis_backend, dis_conf = _predict_disease_bgr(roi)
            behaviors.append(beh)
            diseases.append(dis)
            yolo_conf_values.append(yolo_conf)
            yolo_area_values.append(box_area_ratio)
            beh_conf_values.append(float(beh_conf))
            dis_conf_values.append(float(dis_conf))
            last_beh_backend = beh_backend
            last_dis_backend = dis_backend
            summary = _combined_summary(beh, dis)
            color = (0, 0, 255) if dis == "Injured" else ((0, 165, 255) if dis == "mange" else (0, 255, 0))
            show = _annotate_animal_frame(frame, box, species, beh, dis, summary, color)
            frame_path = temp_dir / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(frame_path), show)
            last_roi_bgr = roi.copy()
            last_species = species
        if frame_count == 0:
            raise RuntimeError("Video had no readable frames.")
        bv = Counter(behaviors).most_common(1)[0][0] if behaviors else ""
        dv = Counter(diseases).most_common(1)[0][0] if diseases else ""
        summary = _combined_summary(bv, dv)
        _encode_frames_to_h264_mp4(temp_dir, fps, output_path)
        _validate_h264_output(output_path)
    finally:
        cap.release()
        shutil.rmtree(temp_dir, ignore_errors=True)

    llava_text, llava_err = None, None
    decision_meta = {
        "yolo_species": last_species or "unknown",
        "yolo_conf_avg": f"{(sum(yolo_conf_values) / max(1, len(yolo_conf_values))):.3f}",
        "crop_area_ratio_avg": f"{(sum(yolo_area_values) / max(1, len(yolo_area_values))):.3f}",
        "behavior_backend": last_beh_backend,
        "behavior_conf_avg": f"{(sum(beh_conf_values) / max(1, len(beh_conf_values))):.3f}",
        "disease_backend": last_dis_backend,
        "disease_conf_avg": f"{(sum(dis_conf_values) / max(1, len(dis_conf_values))):.3f}",
        "vote_behavior": bv,
        "vote_disease": dv,
        "frames_used": str(frame_count),
    }
    if last_dis_backend == "unavailable":
        decision_meta["disease_note"] = "Install TensorFlow + add mobilenetv2_image_classifier.h5"
    if last_roi_bgr is not None:
        llava_text, llava_err = _llava_explain_crop(
            last_roi_bgr,
            last_species,
            bv,
            dv,
            summary,
            decision_meta=decision_meta,
            local_files_only=True,
        )

    out = {
        "media_type": "video",
        "frames_processed": frame_count,
        "behavior": behaviors[-1] if behaviors else "",
        "disease": diseases[-1] if diseases else "",
        "summary": summary,
        "behavior_vote": bv,
        "disease_vote": dv,
        "decision_evidence": decision_meta,
        "llava_xai": llava_text,
        "llava_error": llava_err,
    }
    return out


app = Flask(__name__)
ensure_dirs()


@app.get("/")
def index():
    """Redirect to street sign page"""
    return redirect(url_for("street_sign_page"))


@app.get("/street-sign")
def street_sign_page():
    return render_template("street_sign.html", active_page="street_sign")


@app.post("/street-sign/first-frame")
def street_sign_first_frame():
    """Return the first frame of an uploaded video as a JPEG data-URL for stop-line drawing."""
    file = request.files.get("media")
    if not file or not file.filename:
        return {"ok": False, "error": "No file"}, 400
    safe_name = secure_filename(file.filename)
    token = uuid.uuid4().hex[:8]
    upload_path = UPLOAD_DIR / f"{token}_{safe_name}"
    file.save(upload_path)
    try:
        cap = cv2.VideoCapture(str(upload_path))
        if not cap.isOpened():
            return {"ok": False, "error": "Cannot open video"}, 400
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return {"ok": False, "error": "Cannot read first frame"}, 400
        _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        h, w = frame.shape[:2]
        return {
            "ok": True,
            "frame_data": f"data:image/jpeg;base64,{b64}",
            "width": w,
            "height": h,
            "token": token,
            "filename": safe_name,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 500


@app.post("/street-sign/suggest-stop-line")
def street_sign_suggest_stop_line():
    """
    Get AI-suggested stop line from first frame.
    Expects: {"frame_data": "data:image/jpeg;base64,..."}
    Returns: {"ok": true, "stop_line_points": [[x1,y1],[x2,y2]], "confidence": 0.0-1.0, "explanation": "..."}
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        frame_data = body.get("frame_data", "")
        
        if not frame_data.startswith("data:image/"):
            return {"ok": False, "error": "Invalid frame_data"}, 400
        
        # Decode base64 image
        header, b64_data = frame_data.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"ok": False, "error": "Could not decode image"}, 400
        
        # Get AI suggestion
        suggestion = suggest_stop_line_with_openai(frame)
        
        if "error" in suggestion:
            return {"ok": False, "error": suggestion["error"]}, 500
        
        return {
            "ok": True,
            "stop_line_points": suggestion.get("stop_line_points", []),
            "confidence": suggestion.get("confidence", 0.0),
            "explanation": suggestion.get("explanation", "")
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 500


@app.route("/street-sign/analyze", methods=["GET", "POST"])
def street_sign_analyze():
    if request.method == "GET":
        return redirect(url_for("street_sign"))
    
    token = uuid.uuid4().hex[:8]
    is_image = False
    is_video = False
    upload_path = None
    stop_line = None

    # Check if stop line data was provided
    stop_line_json = request.form.get("stop_line_data", "").strip()
    if stop_line_json:
        try:
            stop_line_data = json.loads(stop_line_json)
            stop_line = [(int(p["x"]), int(p["y"])) for p in stop_line_data]
        except Exception:
            pass

    file = request.files.get("media")
    if file and file.filename:
        safe_name = secure_filename(file.filename)
        ext = Path(safe_name).suffix.lower()
        upload_path = UPLOAD_DIR / f"{token}_{safe_name}"
        file.save(upload_path)
        is_image = ext in STREET_SIGN_IMAGE_EXTS
        is_video = ext in ALLOWED_EXTENSIONS
        if not (is_image or is_video):
            return render_template(
                "street_sign.html",
                active_page="street_sign",
                error="Unsupported file type. Use an image (jpg/png/...) or video (mp4/...).",
            )
    else:
        data_url = request.form.get("camera_image_data", "").strip()
        if not data_url.startswith("data:image/"):
            return render_template(
                "street_sign.html",
                active_page="street_sign",
                error="Please choose a file or capture from camera.",
            )
        try:
            header, b64_data = data_url.split(",", 1)
            image_bytes = base64.b64decode(b64_data)
            upload_path = UPLOAD_DIR / f"{token}_camera.jpg"
            upload_path.write_bytes(image_bytes)
            is_image = True
        except Exception:
            return render_template(
                "street_sign.html",
                active_page="street_sign",
                error="Could not read captured camera image.",
            )

    try:
        if is_image:
            output_name = f"{token}_street_sign.png"
            output_path = OUTPUT_DIR / output_name
            result = analyze_street_sign_image(upload_path, output_path)
        else:
            output_name = f"{token}_street_sign.mp4"
            output_path = OUTPUT_DIR / output_name
            if stop_line:
                # Process with stop line - violations will be marked in red
                result = analyze_street_sign_video_with_violations(upload_path, output_path, token, stop_line)
            else:
                # No stop line - just track for later
                result = analyze_street_sign_video_tracked(upload_path, output_path)
    except Exception as exc:
        return render_template("street_sign.html", active_page="street_sign", error=f"Analysis failed: {exc}")

    output_url = url_for("serve_output", filename=output_name)
    
    # If violations were detected with stop line, save results for optional dashboard
    if result.get("stop_line_used") and result.get("violation_count", 0) > 0:
        # Save violation results for dashboard
        result_path = OUTPUT_DIR / f"{token}_violation_results.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        
        # Add dashboard link to result (only if AI agents available)
        if AI_AGENTS_AVAILABLE and OPENAI_API_KEY:
            result["dashboard_token"] = token
    
    # Auto-delete uploaded file after processing to save space
    try:
        if upload_path.exists():
            upload_path.unlink()
            print(f"✓ Deleted uploaded file: {upload_path.name}")
    except Exception as e:
        print(f"⚠ Could not delete upload: {e}")
    
    # Delete all OLD output videos (keep only the current one)
    try:
        for old_video in OUTPUT_DIR.glob("*_street_sign.mp4"):
            if old_video.name != output_name:
                old_video.unlink()
                print(f"✓ Deleted old output video: {old_video.name}")
        
        # Also delete old animal videos
        for old_video in OUTPUT_DIR.glob("*_animal.mp4"):
            old_video.unlink()
            print(f"✓ Deleted old animal video: {old_video.name}")
    except Exception as e:
        print(f"⚠ Could not delete old videos: {e}")
    
    return render_template("street_sign.html", active_page="street_sign", result=result, output_url=output_url)


@app.post("/street-sign/check-violations")
def street_sign_check_violations():
    """
    Pure-geometry violation check — no YOLO re-run.
    Expects JSON body: { "tracking_token": "...", "stop_line": [{x,y},{x,y}] }
    Returns: { "ok": true, "violation_count": N, "violated_ids": [...], 
               "normal_crossing_count": N, "tracking_data": {...} }
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        tracking_token = str(body.get("tracking_token", "")).strip()
        raw_line = body.get("stop_line", [])
        if not tracking_token:
            return {"ok": False, "error": "tracking_token is required"}, 400
        if len(raw_line) != 2:
            return {"ok": False, "error": "stop_line must have exactly 2 points"}, 400
        stop_line = [(int(p["x"]), int(p["y"])) for p in raw_line]
        result = check_violations_from_tracking(tracking_token, stop_line)
        
        # Also return the tracking data so frontend can draw boxes
        tracking_path = UPLOAD_DIR / f"{tracking_token}_tracking.json"
        tracking_data = None
        if tracking_path.exists():
            tracking_data = json.loads(tracking_path.read_text(encoding="utf-8"))
        
        result["tracking_data"] = tracking_data
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 500


@app.post("/street-sign/realtime-frame")
def street_sign_realtime_frame():
    global _street_rt_roboflow_counter
    data_url = request.form.get("frame_data", "").strip()
    if not data_url.startswith("data:image/"):
        return {"ok": False, "error": "Missing frame_data"}, 400
    try:
        _, b64_data = data_url.split(",", 1)
        image_bytes = base64.b64decode(b64_data)
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"ok": False, "error": "Could not decode frame"}, 400
        _street_rt_roboflow_counter += 1
        dets, road_zone, plate_backend = _detect_combined_street_objects(
            frame,
            sign_conf=0.5,
            traffic_conf=0.35,
            stream_frame_index=_street_rt_roboflow_counter,
        )
        real_xai = request.form.get("real_xai", "0").strip() in {"1", "true", "on"}
        xai_cells = []
        xai_meta = {"faithfulness": 0.0, "baseline_conf": 0.0}
        if real_xai and dets:
            top_det = max(dets, key=lambda d: float(d.get("confidence", 0.0)))
            xai_payload = _compute_real_xai_cells(frame, top_det, grid=4)
            xai_cells = xai_payload.get("cells", [])
            xai_meta = {
                "faithfulness": float(xai_payload.get("faithfulness", 0.0)),
                "baseline_conf": float(xai_payload.get("baseline_conf", 0.0)),
                "target_label": str(top_det.get("label", "")),
            }
        return {
            "ok": True,
            "detections": dets,
            "xai_cells": xai_cells,
            "xai_meta": xai_meta,
            "road_zone": road_zone,
            "plate_detector_backend": plate_backend,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 500


@app.get("/outputs/<path:filename>")
def serve_output(filename: str):
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return "Not found", 404
    ext = file_path.suffix.lower()
    mimetype = {
        ".mp4": "video/mp4",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    resp = send_file(
        file_path,
        mimetype=mimetype,
        as_attachment=False,
        conditional=True,
        etag=True,
        max_age=0,
    )
    if ext == ".mp4":
        resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# AI AGENTS AND DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/violation-dashboard/<token>")
def violation_dashboard(token):
    """Display comprehensive violation dashboard with AI analysis"""
    try:
        if not AI_AGENTS_AVAILABLE:
            return "AI agents require reportlab library. Install: pip install reportlab", 500
        
        if not OPENAI_API_KEY:
            return "OpenAI API key not configured in .env file", 500
        
        # Load violation results
        result_path = OUTPUT_DIR / f"{token}_violation_results.json"
        if not result_path.exists():
            return "Violation results not found", 404
        
        results = json.loads(result_path.read_text(encoding="utf-8"))
        
        # Initialize AI agents
        analysis_agent = ViolationAnalysisAgent(OPENAI_API_KEY)
        summary_agent = ViolationSummaryAgent(OPENAI_API_KEY)
        recommendation_agent = RecommendationAgent(OPENAI_API_KEY)
        pdf_gen = ViolationPDFGenerator(OUTPUT_DIR / f"{token}_reports")
        
        # Process each violation with AI
        violations = []
        ocr_debug_list = []  # Collect OCR debug info
        for car_id in results.get("violated_ids", []):
            violation_data = {
                "car_id": car_id,
                "traffic_light": "red",
                "crossed_stop_line": True,
                "plate_images_count": 0,
                "confidence": 0.85,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frame_number": 0
            }
            
            # Get AI analysis
            ai_result = analysis_agent.analyze_violation(violation_data)
            
            # Get plate images for this car
            plate_images = []
            plates_dir = OUTPUT_DIR / f"{token}_plates"
            if plates_dir.exists():
                for plate_file in plates_dir.glob(f"plate_car{car_id}_*.jpg"):
                    conf_str = plate_file.stem.split("_conf")[-1]
                    try:
                        conf = int(conf_str) / 100.0
                        plate_images.append((plate_file, conf))
                    except:
                        pass
            
            violation_data["plate_images_count"] = len(plate_images)
            
            # OCR Processing: Extract plate number from best plate image
            plate_number = "NO_PLATE"
            plate_verified = False
            ocr_confidence = 0.0
            ocr_notes = ""
            ocr_debug_entry = {
                "car_id": car_id,
                "ocr_raw": None,
                "auto_formatted": None,
                "openai_result": None,
                "final_plate": None,
                "error": None
            }
            
            if OCR_AVAILABLE and OCR_API_KEY and OPENAI_API_KEY and plate_images:
                try:
                    print(f"\n🔍 Running OCR for Car ID {car_id}...")
                    print(f"   OCR_AVAILABLE: {OCR_AVAILABLE}")
                    print(f"   OCR_API_KEY exists: {bool(OCR_API_KEY)}")
                    print(f"   OPENAI_API_KEY exists: {bool(OPENAI_API_KEY)}")
                    print(f"   Plate images: {len(plate_images)}")
                    
                    ocr_result = process_violation_plates(
                        car_id=car_id,
                        plate_images=plate_images,
                        ocr_api_key=OCR_API_KEY,
                        openai_api_key=OPENAI_API_KEY
                    )
                    
                    print(f"   OCR Result: {ocr_result}")
                    
                    plate_number = ocr_result.get("plate_number", "OCR_FAILED")
                    plate_verified = ocr_result.get("is_tunisian_plate", False)
                    ocr_confidence = ocr_result.get("confidence", 0.0)
                    ocr_notes = ocr_result.get("notes", "")
                    ocr_raw = ocr_result.get("ocr_raw", "")  # Raw OCR before OpenAI
                    
                    # Populate debug info
                    ocr_debug_entry["ocr_raw"] = ocr_raw
                    ocr_debug_entry["auto_formatted"] = ocr_raw.replace('-', ' تونس ').replace('_', ' تونس ') if ocr_raw else None
                    ocr_debug_entry["openai_result"] = ocr_result.get("notes", "")
                    ocr_debug_entry["final_plate"] = plate_number
                    
                    print(f"   📝 Raw OCR: {ocr_raw}")
                    print(f"   🤖 OpenAI Verified: {plate_number}")
                    print(f"   ✓ Status: verified={plate_verified}, conf={ocr_confidence:.2%}")
                    
                    # Update violation data with OCR results
                    violation_data["plate_number"] = plate_number
                    violation_data["plate_verified"] = plate_verified
                    violation_data["ocr_confidence"] = ocr_confidence
                    violation_data["ocr_raw"] = ocr_raw  # Store raw OCR for comparison
                    
                except Exception as ocr_error:
                    error_msg = str(ocr_error)
                    print(f"   ✗ OCR failed: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    plate_number = "OCR_ERROR"
                    ocr_debug_entry["error"] = error_msg
            else:
                # Debug why OCR is not running
                reasons = []
                if not OCR_AVAILABLE:
                    reasons.append("OCR module not available")
                if not OCR_API_KEY:
                    reasons.append("OCR_API_KEY not set")
                if not OPENAI_API_KEY:
                    reasons.append("OPENAI_API_KEY not set")
                if not plate_images:
                    reasons.append("No plate images")
                
                error_msg = ", ".join(reasons)
                print(f"\n⚠ OCR skipped for Car ID {car_id}: {error_msg}")
                ocr_debug_entry["error"] = f"Skipped: {error_msg}"
            
            # Add debug entry to list
            ocr_debug_list.append(ocr_debug_entry)
            
            # Database search and email notification
            vehicle_info = None
            email_sent = False
            if DB_EMAIL_AVAILABLE and plate_number and plate_number not in ("NO_PLATE", "OCR_FAILED", "OCR_ERROR"):
                try:
                    print(f"\n🔍 Searching database for plate: {plate_number}")
                    vehicle_info = db.search_vehicle(plate_number)
                    
                    if vehicle_info:
                        print(f"   ✓ Vehicle found: {vehicle_info['owner_name']}")
                        print(f"   Email: {vehicle_info['email']}")
                        
                        # Record violation in database
                        violation_id = db.record_violation(
                            plate_number=plate_number,
                            vehicle_id=vehicle_info['id'],
                            video_token=token,
                            car_tracking_id=car_id,
                            fine_amount=85.0
                        )
                        
                        # Send email notification
                        if EMAIL_USER and EMAIL_PASSWORD:
                            notifier = ViolationEmailNotifier(
                                smtp_host=EMAIL_HOST,
                                smtp_port=EMAIL_PORT,
                                email_user=EMAIL_USER,
                                email_password=EMAIL_PASSWORD
                            )
                            
                            # Get best plate image
                            best_plate_img = sorted(plate_images, key=lambda x: x[1], reverse=True)[0][0] if plate_images else None
                            
                            email_sent = notifier.send_violation_email(
                                to_email=vehicle_info['email'],
                                owner_name=vehicle_info['owner_name'],
                                plate_number=plate_number,
                                violation_date=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                fine_amount=85.0,
                                plate_image_path=best_plate_img
                            )
                            
                            if email_sent:
                                db.mark_email_sent(violation_id)
                        else:
                            print(f"   ⚠ Email not configured")
                    else:
                        print(f"   ⚠ Vehicle not found in database")
                        # Record violation without vehicle_id
                        db.record_violation(
                            plate_number=plate_number,
                            vehicle_id=None,
                            video_token=token,
                            car_tracking_id=car_id,
                            fine_amount=85.0
                        )
                except Exception as db_error:
                    print(f"   ✗ Database/Email error: {db_error}")
                    import traceback
                    traceback.print_exc()
            
            # Generate PDF report
            pdf_path = pdf_gen.generate_violation_report(
                car_id=car_id,
                violation_data=violation_data,
                plate_images=plate_images,
                ai_analysis=ai_result.get("analysis", "Analysis not available")
            )
            
            violations.append({
                "car_id": car_id,
                "timestamp": violation_data["timestamp"],
                "confidence": violation_data["confidence"],
                "plate_count": len(plate_images),
                "plate_number": plate_number,
                "plate_verified": plate_verified,
                "ocr_confidence": int(ocr_confidence * 100),
                "ocr_notes": ocr_notes,
                "ocr_raw": violation_data.get("ocr_raw", ""),  # Raw OCR for comparison
                "owner_name": vehicle_info['owner_name'] if vehicle_info else "Unknown",
                "owner_email": vehicle_info['email'] if vehicle_info else "N/A",
                "email_sent": email_sent,
                "pdf_filename": f"{token}_reports/{pdf_path.name}",
                "ai_analysis": ai_result.get("analysis", "")
            })
        
        # Generate executive summary
        video_metadata = {
            "duration": f"{results.get('frames_processed', 0)} frames",
            "frames_processed": results.get('frames_processed', 0),
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        summary_result = summary_agent.generate_summary(violations, video_metadata)
        
        # Generate recommendations
        violation_stats = {
            "total_violations": len(violations),
            "violation_rate": len(violations) / max(results.get('frames_processed', 1), 1),
            "peak_time": "N/A",
            "common_vehicle": "car",
            "avg_confidence": sum(v["confidence"] for v in violations) / max(len(violations), 1)
        }
        
        recommendations_result = recommendation_agent.get_recommendations(violation_stats)
        
        # Generate executive summary PDF
        pdf_gen.generate_summary_report(
            violations=violations,
            video_metadata=video_metadata,
            ai_summary=summary_result.get("summary", ""),
            ai_recommendations=recommendations_result.get("recommendations", "")
        )
        
        # Prepare stats
        stats = {
            "total_violations": len(violations),
            "unique_cars": len(violations),
            "plates_detected": len(results.get("plate_images", {})),
            "avg_confidence": int(violation_stats["avg_confidence"] * 100)
        }
        
        return render_template(
            "violation_dashboard.html",
            active_page="street_sign",
            token=token,
            violations=violations,
            stats=stats,
            plate_images=results.get("plate_images", {}),
            ai_summary=summary_result.get("summary", "").replace("\n", "<br>"),
            ai_recommendations=recommendations_result.get("recommendations", "").replace("\n", "<br>"),
            ocr_debug=ocr_debug_list  # Add OCR debug info
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Dashboard error: {error_details}")
        return f"Dashboard error: {str(e)}<br><pre>{error_details}</pre>", 500


@app.route("/download-all-reports/<token>")
def download_all_reports(token):
    """Download all reports as a ZIP file"""
    import zipfile
    import io
    
    reports_dir = OUTPUT_DIR / f"{token}_reports"
    if not reports_dir.exists():
        return "Reports not found", 404
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for pdf_file in reports_dir.glob("*.pdf"):
            zip_file.write(pdf_file, pdf_file.name)
    
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'violation_reports_{token}.zip'
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

