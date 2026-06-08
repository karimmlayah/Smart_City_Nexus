import os
import sys
from pathlib import Path

if sys.platform.startswith("win"):
    os.environ.setdefault("PYTHONUTF8", "1")

BASE_DIR = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv, dotenv_values

    # Local dev: .env overrides empty shell vars. Vercel uses platform env vars only.
    if os.environ.get("VERCEL") != "1":
        load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    load_dotenv = None
    dotenv_values = None

_MODEL_RD_ENV_PATH = BASE_DIR / "model_road_damage" / ".env"
_MODEL_RD_ENV: dict[str, str] = {}
if dotenv_values and _MODEL_RD_ENV_PATH.is_file():
    _vals = dotenv_values(_MODEL_RD_ENV_PATH)
    _MODEL_RD_ENV = {str(k): str(v) for k, v in _vals.items() if k and v is not None}

SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "django-insecure-smartcity-dev-change-in-production",
)

VERCEL = os.environ.get("VERCEL") == "1"
VERCEL_ENV = (os.environ.get("VERCEL_ENV") or "").strip()
IS_VERCEL = VERCEL
ML_FEATURES_ENABLED = os.environ.get("ML_FEATURES_ENABLED", "0" if VERCEL else "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

DEBUG = os.environ.get("DEBUG", "False" if VERCEL else "True").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

_allowed_hosts_env = (os.environ.get("ALLOWED_HOSTS") or "").strip()
if _allowed_hosts_env:
    ALLOWED_HOSTS = [h.strip() for h in _allowed_hosts_env.split(",") if h.strip()]
else:
    ALLOWED_HOSTS = ["localhost", "127.0.0.1", ".vercel.app"]
    if DEBUG and not VERCEL:
        ALLOWED_HOSTS.extend(["0.0.0.0"])

CSRF_TRUSTED_ORIGINS: list[str] = []
_csrf_env = (os.environ.get("CSRF_TRUSTED_ORIGINS") or "").strip()
if _csrf_env:
    CSRF_TRUSTED_ORIGINS.extend(h.strip() for h in _csrf_env.split(",") if h.strip())
_vercel_url = (os.environ.get("VERCEL_URL") or "").strip()
if _vercel_url:
    CSRF_TRUSTED_ORIGINS.append(f"https://{_vercel_url}")
_vercel_prod = (os.environ.get("VERCEL_PROJECT_PRODUCTION_URL") or "").strip()
if _vercel_prod:
    CSRF_TRUSTED_ORIGINS.append(f"https://{_vercel_prod}")
CSRF_TRUSTED_ORIGINS = list(dict.fromkeys(CSRF_TRUSTED_ORIGINS))

if VERCEL:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "monitoring",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# WhiteNoise must sit directly after SecurityMiddleware (production + Vercel).
_sec_idx = MIDDLEWARE.index("django.middleware.security.SecurityMiddleware")
if VERCEL or not DEBUG:
    MIDDLEWARE.insert(_sec_idx + 1, "whitenoise.middleware.WhiteNoiseMiddleware")
# CORS before Security is fine; keep it first in the stack.
MIDDLEWARE.insert(0, "corsheaders.middleware.CorsMiddleware")

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "monitoring.context_processors.platform_theme",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip()
if DATABASE_URL:
    import dj_database_url

    DATABASES = {
        "default": dj_database_url.config(
            default=DATABASE_URL,
            conn_max_age=600,
            conn_health_checks=True,
            ssl_require=not DEBUG,
        )
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "fr-fr"
TIME_ZONE = "Europe/Paris"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

_static_dir = BASE_DIR / "static"
STATICFILES_DIRS = [_static_dir] if _static_dir.is_dir() else []

_static_manifest = STATIC_ROOT / "staticfiles.json"
_has_static_manifest = _static_manifest.is_file()

_USE_WHITENOISE_STORAGE = VERCEL or not DEBUG
if _USE_WHITENOISE_STORAGE:
    if _has_static_manifest:
        _staticfiles_backend = "whitenoise.storage.CompressedManifestStaticFilesStorage"
    elif VERCEL:
        # Fallback when collectstatic has not run yet (avoids 500 on {% static %}).
        _staticfiles_backend = "whitenoise.storage.CompressedStaticFilesStorage"
    else:
        _staticfiles_backend = "whitenoise.storage.CompressedManifestStaticFilesStorage"
    STORAGES = {
        "default": {
            "BACKEND": os.environ.get(
                "DEFAULT_FILE_STORAGE",
                "django.core.files.storage.FileSystemStorage",
            ),
        },
        "staticfiles": {
            "BACKEND": _staticfiles_backend,
        },
    }
    WHITENOISE_MANIFEST_STRICT = False
    WHITENOISE_MAX_AGE = 31536000
    if VERCEL and not _has_static_manifest and _static_dir.is_dir():
        WHITENOISE_USE_FINDERS = True
    STATIC_ROOT.mkdir(parents=True, exist_ok=True)
else:
    STORAGES = {
        "default": {
            "BACKEND": os.environ.get(
                "DEFAULT_FILE_STORAGE",
                "django.core.files.storage.FileSystemStorage",
            ),
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
        },
    }

# Slash initial : évite les URLs résolues sous /reclamations/.../ au lieu de la racine site.
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Optional external media for production (Vercel has no persistent disk).
# Example later: django-cloudinary-storage + CLOUDINARY_URL in env.
# File uploads use STORAGES["default"] (see above).

if VERCEL:
    FILE_UPLOAD_TEMP_DIR = "/tmp/medinamind_uploads"
    os.makedirs(FILE_UPLOAD_TEMP_DIR, exist_ok=True)
else:
    FILE_UPLOAD_TEMP_DIR = BASE_DIR / "media" / "tmp_uploads"
    FILE_UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Traffic light violation pipeline (hazemproj / Smart city — YOLO, OCR, SQLite vehicles.db)
SMART_CITY_VIOLATION_ROOT = BASE_DIR / "hazemproj" / "Smart city"
_SC_VIOLATION_ENV = SMART_CITY_VIOLATION_ROOT / ".env"
if _SC_VIOLATION_ENV.is_file():
    try:
        from dotenv import load_dotenv as _load_dotenv_sc

        _load_dotenv_sc(_SC_VIOLATION_ENV, override=False)
    except ImportError:
        pass


def _resolve_fire_onnx_model_path() -> Path:
    """Chemin du modèle feu/fumée : env FIRE_ONNX_MODEL, sinon premier fichier .onnx trouvé parmi les candidats."""
    env = (os.environ.get("FIRE_ONNX_MODEL") or "").strip()
    if env:
        return Path(env).expanduser()
    candidates = [
        BASE_DIR / "models" / "best.onnx",
        BASE_DIR / "assets" / "models" / "best.onnx",
        BASE_DIR / "smart_crowd_safety_ai" / "assets" / "models" / "best.onnx",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


# Caméra ESP32 — feu / fumée (ONNX). Priorité : models/best.onnx, puis assets/models/, ou FIRE_ONNX_MODEL dans .env.
FIRE_ONNX_MODEL_PATH = str(_resolve_fire_onnx_model_path())

# Mobile / Expo (multipart uploads for citizen reports)
DATA_UPLOAD_MAX_MEMORY_SIZE = 104_857_600
FILE_UPLOAD_MAX_MEMORY_SIZE = 104_857_600

CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_HEADERS = [
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "dnt",
    "origin",
    "user-agent",
    "x-csrftoken",
    "x-requested-with",
]

REST_FRAMEWORK = {
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.MultiPartParser",
        "rest_framework.parsers.FormParser",
    ],
}

# Poids entraîné du projet (détection sur chaque frame du flux MJPEG).
YOLO_MODEL_PATH = str(BASE_DIR / "best.pt")

# Classificateur combat / non-combat (YOLO classify, page dédiée + URL YouTube).
FIGHT_CLASSIFIER_PATH = str(BASE_DIR / "best_fight_classifier.pt")
FIGHT_SAMPLE_FRAMES = 16
# Points sur la timeline « temps réel » (plus = plus fluide, analyse plus longue)
FIGHT_TIMELINE_FRAMES = 40
FIGHT_SAMPLE_SKIP_EVERY = 10
FIGHT_YT_TIMEOUT_SECONDS = 120
FIGHT_YT_DOWNLOAD_TIMEOUT = 600
FIGHT_YT_MAX_FILESIZE = "120m"
FIGHT_YT_EXTRACTOR_ARGS = "youtube:player_client=android,web"
# Optionnel: "node" si Node.js est installe (sinon laissez vide)
FIGHT_YT_JS_RUNTIMES = ""
# Upload local pour la page combat (Mo)
FIGHT_UPLOAD_MAX_MB = 200
# Liste de codes 4 lettres OpenCV pour l’export vidéo annotée (None = défaut : mp4v sous Windows, H.264 puis mp4v sinon)
FIGHT_ANNOTATED_VIDEO_CODECS: tuple[str, ...] | list[str] | None = None
# Analyse « caméra » : toute la vidéo, image par image (stride 1 = chaque frame)
FIGHT_FULL_SCAN_STRIDE = 1
FIGHT_FULL_SCAN_MAX_FRAMES = 20000
# Alerte site : combat « confirmé » (probabilité du modèle, ex. 99 %)
FIGHT_ALERT_MIN_CONFIDENCE = 0.99
FIGHT_ALERT_MIN_TOP1_CONF = 0.99
FIGHT_ALERT_COOLDOWN_SECONDS = 45
# Points max pour le JSON timeline (HUD) — sous-échantillonnage si la vidéo est longue
FIGHT_TIMELINE_MAX_POINTS = 800
# Dossier model_fight/ : .h5 (Keras / TensorFlow), .pth (PyTorch nn.Module) — voir fight_backends.py
FIGHT_CUSTOM_KERAS_NORM_MODE = ""  # ex. "imagenet" pour normalisation MobileNet
# .h5 dont l’entrée est (batch, T, dim) features : pré-encode les images avec ImageNet —
# valeur "" ou "auto" choisit le backbone canonique dont la dimension de sortie = dim attendue ;
# sinon : resnet50 | mobilenet_v2 | inception_resnet_v2 | densenet121
FIGHT_CUSTOM_KERAS_SEQ_EMBEDDING = ""
# Si le graphe a des dims None pour T ou les features : forcer ces entiers (>0), sinon ignorer (0).
FIGHT_CUSTOM_KERAS_SEQ_LEN = 0
FIGHT_CUSTOM_KERAS_SEQ_FEAT_DIM = 0
# Tentatives si le téléchargement des .h5 ImageNet (backbone fusion) est tronqué.
FIGHT_KERAS_IMAGENET_WEIGHT_RETRIES = 5
# YOLO COCO générique (detect) pour repérer les personnes lorsque combat + arme sont vrais ensemble
FIGHT_PERSON_MODEL_PATH = "yolov8n.pt"
FIGHT_PERSON_YOLO_CONF = 0.35
FIGHT_PERSON_YOLO_IMGSZ = 640
FIGHT_PERSON_MAX_PER_FRAME = 2
# Double détection fusion : même image combat probable + au moins une arme assez confiante
FIGHT_DUAL_EVIDENCE_FIGHT_P = 0.52
FIGHT_DUAL_EVIDENCE_WEAPON_CONF = 0.42
FIGHT_DUAL_EVIDENCE_MAX_SNAPSHOTS = 16
FIGHT_DUAL_CROP_PAD = 0.14
FIGHT_DUAL_LIVE_CROP_MAX_W = 288
# Fusion Hub : scan biométrique — extraction faciale (DeepFace + repli Haar)
FUSION_FACE_SCAN_MAX_PEOPLE = 6
FUSION_FACE_SCAN_PAD = 0.18
FUSION_FACE_DETECT_MAX = 8
FACE_EXTRACT_DETECTOR_BACKEND = "opencv"
# Sortie carte biométrique (carré) sauvegardée + embedding
FACE_BIOMETRIC_CROP_SIZE = 256
FACE_BIOMETRIC_BBOX_PADDING = 0.20
FACE_BIOMETRIC_LIVE_PREVIEW = 140
FACE_BIOMETRIC_DEBUG_LIVE_CROPS = False

# Base visages Django (fusion : comparaison sur recadrages visage)
FACE_RECOGNITION_ENABLED = True
FACE_EMBED_MODEL_NAME = "Facenet"
FACE_EMBED_DETECTOR_BACKEND = "opencv"
FACE_EMBED_FALLBACK_SKIP = True
FACE_MATCH_MAX_RESULTS = 5
# Seuil cosinus DeepFace (plus haut = plus permissif). 0.45 = tests / démo ; prod souvent 0.35–0.4.
FACE_MATCH_MAX_COSINE_DISTANCE = 0.45
# Logs INFO détaillés lors du matching (galerie, distances, seuil). Désactiver en prod si trop verbeux.
FACE_MATCH_DEBUG_LOG = True

# Fusion Hub — analyse vidéo progressive (~1 cadre / seconde pendant lecture, sans blocage POST)
FUSION_PROGRESSIVE_VIDEO_ANALYSIS = True
FUSION_VIDEO_ANALYSIS_INTERVAL_SECONDS = 1.0
FUSION_PROGRESSIVE_UPLOAD_CACHE_SECONDS = int(os.environ.get("FUSION_PROGRESSIVE_UPLOAD_CACHE_SECONDS", "7200"))
FUSION_PROGRESSIVE_FRAME_CACHE_SECONDS = int(os.environ.get("FUSION_PROGRESSIVE_FRAME_CACHE_SECONDS", "86400"))
# Caméra live Fusion Hub : même pipeline « image fusion » que upload / frame vidéo quand combat ou arme dépasse le seuil
FUSION_LIVE_IMAGE_FUSION_FIGHT_P = float(os.environ.get("FUSION_LIVE_IMAGE_FUSION_FIGHT_P", "0.38"))
FUSION_LIVE_IMAGE_FUSION_WEAPON_MAX = float(os.environ.get("FUSION_LIVE_IMAGE_FUSION_WEAPON_MAX", "0.30"))
# Cooldown entre deux passes complètes (0 = chaque frame déclenchée ; ex. 0.6 pour limiter la charge CPU/GPU)
FUSION_LIVE_IMAGE_FUSION_COOLDOWN_SEC = float(os.environ.get("FUSION_LIVE_IMAGE_FUSION_COOLDOWN_SEC", "0"))
# Caméra live : si le modèle met « fight » en top-1 mais P(fight) reste bas (gros plan visage, faux positif),
# on affiche plutôt « nonfight » pour le libellé temps réel (la probabilité reste visible).
FUSION_LIVE_FIGHT_DISPLAY_MIN_P = float(os.environ.get("FUSION_LIVE_FIGHT_DISPLAY_MIN_P", "0.52"))
FIGHT_CUSTOM_TORCH_INPUT_HW: tuple[int, ...] = ()  # ex. (224, 224) pour .pth inconnu
FIGHT_CUSTOM_TORCH_NORMALIZE_TO_N = ""  # "" ou "neg1" (−1 … 1 avant forward)

# Détecteur d’armes (YOLO detect) — fusionné avec l’analyse combat sur la même page
WEAPON_DETECTOR_PATH = str(BASE_DIR / "best_weapon_detector.pt")
# Détecteur dédié à la page "weapon test" (YOLOv8 direct).
# Utilisez un nom de poids Ultralytics (ex: yolov8n.pt) ou un chemin local.
WEAPON_TEST_MODEL_PATH = "yolov8n.pt"
WEAPON_GUN_TEST_MODEL_PATH = str(BASE_DIR / "model_gun.pt")
WEAPON_TEST_TARGET_LABELS = ["knife"]
WEAPON_GUN_TEST_TARGET_LABELS = ["gun", "pistol", "rifle", "weapon"]
# Réglages rapides dédiés à la page gun (réponse web plus fluide).
WEAPON_GUN_SCAN_STRIDE = 4
WEAPON_GUN_MAX_FRAMES = 900
WEAPON_GUN_TIMELINE_MAX_POINTS = 240
WEAPON_GUN_ALERT_MIN_CONF = 0.55
WEAPON_GUN_DRAW_MIN_CONF = 0.40
WEAPON_GUN_TRANSCODE_OUTPUT = True
WEAPON_YOLO_CONF = 0.35
WEAPON_YOLO_IOU = 0.45
WEAPON_YOLO_IMGSZ = 640
# Seuil alerte knife (verdict / frame avec knife)
WEAPON_ALERT_MIN_CONF = 0.82
# Seuil plus bas juste pour dessiner le cercle / crops
WEAPON_DRAW_MIN_CONF = 0.5
WEAPON_ALERT_COOLDOWN_SECONDS = 45
WEAPON_TEST_SCAN_STRIDE = 2
WEAPON_TEST_MAX_FRAMES = 20000
WEAPON_TEST_TIMELINE_MAX_POINTS = 800
# Extraits image (zoom knife) affiches a droite de la video
WEAPON_TEST_MAX_CROP_SNAPSHOTS = 24
WEAPON_TEST_CROP_PADDING = 0.18
WEAPON_IMAGE_UPLOAD_MAX_MB = 25

# Road damage detect (laboratoire / segmentation — fixe)
ROAD_DAMAGE_MODEL_PATH = str(BASE_DIR / "best (2).pt")

# Classification (page détail réclamation + inférence détection) — ?model=<clé>
_ROAD_MD = BASE_DIR / "model_road_damage"
ROAD_VISION_CLASSIFICATION_MODELS: dict[str, str] = {
    "rtdetr": str(_ROAD_MD / "rtdetr.pt"),
    "yolov8s_best": str(_ROAD_MD / "yolov8s_best.pt"),
    "fasterrcnn_weights": str(_ROAD_MD / "fasterrcnn_weights (1).pth"),
    "yolov8n_100ep": str(_ROAD_MD / "yolov8n_100ep.pt"),
}
ROAD_VISION_CLASSIFICATION_LABELS: dict[str, str] = {
    "rtdetr": "RT-DETR — model_road_damage/rtdetr.pt",
    "yolov8s_best": "YOLOv8s best — model_road_damage/yolov8s_best.pt",
    "fasterrcnn_weights": "Faster R-CNN — model_road_damage/fasterrcnn_weights (1).pth",
    "yolov8n_100ep": "YOLOv8n 100 ep — model_road_damage/yolov8n_100ep.pt",
}
ROAD_VISION_CLASSIFICATION_UI_ORDER: tuple[str, ...] = (
    "rtdetr",
    "yolov8s_best",
    "fasterrcnn_weights",
    "yolov8n_100ep",
)
ROAD_VISION_CLASSIFICATION_FALLBACK_ORDER: tuple[str, ...] = ROAD_VISION_CLASSIFICATION_UI_ORDER

# Segmentation : affichage fixe (aligné sur ROAD_DAMAGE_MODEL_PATH) — non sélectionnable sur la page IA
ROAD_VISION_SEGMENTATION_INFO: dict[str, str] = {
    "title": "Modèle segmentation",
    "label": "best (2).pt",
    "path": str(BASE_DIR / "best (2).pt"),
    "hint": "Fixe — non modifiable sur cette page (laboratoire route / segmentation).",
}

# Détection déchets rue — Waste Street Detection (`/waste/street-detection/`, `/api/waste/detect/`)
# WASTE_MODEL_PATH : chemin vers fichier Ultralytics .pt (vide = erreur explicite côté API).
_WASTE_PT = BASE_DIR / "models" / "best (1) 1.pt"
WASTE_MODEL_PATH = str(_WASTE_PT) if _WASTE_PT.is_file() else ""
WASTE_YOLO_CONF = float(os.environ.get("WASTE_YOLO_CONF", "0.25"))
WASTE_YOLO_IOU = 0.45
WASTE_YOLO_IMGSZ = 640

# Twilio — SMS municipalité (voir aussi monitoring/services/waste_sms.py)
# IMPORTANT: values are sourced ONLY from model_road_damage/.env (as requested).
TWILIO_ACCOUNT_SID = (_MODEL_RD_ENV.get("TWILIO_ACCOUNT_SID", "") or "").strip()
TWILIO_AUTH_TOKEN = (_MODEL_RD_ENV.get("TWILIO_AUTH_TOKEN", "") or "").strip()
TWILIO_PHONE_NUMBER = (_MODEL_RD_ENV.get("TWILIO_PHONE_NUMBER", "") or "").strip()
MUNICIPALITY_PHONE_NUMBER = (_MODEL_RD_ENV.get("MUNICIPALITY_PHONE_NUMBER", "") or "").strip()
# Backward-compatible aliases inside same env file only.
TWILIO_FROM_NUMBER = TWILIO_PHONE_NUMBER or (_MODEL_RD_ENV.get("TWILIO_FROM", "") or "").strip()
WASTE_MUNICIPALITY_PHONE = MUNICIPALITY_PHONE_NUMBER or (_MODEL_RD_ENV.get("TWILIO_TO", "") or "").strip()
TWILIO_PHONE_NUMBER = TWILIO_PHONE_NUMBER or TWILIO_FROM_NUMBER
MUNICIPALITY_PHONE_NUMBER = MUNICIPALITY_PHONE_NUMBER or WASTE_MUNICIPALITY_PHONE

# SMTP email settings (waste alerts + other notifications)
# Uses .env keys shown in project: EMAIL_HOST / EMAIL_PORT / EMAIL_USER / EMAIL_PASSWORD
EMAIL_HOST = (os.environ.get("EMAIL_HOST", "") or "").strip()
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", "587") or "587")
EMAIL_HOST_USER = (
    os.environ.get("EMAIL_HOST_USER", "")
    or os.environ.get("EMAIL_USER", "")
    or ""
).strip()
EMAIL_HOST_PASSWORD = (
    os.environ.get("EMAIL_HOST_PASSWORD", "")
    or os.environ.get("EMAIL_PASSWORD", "")
    or ""
).strip()
EMAIL_USE_TLS = str(os.environ.get("EMAIL_USE_TLS", "1")).strip().lower() in {"1", "true", "yes", "on"}
EMAIL_USE_SSL = str(os.environ.get("EMAIL_USE_SSL", "0")).strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_FROM_EMAIL = (os.environ.get("DEFAULT_FROM_EMAIL", "") or EMAIL_HOST_USER or "noreply@smartcity.local").strip()

# OpenAI — assistant chat (dashboard) + autres modules
OPENAI_API_KEY = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
OPENAI_CHAT_MODEL = (os.environ.get("OPENAI_CHAT_MODEL", "") or "gpt-4o-mini").strip()
# Groq — optionnel (ex. rapports road damage XAI)
GROQ_API_KEY = (os.environ.get("GROQ_API_KEY", "") or "").strip()
GROQ_MODEL = (os.environ.get("GROQ_MODEL", "") or "llama-3.3-70b-versatile").strip()

# Email municipalité (optionnel — préremplissage UI)
WASTE_DEFAULT_MUNICIPALITY_EMAIL = (
    os.environ.get("WASTE_DEFAULT_MUNICIPALITY_EMAIL", "")
    or "hazem.jbali@esprit.tn"
)
# Surcharge possible des mots-clés zones sensibles (waste_severity.py)
WASTE_SENSITIVE_AREA_KEYWORDS: tuple[str, ...] = ()

# Geocoding (Nominatim / OpenStreetMap — no API key required)
GEOCODING_NOMINATIM_URL = os.environ.get("GEOCODING_NOMINATIM_URL", "https://nominatim.openstreetmap.org")
GEOCODING_USER_AGENT = os.environ.get("GEOCODING_USER_AGENT", "MedinaMind/1.0 (smartcity)")
# Chemin absolu ou relatif à BASE_DIR ; résolution avec repli à l'exécution (uav_cnn_inference).
_uav_env = (os.environ.get("UAV_MODEL_PATH", "") or "").strip()
if _uav_env:
    _uav_cfg = Path(_uav_env).expanduser()
    if not _uav_cfg.is_absolute():
        _uav_cfg = BASE_DIR / _uav_cfg
    UAV_MODEL_PATH = str(_uav_cfg)
else:
    UAV_MODEL_PATH = str(BASE_DIR / "models" / "best_CNN.keras")
# Réservé si vous réintroduisez un vrai Grad-CAM conv ; l’UI utilise un proxy gradient-entrée lissé.
UAV_GRADCAM_LAYER = os.environ.get("UAV_GRADCAM_LAYER", "conv2d_3")
# Libellés FR pour les 3 sorties du modèle [indice 0, 1, 2] — alignez avec votre entraînement.
UAV_CLASS_LABELS: tuple[str, str, str] = ("Intact", "Damaged", "Collapsed")
# Badge affiché dans l’en-tête (précision validation — valeur indicative).
UAV_DISPLAY_MODEL_ACCURACY = os.environ.get("UAV_DISPLAY_MODEL_ACCURACY", "96.06%")

# Traffic Nexus (YOLO + tdss) — modèles dans models_traffic/, traffic_nexus/traffic_nexus/ ou racine projet
_TRAFFIC_NEXUS_DIR = BASE_DIR / "traffic_nexus" / "traffic_nexus"
_TRAFFIC_MODELS_DIR = BASE_DIR / "models_traffic"


def _traffic_model_path(filename: str, fallback: Path | None = None) -> str:
    p_models = _TRAFFIC_MODELS_DIR / filename
    if p_models.is_file():
        return str(p_models)
    p = _TRAFFIC_NEXUS_DIR / filename
    if p.is_file():
        return str(p)
    if fallback and fallback.is_file():
        return str(fallback)
    return ""


TRAFFIC_NEXUS_MODEL_PATHS: dict[str, str] = {
    "yolov8n": _traffic_model_path("yolov8n.pt", BASE_DIR / "yolov8n.pt"),
    "yolov8m": _traffic_model_path("yolov8m.pt"),
    "yolo_congestion": _traffic_model_path("yolo_congestion.pt"),
    # Ambulance / custom — même ordre que Streamlit : models_traffic/, puis dossier traffic_nexus/, puis racine projet (cwd équivalent)
    "best": _traffic_model_path("best.pt", BASE_DIR / "best.pt"),
}
TRAFFIC_NEXUS_FRAME_MAX_SIDE = int(os.environ.get("TRAFFIC_NEXUS_FRAME_MAX_SIDE", "1280"))
# Libellés YOLO (sous-chaîne) pour filtrer les détections ambulance depuis best.pt (mode véhicule / dual uniquement)
TRAFFIC_AMBULANCE_LABEL_KEYWORDS: tuple[str, ...] = tuple(
    x.strip().lower()
    for x in os.environ.get("TRAFFIC_AMBULANCE_LABEL_KEYWORDS", "ambulance,emergency,ems").split(",")
    if x.strip()
)
# Détection ambulance (boîtes plus fines / moins de doublons avec YOLO générique)
TRAFFIC_AMBULANCE_MIN_CONF = float(os.environ.get("TRAFFIC_AMBULANCE_MIN_CONF", "0.85"))
TRAFFIC_AMBULANCE_MIN_BOX_AREA = int(os.environ.get("TRAFFIC_AMBULANCE_MIN_BOX_AREA", "110"))
TRAFFIC_AMBULANCE_NMS_IOU = float(os.environ.get("TRAFFIC_AMBULANCE_NMS_IOU", "0.40"))
TRAFFIC_AMBULANCE_FALLBACK_MIN_CONF = float(os.environ.get("TRAFFIC_AMBULANCE_FALLBACK_MIN_CONF", "0.20"))
TRAFFIC_AMBULANCE_FALLBACK_MAX_DET = int(os.environ.get("TRAFFIC_AMBULANCE_FALLBACK_MAX_DET", "6"))
TRAFFIC_AMBULANCE_OVERLAP_IOU = float(os.environ.get("TRAFFIC_AMBULANCE_OVERLAP_IOU", "0.48"))
# Classes YOLO toujours ignorées (Traffic Nexus — ex. COCO « person », « train »)
TRAFFIC_YOLO_EXCLUDED_CLASSES: tuple[str, ...] = tuple(
    x.strip().lower()
    for x in os.environ.get("TRAFFIC_YOLO_EXCLUDED_CLASSES", "person,train").split(",")
    if x.strip()
)
# Temps de feu de référence (DecisionEngine) — rapport conservé quand on met à l’échelle (ex. 45 s vert / 25 s rouge)
TRAFFIC_SIGNAL_GREEN_REF_S = int(os.environ.get("TRAFFIC_SIGNAL_GREEN_REF_S", "45"))
TRAFFIC_SIGNAL_RED_REF_S = int(os.environ.get("TRAFFIC_SIGNAL_RED_REF_S", "25"))

# Page autonome Feux IA -> ESP32 (upload image/vidéo + traffic_sign_detector.pt)
_TRAFFIC_SIGNAL_MODEL_CANDIDATES = (
    BASE_DIR / "hazemproj" / "traffic_sign_detector.pt",
    BASE_DIR / "hazemproj" / "Smart city" / "traffic_sign_detector.pt",
)
_TRAFFIC_SIGNAL_MODEL_DEFAULT = next(
    (p for p in _TRAFFIC_SIGNAL_MODEL_CANDIDATES if p.is_file()),
    None,
)
TRAFFIC_SIGNAL_MODEL_PATH = (
    os.environ.get("TRAFFIC_SIGNAL_MODEL_PATH", "")
    or (str(_TRAFFIC_SIGNAL_MODEL_DEFAULT) if _TRAFFIC_SIGNAL_MODEL_DEFAULT else "")
)
TRAFFIC_SIGNAL_CONF = float(os.environ.get("TRAFFIC_SIGNAL_CONF", "0.25"))
TRAFFIC_SIGNAL_IOU = float(os.environ.get("TRAFFIC_SIGNAL_IOU", "0.45"))
TRAFFIC_SIGNAL_IMGSZ = int(os.environ.get("TRAFFIC_SIGNAL_IMGSZ", "640"))
TRAFFIC_SIGNAL_VIDEO_FRAME_STEP = int(os.environ.get("TRAFFIC_SIGNAL_VIDEO_FRAME_STEP", "3"))
TRAFFIC_SIGNAL_VIDEO_MAX_FRAMES = int(os.environ.get("TRAFFIC_SIGNAL_VIDEO_MAX_FRAMES", "900"))
TRAFFIC_SIGNAL_RED_LABELS: tuple[str, ...] = tuple(
    x.strip().lower()
    for x in os.environ.get("TRAFFIC_SIGNAL_RED_LABELS", "red light,red,rouge").split(",")
    if x.strip()
)
TRAFFIC_SIGNAL_GREEN_LABELS: tuple[str, ...] = tuple(
    x.strip().lower()
    for x in os.environ.get("TRAFFIC_SIGNAL_GREEN_LABELS", "green light,green,vert").split(",")
    if x.strip()
)
TRAFFIC_SIGNAL_ESP32_URL = os.environ.get("TRAFFIC_SIGNAL_ESP32_URL", "http://192.168.0.188/signal")

# Alias : même ensemble que la classification (inventaire, pick_model_path)
ROAD_VISION_MODELS: dict[str, str] = dict(ROAD_VISION_CLASSIFICATION_MODELS)
ROAD_VISION_MODEL_LABELS: dict[str, str] = dict(ROAD_VISION_CLASSIFICATION_LABELS)
# Défaut : VISION_MODEL_KEY ou yolov8n_100ep
ROAD_VISION_MODEL_DEFAULT_KEY = os.environ.get("VISION_MODEL_KEY", "yolov8n_100ep")
# Laissez [] pour accepter toutes les classes du modele.
ROAD_DAMAGE_TARGET_LABELS = []
ROAD_DAMAGE_YOLO_CONF = 0.25
# Seuil YOLO par défaut (page détail réclamation) — surchargé par ?conf= ou VISION_YOLO_CONF
RECLAMATION_VISION_CONF_DEFAULT = float(
    os.environ.get("VISION_YOLO_CONF", str(ROAD_DAMAGE_YOLO_CONF))
)
ROAD_DAMAGE_YOLO_IOU = 0.45
ROAD_DAMAGE_YOLO_IMGSZ = 640
ROAD_DAMAGE_ALERT_MIN_CONF = 0.5
ROAD_DAMAGE_DRAW_MIN_CONF = 0.35
ROAD_DAMAGE_SCAN_STRIDE = 2
ROAD_DAMAGE_MAX_FRAMES = 20000
ROAD_DAMAGE_TIMELINE_MAX_POINTS = 800
# XAI occlusion (road damage)
ROAD_DAMAGE_XAI_GRID_X = 10
ROAD_DAMAGE_XAI_GRID_Y = 10
ROAD_DAMAGE_XAI_OCCLUSION_RATIO = 0.18
ROAD_DAMAGE_QUALITY_MIN_BLUR = 70.0
ROAD_DAMAGE_QUALITY_MIN_BRIGHTNESS = 50.0
ROAD_DAMAGE_QUALITY_MAX_BRIGHTNESS = 220.0
ROAD_DAMAGE_ALERT_NOTIFY_MIN_SCORE = 0.75
ROAD_DAMAGE_ALERT_EMAILS = ""
ROAD_DAMAGE_TELEGRAM_WEBHOOK = ""
ROAD_DAMAGE_WHATSAPP_WEBHOOK = ""
# LLM rapport (OpenAI/Groq) — autres usages (XAI, copilot…)
OPENAI_MODEL = (os.environ.get("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
# Identifiants Sightengine (NE PAS commiter en production)
SIGHTENGINE_API_USER = "1098611735"
SIGHTENGINE_API_KEY = "HfQ2aJNEe8WAbk7o3qTLFf5rMHExmxtU"
# Vidéo Sightengine (soumission async + polling) — max 50 Mo côté API upload direct
SIGHTENGINE_VIDEO_MAX_UPLOAD_MB = 50
SIGHTENGINE_VIDEO_POLL_MAX_SEC = 120
SIGHTENGINE_VIDEO_POLL_INTERVAL_SEC = 2.0
# Plus bas = plus de détections (sacs petits / loin) mais plus de faux positifs
YOLO_CONFIDENCE = 0.22
YOLO_IOU = 0.5
# Résolution d’inférence : 640 défaut ; 800–960 aide les petits objets (plus lent)
YOLO_IMGSZ = 800
# Pièce sombre / faible lumière : renforce contraste avant YOLO (CLAHE + gamma léger).
# Désactivez (False) si la vidéo est déjà très claire (évite un rendu trop « dur »).
YOLO_LOW_LIGHT_ENHANCE = True
YOLO_LOW_LIGHT_GAMMA = 0.72
YOLO_CLAHE_CLIP_LIMIT = 2.8
YOLO_CLAHE_TILE = 8
# Seuil personnes → alerte foule (défaut modèle VideoSource)
DEFAULT_CROWD_THRESHOLD = 12
# Mouvement rapide : nb d’analyses consécutives (1 frame sur 3) où la vitesse max dépasse le seuil source
FAST_MOVEMENT_STREAK = 2
ALERT_COOLDOWN_SECONDS = 45
# Heuristique sac ↔ personne (piste vol) : distance max = facteur × taille du sac ; stabilité en frames
BAG_PERSON_MAX_DIST_FACTOR = 3.2
BAG_OWNER_STABLE_FRAMES = 5

# Page web Smart Crowd Safety AI (upload + pipeline)
SMARTCROWD_UPLOAD_MAX_MB = 150
SMARTCROWD_WEB_MAX_FRAMES_DEFAULT = 400
SMARTCROWD_WEB_MAX_FRAMES_CAP = 2500

# Logs terminal (runserver) : esp-analyse / bouton capture manuelle
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "monitoring.views": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "medinamind.startup": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
}

import logging as _logging

_startup_log = _logging.getLogger("medinamind.startup")
_startup_log.info(
    "Running on Vercel: %s | DEBUG: %s | DATABASE_URL configured: %s | OPENAI configured: %s | static manifest: %s",
    IS_VERCEL,
    DEBUG,
    bool(DATABASE_URL),
    bool((os.environ.get("OPENAI_API_KEY") or "").strip()),
    _static_manifest.is_file(),
)
