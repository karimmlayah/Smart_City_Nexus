"""Chemins et constantes globaux."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
# Racine du dépôt Django parent (smartcity/) — pour best_fight_classifier.pt
SMARTCITY_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_fight_classifier_path() -> Path:
    """YOLO classify combat (pré-entraîné), sans entraînement local."""
    env = (os.environ.get("SMARTCROWD_FIGHT_CLASSIFIER") or "").strip()
    if env:
        return Path(env)
    return SMARTCITY_REPO_ROOT / "best_fight_classifier.pt"

# COCO (Ultralytics) — indices utiles
CLASS_PERSON = 0
# Sacs / valises (approximation objets « abandonnables »)
BAG_CLASSES = {24, 26, 28}  # backpack, handbag, suitcase

# Seuils comportement
PANIC_VELOCITY_PX_PER_S = 180.0  # mouvement brusque
ABANDON_DISTANCE_PX = 120.0
ABANDON_FRAMES = 45
# Lissage P(fight) sur les dernières frames (YOLO classify frame par frame)
FIGHT_PROB_ROLLING_MAX = 5
