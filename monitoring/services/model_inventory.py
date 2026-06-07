"""
Inventaire des poids du projet : .pt Ultralytics, dossier model_fight/ (.h5, .pth), hub yolov8.

Utilisé par les dropdowns combat / armes (page fusion et analyse combat).
"""

from __future__ import annotations

import re
from pathlib import Path

from django.conf import settings

# Menu combat fixe (dropdown Violence Model) — ordre et libellés détaillés
_FIGHT_MENU_CHOICES: tuple[tuple[str, str], ...] = (
    ("model_fight/fd_v2.h5", "model_fight/fd_v2.h5 • Keras TensorFlow (.h5)"),
    (
        "best_fight_classifier.pt",
        "best_fight_classifier.pt • (réglages) classify combat",
    ),
)
FIGHT_MENU_VALUES = frozenset(v for v, _ in _FIGHT_MENU_CHOICES)
DEFAULT_FIGHT_WEIGHTS = "model_fight/fd_v2.h5"

# Autres entrées model_fight (hors menu dropdown combat)
_MODEL_FIGHT_ALWAYS_LIST: tuple[tuple[str, str], ...] = (
    ("model_fight/fd_v2.h5", "model_fight/fd_v2.h5 • Keras TensorFlow (.h5)"),
)

SKIP_DIR_PARTS = frozenset({
    ".venv",
    "venv",
    ".git",
    "node_modules",
    "__pycache__",
    "site-packages",
    ".mypy_cache",
    ".pytest_cache",
})

# Réglages Django connus qui pointent vers des poids (.pt ou nom Ultralytics).
_SETTINGS_PATH_KEYS: tuple[tuple[str, str], ...] = (
    ("FIGHT_CLASSIFIER_PATH", "(réglages) classify combat"),
    ("WEAPON_DETECTOR_PATH", "(réglages) détection armes"),
    ("WEAPON_TEST_MODEL_PATH", "(réglages) weapon test"),
    ("YOLO_MODEL_PATH", "(réglages) MJPEG foule"),
    ("WEAPON_GUN_TEST_MODEL_PATH", "(réglages) gun test"),
    ("ROAD_DAMAGE_MODEL_PATH", "(réglages) route"),
)

# Noms de fichier / chemins interprétés comme classificateur « combat » (YOLO classify, etc.)
_CLASSIFY_NAME_HINT = re.compile(
    r"fight|classif|combat|cnn_lstm|cnn-lstm|^cf_|clf",
    re.I,
)
# Poids détection « armes / guns » (YOLO detect, modèles armes dédiés, YOLO génériques)
_WEAPON_DETECT_HINT = re.compile(
    r"weapon|gun|knife|pistol|rifle|model_gun|detector|yolov8|yolo11|yolo26|yolo9|yolo10|rtdetr",
    re.I,
)
# Modèles route / foule MJPEG : hors menu combat et hors menu armes
_ROAD_CROWD_EXCLUDE = re.compile(
    r"model_road_damage|/road_damage|road_damage\.pt|best \(2\)\.pt|"
    r"reclamation|mjpeg|foule|best\.pt$",
    re.I,
)


def _collect_settings_paths() -> list[tuple[Path, str]]:
    """Chemins configurés avec petit badge libellé."""
    out: list[tuple[Path, str]] = []
    base = Path(getattr(settings, "BASE_DIR", ".")).resolve()

    for key, tag in _SETTINGS_PATH_KEYS:
        raw = getattr(settings, key, None)
        if not raw:
            continue
        s = str(raw).strip()
        if not s or not (s.endswith(".pt") or "yolov" in s.lower()):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (base / s).resolve()
        else:
            p = p.resolve()
        out.append((p, tag))

    rvm = getattr(settings, "ROAD_VISION_MODELS", None)
    if isinstance(rvm, dict):
        rl = getattr(settings, "ROAD_VISION_MODEL_LABELS", {})
        for k, v in rvm.items():
            pv = Path(str(v)).resolve()
            label = rl.get(k, k)
            out.append((pv, f"(route) {label}"))
    extra = getattr(settings, "BASE_DIR", None)
    if extra:
        cand = Path(extra) / "models" / "road_damage.pt"
        if cand.is_file():
            out.append((cand.resolve(), "(scan) models/road_damage.pt"))
    smart_fight = (
        Path(extra) / "smart_crowd_safety_ai" / "models" / "fight_cnn_lstm.pt"
    )
    if smart_fight.is_file():
        out.append((smart_fight.resolve(), "(Smart Crowd) fight_cnn_lstm.pt"))

    uniq: dict[str, tuple[Path, str]] = {}
    for pth, lbl in out:
        k = str(pth)
        if k not in uniq:
            uniq[k] = (pth, lbl)
    return list(uniq.values())


def _scan_model_fight_formats(base: Path) -> dict[str, str]:
    """.h5 et .pth sous model_fight/."""
    items: dict[str, str] = {}
    d = (base / "model_fight").resolve()
    if not d.is_dir():
        return items
    for pattern in ("*.h5", "*.pth"):
        for abs_p in d.glob(pattern):
            if abs_p.is_file():
                try:
                    rel = relative_posix_under_base(abs_p.resolve(), base)
                except ValueError:
                    continue
                items.setdefault(
                    rel,
                    f"{rel} • dossier model_fight",
                )
    return items


def _discovered_pt_files(base: Path) -> list[Path]:
    found: list[Path] = []
    base = base.resolve()
    if not base.is_dir():
        return found
    for p in base.rglob("*.pt"):
        try:
            p = p.resolve()
        except OSError:
            continue
        if not p.is_file():
            continue
        parts = set(p.parts)
        if SKIP_DIR_PARTS & parts:
            continue
        try:
            p.relative_to(base)
        except ValueError:
            continue
        found.append(p)
    return sorted(set(found))


def relative_posix_under_base(abs_path: Path, base: Path) -> str:
    try:
        return abs_path.relative_to(base.resolve()).as_posix()
    except ValueError:
        return abs_path.as_posix()


def _label_text(rel_posix: str, label: str) -> str:
    return f"{rel_posix} {label}"


def _is_under_model_fight_bundle(rel_posix: str) -> bool:
    low = rel_posix.replace("\\", "/").lower().strip()
    if not low.startswith("model_fight/"):
        return False
    return Path(low).suffix.lower() in (".h5", ".pth", ".pt")


def is_fight_only_entry(rel_posix: str, label: str = "") -> bool:
    """Uniquement poids destinés au flux « combat » (classify / fight), pas gun ni route."""
    if _is_under_model_fight_bundle(rel_posix):
        return True

    blob = _label_text(rel_posix, label)
    if _ROAD_CROWD_EXCLUDE.search(rel_posix) or _ROAD_CROWD_EXCLUDE.search(label):
        return False
    low_rel = rel_posix.lower()
    if "yolov8" in low_rel or "yolo11" in low_rel or "yolo26" in low_rel:
        if "cls" not in low_rel and "classify" not in low_rel:
            return False
    if rel_posix in ("yolov8n.pt", "yolov8s.pt"):
        return False
    if _CLASSIFY_NAME_HINT.search(rel_posix) or _CLASSIFY_NAME_HINT.search(Path(rel_posix).stem):
        return True
    low = label.lower()
    if "classify combat" in low:
        return True
    if "smart crowd" in low and "fight" in low:
        return True
    if "fight_cnn" in rel_posix.lower():
        return True
    return False


def _is_weapon_name_in_model_fight(rel_posix: str, label: str) -> bool:
    blob = (_label_text(rel_posix, label)).lower()
    return "model_fight/" in blob and _WEAPON_DETECT_HINT.search(blob)


def is_weapon_only_entry(rel_posix: str, label: str = "") -> bool:
    """Uniquement poids détection armes / guns (ou YOLO detect générique), pas classify combat ni route."""
    if _is_under_model_fight_bundle(rel_posix):
        # model_fight/ est réservé aux classifiers combat (.h5 / .pth) sauf explicite arme dans le nom
        return _is_weapon_name_in_model_fight(rel_posix, label)

    blob = _label_text(rel_posix, label)
    if _ROAD_CROWD_EXCLUDE.search(rel_posix) or _ROAD_CROWD_EXCLUDE.search(label):
        return False
    # Ne pas proposer un classificateur combat dans le menu « gun »
    if (
        (_CLASSIFY_NAME_HINT.search(rel_posix) or _CLASSIFY_NAME_HINT.search(Path(rel_posix).stem))
        and not _WEAPON_DETECT_HINT.search(rel_posix)
        and not _WEAPON_DETECT_HINT.search(label)
    ):
        return False
    if _WEAPON_DETECT_HINT.search(rel_posix) or _WEAPON_DETECT_HINT.search(label):
        return True
    if rel_posix in ("yolov8n.pt", "yolov8s.pt"):
        return True
    low = label.lower()
    if "gun test" in low or "weapon test" in low or "détection armes" in low or "weapon" in low:
        return True
    return False


def build_inventory() -> list[tuple[str, str]]:
    """
    Liste (valeur, libellé) pour les <select>.
    valeur = chemin posix relatif à BASE_DIR, ou fichier .pt sans slash (Ultralytics).
    """
    base = Path(getattr(settings, "BASE_DIR", ".")).resolve()
    items: dict[str, str] = {}

    for abs_p, badge in _collect_settings_paths():
        exists = abs_p.is_file()
        suffix = "" if exists else " — absent"
        rel = relative_posix_under_base(abs_p, base)
        if rel not in items or badge.startswith("(réglages)"):
            items[rel] = f"{rel} • {badge}{suffix}"

    for abs_p in _discovered_pt_files(base):
        rel = relative_posix_under_base(abs_p, base)
        if rel not in items:
            items[rel] = rel

    for mf_rel, mf_lbl in _MODEL_FIGHT_ALWAYS_LIST:
        if mf_rel not in items:
            exists = (base / mf_rel).is_file()
            items[mf_rel] = mf_lbl + ("" if exists else " — absent")

    for mf_rel, mf_lbl in _scan_model_fight_formats(base).items():
        if mf_rel not in items:
            items[mf_rel] = mf_lbl

    extra_ultra = [
        ("yolov8n.pt", "yolov8n.pt • Ultralytics (téléchargement auto)"),
        ("yolov8s.pt", "yolov8s.pt • Ultralytics (téléchargement auto)"),
    ]
    for val, lbl in extra_ultra:
        if val not in items:
            items[val] = lbl

    return sorted(items.items(), key=lambda x: x[0].lower())


def fight_model_choices() -> list[tuple[str, str]]:
    """Dropdown Violence Model : .h5 par défaut, puis best_fight_classifier.pt."""
    base = Path(getattr(settings, "BASE_DIR", ".")).resolve()
    out: list[tuple[str, str]] = []
    for val, lbl in _FIGHT_MENU_CHOICES:
        exists = (base / val).is_file()
        out.append((val, lbl + ("" if exists else " — absent")))
    return out


def coerce_fight_weights_pick(pick: str | None) -> str:
    """Réinitialise les anciens modèles retirés du menu vers fd_v2.h5."""
    norm = (pick or "").strip().replace("\\", "/")
    if norm in FIGHT_MENU_VALUES:
        return norm
    base_name = Path(norm).name
    for val in FIGHT_MENU_VALUES:
        if base_name == Path(val).name:
            return val
    return DEFAULT_FIGHT_WEIGHTS


def weapon_model_choices() -> list[tuple[str, str]]:
    """Liste dédiée : armes, guns et YOLO detect — sans classifiers combat ni route."""
    pairs = [(v, lbl) for v, lbl in build_inventory() if is_weapon_only_entry(v, lbl)]
    pairs.sort(key=lambda x: x[0].lower())
    return pairs


def default_relative_fight_weights() -> str:
    """Valeur par défaut du menu combat ( posix relatif à BASE_DIR )."""
    return DEFAULT_FIGHT_WEIGHTS


def default_relative_weapon_weights() -> str:
    base = Path(getattr(settings, "BASE_DIR", ".")).resolve()
    wp = Path(getattr(settings, "WEAPON_DETECTOR_PATH", "") or "").resolve()
    try:
        return wp.relative_to(base).as_posix()
    except ValueError:
        return wp.name


def with_default_choice(
    choices: list[tuple[str, str]],
    default_key: str,
    label_tag: str = "défaut config",
) -> list[tuple[str, str]]:
    if not default_key:
        return choices
    keys = {k for k, _ in choices}
    if default_key not in keys:
        return [(default_key, f"{default_key} ({label_tag})")] + list(choices)
    return choices


def resolve_weights_for_ultralytics(
    user_pick: str | None,
    default_setting_absolute: str,
) -> str:
    """
    Chemin absolu d'un fichier projet OU nom « yolov8n.pt » pour le hub Ultralytics.

    Sécurité : pas de fichier résolu hors BASE_DIR pour les chemins issus du menu.
    """
    base = Path(getattr(settings, "BASE_DIR", ".")).resolve()
    picked = (user_pick or "").strip()
    fb_abs = Path(default_setting_absolute)

    def safe_fb() -> str:
        p = fb_abs.expanduser().resolve()
        return str(p) if p.exists() else str(default_setting_absolute)

    if not picked:
        return safe_fb()

    def under_base(candidate: Path) -> bool:
        try:
            return candidate.expanduser().resolve().is_relative_to(base)
        except (ValueError, OSError):
            return False

    norm = picked.replace("\\", "/").strip()

    # Nom sans slash : fichier à la racine du projet OU hub Ultralytics (.pt uniquement)
    if "/" not in norm:
        suf = Path(norm).suffix.lower()
        rooted = (base / norm).resolve()
        if suf == ".pt":
            if under_base(rooted) and rooted.is_file():
                return str(rooted)
            return norm
        if suf in (".h5", ".pth"):
            if under_base(rooted) and rooted.is_file():
                return str(rooted)
            return safe_fb()

    if not Path(picked).is_absolute():
        full = (base / norm).resolve()
        if under_base(full) and full.is_file():
            return str(full)

    cand = Path(picked).expanduser().resolve()
    if cand.is_file() and under_base(cand):
        return str(cand)

    return safe_fb()
