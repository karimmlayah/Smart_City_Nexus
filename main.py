#!/usr/bin/env python3
"""
Lanceur à la racine du dépôt Smart City → délègue à smart_crowd_safety_ai/main.py.

Usage (depuis D:\\downloads\\smartcity) :
  python main.py train --epochs 12
  python main.py demo --video video.mp4 --output sortie.mp4
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "smart_crowd_safety_ai"
MAIN = ROOT / "main.py"

if not MAIN.is_file():
    print(
        f"Erreur: {MAIN} introuvable. "
        "Le projet Smart Crowd Safety AI doit être dans le dossier smart_crowd_safety_ai/.",
        file=sys.stderr,
    )
    sys.exit(1)

os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("smartcrowd_main", MAIN)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)
mod.main()
