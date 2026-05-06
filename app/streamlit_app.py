"""
Proxy Streamlit : exécute l'app du sous-projet smart_crowd_safety_ai.

Depuis la racine Smart City :
  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "smart_crowd_safety_ai" / "app" / "streamlit_app.py"

if not TARGET.is_file():
    print(
        f"Erreur: {TARGET} introuvable.",
        file=sys.stderr,
    )
    sys.exit(1)

runpy.run_path(str(TARGET), run_name="__main__")
