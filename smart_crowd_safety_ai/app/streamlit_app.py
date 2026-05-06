"""
Dashboard Streamlit — Smart Crowd Safety AI.

Lancer depuis le dossier du projet:
  streamlit run app/streamlit_app.py

Ou:
  cd smart_crowd_safety_ai && streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run_video_pipeline


def main() -> None:
    st.set_page_config(page_title="Smart Crowd Safety AI", layout="wide")
    st.title("Smart Crowd Safety AI — Dashboard")
    st.caption(
        "YOLOv8 détection/suivi, combat via YOLO classify pré-entraîné, panique, objets abandonnés."
    )

    uploaded = st.file_uploader("Vidéo (mp4, avi…)", type=["mp4", "avi", "mov", "mkv", "webm"])
    max_frames = st.number_input("Max frames (0 = tout)", min_value=0, value=0, step=100)
    fight_path = st.text_input(
        "Modèle combat (.pt classify, optionnel)",
        value="",
        help="Vide = ../best_fight_classifier.pt (racine du dépôt smartcity) ou variable SMARTCROWD_FIGHT_CLASSIFIER",
    )
    no_show = True

    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        fc = Path(fight_path.strip()).expanduser() if fight_path.strip() else None

        with st.spinner("Analyse en cours…"):
            out_mp4 = Path(tempfile.mkdtemp()) / "out.mp4"
            mf = int(max_frames) if max_frames > 0 else None
            result = run_video_pipeline(
                tmp_path,
                output_path=out_mp4,
                fight_classifier_path=fc,
                show_preview=not no_show,
                max_frames=mf,
            )

        st.subheader("Statistiques")
        st.json(result.get("stats", {}))
        st.caption(f"Combat: {result.get('fight_backend')} — {result.get('fight_model')}")

        st.subheader("Explication (XAI)")
        st.json(result.get("explanation", {}))

        if out_mp4.is_file():
            st.subheader("Vidéo annotée")
            st.video(str(out_mp4))
    else:
        st.info("Téléversez une vidéo pour lancer l’analyse.")

    st.markdown("---")
    st.markdown(
        "Aucun entraînement local requis : placez **best_fight_classifier.pt** "
        "à la racine du dépôt `smartcity` (comme pour la page Combat), ou définissez "
        "`SMARTCROWD_FIGHT_CLASSIFIER`."
    )


if __name__ == "__main__":
    main()
