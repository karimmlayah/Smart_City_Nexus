# YOLO Traffic Dashboard

Dashboard Streamlit pour detection des vehicules avec YOLO, selection manuelle d'une zone (ROI), et comptage des types de vehicules dans cette zone.

## Fonctions

- Interface style "monitoring" (theme sombre)
- Champ pour coller un lien YouTube
- Recuperation automatique du flux video via `yt-dlp`
- Dessin manuel de la zone ROI (polygone)
- Comptage en temps reel des classes de vehicules dans la zone:
  - car
  - bus
  - truck
  - motorcycle
  - bicycle

## Installation

```bash
cd yolo-dashboard
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Utilisation

1. Coller le lien YouTube (ou une source video locale/RTSP).
2. Cliquer sur **"1) Charger image pour dessiner zone"**.
3. Dessiner un polygone sur l'image.
4. Cliquer sur **"2) Lancer detection"**.
5. Voir le total et le detail par type dans la zone.

## Notes

- Le modele par defaut est `yolov8n.pt` (rapide). Tu peux choisir `yolov8s.pt` ou `yolov8m.pt`.
- Le "frame skip" augmente la fluidite sur PC moins puissant.
