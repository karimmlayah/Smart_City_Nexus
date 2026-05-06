# Smart Crowd Safety AI System

Analyse de flux vidéo : **personnes**, **densité**, **combat (YOLO classify pré-entraîné)**, **panique (vitesses)**, **objets abandonnés (sacs)**, **score de risque** et **explications**.

## Prérequis

- Python 3.10+
- GPU optionnel (CUDA pour PyTorch)

## Modèles pré-entraînés (sans entraînement local)

Le pipeline utilise :

- **YOLOv8n** (`yolov8n.pt`, téléchargé automatiquement) — détection / suivi
- **Combat** : le même fichier **`best_fight_classifier.pt`** que la page « Combat » du site, attendu à la **racine du dépôt `smartcity/`** (dossier parent de `smart_crowd_safety_ai/`).

Vous pouvez aussi définir la variable d’environnement :

```text
SMARTCROWD_FIGHT_CLASSIFIER=C:\chemin\vers\votre_modele_classify.pt
```

Ou passer `--fight-classifier chemin` à la commande `demo`.

Si aucun fichier combat n’est trouvé, P(fight) reste à 0 (les autres signaux panique / abandonné fonctionnent toujours).

## Installation

```bash
cd smart_crowd_safety_ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**À la racine du dépôt parent** (`smartcity/`), vous pouvez aussi utiliser les relais `main.py` et `app/streamlit_app.py` (sans `cd smart_crowd_safety_ai`) après avoir installé les dépendances dans ce sous-dossier.

Au premier lancement, Ultralytics télécharge `yolov8n.pt`.

## 1. Démo sur une vidéo

```bash
python main.py demo --video chemin/vers/video.mp4 --output sortie.mp4
```

Touches : fenêtre OpenCV — **q** pour quitter.

Sans affichage graphique :

```bash
python scripts/demo.py chemin/vers/video.mp4 -o sortie.mp4 --no-show
```

## 2. Dashboard Streamlit

```bash
streamlit run app/streamlit_app.py
```

## 3. (Optionnel) Entraînement CNN+LSTM sur données synthétiques

Réservé aux utilisateurs avancés ; **le pipeline par défaut ne l’utilise pas**.

```bash
python main.py train --epochs 12
```

Génère `models/fight_cnn_lstm.pt` (non branché au pipeline actuel).

## Intégration site Django (Smart City)

Une page **« Smart Crowd Safety AI »** décrit ce module.

## Architecture

- `src/detection/` — YOLOv8 personnes
- `src/tracking/` — suivi Ultralytics (ByteTrack/BoT-SORT)
- `src/behavior/` — combat `fight_pretrained.py` (YOLO classify), panique, abandonné
- `src/utils/` — score, explicabilité
- `src/pipeline.py` — orchestration vidéo

## Notes

- Les seuils (panique, abandon) sont dans `src/utils/config.py`.
