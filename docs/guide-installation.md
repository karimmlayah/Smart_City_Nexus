# Guide d'installation — MedinaMind

Temps estimé : **15 à 30 minutes** (selon téléchargement des dépendances et présence des modèles IA).

## Prérequis

- Windows 10/11, Linux ou macOS
- **Python 3.10+** (3.11 ou 3.12 recommandé)
- **pip** à jour
- (Optionnel) GPU CUDA pour accélérer YOLO / TensorFlow
- Espace disque : ~2–5 Go (venv + modèles)

## 1. Cloner le dépôt

```bash
git clone https://github.com/VOTRE_USERNAME/Esprit-PI-Classe-2526-MedinaMind.git
cd Esprit-PI-Classe-2526-MedinaMind
```

## 2. Environnement virtuel

**Windows (PowerShell) :**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS :**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Configuration

```bash
copy .env.example .env    # Windows
# cp .env.example .env    # Linux/macOS
```

Variables optionnelles (voir `.env.example`) :

| Variable | Usage |
|----------|--------|
| `FIRE_ONNX_MODEL` | Chemin modèle feu/fumée ONNX |
| `GROQ_API_KEY` | Assistant vocal AI Orb |
| `TWILIO_*` | Alertes SMS / voix |
| `DRONE_HOST` | IP caméra ESP32 |

## 4. Modèles IA (obligatoire pour l'inférence)

Les poids sont **exclus du Git** (`.gitignore`). Placez-les localement :

| Fichier / dossier | Emplacement | Module |
|-------------------|-------------|--------|
| `best.onnx` + `classes.txt` | `models/` | Fire & Smoke |
| `best.pt` (ambulance) | `models_traffic/` | Traffic / Mission Mayor ambulance |
| `yolov8n.pt` | `models_traffic/` | Traffic Nexus |
| `best (2).pt` | racine ou `ROAD_DAMAGE_MODEL_PATH` | Road damage |
| Modèle waste `.pt` | voir `WASTE_MODEL_PATH` dans settings | Street waste |
| `best_CNN.keras` | voir `UAV_MODEL_PATH` | UAV / bâtiment |

Sans modèles : l'application **démarre**, mais certaines pages affichent un fallback ou une erreur explicite.

## 5. Images démo Mission Mayor

Incluses dans le dépôt :

```text
images/game/
  ambulance.jpg
  cracks.jpg
  fire.jpg
  garbage.jpg
  batiment.png
```

Copie miroir servie en statique : `static/images/game/`

## 6. Base de données

```bash
python manage.py migrate
```

Crée `db.sqlite3` localement (non versionné).

Superutilisateur (optionnel) :

```bash
python manage.py createsuperuser
```

## 7. Lancer le serveur

```bash
python manage.py runserver
```

Ouvrir : [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## 8. Vérification rapide

| URL | Attendu |
|-----|---------|
| `/` ou `/introduction/` | Page d'accueil MedinaMind |
| `/ai-dashboard/` | Tableau de bord IA |
| `/mayor-mission/` | Jeu jury interactif |
| `/admin/` | Interface Django (si superuser créé) |

## Dépannage

| Problème | Solution |
|----------|----------|
| `No module named 'yt_dlp'` | `pip install yt-dlp` |
| Modèle waste introuvable | Configurer `WASTE_MODEL_PATH` dans `config/settings.py` |
| TensorFlow lent au 1er lancement | Normal — chargement lazy du modèle UAV |
| Caméra refusée (Mission Mayor) | Autoriser le navigateur ou utiliser « Generic Avatar » |
