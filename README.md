# Smart City AI Command Platform

## Overview

Cette plateforme Django est concue comme un **Smart City AI Command Center unifie**: un cockpit unique qui connecte la securite urbaine, l'environnement, l'infrastructure, la mobilite, l'analyse UAV et l'observabilite IA.

Au lieu d'avoir des outils separes, le projet centralise les flux terrain (camera, upload, signalements citoyens), execute les inferences IA, puis transforme ces resultats en informations operationnelles exploitables par une municipalite: detection des risques, priorisation des interventions, generation de preuves/rapports et supervision continue des systemes IA.

## Project Value

Pour une municipalite, la valeur pratique du projet depasse la simple detection:

- **Decision support**: les resultats IA alimentent la priorisation et la planification des actions.
- **Alerting operationnel**: notifications et declenchements (ex: Twilio) pour les situations critiques.
- **Evidence generation**: production de sorties annotees, rapports PDF et exports pour traçabilite.
- **Operational dashboards**: vue consolidee pour equipes securite, mobilite, environnement et infrastructure.
- **Workflow terrain -> centre de commandement**: des capteurs et rapports citoyens jusqu'aux decisions d'intervention.

## Global Architecture

```text
Camera / Upload / Citizen Report
     ↓
AI Model Inference
     ↓
Risk Scoring + Decision Support
     ↓
Dashboard + Alerts + Reports
```

## Features

| Module | Description |
|---|---|
| Surveillance Dashboard | Vision live, alertes menaces, suivi incidents, reconnaissance faciale |
| Fire Camera | Reglages ESP32, inference ONNX feu/fumee, XAI, test/declenchement Twilio |
| Road Command Center | Priorisation IA des reclamations, KPIs, carte/heatmap, suivi intervention |
| Waste Street Detection | Detection dechets, severite, statistiques, envoi SMS/email, reporting |
| UAV Dashboard | Analyse structurelle (`Intact`, `Endommage`, `Effondre`), historique, exports |
| Traffic Nexus | Analyse trafic, mode live, decision support, rapport PDF |
| Traffic Violations | Stop-line, detection infractions feu rouge, OCR plaques, rapports PDF/ZIP |
| AI Dashboard | KPIs modeles, latence/erreurs, charts, filtres, activity feed |

## Tech Stack

### Frontend

- Django Templates (`templates/monitoring/*.html`)
- JavaScript vanilla (fetch/XHR, canvas, webcam APIs)
- CSS custom + themes cockpit
- Chart.js (visualisation du `ai-dashboard`)
- Leaflet (cartographie: road damage / waste / UAV)

### Backend

- Django 5 + Django REST Framework
- Python 3.x
- SQLite (`db.sqlite3`) pour les donnees Django
- Endpoints API Django (`/api/waste/`, `/api/uav/`, `/api/traffic/`)
- Integration avec module Smart City violation (dossier `hazemproj/Smart city`)

### Other Tools

- Ultralytics YOLO (plusieurs pipelines CV)
- OpenCV, NumPy, Pillow
- ONNX Runtime (fire/smoke inference)
- TensorFlow/Keras + DeepFace (selon module)
- Twilio (alertes vocales/SMS)
- yt-dlp (ingestion video selon flux)
- reportlab (generation PDF)

## Directory Structure

```text
smartcity/
|- config/                      # settings.py, urls.py (routing global)
|- monitoring/                  # app principale: views, api, services, urls
|  |- services/                 # pipelines IA (fire, waste, UAV, road, traffic, etc.)
|  |- traffic_tdss/             # logique traffic decision support
|  |- waste_api.py              # endpoints dechets
|  |- uav_api.py                # endpoints UAV
|  |- traffic_nexus_api.py      # endpoints Traffic Nexus
|  |- fire_camera_views.py      # settings camera + detect feu/fumee
|  |- traffic_violation_views.py# infractions feu rouge + OCR + rapports
|- templates/
|  |- monitoring/               # pages UI cockpit
|- static/                      # CSS/JS/images statiques
|- media/                       # uploads, outputs annotes, rapports
|- models/                      # modeles IA (waste, UAV, fire onnx, etc.)
|- models_traffic/              # modeles traffic nexus
|- model_road_damage/           # modeles route + env local
|- hazemproj/Smart city/        # module externe infractions trafic
|- requirements.txt
|- .env.example
|- manage.py
```

## Getting Started

### 1) Prerequisites

- Windows/Linux/macOS
- Python 3.10+ recommande
- Pip a jour
- (Optionnel) GPU/CUDA selon vos modeles

### 2) Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3) Configuration

Copier le fichier d'exemple:

```bash
copy .env.example .env
```

Configurer selon vos besoins:

- `FIRE_ONNX_MODEL` (si vous ne placez pas `models/best.onnx`)
- `DRONE_HOST` (IP ESP32-CAM)
- Variables Twilio (`TWILIO_*`) pour alertes fire-camera
- `OPENAI_API_KEY` (si vous activez les agents IA de synthese)
- Variables OCR/email si pipeline infractions complet

### 4) Migrations + Run Server

```bash
python manage.py migrate
python manage.py runserver
```

Application disponible sur: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Operational Scenario (End-to-End)

Ce scenario couvre toutes les pages operatoires du projet.

### Step 1 - Surveillance temps reel

- **URL**: [http://127.0.0.1:8000/surveillance/](http://127.0.0.1:8000/surveillance/)
- **But demo**: montrer la detection live des menaces et la supervision incidents.
- **Actions cle**: start camera, toggle fight/weapon, verifier liste des menaces, generer rapport incident.

### Step 2 - Reglage camera ESP32

- **URL**: [http://127.0.0.1:8000/camera/settings/](http://127.0.0.1:8000/camera/settings/)
- **But demo**: preparer la source terrain et valider la connectivite camera.
- **Actions cle**: configurer IP ESP32-CAM, tester ports 80/81, verifier `/capture` et stream, sauvegarder sync Django + ESP.

### Step 3 - Detection feu/fumee + alerting

- **URL**: [http://127.0.0.1:8000/fire-camera/detect/](http://127.0.0.1:8000/fire-camera/detect/)
- **But demo**: illustrer l'inference ONNX et la logique d'alerte.
- **Actions cle**: verifier `best.onnx`, ajuster seuils (`Conf`, `IoU`, delais), activer appel Twilio, lire table XAI.

### Step 4 - Road Command Center / reclamations

- **URL**: [http://127.0.0.1:8000/road-damage-test/](http://127.0.0.1:8000/road-damage-test/)
- **But demo**: montrer la priorisation IA pour interventions infrastructure.
- **Actions cle**: presenter KPIs, carte/heatmap, ouvrir une fiche "Analyse IA", lancer verification drone live.

### Step 5 - Waste detection urbain

- **URL**: [http://127.0.0.1:8000/waste/street-detection/](http://127.0.0.1:8000/waste/street-detection/)
- **But demo**: prouver la chaine detection -> severite -> notification.
- **Actions cle**: soumettre media, afficher resultat inference, envoyer SMS/email, consulter statistiques et rapports.

### Step 6 - UAV Structural Assessment

- **URL**: [http://127.0.0.1:8000/uav/dashboard/](http://127.0.0.1:8000/uav/dashboard/)
- **But demo**: visualiser l'evaluation structurelle assistee par UAV.
- **Actions cle**: charger image/frame, lire classe predite, consulter historique, exporter CSV/PDF.

### Step 7 - Traffic Nexus (gestion trafic)

- **URL**: [http://127.0.0.1:8000/traffic/nexus/](http://127.0.0.1:8000/traffic/nexus/)
- **But demo**: montrer l'analyse trafic et l'aide a la decision.
- **Actions cle**: lancer session live (`init/tick/close`) ou analyse media, changer modele, lire resultats, exporter PDF.

### Step 8 - Violations feu rouge (preuve + enforcement)

- **URL**: [http://127.0.0.1:8000/traffic/violations/](http://127.0.0.1:8000/traffic/violations/)
- **But demo**: demonstrer la detection d'infraction avec generation de preuve.
- **Actions cle**: upload/capture, tracer stop-line (manuel/AI Suggest), traiter, verifier output annote, consulter registre SQLite, generer PDF/ZIP.

### Step 9 - AI observability global

- **URL**: [http://127.0.0.1:8000/ai-dashboard/](http://127.0.0.1:8000/ai-dashboard/)
- **But demo**: piloter performance et fiabilite des modeles IA.
- **Actions cle**: filtrer modele/periode, lire KPIs latence/erreurs, presenter charts et table, activer auto-refresh.

### Recommended Demo Path (5 minutes)

1. `surveillance/` (45s): lancer camera et montrer alertes live.  
2. `camera/settings/` + `fire-camera/detect/` (60s): valider flux ESP32 puis inference ONNX + Twilio.  
3. `road-damage-test/` (45s): KPIs + priorisation + carte.  
4. `waste/street-detection/` (40s): inference + notification.  
5. `uav/dashboard/` (35s): prediction structurelle + export.  
6. `traffic/nexus/` (35s): analyse trafic + rapport.  
7. `traffic/violations/` (50s): stop-line, violation, preuves.  
8. `ai-dashboard/` (30s): observabilite globale des modeles.

## API Quick Links

### Core APIs

| Domain | Method | Endpoint |
|---|---|---|
| Reclamations | `POST` | `/api/reclamations/` |
| Waste | `POST` | `/api/waste/detect/` |
| Waste | `POST` | `/api/waste/send-sms/` |
| Waste | `POST` | `/api/waste/send-email/` |
| Waste | `GET` | `/api/waste/reports/` |
| Waste | `GET` | `/api/waste/statistics/` |
| UAV | `POST` | `/api/uav/analyze/` |
| UAV | `GET` | `/api/uav/history/` |
| UAV | `GET` | `/api/uav/export/csv/` |
| UAV | `GET` | `/api/uav/export/pdf/` |
| Traffic Nexus | `POST` | `/api/traffic/first-frame/` |
| Traffic Nexus | `POST` | `/api/traffic/analyze/` |
| Traffic Nexus | `GET` | `/api/traffic/models/` |
| Traffic Nexus | `POST` | `/api/traffic/live/init/` |
| Traffic Nexus | `POST` | `/api/traffic/live/tick/` |
| Traffic Nexus | `POST` | `/api/traffic/live/close/` |
| Traffic Nexus | `GET` | `/api/traffic/report/pdf/` |

## Acknowledgments

- Equipe Smart City / Computer Vision.
- Ecosysteme open-source: Django, Ultralytics, OpenCV, ONNX Runtime, TensorFlow, DeepFace, Chart.js, Leaflet.
- Integrations externes: Twilio et outils OCR/LLM selon configuration.

