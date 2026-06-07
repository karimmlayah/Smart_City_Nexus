# Architecture — MedinaMind Smart City

## Vue d'ensemble

MedinaMind est un **centre de commandement IA** pour ville intelligente. Il unifie détection computer vision, scoring des risques, tableaux de bord et alertes opérationnelles.

```text
Capteurs / Upload / Signalement citoyen
              ↓
       Inférence IA (YOLO, ONNX, CNN)
              ↓
    Scoring · Priorisation · Décision
              ↓
  Dashboard · Alertes · Rapports · APIs
```

## Stack technique

| Couche | Technologies |
|--------|--------------|
| Frontend | Django Templates, JavaScript, Chart.js, Leaflet, Three.js |
| Backend | Python 3.10+, Django 5, Django REST Framework |
| Base de données | SQLite (dev) — migrations Django |
| Vision | Ultralytics YOLO, OpenCV, ONNX Runtime, TensorFlow/Keras |
| Intégrations | Twilio (SMS/voix), ESP32-CAM (IoT feu) |

## Structure du dépôt

```text
smartcity/
├── config/                 # Settings Django, routing global
├── monitoring/             # App principale (views, APIs, services IA)
│   ├── services/           # Pipelines : fire, waste, road, uav, fight…
│   ├── traffic_tdss/       # Traffic Nexus — détection & décision
│   ├── demo_mayor_service.py   # Jeu jury Mission Mayor
│   └── migrations/         # Schéma base de données
├── templates/monitoring/   # Pages UI cockpit
├── static/                 # CSS, JS, images
├── images/game/            # Assets démo Mission Mayor (crises)
├── models/                 # Modèles ONNX feu/fumée (local, hors Git)
├── models_traffic/         # YOLO trafic / ambulance (local, hors Git)
├── docs/                   # Cette documentation
└── requirements.txt
```

## Modules IA

| Module | Modèle | Entrée | Sortie |
|--------|--------|--------|--------|
| Fire & Smoke | ONNX YOLO (`models/best.onnx`) | Image / flux ESP32 | Classes feu, fumée + alerte Twilio |
| Road Damage | YOLO segmentation (`best (2).pt`) | Photo route | Fissures / nids-de-poule annotés |
| Street Waste | YOLO déchets | Photo rue | Comptage, sévérité, SMS/email |
| UAV / Bâtiment | CNN Keras 3 classes | Image aérienne | Intact / Endommagé / Effondré + Grad-CAM |
| Traffic Nexus | YOLO dual (`best.pt` + `yolov8n.pt`) | Image / vidéo | Véhicules, ambulance, congestion |
| Surveillance | Fight / weapon classifiers | Flux webcam | Alertes sécurité |

## Mission Mayor (démo jury)

Jeu interactif : `/mayor-mission/`

```text
Scan visage → Crise (image réelle) → Choix Classic vs MedinaMind
       → Inférence modèle réel → Score / animation → Résultat final
```

API dédiées :

- `GET /api/demo/mayor-mission/challenges/`
- `POST /api/demo/mayor-mission/run-classic/`
- `POST /api/demo/mayor-mission/run-medinamind/`
- `POST /api/demo/mayor-mission/finish/`

## APIs principales

| Domaine | Endpoint |
|---------|----------|
| Déchets | `POST /api/waste/detect/` |
| UAV | `POST /api/uav/analyze/` |
| Trafic | `POST /api/traffic/analyze/` |
| Assistant | `POST /api/assistant/groq/` |

Liste complète : voir [README.md](../README.md) à la racine du dépôt.

## Sécurité

- Secrets dans `.env` (jamais versionné) — voir `.env.example`
- Modèles lourds exclus du Git (`.gitignore`)
- `media/` généré localement (uploads, sorties annotées)
