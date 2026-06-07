# Guide démo jury — MedinaMind

Scénario recommandé pour la soutenance / démonstration ESPRIT (~10–15 min).

## Parcours principal : Mission Mayor (jeu interactif)

**URL :** [http://127.0.0.1:8000/mayor-mission/](http://127.0.0.1:8000/mayor-mission/)

### Déroulé

1. **Start Mission** — lancer le jeu
2. **Scan visage** — le jury devient « Maire » (ou avatar générique)
3. **Crise** — image réelle (`images/game/`) + alerte « CITY INCIDENT DETECTED »
4. **Choix obligatoire** :
   - **Classic Method** → situation empire, perte de points, overlay rouge
   - **MedinaMind AI** → **vrai modèle IA** + image annotée + gain de points
5. **5 rounds** — feu, fissures, déchets, ambulance, bâtiment
6. **Résultat final** — CITY SAVED / CITY AT RISK + certificat téléchargeable

### Modèles utilisés par crise

| Image | Crise | Modèle MedinaMind |
|-------|-------|-------------------|
| `ambulance.jpg` | Accident / corridor bloqué | Traffic Nexus (ambulance + véhicules) |
| `cracks.jpg` | Route endommagée | Road Damage YOLO |
| `garbage.jpg` | Hotspot déchets | Street Waste YOLO |
| `fire.jpg` | Feu / fumée | Fire & Smoke ONNX |
| `batiment.png` | État structurel | UAV CNN + Grad-CAM |

### Points à montrer au jury

- Timer par round (choix sous pression)
- Métriques en **TND** (dinars tunisiens)
- Panneau « Powered by real MedinaMind AI »
- Comparaison Classic vs IA sur la **même** image

---

## Parcours complémentaire (modules live)

### 1. AI Dashboard — observabilité

`/ai-dashboard/` — KPIs modèles, latence, activité globale.

### 2. Fire & Smoke + ESP32

1. `/camera/settings/` — config caméra
2. `/fire-camera/detect/` — inference ONNX, seuils, alerte Twilio (si configuré)

### 3. Road Command Center

`/road-damage-test/` — upload image, détection fissures, carte / priorités.

### 4. Street Waste

`/waste/street-detection/` — détection déchets, sévérité, notification.

### 5. UAV Dashboard

`/uav/dashboard/` — prédiction Intact / Endommagé / Effondré, Grad-CAM.

### 6. Traffic Nexus

`/traffic/nexus/` — analyse congestion, ambulance, rapport PDF.

---

## Script oral court (2 min)

> « MedinaMind centralise la vision par ordinateur pour une municipalité : feu, routes, déchets, trafic et infrastructures.  
> Le jury vit l'expérience via **Mission Mayor** : chaque crise est une vraie photo, chaque choix MedinaMind appelle le modèle correspondant et affiche le résultat annoté.  
> Classic simule la réponse manuelle lente ; l'IA réduit le risque et améliore le score ville. »

---

## Checklist avant la soutenance

- [ ] `python manage.py runserver` démarre sans erreur
- [ ] Modèles IA présents (`models/`, `models_traffic/`, etc.)
- [ ] `/mayor-mission/` testé sur les 5 scénarios
- [ ] Micro / caméra autorisés dans le navigateur
- [ ] `.env` configuré (Twilio / Groq optionnels)
- [ ] Captures dans `docs/captures/` pour le rapport PDF
