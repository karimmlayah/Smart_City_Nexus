# Documentation MedinaMind — Smart City AI Platform

Index de la documentation du projet (checklist ESPRIT / soutenance jury).

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | Architecture globale, modules IA, flux de données |
| [guide-installation.md](guide-installation.md) | Installation pas à pas (venv, modèles, lancement) |
| [guide-demo-jury.md](guide-demo-jury.md) | Scénario de démonstration jury (Mission Mayor + modules) |
| [captures/](captures/) | Captures d'écran pour rapport et soutenance |

## Nom du projet (format ESPRIT)

```
Esprit-[PI]-[Classe]-2526-MedinaMind
```

Remplacez `[PI]` par votre numéro de Projet Intégrateur et `[Classe]` par votre classe (ex. `4SIM`).

## Lancement rapide

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env
python manage.py migrate
python manage.py runserver
```

Application : [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Rapport PDF

Le rapport de projet intégrateur (PDF) est à déposer séparément selon les consignes ESPRIT.  
Les captures dans `docs/captures/` peuvent être intégrées au rapport Word/LaTeX.
