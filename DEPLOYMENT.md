# MedinaMind — Deployment

## Vercel (lightweight web + API)

Vercel installs **`requirements.txt`** automatically (lightweight, no TensorFlow/OpenCV/Ultralytics).

| Vercel setting | Value |
|----------------|--------|
| Root Directory | `.` (repo root) |
| Framework Preset | Django |
| Install Command | `pip install -r requirements.txt` |
| Build Command | `bash build_files.sh` |
| Output Directory | *(leave empty)* |

### Required environment variables

- `SECRET_KEY` — Django secret
- `DEBUG` — `False`
- `DATABASE_URL` — PostgreSQL connection string
- `OPENAI_API_KEY` — AI dashboard assistant (server-side only)
- `CSRF_TRUSTED_ORIGINS` — e.g. `https://your-app.vercel.app`

Optional: `GROQ_API_KEY`, Twilio, email settings.

### After deploy

```bash
DATABASE_URL=postgresql://... python manage.py migrate
DATABASE_URL=postgresql://... python manage.py createsuperuser
```

## Local development (full ML stack)

```bash
pip install -r requirements-local.txt
python manage.py migrate
python manage.py runserver
```

## Dependency files

| File | Purpose |
|------|---------|
| `requirements.txt` | **Vercel / production** — Django, WhiteNoise, Postgres, OpenAI, etc. |
| `requirements-local.txt` | **Local dev** — full ML: TensorFlow, DeepFace, Ultralytics, OpenCV, … |

## ML features on Vercel

These require **`requirements-local.txt`** on Railway, Render, Fly.io, or a GPU worker:

- Surveillance / fusion (YOLO fight + weapons + DeepFace)
- Face registry embeddings (DeepFace)
- Road damage / waste / UAV CNN inference
- Traffic Nexus / violation pipeline
- Fire camera ONNX detection
- Smart crowd safety YOLO stream
- Mayor mission demo (OpenCV simulations)

On Vercel, pages load but ML endpoints return **503** with a clear message.

## Media uploads

Vercel has **no persistent disk**. Use Cloudinary/S3 for production `MEDIA_ROOT` (configure `DEFAULT_FILE_STORAGE` later).
