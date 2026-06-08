#!/usr/bin/env bash
set -euo pipefail

export VERCEL="${VERCEL:-1}"
export DEBUG="${DEBUG:-False}"

echo "Collecting static files (VERCEL=${VERCEL}, DEBUG=${DEBUG})..."
python manage.py collectstatic --noinput --verbosity 2
echo "Static files collected to $(python -c "from pathlib import Path; print(Path('staticfiles').resolve())")"
ls -la staticfiles/ | head -20
