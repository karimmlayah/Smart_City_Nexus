#!/usr/bin/env bash
set -euo pipefail

export VERCEL="${VERCEL:-1}"
export DEBUG="${DEBUG:-False}"

mkdir -p staticfiles

echo "Collecting static files (VERCEL=${VERCEL}, DEBUG=${DEBUG})..."
python manage.py collectstatic --noinput

if [ -f staticfiles/staticfiles.json ]; then
  echo "staticfiles.json created ($(wc -c < staticfiles/staticfiles.json) bytes)"
else
  echo "WARNING: staticfiles/staticfiles.json missing after collectstatic"
fi

echo "Static file count: $(find staticfiles -type f | wc -l)"
