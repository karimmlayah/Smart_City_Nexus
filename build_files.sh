#!/usr/bin/env bash
set -euo pipefail

echo "Collecting static files..."
python manage.py collectstatic --noinput
