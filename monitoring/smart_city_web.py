"""
Lazy-load `hazemproj/Smart city/web_app.py` (YOLO violation pipeline).
Adds SMART_CITY_VIOLATION_ROOT to sys.path once.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_MODULE: Any = None


def get_smart_city_web():
    """Return the web_app module (Flask app + analysis helpers). Thread-unsafe lazy init."""
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    from django.conf import settings

    root = Path(settings.SMART_CITY_VIOLATION_ROOT)
    path = root / "web_app.py"
    if not path.is_file():
        raise FileNotFoundError(f"Smart City web_app not found: {path}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec = importlib.util.spec_from_file_location("smart_city_web_app", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smart_city_web_app"] = mod
    spec.loader.exec_module(mod)
    _MODULE = mod
    return mod
