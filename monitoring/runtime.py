"""Deployment/runtime helpers — Vercel vs local ML availability."""
from __future__ import annotations

import os
from functools import wraps
from typing import Any, Callable

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render

IS_VERCEL = os.environ.get("VERCEL") == "1"

ML_UNAVAILABLE_MESSAGE = (
    "This ML feature is not available on Vercel. "
    "Run the full stack locally (requirements-local.txt) or use an external ML service."
)

_cv2_checked: bool | None = None
_cv2_available: bool = False


def ml_deps_available() -> bool:
    """True when OpenCV (proxy for ML stack) is importable and ML is enabled."""
    global _cv2_checked, _cv2_available
    if _cv2_checked:
        return _cv2_available
    _cv2_checked = True
    if IS_VERCEL and not getattr(settings, "ML_FEATURES_ENABLED", False):
        _cv2_available = False
        return False
    try:
        import cv2  # noqa: F401

        _cv2_available = True
    except ImportError:
        _cv2_available = False
    return _cv2_available


def ml_unavailable_json(feature: str = "ML feature", status: int = 503) -> JsonResponse:
    return JsonResponse(
        {
            "success": False,
            "error": ML_UNAVAILABLE_MESSAGE,
            "feature": feature,
            "vercel": IS_VERCEL,
        },
        status=status,
    )


def ml_unavailable_http(request: HttpRequest, feature: str = "ML feature") -> HttpResponse:
    if _wants_json(request):
        return ml_unavailable_json(feature)
    return render(
        request,
        "monitoring/ml_unavailable.html",
        {
            "feature": feature,
            "message": ML_UNAVAILABLE_MESSAGE,
            "is_vercel": IS_VERCEL,
        },
        status=503,
    )


def _wants_json(request: HttpRequest) -> bool:
    accept = (request.headers.get("Accept") or "").lower()
    if "application/json" in accept:
        return True
    path = (request.path or "").lower()
    return path.startswith("/api/") or "/api/" in path


def ml_stub_view(feature: str = "ML feature") -> Callable[[HttpRequest], HttpResponse]:
    def _view(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        return ml_unavailable_http(request, feature=feature)

    _view.__name__ = f"ml_stub_{feature.replace(' ', '_')[:40]}"
    return _view


def require_ml(feature: str = "ML feature") -> Callable:
    """Decorator: return 503 on Vercel / when ML deps are missing."""

    def decorator(view_func: Callable) -> Callable:
        @wraps(view_func)
        def wrapper(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
            if not ml_deps_available():
                return ml_unavailable_http(request, feature=feature)
            return view_func(request, *args, **kwargs)

        return wrapper

    return decorator
