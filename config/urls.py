from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

from monitoring.api_views import (
    ReclamationCreateView,
    api_health,
    platform_diagnostics,
    reclamation_geocode,
    reclamation_reverse_geocode,
)
from monitoring.face_registry_api import face_registry_enroll

# Mobile + core API routes first (must load even if optional modules fail).
urlpatterns = [
    path(
        "favicon.ico",
        RedirectView.as_view(url=f"{settings.STATIC_URL}images/medinamind-logo.png", permanent=False),
        name="favicon_ico",
    ),
    path(
        "favicon.png",
        RedirectView.as_view(url=f"{settings.STATIC_URL}images/medinamind-logo.png", permanent=False),
        name="favicon_png",
    ),
    path("admin/", admin.site.urls),
    path("api/health/", api_health, name="api_health"),
    path("api/diagnostics/", platform_diagnostics, name="platform_diagnostics"),
    path("api/reclamations/", ReclamationCreateView.as_view(), name="api_reclamations"),
    path("api/reclamations/geocode/", reclamation_geocode, name="api_reclamations_geocode"),
    path("api/reclamations/reverse-geocode/", reclamation_reverse_geocode, name="api_reclamations_reverse_geocode"),
    path("api/face-registry/enroll/", face_registry_enroll, name="api_face_registry_enroll"),
]

try:
    urlpatterns.append(
        path("api/waste/", include(("monitoring.waste_urls", "waste_api"), namespace="waste_api")),
    )
except Exception:
    import logging

    logging.getLogger(__name__).warning(
        "Waste API routes disabled (ML / optional dependency missing).",
        exc_info=True,
    )

try:
    urlpatterns.append(
        path("api/uav/", include(("monitoring.uav_urls", "uav_api"), namespace="uav_api")),
    )
except Exception:
    import logging

    logging.getLogger(__name__).warning(
        "UAV API routes disabled (ML / optional dependency missing).",
        exc_info=True,
    )

urlpatterns.append(
    path("api/traffic/", include(("monitoring.traffic_urls", "traffic_api"), namespace="traffic_api")),
)

try:
    urlpatterns.append(path("", include("monitoring.urls")))
except Exception:
    import logging

    logging.getLogger(__name__).exception(
        "monitoring.urls failed to load — homepage routes unavailable.",
    )
    from django.http import HttpResponse

    def _startup_fallback_home(_request):
        return HttpResponse(
            "MedinaMind backend is online but page routes failed to load. Check Vercel runtime logs.",
            status=503,
            content_type="text/plain",
        )

    urlpatterns.append(path("", _startup_fallback_home, name="home_fallback"))

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
