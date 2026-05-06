from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

from monitoring.api_views import ReclamationCreateView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/reclamations/", ReclamationCreateView.as_view(), name="api_reclamations"),
    path("api/waste/", include(("monitoring.waste_urls", "waste_api"), namespace="waste_api")),
    path("api/uav/", include(("monitoring.uav_urls", "uav_api"), namespace="uav_api")),
    path("api/traffic/", include(("monitoring.traffic_urls", "traffic_api"), namespace="traffic_api")),
    path("", include("monitoring.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
