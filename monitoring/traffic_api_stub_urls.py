"""Fallback URLconf so traffic_api namespace always exists on Vercel."""
from django.urls import path

from monitoring import traffic_api_stub as _api

app_name = "traffic_api"

urlpatterns = [
    path("first-frame/", _api.traffic_first_frame, name="traffic_first_frame"),
    path("analyze/", _api.traffic_analyze, name="traffic_analyze"),
    path("models/", _api.traffic_models, name="traffic_models"),
    path("live/init/", _api.traffic_live_init, name="traffic_live_init"),
    path("live/tick/", _api.traffic_live_tick, name="traffic_live_tick"),
    path("live/close/", _api.traffic_live_close, name="traffic_live_close"),
    path("report/pdf/", _api.traffic_report_pdf, name="traffic_report_pdf"),
]
