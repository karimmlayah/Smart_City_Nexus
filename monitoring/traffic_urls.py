from django.urls import path

from monitoring import traffic_nexus_api

app_name = "traffic_api"

urlpatterns = [
    path("first-frame/", traffic_nexus_api.traffic_first_frame, name="traffic_first_frame"),
    path("analyze/", traffic_nexus_api.traffic_analyze, name="traffic_analyze"),
    path("models/", traffic_nexus_api.traffic_models, name="traffic_models"),
    path("live/init/", traffic_nexus_api.traffic_live_init, name="traffic_live_init"),
    path("live/tick/", traffic_nexus_api.traffic_live_tick, name="traffic_live_tick"),
    path("live/close/", traffic_nexus_api.traffic_live_close, name="traffic_live_close"),
    path("report/pdf/", traffic_nexus_api.traffic_report_pdf, name="traffic_report_pdf"),
]
