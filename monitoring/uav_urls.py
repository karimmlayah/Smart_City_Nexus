from django.urls import path

from monitoring import uav_api

app_name = "uav_api"

urlpatterns = [
    path("analyze/", uav_api.uav_analyze, name="uav_analyze"),
    path("history/", uav_api.uav_history, name="uav_history"),
    path("export/csv/", uav_api.uav_export_csv, name="uav_export_csv"),
    path("export/pdf/", uav_api.uav_export_pdf, name="uav_export_pdf"),
    path("sim-buildings/", uav_api.uav_sim_buildings, name="uav_sim_buildings"),
]
