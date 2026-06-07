"""Page dashboard UAV."""

from django.shortcuts import render
from django.urls import reverse

from monitoring.runtime import ml_deps_available


def uav_dashboard_page(request):
    model_status = {"ready": False, "model_path": "", "message": "UAV model unavailable on this host."}
    if ml_deps_available():
        from monitoring.services.uav_cnn_inference import uav_model_status

        model_status = uav_model_status(verify_load=False)
    return render(
        request,
        "monitoring/uav_dashboard.html",
        {
            "analyze_url": reverse("uav_api:uav_analyze"),
            "history_url": reverse("uav_api:uav_history"),
            "csv_url": reverse("uav_api:uav_export_csv"),
            "pdf_url": reverse("uav_api:uav_export_pdf"),
            "sim_buildings_url": reverse("uav_api:uav_sim_buildings"),
            "model_status_url": reverse("uav_api:uav_model_status"),
            "uav_model_ready": model_status.get("ready", False),
            "uav_model_path": model_status.get("model_path", ""),
            "uav_model_status_message": model_status.get("message", "UAV model missing"),
            "page_title": "Drone Mission Intelligence",
        },
    )
