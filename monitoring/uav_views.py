"""Page dashboard UAV."""

from django.shortcuts import render
from django.urls import reverse


def uav_dashboard_page(request):
    return render(
        request,
        "monitoring/uav_dashboard.html",
        {
            "analyze_url": reverse("uav_api:uav_analyze"),
            "history_url": reverse("uav_api:uav_history"),
            "csv_url": reverse("uav_api:uav_export_csv"),
            "pdf_url": reverse("uav_api:uav_export_pdf"),
            "sim_buildings_url": reverse("uav_api:uav_sim_buildings"),
            "page_title": "Drone Mission Intelligence",
        },
    )
