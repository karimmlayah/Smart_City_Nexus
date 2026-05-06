"""Page Django Traffic Nexus (intégration au lieu de Streamlit)."""

from django.shortcuts import render
from django.urls import reverse


def traffic_nexus_dashboard(request):
    return render(
        request,
        "monitoring/traffic_nexus_dashboard.html",
        {
            "first_frame_url": reverse("traffic_api:traffic_first_frame"),
            "analyze_url": reverse("traffic_api:traffic_analyze"),
            "models_url": reverse("traffic_api:traffic_models"),
            "live_init_url": reverse("traffic_api:traffic_live_init"),
            "live_tick_url": reverse("traffic_api:traffic_live_tick"),
            "live_close_url": reverse("traffic_api:traffic_live_close"),
            "report_pdf_url": reverse("traffic_api:traffic_report_pdf"),
            "page_title": "Traffic Nexus",
        },
    )
