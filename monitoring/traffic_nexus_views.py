"""Page Django Traffic Nexus (intégration au lieu de Streamlit)."""

from django.shortcuts import render
from django.urls import reverse


def traffic_nexus_dashboard(request):
    from monitoring.runtime import ML_UNAVAILABLE_MESSAGE, ml_deps_available

    traffic_api_ok = ml_deps_available()
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
            "traffic_api_unavailable": not traffic_api_ok,
            "traffic_api_message": ML_UNAVAILABLE_MESSAGE if not traffic_api_ok else "",
        },
    )
