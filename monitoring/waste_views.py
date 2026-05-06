"""Page UI Waste Street Detection."""

from django.shortcuts import render
from django.urls import reverse


def waste_street_detection_page(request):
    return render(
        request,
        "monitoring/waste_street_detection.html",
        {
            "detect_url": reverse("waste_api:waste_detect"),
            "sms_url": reverse("waste_api:waste_send_sms"),
            "reports_url": reverse("waste_api:waste_reports"),
            "statistics_url": reverse("waste_api:waste_statistics"),
            "page_title": "Waste Street Detection",
        },
    )
