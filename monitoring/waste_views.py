"""Page UI Waste Street Detection."""

from django.conf import settings
from django.shortcuts import render
from django.urls import reverse


def waste_street_detection_page(request):
    return render(
        request,
        "monitoring/waste_street_detection.html",
        {
            "detect_url": reverse("waste_api:waste_detect"),
            "email_url": reverse("waste_api:waste_send_email"),
            "reports_url": reverse("waste_api:waste_reports"),
            "statistics_url": reverse("waste_api:waste_statistics"),
            "default_email_to": (
                getattr(settings, "WASTE_DEFAULT_MUNICIPALITY_EMAIL", "")
                or getattr(settings, "EMAIL_HOST_USER", "")
                or ""
            ),
            "page_title": "Waste Street Detection",
        },
    )
