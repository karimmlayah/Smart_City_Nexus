"""Persistance rapports déchets + agrégats statistiques dashboard."""
from __future__ import annotations

from typing import Any

from django.db.models import Count, Sum

from monitoring.models import WasteReport


def format_location_line(city: str, address: str, legacy_location: str = "") -> str:
    parts = [p.strip() for p in [city or "", address or ""] if p and p.strip()]
    if parts:
        return ", ".join(parts)
    return (legacy_location or "").strip() or "Unknown location"


def report_to_api_dict(r: WasteReport) -> dict[str, Any]:
    return {
        "id": r.pk,
        "city": r.city,
        "address": r.address,
        "location": r.location or format_location_line(r.city, r.address),
        "latitude": r.latitude,
        "longitude": r.longitude,
        "notes": r.notes,
        "detected_objects": r.detected_objects or [],
        "count": r.object_count,
        "max_confidence": r.max_confidence,
        "severity": r.severity,
        "recommended_action": r.recommended_action,
        "annotated_image_url": r.annotated_image_url,
        "source_image_url": r.source_image_url or "",
        "sms_status": r.sms_status,
        "email_status": r.email_status,
        "sensitive_area_boost": r.sensitive_area_boost,
        "created_at": r.created_at.isoformat(),
    }


def list_reports_for_api() -> list[dict[str, Any]]:
    return [report_to_api_dict(r) for r in WasteReport.objects.all().order_by("-created_at")]


def compute_statistics() -> dict[str, Any]:
    qs = WasteReport.objects.all()
    total = qs.count()
    high = qs.filter(severity=WasteReport.Severity.HIGH).count()
    medium = qs.filter(severity=WasteReport.Severity.MEDIUM).count()
    low = qs.filter(severity=WasteReport.Severity.LOW).count()
    agg = qs.aggregate(sum_obj=Sum("object_count"))
    total_objects = int(agg["sum_obj"] or 0)
    sms_sent = qs.filter(
        sms_status__in=[
            WasteReport.SmsStatus.SENT,
            WasteReport.SmsStatus.MOCK_SENT,
        ]
    ).count()
    emails_sent = qs.filter(
        email_status__in=[
            WasteReport.EmailStatus.SENT,
            WasteReport.EmailStatus.MOCK_SENT,
        ]
    ).count()

    top_city = (
        qs.exclude(city="")
        .values("city")
        .annotate(n=Count("id"))
        .order_by("-n")
        .first()
    )
    most_zone = top_city["city"] if top_city else ""

    return {
        "total_reports": total,
        "high_severity": high,
        "medium_severity": medium,
        "low_severity": low,
        "total_detected_objects": total_objects,
        "sms_sent": sms_sent,
        "emails_sent": emails_sent,
        "most_reported_zone": most_zone or "—",
    }
