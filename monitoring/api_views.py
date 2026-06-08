from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics, parsers, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from .geocoding_service import geocode_address, reverse_geocode
from .models import CitizenReclamation
from .serializers import CitizenReclamationSerializer


@csrf_exempt
@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def api_health(request):
    """GET /api/health/ — mobile app connection check."""
    return Response({"status": "ok", "message": "MedinaMind backend online"})


@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def platform_diagnostics(request):
    """GET /api/diagnostics/ — safe deployment checks (no secrets)."""
    import os

    from django.urls import NoReverseMatch, reverse

    from monitoring.openai_client import get_openai_api_key
    from monitoring.runtime import IS_VERCEL, ml_deps_available

    def _route_ok(name: str) -> bool:
        try:
            reverse(name)
            return True
        except NoReverseMatch:
            return False

    return Response(
        {
            "vercel": IS_VERCEL,
            "ml_deps_available": ml_deps_available(),
            "openai_configured": bool(get_openai_api_key()),
            "openai_chat_model": (
                os.environ.get("OPENAI_CHAT_MODEL")
                or "gpt-4o-mini"
            ).strip(),
            "groq_configured": bool((os.environ.get("GROQ_API_KEY") or "").strip()),
            "database_url_configured": bool((os.environ.get("DATABASE_URL") or "").strip()),
            "routes": {
                "home": _route_ok("monitoring:home"),
                "surveillance": _route_ok("monitoring:surveillance_dashboard"),
                "road_damage_test": _route_ok("monitoring:road_damage_test"),
                "traffic_nexus": _route_ok("monitoring:traffic_nexus_dashboard"),
                "ai_dashboard_chatbot": _route_ok("monitoring:ai_dashboard_chatbot"),
                "traffic_api_first_frame": _route_ok("traffic_api:traffic_first_frame"),
            },
        }
    )


@method_decorator(csrf_exempt, name="dispatch")
class ReclamationCreateView(generics.CreateAPIView):
    """
    POST multipart/form-data vers /api/reclamations/
    Champs : media, media_type, latitude, longitude, address, location_source, description, category
    """

    queryset = CitizenReclamation.objects.all()
    serializer_class = CitizenReclamationSerializer
    parser_classes = (parsers.MultiPartParser, parsers.FormParser)
    permission_classes = (permissions.AllowAny,)
    authentication_classes = ()


@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def reclamation_geocode(request):
    """GET /api/reclamations/geocode/?q=address"""
    query = (request.GET.get("q") or request.GET.get("address") or "").strip()
    if not query:
        return Response(
            {"success": False, "message": "Query parameter q is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    result = geocode_address(query)
    if not result:
        return Response(
            {"success": False, "message": "Address not found.", "query": query},
            status=status.HTTP_404_NOT_FOUND,
        )
    return Response({"success": True, **result})


@api_view(["GET"])
@permission_classes([permissions.AllowAny])
def reclamation_reverse_geocode(request):
    """GET /api/reclamations/reverse-geocode/?lat=&lng="""
    try:
        lat = float(request.GET.get("lat", ""))
        lng = float(request.GET.get("lng", ""))
    except (TypeError, ValueError):
        return Response(
            {"success": False, "message": "Valid lat and lng parameters are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    result = reverse_geocode(lat, lng)
    if not result:
        return Response(
            {"success": False, "message": "Reverse geocoding unavailable.", "latitude": lat, "longitude": lng},
            status=status.HTTP_404_NOT_FOUND,
        )
    return Response({"success": True, **result})
