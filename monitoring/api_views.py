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
