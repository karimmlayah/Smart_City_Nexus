from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import generics, parsers, permissions

from .models import CitizenReclamation
from .serializers import CitizenReclamationSerializer


@method_decorator(csrf_exempt, name="dispatch")
class ReclamationCreateView(generics.CreateAPIView):
    """
    POST multipart/form-data vers /api/reclamations/
    Champs : media, media_type, latitude, longitude, description, category
    """

    queryset = CitizenReclamation.objects.all()
    serializer_class = CitizenReclamationSerializer
    parser_classes = (parsers.MultiPartParser, parsers.FormParser)
    permission_classes = (permissions.AllowAny,)
    authentication_classes = ()

