from rest_framework import generics, parsers, permissions

from .models import CitizenReclamation
from .serializers import CitizenReclamationSerializer


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

