from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework import parsers, permissions, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response

from monitoring.services.face_registry_enroll import (
    FaceRegistryEnrollError,
    enroll_face_to_registry,
)


@csrf_exempt
@api_view(["POST"])
@permission_classes([permissions.AllowAny])
def face_registry_enroll(request):
    """
    POST /api/face-registry/enroll/
    multipart: name, category, unique_id, crop_url (optional), cropped_face_image, reference_image
    """
    data = request.data
    name = (data.get("name") or data.get("full_name") or "").strip()
    category = (data.get("category") or "").strip()
    unique_id = (data.get("unique_id") or data.get("person_code") or "").strip()
    crop_url = (data.get("crop_url") or data.get("cropped_face_url") or "").strip()

    cropped = request.FILES.get("cropped_face_image") or request.FILES.get("cropped_face")
    reference = request.FILES.get("reference_image") or request.FILES.get("additional_reference")

    try:
        payload = enroll_face_to_registry(
            display_name=name,
            person_code=unique_id,
            category=category,
            cropped_upload=cropped,
            reference_upload=reference,
            crop_media_url=crop_url,
        )
        return Response(payload, status=status.HTTP_201_CREATED)
    except FaceRegistryEnrollError as exc:
        return Response(
            {"success": False, "message": exc.message},
            status=exc.status,
        )
    except Exception as exc:
        return Response(
            {"success": False, "message": str(exc)[:500] or "Enrollment failed."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
