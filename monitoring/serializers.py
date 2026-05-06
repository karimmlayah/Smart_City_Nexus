from pathlib import Path

from rest_framework import serializers

from .models import CitizenReclamation


class CitizenReclamationSerializer(serializers.ModelSerializer):
    """Multipart : media, media_type, latitude, longitude, description, category."""

    class Meta:
        model = CitizenReclamation
        fields = (
            "id",
            "media",
            "media_type",
            "latitude",
            "longitude",
            "description",
            "category",
            "created_at",
        )
        read_only_fields = ("id", "created_at")

    def validate_media_type(self, value: str) -> str:
        allowed = {c.value for c in CitizenReclamation.MediaType}
        if value not in allowed:
            raise serializers.ValidationError(f"media_type doit être parmi {sorted(allowed)}.")
        return value

    def validate_category(self, value: str) -> str:
        allowed = {c.value for c in CitizenReclamation.Category}
        if value not in allowed:
            raise serializers.ValidationError(f"category doit être parmi {sorted(allowed)}.")
        return value

    def validate(self, attrs):
        media = attrs.get("media")
        media_type = attrs.get("media_type")
        if not media or media_type is None:
            return attrs

        name = getattr(media, "name", "") or ""
        ext = Path(name).suffix.lower().lstrip(".")
        image_ext = {"jpg", "jpeg", "png", "webp"}
        video_ext = {"mp4", "mov", "m4v"}

        if media_type == CitizenReclamation.MediaType.IMAGE and ext not in image_ext:
            raise serializers.ValidationError(
                {"media": f"Extension « {ext or '?'} » non acceptée pour une image."}
            )
        if media_type == CitizenReclamation.MediaType.VIDEO and ext not in video_ext:
            raise serializers.ValidationError(
                {"media": f"Extension « {ext or '?'} » non acceptée pour une vidéo."}
            )
        return attrs

