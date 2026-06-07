from pathlib import Path

from rest_framework import serializers

from .models import CitizenReclamation


class CitizenReclamationSerializer(serializers.ModelSerializer):
    """Multipart : media, media_type, latitude, longitude, address, location_source, description, category."""

    latitude = serializers.FloatField(required=False, allow_null=True)
    longitude = serializers.FloatField(required=False, allow_null=True)
    address = serializers.CharField(required=False, allow_blank=True, default="")
    location_source = serializers.CharField(required=False, allow_blank=True, default="gps")

    class Meta:
        model = CitizenReclamation
        fields = (
            "id",
            "media",
            "media_type",
            "latitude",
            "longitude",
            "address",
            "location_source",
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

    def validate_location_source(self, value: str) -> str:
        raw = (value or "gps").strip().lower()
        allowed = {c.value for c in CitizenReclamation.LocationSource}
        if raw not in allowed:
            return CitizenReclamation.LocationSource.GPS
        return raw

    def validate(self, attrs):
        media = attrs.get("media")
        media_type = attrs.get("media_type")
        if media and media_type is not None:
            name = getattr(media, "name", "") or ""
            ext = Path(name).suffix.lower().lstrip(".")
            image_ext = {"jpg", "jpeg", "png", "webp", "heic", "heif"}
            video_ext = {"mp4", "mov", "m4v"}

            if media_type == CitizenReclamation.MediaType.IMAGE and ext in {"heic", "heif"}:
                try:
                    from .services.media_upload_utils import maybe_convert_heic_upload

                    media = maybe_convert_heic_upload(media)
                    attrs["media"] = media
                    name = getattr(media, "name", "") or name
                    ext = Path(name).suffix.lower().lstrip(".")
                except ValueError as exc:
                    raise serializers.ValidationError({"media": str(exc)}) from exc

            if media_type == CitizenReclamation.MediaType.IMAGE and ext not in image_ext:
                raise serializers.ValidationError(
                    {"media": f"Extension « {ext or '?'} » non acceptée pour une image."}
                )
            if media_type == CitizenReclamation.MediaType.VIDEO and ext not in video_ext:
                raise serializers.ValidationError(
                    {"media": f"Extension « {ext or '?'} » non acceptée pour une vidéo."}
                )

        lat = attrs.get("latitude")
        lng = attrs.get("longitude")
        if lat is None and self.initial_data.get("latitude") == "":
            attrs["latitude"] = None
        if lng is None and self.initial_data.get("longitude") == "":
            attrs["longitude"] = None

        address = (attrs.get("address") or "").strip()
        attrs["address"] = address
        has_coords = lat is not None and lng is not None
        has_address = bool(address)

        if not has_coords and not has_address:
            raise serializers.ValidationError(
                "Provide location via GPS coordinates, map selection, or address."
            )

        return attrs