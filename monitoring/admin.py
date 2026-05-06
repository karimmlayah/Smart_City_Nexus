from django.contrib import admin

from .models import (
    Alert,
    CitizenReclamation,
    FaceIdentity,
    FacePersonProfile,
    FaceReferenceImage,
    FireCameraConfig,
    UavStructuralAnalysis,
    VideoSource,
    WasteReport,
)


class FaceReferenceImageInline(admin.TabularInline):
    model = FaceReferenceImage
    extra = 1
    fields = ("image", "is_primary", "embedding_short")
    readonly_fields = ("embedding_short",)

    @admin.display(description="Embedding")
    def embedding_short(self, obj: FaceReferenceImage) -> str:
        if not obj or not getattr(obj, "pk", None):
            return "—"
        emb = getattr(obj, "embedding", None)
        if emb and isinstance(emb, list) and len(emb) >= 16:
            return f"OK · dim {len(emb)}"
        if emb and isinstance(emb, list):
            return f"short · len {len(emb)}"
        return "empty"


@admin.register(FaceIdentity)
class FaceIdentityAdmin(admin.ModelAdmin):
    list_display = ("display_name", "person_code", "category", "is_active", "slug", "created_at")
    list_filter = ("is_active", "category")
    search_fields = ("display_name", "person_code", "slug", "notes", "category")
    fields = ("slug", "display_name", "person_code", "category", "is_active", "notes")
    prepopulated_fields = {"slug": ("display_name",)}
    inlines = [FaceReferenceImageInline]


@admin.register(FacePersonProfile)
class FacePersonProfileAdmin(admin.ModelAdmin):
    list_display = ("identity_code", "full_name", "role", "risk_level", "last_seen", "updated_at")
    search_fields = ("identity_code", "full_name", "notes", "role")
    readonly_fields = ("created_at", "updated_at")

    def get_changeform_initial_data(self, request):
        initial = super().get_changeform_initial_data(request)
        ic = (request.GET.get("identity_code") or "").strip()
        if ic and request.path.rstrip("/").endswith("/add"):
            initial["identity_code"] = ic
        return initial


@admin.register(VideoSource)
class VideoSourceAdmin(admin.ModelAdmin):
    list_display = ("name", "source_type", "is_active", "crowd_threshold", "created_at")
    list_filter = ("source_type", "is_active")
    search_fields = ("name", "url", "notes")
    fields = (
        "name",
        "source_type",
        "video_file",
        "url",
        "webcam_index",
        "crowd_threshold",
        "fast_movement_threshold",
        "is_active",
        "notes",
    )


@admin.register(CitizenReclamation)
class CitizenReclamationAdmin(admin.ModelAdmin):
    list_display = ("category", "media_type", "latitude", "longitude", "created_at")
    list_filter = ("category", "media_type", "created_at")
    readonly_fields = ("created_at",)
    search_fields = ("description",)


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("alert_type", "source", "person_count", "created_at")
    list_filter = ("alert_type", "created_at")
    readonly_fields = ("created_at",)
    search_fields = ("message", "source__name")


@admin.register(FireCameraConfig)
class FireCameraConfigAdmin(admin.ModelAdmin):
    list_display = ("esp_ip", "updated_at")
    search_fields = ("esp_ip",)


@admin.register(UavStructuralAnalysis)
class UavStructuralAnalysisAdmin(admin.ModelAdmin):
    list_display = (
        "pk",
        "prediction_label",
        "zone",
        "confidence",
        "risk_score",
        "operational_level",
        "surface_m2",
        "created_at",
    )
    list_filter = ("zone", "prediction_label", "created_at")
    readonly_fields = ("created_at",)
    search_fields = ("primary_filename", "comparison_filename", "prediction_label")


@admin.register(WasteReport)
class WasteReportAdmin(admin.ModelAdmin):
    list_display = (
        "pk",
        "city",
        "severity",
        "object_count",
        "sms_status",
        "email_status",
        "created_at",
    )
    list_filter = ("severity", "sms_status", "email_status", "created_at")
    readonly_fields = ("created_at",)
    search_fields = ("city", "address", "location", "notes", "sms_twilio_sid")
