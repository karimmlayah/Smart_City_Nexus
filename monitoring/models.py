import logging

from django.core.validators import FileExtensionValidator
from django.db import models
from django.utils.text import slugify

logger = logging.getLogger(__name__)


def upload_face_registry_reference(instance, filename: str) -> str:
    base = getattr(instance.face_identity, "slug", "") or "identite"
    return f"face_registry/{base}/{filename}"


class FaceIdentity(models.Model):
    """Individu enregistré pour la correspondance biométrique (recadrage facial)."""

    slug = models.SlugField(unique=True, max_length=96)
    display_name = models.CharField(
        "Nom affiché (nom complet)",
        max_length=160,
        help_text="Pour l’UI : équivalent au « full name » métier.",
    )
    person_code = models.CharField(
        "Code unique / matricule",
        max_length=64,
        blank=True,
        default="",
        db_index=True,
        help_text="Identifiant métier unique (ex. BOXER-001) — utilisé pour le profil étendu.",
    )
    category = models.CharField(
        "Catégorie",
        max_length=120,
        blank=True,
        default="",
        help_text="Ex. Boxer, Personnel, Visiteur.",
    )
    notes = models.TextField("Notes", blank=True, default="")
    is_active = models.BooleanField("Actif dans la recherche", default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["display_name"]
        verbose_name = "Identité faciale"
        verbose_name_plural = "Identités faciales"

    def __str__(self) -> str:
        code = self.person_code.strip()
        return f"{self.display_name}" + (f" ({code})" if code else "")

    def save(self, *args, **kwargs):
        if not self.slug.strip():
            self.slug = slugify(self.display_name)[:88] or "identite"
        super().save(*args, **kwargs)


class FaceReferenceImage(models.Model):
    """Photo visage de référence (JPEG/PNG…) liée à une identité — embedding calculé automatiquement."""

    face_identity = models.ForeignKey(
        FaceIdentity,
        on_delete=models.CASCADE,
        related_name="reference_images",
        verbose_name="Identité",
    )
    image = models.ImageField(
        "Photo référence",
        upload_to=upload_face_registry_reference,
        validators=[
            FileExtensionValidator(
                allowed_extensions=["jpg", "jpeg", "png", "webp"],
            ),
        ],
    )
    is_primary = models.BooleanField(
        "Principale",
        default=False,
        help_text="Indique une photo favorisée (affichage admin / UI futures).",
    )
    embedding = models.JSONField(
        "Vecteur d’embedding",
        null=True,
        blank=True,
        help_text="Rempli automatiquement (DeepFace) lors de la sauvegarde si activé.",
    )
    embedding_model = models.CharField(
        "Modèle d’embedding",
        max_length=64,
        blank=True,
        default="",
    )
    embedding_error = models.CharField(max_length=500, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-is_primary", "-pk"]
        verbose_name = "Photo référence visage"
        verbose_name_plural = "Photos références visages"

    def __str__(self) -> str:
        return f"{self.face_identity.display_name} — #{self.pk}"

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        try:
            from monitoring.services.face_recognition_gallery import compute_embedding_and_store

            ok = compute_embedding_and_store(int(self.pk))
            if not ok:
                logger.warning(
                    "FaceReferenceImage pk=%s embedding not computed (see embedding_error on row)",
                    self.pk,
                )
        except Exception as exc:
            logger.exception("FaceReferenceImage pk=%s embedding exception: %s", self.pk, exc)


class FacePersonProfile(models.Model):
    """Profil métier enrichi lié au code identité (`FaceIdentity.person_code`)."""

    identity_code = models.CharField(
        "Code identité (unique)",
        max_length=64,
        unique=True,
        db_index=True,
        help_text="Identique à FaceIdentity.person_code pour la jointure après match.",
    )
    full_name = models.CharField("Nom complet", max_length=200, blank=True, default="")
    age = models.PositiveSmallIntegerField("Âge", null=True, blank=True)
    role = models.CharField("Rôle / fonction", max_length=160, blank=True, default="")
    risk_level = models.CharField("Niveau de risque", max_length=64, blank=True, default="")
    notes = models.TextField("Notes profil", blank=True, default="")
    last_seen = models.DateTimeField("Dernière vue", null=True, blank=True)
    extra_info = models.JSONField("Métadonnées", null=True, blank=True)
    face_identity = models.ForeignKey(
        FaceIdentity,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="person_profiles",
        verbose_name="Identité liée (optionnel)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["identity_code"]
        verbose_name = "Profil personne (visage)"
        verbose_name_plural = "Profils personnes (visage)"

    def __str__(self) -> str:
        return f"{self.identity_code} · {self.full_name or '—'}"


class VideoSource(models.Model):
    """Source vidéo : webcam locale, flux RTSP ou fichier."""

    class SourceType(models.TextChoices):
        WEBCAM = "webcam", "Webcam"
        RTSP = "rtsp", "Flux RTSP / IP"
        FILE = "file", "Fichier vidéo"

    name = models.CharField("Nom", max_length=120)
    source_type = models.CharField(
        "Type",
        max_length=20,
        choices=SourceType.choices,
        default=SourceType.WEBCAM,
    )
    # RTSP URL ou chemin absolu/relatif vers un fichier vidéo (si pas d’upload)
    url = models.CharField("URL / chemin", max_length=500, blank=True)
    video_file = models.FileField(
        "Fichier vidéo (PC)",
        upload_to="sources/videos/%Y/%m/",
        blank=True,
        null=True,
        validators=[
            FileExtensionValidator(
                allowed_extensions=["mp4", "mov", "avi", "webm", "mkv"],
            )
        ],
        help_text="Téléversez un MP4 (ou MOV, AVI, WebM). Prioritaire sur le champ chemin pour le type « Fichier vidéo ».",
    )
    webcam_index = models.PositiveSmallIntegerField("Index webcam", default=0)
    crowd_threshold = models.PositiveSmallIntegerField(
        "Seuil alerte foule (nb personnes)",
        default=12,
        help_text="Nombre de personnes détectées déclenchant une alerte foule.",
    )
    fast_movement_threshold = models.FloatField(
        "Seuil mouvement rapide (px/frame)",
        default=18.0,
        help_text="Vitesse max (centre de la boîte) d’au moins une personne suivie ; au-delà → alerte course / mouvement brusque. Plus bas = plus sensible.",
    )
    is_active = models.BooleanField("Actif", default=True)
    notes = models.TextField("Notes", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_active", "name"]
        verbose_name = "Source vidéo"
        verbose_name_plural = "Sources vidéo"

    def __str__(self):
        return self.name

    def get_file_path(self) -> str:
        """Chemin absolu pour OpenCV : upload serveur ou champ URL / chemin."""
        if self.video_file:
            return self.video_file.path
        return (self.url or "").strip()


class Alert(models.Model):
    """Événement détecté (foule dense, mouvement anormal, etc.)."""

    class AlertType(models.TextChoices):
        CROWD = "crowd", "Foule dense"
        FAST_MOVEMENT = "fast_movement", "Mouvement rapide / dispersion"
        HIGH_OVERLAP = "high_overlap", "Regroupement serré (piste bagarre)"
        THEFT_SUSPICIOUS = (
            "theft_suspicious",
            "Piste vol (sac / objet — changement de proximité)",
        )
        FIGHT = "fight", "Combat détecté (analyse vidéo)"
        WEAPON = "weapon", "Arme détectée (fusil / couteau)"
        SYSTEM = "system", "Système"

    source = models.ForeignKey(
        VideoSource,
        on_delete=models.CASCADE,
        related_name="alerts",
        verbose_name="Source",
    )
    alert_type = models.CharField("Type", max_length=32, choices=AlertType.choices)
    message = models.CharField("Message", max_length=500)
    person_count = models.PositiveIntegerField("Personnes", null=True, blank=True)
    confidence_max = models.FloatField("Confiance max", null=True, blank=True)
    metadata = models.JSONField("Métadonnées", default=dict, blank=True)
    thumbnail = models.ImageField(
        "Aperçu",
        upload_to="alerts/%Y/%m/",
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Alerte"
        verbose_name_plural = "Alertes"

    def __str__(self):
        return f"{self.get_alert_type_display()} — {self.source.name} ({self.created_at:%Y-%m-%d %H:%M})"


class RoadDamageAnalysis(models.Model):
    """Historique d'analyse road damage enrichie (XAI, ticketing, geotag, plan)."""

    route_name = models.CharField("Route", max_length=160, blank=True, default="")
    source_type = models.CharField("Source type", max_length=32, default="unknown")
    source_ref = models.CharField("Source ref", max_length=600, blank=True, default="")
    annotated_image = models.CharField("Image annotee", max_length=600, blank=True, default="")
    heatmap_image = models.CharField("Heatmap XAI", max_length=600, blank=True, default="")
    verdict = models.CharField("Verdict", max_length=32, default="no_damage")
    risk_score = models.FloatField("Score risque", default=0.0)
    risk_level = models.CharField("Niveau risque", max_length=32, default="low")
    confidence_max = models.FloatField("Confiance max", default=0.0)
    damage_ratio = models.FloatField("Ratio dommages", default=0.0)
    priority = models.CharField("Priorite maintenance", max_length=32, default="medium")
    estimated_budget_min = models.PositiveIntegerField("Budget min", default=0)
    estimated_budget_max = models.PositiveIntegerField("Budget max", default=0)
    action_plan = models.JSONField("Plan d'action", default=list, blank=True)
    quality = models.JSONField("Qualite donnees", default=dict, blank=True)
    xai_summary = models.JSONField("XAI", default=dict, blank=True)
    llm_report = models.TextField("Rapport LLM", blank=True, default="")
    ticket_id = models.CharField("Ticket maintenance", max_length=64, blank=True, default="")
    ticket_deadline_days = models.PositiveSmallIntegerField("Delai ticket (jours)", default=30)
    latitude = models.FloatField("Latitude", null=True, blank=True)
    longitude = models.FloatField("Longitude", null=True, blank=True)
    anti_fp_flag = models.BooleanField("Anti faux-positifs corrige", default=False)
    pdf_report_path = models.CharField("PDF rapport", max_length=600, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Analyse road damage"
        verbose_name_plural = "Analyses road damage"

    def __str__(self):
        route = self.route_name or "Route inconnue"
        return f"{route} — {self.risk_level} ({self.created_at:%Y-%m-%d %H:%M})"


class CitizenReclamation(models.Model):
    """Signalement citoyen (app mobile) : photo / vidéo + géoloc + catégorie."""

    class Category(models.TextChoices):
        POTHOLE = "pothole", "Nid de poule"
        GARBAGE = "garbage", "Déchets"
        SMOKE = "smoke", "Fumée"
        FLOOD = "flood", "Inondation"

    class MediaType(models.TextChoices):
        IMAGE = "image", "Image"
        VIDEO = "video", "Vidéo"

    media = models.FileField(
        "Média",
        upload_to="reclamations/%Y/%m/",
        validators=[
            FileExtensionValidator(
                allowed_extensions=["jpg", "jpeg", "png", "webp", "mp4", "mov", "m4v"],
            )
        ],
    )
    media_type = models.CharField("Type média", max_length=10, choices=MediaType.choices)
    latitude = models.FloatField("Latitude")
    longitude = models.FloatField("Longitude")
    description = models.TextField("Description", blank=True, default="")
    category = models.CharField("Catégorie", max_length=20, choices=Category.choices)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Réclamation citoyenne"
        verbose_name_plural = "Réclamations citoyennes"

    def __str__(self):
        return f"{self.get_category_display()} — {self.created_at:%Y-%m-%d %H:%M}"


class FireCameraConfig(models.Model):
    """
    Une ligne (singleton via store.get_fire_config) : IP + préférences caméra ESP32-CAM.

    Les réglages image sont dans ``ui_prefs`` (clés = noms paramètres ESP ``/cmd``), notamment :
    framesize, quality, brightness, contrast, saturation, sharpness,
    awb, awb_gain, wb_mode, aec, aec2, ae_level, agc, gainceiling,
    bpc, wpc, raw_gma, lenc, dcw, hmirror, vflip, special_effect, led_intensity.
    """

    esp_ip = models.CharField("IP ESP32-CAM", max_length=128, blank=True, default="")
    ui_prefs = models.JSONField("Préférences (JSON)", blank=True, default=dict)
    road_live_model_key = models.CharField(
        "Clé modèle road damage live (YOLO drone/ESP)",
        max_length=96,
        blank=True,
        default="",
        help_text="Pour /reclamations/…/ai/?live=1 — clé registre ROAD_VISION_MODELS (ex. yolov8n_100ep).",
    )
    road_live_conf = models.FloatField(
        "Confiance seuil inference live ",
        blank=True,
        null=True,
        help_text="Seuil 0–1 pour esp-analyze live ; vide = valeur Django RECLAMATION_VISION_CONF_DEFAULT.",
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Configuration caméra feu ESP32"
        verbose_name_plural = "Configuration caméra feu ESP32"

    def __str__(self) -> str:
        return f"Feu ESP — {self.esp_ip or '—'}"


class WasteReport(models.Model):
    """Rapport détection déchets rue — carte, SMS, email, statistiques."""

    class Severity(models.TextChoices):
        LOW = "Low", "Low"
        MEDIUM = "Medium", "Medium"
        HIGH = "High", "High"

    class SmsStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        MOCK_SENT = "mock_sent", "Mock sent"
        SENT = "sent", "Sent"
        FAILED = "failed", "Failed"

    class EmailStatus(models.TextChoices):
        PENDING = "pending", "Pending"
        MOCK_SENT = "mock_sent", "Mock sent"
        SENT = "sent", "Sent"
        FAILED = "failed", "Failed"

    source_image = models.ImageField(
        "Image téléversée",
        upload_to="waste_reports/%Y/%m/",
        blank=True,
        null=True,
        validators=[
            FileExtensionValidator(
                allowed_extensions=["jpg", "jpeg", "png", "webp"],
            ),
        ],
    )
    source_image_url = models.URLField(
        "URL image source",
        max_length=1024,
        blank=True,
        default="",
    )
    annotated_image_url = models.CharField(
        "URL image annotée",
        max_length=512,
        blank=True,
        default="",
    )
    location = models.CharField(
        "Localisation (résumé)",
        max_length=400,
        blank=True,
        default="",
        help_text="Souvent « ville, adresse » pour affichage rapide.",
    )
    city = models.CharField("Ville", max_length=160, blank=True, default="")
    address = models.CharField("Adresse / zone", max_length=400, blank=True, default="")
    latitude = models.FloatField("Latitude", null=True, blank=True)
    longitude = models.FloatField("Longitude", null=True, blank=True)
    notes = models.TextField("Notes", blank=True, default="")
    detected_objects = models.JSONField("Détections", default=list, blank=True)
    object_count = models.PositiveIntegerField("Nombre d’objets", default=0)
    max_confidence = models.FloatField("Confiance max (0–1)", default=0.0)
    severity = models.CharField(
        "Gravité",
        max_length=16,
        choices=Severity.choices,
        default=Severity.MEDIUM,
    )
    recommended_action = models.TextField("Action recommandée", blank=True, default="")
    sensitive_area_boost = models.BooleanField("Zone sensible (+1 niveau)", default=False)
    sms_status = models.CharField(
        "Statut SMS",
        max_length=24,
        choices=SmsStatus.choices,
        default=SmsStatus.PENDING,
    )
    email_status = models.CharField(
        "Statut email",
        max_length=24,
        choices=EmailStatus.choices,
        default=EmailStatus.PENDING,
    )
    sms_body = models.TextField("Texte SMS", blank=True, default="")
    sms_twilio_sid = models.CharField("Twilio SID", max_length=64, blank=True, default="")
    sms_detail = models.TextField("Détail SMS / erreur", blank=True, default="")
    email_detail = models.TextField("Détail email / erreur", blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Rapport déchets (Waste Report)"
        verbose_name_plural = "Rapports déchets (Waste)"

    def __str__(self) -> str:
        loc = self.city or self.location or "—"
        return f"Déchets #{self.pk} · {loc} ({self.created_at:%Y-%m-%d %H:%M})"


class UavStructuralAnalysis(models.Model):
    """Historique analyse structurelle UAV (CNN Keras + métriques de risque)."""

    class Zone(models.TextChoices):
        A = "A", "Zone A"
        B = "B", "Zone B"
        C = "C", "Zone C"

    created_at = models.DateTimeField(auto_now_add=True)
    primary_filename = models.CharField("Fichier image UAV", max_length=260, default="")
    comparison_filename = models.CharField("Image comparaison", max_length=260, blank=True, default="")
    zone = models.CharField("Zone", max_length=4, choices=Zone.choices, default=Zone.B)
    surface_m2 = models.FloatField("Surface estimée (m²)", default=0.0)

    prediction_index = models.PositiveSmallIntegerField(default=0)
    prediction_label = models.CharField(max_length=64, default="")
    probabilities = models.JSONField(default=list, blank=True)
    confidence = models.FloatField(default=0.0)

    comparison_index = models.PositiveSmallIntegerField(null=True, blank=True)
    comparison_label = models.CharField(max_length=64, blank=True, default="")
    comparison_probabilities = models.JSONField(default=list, blank=True)
    comparison_confidence = models.FloatField(null=True, blank=True)

    primary_original_url = models.CharField(max_length=512, blank=True, default="")
    primary_gradcam_url = models.CharField(max_length=512, blank=True, default="")
    primary_saliency_url = models.CharField(max_length=512, blank=True, default="")
    comparison_original_url = models.CharField(max_length=512, blank=True, default="")
    comparison_gradcam_url = models.CharField(max_length=512, blank=True, default="")
    comparison_saliency_url = models.CharField(max_length=512, blank=True, default="")

    risk_score = models.FloatField(default=0.0)
    operational_level = models.CharField(max_length=32, blank=True, default="")
    operational_level_key = models.CharField(max_length=24, blank=True, default="")
    co2_estimate_kg = models.FloatField(default=0.0)
    recommendations = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Analyse UAV (structure)"
        verbose_name_plural = "Analyses UAV (structure)"

    def __str__(self) -> str:
        return f"UAV #{self.pk} · {self.prediction_label} ({self.created_at:%Y-%m-%d %H:%M})"
