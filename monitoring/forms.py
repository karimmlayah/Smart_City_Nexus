from django import forms
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator

from .models import VideoSource
from .services.fight_classifier import is_youtube_url

_video_extensions = FileExtensionValidator(
    allowed_extensions=["mp4", "mov", "avi", "webm", "mkv"],
)
_image_extensions = FileExtensionValidator(
    allowed_extensions=["jpg", "jpeg", "png", "webp"],
)


class FightAnalyzeForm(forms.Form):
    """URL YouTube ou fichier vidéo local pour analyse combat / non-combat."""

    fight_weights = forms.ChoiceField(
        label="Combat — classifier (.pt classify)",
        required=True,
        choices=[],
        widget=forms.Select(attrs={"class": "fight-model-select fight-model-select--combat"}),
    )
    weapon_weights = forms.ChoiceField(
        label="Armes & guns — détection YOLO (.pt detect)",
        required=True,
        choices=[],
        widget=forms.Select(attrs={"class": "fight-model-select fight-model-select--weapon"}),
    )
    youtube_url = forms.CharField(
        label="URL YouTube",
        max_length=500,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "https://www.youtube.com/watch?v=…",
                "class": "fight-url-input",
                "autocomplete": "off",
                "inputmode": "url",
            }
        ),
    )
    video_file = forms.FileField(
        label="Ou vidéo depuis votre PC",
        required=False,
        validators=[_video_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/*",
                "class": "fight-file-input",
            }
        ),
    )

    def __init__(
        self,
        *args,
        fight_choices: list[tuple[str, str]] | None = None,
        weapon_choices: list[tuple[str, str]] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if fight_choices is not None:
            self.fields["fight_weights"].choices = fight_choices
        if weapon_choices is not None:
            self.fields["weapon_weights"].choices = weapon_choices

    def clean(self):
        data = super().clean()
        url = (data.get("youtube_url") or "").strip()
        vf = data.get("video_file")
        has_file = bool(
            vf
            and getattr(vf, "name", "")
            and getattr(vf, "size", 0) > 0
        )

        if has_file and url:
            raise ValidationError("Choisissez soit une URL YouTube, soit un fichier vidéo, pas les deux.")

        if not has_file and not url:
            raise ValidationError("Indiquez une URL YouTube ou sélectionnez un fichier vidéo.")

        max_mb = int(getattr(settings, "FIGHT_UPLOAD_MAX_MB", 200))
        if has_file and vf.size > max_mb * 1024 * 1024:
            raise ValidationError(f"Fichier trop volumineux (maximum {max_mb} Mo).")

        if url and not has_file:
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            if not is_youtube_url(url):
                raise ValidationError(
                    {"youtube_url": ["Indiquez un lien YouTube (youtube.com ou youtu.be)."]}
                )
            data["youtube_url"] = url

        return data


class FusionHubForm(FightAnalyzeForm):
    """Même entrée vidéo que la page combat + activation indépendante combat / armes."""

    def __init__(
        self,
        *args,
        fight_choices: list[tuple[str, str]] | None = None,
        weapon_choices: list[tuple[str, str]] | None = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            fight_choices=fight_choices,
            weapon_choices=weapon_choices,
            **kwargs,
        )

    use_fight = forms.BooleanField(
        label="Activer l’analyse combat",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "fusion-model-check"}),
    )
    use_weapon = forms.BooleanField(
        label="Activer la détection d’armes",
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={"class": "fusion-model-check"}),
    )
    use_openai_threat = forms.BooleanField(
        label="Activer API Chat Threat",
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={"class": "fusion-model-check"}),
    )
    fusion_image_file = forms.FileField(
        label="Ou image (JPEG / PNG / WebP)",
        required=False,
        validators=[_image_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp",
                "class": "fusion-image-file-input",
            }
        ),
    )

    def clean(self):
        """
        Source : exactement une option parmi YouTube, vidéo, image.
        (Ne pas appeler FightAnalyzeForm.clean — logique étendue ici.)
        """
        if self.errors:
            return self.cleaned_data
        data = self.cleaned_data
        url = (data.get("youtube_url") or "").strip()
        vf = data.get("video_file")
        imgf = data.get("fusion_image_file")
        has_vid = bool(
            vf
            and getattr(vf, "name", "")
            and getattr(vf, "size", 0) > 0
        )
        has_img = bool(
            imgf
            and getattr(imgf, "name", "")
            and getattr(imgf, "size", 0) > 0
        )
        n_sources = sum([1 if url else 0, 1 if has_vid else 0, 1 if has_img else 0])
        if n_sources != 1:
            raise ValidationError(
                "Indiquez exactement une source : URL YouTube, fichier vidéo, ou fichier image."
            )

        max_mb = int(getattr(settings, "FIGHT_UPLOAD_MAX_MB", 200))
        if has_vid and vf.size > max_mb * 1024 * 1024:
            raise ValidationError(f"Fichier trop volumineux (maximum {max_mb} Mo).")
        if has_img and imgf.size > max_mb * 1024 * 1024:
            raise ValidationError(f"Image trop volumineuse (maximum {max_mb} Mo).")

        if url and not has_vid and not has_img:
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            if not is_youtube_url(url):
                raise ValidationError(
                    {"youtube_url": ["Indiquez un lien YouTube (youtube.com ou youtu.be)."]}
                )
            data["youtube_url"] = url

        if (
            not data.get("use_fight")
            and not data.get("use_weapon")
            and not data.get("use_openai_threat")
        ):
            raise ValidationError(
                "Cochez au moins une option : combat, armes, ou Chat API threat."
            )
        return data


class WeaponAnalyzeForm(forms.Form):
    """URL YouTube ou fichier vidéo local pour analyse arme / non-arme (modèle interne)."""

    youtube_url = forms.CharField(
        label="URL YouTube",
        max_length=500,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "https://www.youtube.com/watch?v=…",
                "class": "fight-url-input",
                "autocomplete": "off",
                "inputmode": "url",
            }
        ),
    )
    video_file = forms.FileField(
        label="Ou vidéo depuis votre PC",
        required=False,
        validators=[_video_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/*",
                "class": "fight-file-input",
            }
        ),
    )
    image_file = forms.FileField(
        label="Ou image depuis votre PC",
        required=False,
        validators=[_image_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "image/jpeg,image/png,image/webp,image/*",
                "class": "fight-file-input",
            }
        ),
    )

    def clean(self):
        data = super().clean()
        url = (data.get("youtube_url") or "").strip()
        vf = data.get("video_file")
        img = data.get("image_file")
        has_file = bool(
            vf
            and getattr(vf, "name", "")
            and getattr(vf, "size", 0) > 0
        )
        has_image = bool(
            img
            and getattr(img, "name", "")
            and getattr(img, "size", 0) > 0
        )

        picked = int(bool(url)) + int(has_file) + int(has_image)
        if picked > 1:
            raise ValidationError(
                "Choisissez une seule source: URL YouTube, vidéo ou image."
            )

        if picked == 0:
            raise ValidationError(
                "Indiquez une URL YouTube, ou sélectionnez une vidéo, ou une image."
            )

        max_mb = int(getattr(settings, "FIGHT_UPLOAD_MAX_MB", 200))
        if has_file and vf.size > max_mb * 1024 * 1024:
            raise ValidationError(f"Fichier trop volumineux (maximum {max_mb} Mo).")
        img_max_mb = int(getattr(settings, "WEAPON_IMAGE_UPLOAD_MAX_MB", 25))
        if has_image and img.size > img_max_mb * 1024 * 1024:
            raise ValidationError(f"Image trop volumineuse (maximum {img_max_mb} Mo).")

        if url and not has_file and not has_image:
            if not url.startswith(("http://", "https://")):
                url = f"https://{url}"
            if not is_youtube_url(url):
                raise ValidationError(
                    {"youtube_url": ["Indiquez un lien YouTube (youtube.com ou youtu.be)."]}
                )
            data["youtube_url"] = url

        return data


class SightengineWeaponForm(forms.Form):
    """Image, URL image, ou vidéo (fichier / URL) pour analyse Sightengine."""

    image_file = forms.FileField(
        label="Image depuis votre PC",
        required=False,
        validators=[_image_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "image/jpeg,image/png,image/webp,image/*",
                "class": "fight-file-input",
            }
        ),
    )
    image_url = forms.CharField(
        label="Ou URL d'image",
        max_length=500,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "https://… (jpg/png)",
                "class": "fight-url-input",
                "autocomplete": "off",
                "inputmode": "url",
            }
        ),
    )
    video_file = forms.FileField(
        label="Vidéo depuis votre PC",
        required=False,
        validators=[_video_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/*",
                "class": "fight-file-input",
            }
        ),
    )
    video_url = forms.CharField(
        label="Ou URL vidéo directe (mp4, etc.)",
        max_length=800,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "https://…/video.mp4 (pas YouTube)",
                "class": "fight-url-input",
                "autocomplete": "off",
                "inputmode": "url",
            }
        ),
    )
    youtube_url = forms.CharField(
        label="Ou URL YouTube (sera téléchargée)",
        max_length=800,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "https://www.youtube.com/watch?v=…",
                "class": "fight-url-input",
                "autocomplete": "off",
                "inputmode": "url",
            }
        ),
    )

    def clean(self):
        data = super().clean()
        img_url = (data.get("image_url") or "").strip()
        vid_url = (data.get("video_url") or "").strip()
        img = data.get("image_file")
        vf = data.get("video_file")
        yt = (data.get("youtube_url") or "").strip()

        has_image = bool(
            img
            and getattr(img, "name", "")
            and getattr(img, "size", 0) > 0
        )
        has_image_url = bool(img_url)
        has_video = bool(
            vf
            and getattr(vf, "name", "")
            and getattr(vf, "size", 0) > 0
        )
        has_video_url = bool(vid_url)
        has_youtube = bool(yt)

        modes = [
            int(has_image),
            int(has_image_url),
            int(has_video),
            int(has_video_url),
            int(has_youtube),
        ]
        if sum(modes) > 1:
            raise ValidationError(
                "Choisissez une seule source: image, URL image, vidéo ou URL vidéo."
            )
        if sum(modes) == 0:
            raise ValidationError(
                "Indiquez une image, une URL image, une vidéo, une URL vidéo ou une URL YouTube."
            )

        img_max_mb = int(getattr(settings, "WEAPON_IMAGE_UPLOAD_MAX_MB", 25))
        if has_image and img.size > img_max_mb * 1024 * 1024:
            raise ValidationError(f"Image trop volumineuse (maximum {img_max_mb} Mo).")

        vid_max_mb = int(getattr(settings, "FIGHT_UPLOAD_MAX_MB", 200))
        if has_video and vf.size > vid_max_mb * 1024 * 1024:
            raise ValidationError(f"Vidéo trop volumineuse (maximum {vid_max_mb} Mo).")

        if has_image_url and not img_url.startswith(("http://", "https://")):
            img_url = f"https://{img_url}"
        data["image_url"] = img_url

        if has_video_url and not vid_url.startswith(("http://", "https://")):
            vid_url = f"https://{vid_url}"
        data["video_url"] = vid_url

        if has_youtube and not yt.startswith(("http://", "https://")):
            yt = f"https://{yt}"
        data["youtube_url"] = yt

        return data


class RoadDamageAnalyzeForm(forms.Form):
    """Image, URL image, video locale ou URL video pour detection road damage."""

    image_file = forms.FileField(
        label="Image depuis votre PC",
        required=False,
        validators=[_image_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "image/jpeg,image/png,image/webp,image/*",
                "class": "fight-file-input",
            }
        ),
    )
    image_url = forms.CharField(
        label="Ou URL d'image",
        max_length=500,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "https://... (jpg/png)",
                "class": "fight-url-input",
                "autocomplete": "off",
                "inputmode": "url",
            }
        ),
    )
    video_file = forms.FileField(
        label="Ou video depuis votre PC",
        required=False,
        validators=[_video_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/*",
                "class": "fight-file-input",
            }
        ),
    )
    video_url = forms.CharField(
        label="Ou URL video (YouTube ou lien direct)",
        max_length=800,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "https://www.youtube.com/watch?v=... ou https://.../video.mp4",
                "class": "fight-url-input",
                "autocomplete": "off",
                "inputmode": "url",
            }
        ),
    )
    route_name = forms.CharField(
        label="Nom de route / zone",
        max_length=160,
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "Ex: Avenue Habib Bourguiba, Sfax",
                "class": "fight-url-input",
                "autocomplete": "off",
            }
        ),
    )
    latitude = forms.FloatField(
        label="Latitude (optionnel)",
        required=False,
        widget=forms.NumberInput(attrs={"step": "0.000001", "class": "fight-url-input"}),
    )
    longitude = forms.FloatField(
        label="Longitude (optionnel)",
        required=False,
        widget=forms.NumberInput(attrs={"step": "0.000001", "class": "fight-url-input"}),
    )

    def clean(self):
        data = super().clean()
        img_url = (data.get("image_url") or "").strip()
        vid_url = (data.get("video_url") or "").strip()
        img = data.get("image_file")
        vf = data.get("video_file")

        has_image = bool(img and getattr(img, "name", "") and getattr(img, "size", 0) > 0)
        has_image_url = bool(img_url)
        has_video = bool(vf and getattr(vf, "name", "") and getattr(vf, "size", 0) > 0)
        has_video_url = bool(vid_url)

        modes = [int(has_image), int(has_image_url), int(has_video), int(has_video_url)]
        if sum(modes) > 1:
            raise ValidationError(
                "Choisissez une seule source: image, URL image, video ou URL video."
            )
        if sum(modes) == 0:
            raise ValidationError(
                "Indiquez une image, une URL image, une video ou une URL video."
            )

        img_max_mb = int(getattr(settings, "WEAPON_IMAGE_UPLOAD_MAX_MB", 25))
        if has_image and img.size > img_max_mb * 1024 * 1024:
            raise ValidationError(f"Image trop volumineuse (maximum {img_max_mb} Mo).")

        vid_max_mb = int(getattr(settings, "FIGHT_UPLOAD_MAX_MB", 200))
        if has_video and vf.size > vid_max_mb * 1024 * 1024:
            raise ValidationError(f"Video trop volumineuse (maximum {vid_max_mb} Mo).")

        if has_image_url and not img_url.startswith(("http://", "https://")):
            img_url = f"https://{img_url}"
        data["image_url"] = img_url

        if has_video_url and not vid_url.startswith(("http://", "https://")):
            vid_url = f"https://{vid_url}"
        data["video_url"] = vid_url

        lat = data.get("latitude")
        lon = data.get("longitude")
        if lat is not None and not (-90.0 <= float(lat) <= 90.0):
            raise ValidationError({"latitude": ["Latitude invalide (-90 a 90)."]})
        if lon is not None and not (-180.0 <= float(lon) <= 180.0):
            raise ValidationError({"longitude": ["Longitude invalide (-180 a 180)."]})

        return data


class SmartCrowdSafetyForm(forms.Form):
    """Test web Smart Crowd Safety AI : upload vidéo + limite de frames."""

    video_file = forms.FileField(
        label="Vidéo à analyser",
        required=True,
        validators=[_video_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/*",
                "class": "fight-file-input",
            }
        ),
    )
    max_frames = forms.IntegerField(
        label="Frames max à analyser (0 = défaut rapide)",
        required=False,
        min_value=0,
        initial=0,
        widget=forms.NumberInput(
            attrs={
                "min": 0,
                "max": 5000,
                "class": "fight-url-input",
                "style": "max-width:8rem",
            }
        ),
    )

    def clean(self):
        data = super().clean()
        vf = data.get("video_file")
        if not vf or not getattr(vf, "size", 0):
            raise ValidationError("Sélectionnez un fichier vidéo.")

        max_mb = int(getattr(settings, "SMARTCROWD_UPLOAD_MAX_MB", 150))
        if vf.size > max_mb * 1024 * 1024:
            raise ValidationError(f"Vidéo trop volumineuse (maximum {max_mb} Mo).")

        raw = data.get("max_frames")
        mf = int(raw) if raw is not None else 0
        cap = int(getattr(settings, "SMARTCROWD_WEB_MAX_FRAMES_CAP", 2500))
        default = int(getattr(settings, "SMARTCROWD_WEB_MAX_FRAMES_DEFAULT", 400))
        if mf <= 0:
            data["max_frames"] = default
        else:
            data["max_frames"] = min(mf, cap)
        return data


class QuickVideoForm(forms.Form):
    """Ajout rapide : uniquement la vidéo (nom dérivé du fichier si vide)."""

    name = forms.CharField(
        label="Nom affiché (optionnel)",
        max_length=120,
        required=False,
        widget=forms.TextInput(
            attrs={"placeholder": "Laissez vide pour utiliser le nom du fichier"}
        ),
    )
    video_file = forms.FileField(
        label="Fichier vidéo",
        validators=[_video_extensions],
        widget=forms.FileInput(
            attrs={
                "accept": "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/*",
            }
        ),
    )


class QuickVideoEditForm(forms.ModelForm):
    """Modifier une source fichier : nom, remplacer la vidéo, activer / désactiver."""

    class Meta:
        model = VideoSource
        fields = ["name", "video_file", "is_active", "fast_movement_threshold"]
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "Nom de la vidéo"}),
            "video_file": forms.ClearableFileInput(
                attrs={
                    "accept": "video/mp4,video/quicktime,video/x-msvideo,video/webm,video/*",
                }
            ),
            "fast_movement_threshold": forms.NumberInput(
                attrs={"min": 4, "max": 80, "step": "1"}
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk and self.instance.source_type != VideoSource.SourceType.FILE:
            self.fields.pop("video_file", None)

    def clean(self):
        data = super().clean()
        inst = self.instance
        if inst.source_type != VideoSource.SourceType.FILE:
            return data
        vf = data.get("video_file")
        cleared = vf is False
        has_new = vf not in (None, False) and bool(getattr(vf, "name", ""))
        has_stored = (
            not cleared
            and bool(getattr(inst.video_file, "name", "") or "")
        )
        has_url = bool((inst.url or "").strip())
        if not (has_new or has_stored or has_url):
            raise ValidationError("Il faut une vidéo ou un chemin sur le serveur.")
        return data
