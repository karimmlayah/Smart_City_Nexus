from django import forms

from monitoring.models import SiteConfiguration


class PlatformSettingsForm(forms.ModelForm):
    smtp_password = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False, attrs={"placeholder": "••••••••", "autocomplete": "new-password"}),
        label="SMTP password",
    )
    twilio_auth_token = forms.CharField(
        required=False,
        widget=forms.PasswordInput(render_value=False, attrs={"placeholder": "••••••••", "autocomplete": "new-password"}),
        label="Twilio auth token",
    )

    class Meta:
        model = SiteConfiguration
        exclude = ("updated_at",)
        widgets = {
            "role_permissions_summary": forms.Textarea(attrs={"rows": 3}),
            "report_footer_text": forms.TextInput(),
            "primary_accent_color": forms.TextInput(attrs={"type": "color"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            css = "ps-field__input"
            if isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs["class"] = "ps-field__check"
            elif isinstance(field.widget, forms.Select):
                field.widget.attrs["class"] = css
            elif not isinstance(field.widget, forms.PasswordInput):
                field.widget.attrs.setdefault("class", css)

    def save(self, commit=True):
        instance = super().save(commit=False)
        if self.instance.pk:
            if not self.cleaned_data.get("smtp_password"):
                instance.smtp_password = self.instance.smtp_password
            if not self.cleaned_data.get("twilio_auth_token"):
                instance.twilio_auth_token = self.instance.twilio_auth_token
        if commit:
            instance.save()
        return instance
