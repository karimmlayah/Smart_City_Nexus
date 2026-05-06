from django.urls import path

from monitoring import waste_api

app_name = "waste_api"

urlpatterns = [
    path("detect/", waste_api.waste_detect, name="waste_detect"),
    path("send-sms/", waste_api.waste_send_sms, name="waste_send_sms"),
    path("send-email/", waste_api.waste_send_email, name="waste_send_email"),
    path("reports/", waste_api.waste_reports_dispatch, name="waste_reports"),
    path("statistics/", waste_api.waste_statistics, name="waste_statistics"),
]
