from django.urls import path

from . import (
    assistant_api,
    camera_settings_api,
    fire_camera_views,
    mayor_mission_views,
    platform_settings_views,
    traffic_nexus_views,
    traffic_signal_views,
    traffic_violation_views,
    uav_views,
    views,
    waste_views,
)

app_name = "monitoring"

urlpatterns = [
    path("", views.homepage, name="home"),
    path("home/", views.homepage, name="home_page"),
    path("surveillance/", views.surveillance_dashboard, name="surveillance_dashboard"),
    path("road-damage-test/", views.road_damage_test, name="road_damage_test"),
    path(
        "waste/street-detection/",
        waste_views.waste_street_detection_page,
        name="waste_street_detection",
    ),
    path("uav/dashboard/", uav_views.uav_dashboard_page, name="uav_dashboard"),
    path(
        "traffic/nexus/",
        traffic_nexus_views.traffic_nexus_dashboard,
        name="traffic_nexus_dashboard",
    ),
    path(
        "traffic/signal-control/",
        traffic_signal_views.traffic_signal_control_page,
        name="traffic_signal_control",
    ),
    path(
        "traffic/violations/",
        traffic_violation_views.traffic_violation_page,
        name="traffic_violation",
    ),
    path(
        "traffic/violations/analyze/",
        traffic_violation_views.traffic_violation_analyze,
        name="traffic_violation_analyze",
    ),
    path(
        "traffic/violations/api/suggest-stop-line/",
        traffic_violation_views.traffic_violation_suggest_stop_line,
        name="traffic_violation_suggest_stop_line",
    ),
    path(
        "traffic/violations/api/check-violations/",
        traffic_violation_views.traffic_violation_check_violations,
        name="traffic_violation_check_violations",
    ),
    path(
        "traffic/violations/api/realtime-frame/",
        traffic_violation_views.traffic_violation_realtime_frame,
        name="traffic_violation_realtime_frame",
    ),
    path(
        "traffic/violations/output/<path:filename>",
        traffic_violation_views.traffic_violation_output,
        name="traffic_violation_output",
    ),
    path(
        "traffic/violations/dashboard/<str:token>/",
        traffic_violation_views.violation_dashboard,
        name="violation_dashboard",
    ),
    path(
        "traffic/violations/download/<str:token>/",
        traffic_violation_views.violation_download_reports,
        name="violation_download_reports",
    ),
    path(
        "reclamations/<int:pk>/ai/",
        views.reclamation_ai_detail,
        name="reclamation_ai_detail",
    ),
    path(
        "reclamations/<int:pk>/ai/live/",
        views.reclamation_ai_live,
        name="reclamation_ai_live",
    ),
    path(
        "reclamations/<int:pk>/ai/esp-analyze/",
        views.reclamation_ai_esp_analyze,
        name="reclamation_ai_esp_analyze",
    ),
    path("fusion/analyze-frame/", views.fusion_analyze_frame, name="fusion_analyze_frame"),
    path("fusion/frame/", views.fusion_live_frame, name="fusion_live_frame"),
    path("fusion/", views.fusion_hub, name="fusion_hub"),
    path(
        "camera/settings/",
        fire_camera_views.fire_camera_settings,
        name="camera_settings",
    ),
    path(
        "fire-camera/settings/",
        fire_camera_views.fire_camera_settings,
        name="fire_camera_settings",
    ),
    path(
        "fire-camera/settings/patch/",
        fire_camera_views.fire_camera_settings_patch,
        name="fire_camera_settings_patch",
    ),
    path(
        "fire-camera/detect/",
        fire_camera_views.fire_camera_detect,
        name="fire_camera_detect",
    ),
    path(
        "fire-camera/analyze/",
        fire_camera_views.fire_camera_analyze,
        name="fire_camera_analyze",
    ),
    path(
        "fire-camera/twilio-test/",
        fire_camera_views.fire_camera_twilio_test,
        name="fire_camera_twilio_test",
    ),
    path(
        "fire-camera/api/state/",
        fire_camera_views.fire_camera_settings_state,
        name="fire_camera_settings_state",
    ),
    path(
        "api/esp/capture/",
        fire_camera_views.esp_capture_proxy,
        name="esp_capture_proxy",
    ),
    path(
        "api/esp/stream/",
        fire_camera_views.esp_stream_proxy,
        name="esp_stream_proxy",
    ),
    path(
        "camera/settings/update/",
        camera_settings_api.camera_settings_update,
        name="camera_settings_update",
    ),
    path(
        "camera/settings/sync-full/",
        camera_settings_api.camera_settings_sync_full,
        name="camera_settings_sync_full",
    ),
    path(
        "camera/capture-proxy/",
        camera_settings_api.camera_capture_proxy,
        name="camera_capture_proxy",
    ),
    path("introduction/", views.introduction_page, name="introduction"),
    path("api/ar/live-stats/", views.ar_live_stats, name="ar_live_stats"),
    path("api/ar/live-info/", views.ar_live_info, name="ar_live_info"),
    path("api/ar/modules/", views.ar_modules, name="ar_modules"),
    path("ai-dashboard/", views.ai_dashboard, name="ai_dashboard"),
    path(
        "ai-dashboard/api/summary/",
        views.ai_dashboard_api_summary,
        name="ai_dashboard_api_summary",
    ),
    path(
        "ai-dashboard/api/chat/",
        views.ai_dashboard_chatbot,
        name="ai_dashboard_chatbot",
    ),
    path(
        "api/assistant/groq/",
        assistant_api.assistant_groq,
        name="assistant_groq",
    ),
    path("mayor-mission/", mayor_mission_views.mayor_mission_page, name="mayor_mission"),
    path(
        "api/demo/mayor-mission/challenges/",
        mayor_mission_views.mayor_mission_challenges,
        name="mayor_mission_challenges",
    ),
    path(
        "api/demo/mayor-mission/scenarios/",
        mayor_mission_views.mayor_mission_scenarios,
        name="mayor_mission_scenarios",
    ),
    path(
        "api/demo/mayor-mission/run-classic/",
        mayor_mission_views.mayor_mission_run_classic,
        name="mayor_mission_run_classic",
    ),
    path(
        "api/demo/mayor-mission/run-medinamind/",
        mayor_mission_views.mayor_mission_run_medinamind,
        name="mayor_mission_run_medinamind",
    ),
    path(
        "api/demo/mayor-mission/finish/",
        mayor_mission_views.mayor_mission_finish,
        name="mayor_mission_finish",
    ),
    path(
        "api/demo/mayor-mission/run/",
        mayor_mission_views.mayor_mission_run,
        name="mayor_mission_run",
    ),
    path(
        "api/demo/mayor-mission/certificate/",
        mayor_mission_views.mayor_mission_certificate,
        name="mayor_mission_certificate",
    ),
    path("settings/", platform_settings_views.platform_settings_page, name="platform_settings"),
    path("settings/test/esp32/", platform_settings_views.platform_settings_test_esp32, name="platform_settings_test_esp32"),
    path("settings/test/email/", platform_settings_views.platform_settings_test_email, name="platform_settings_test_email"),
    path("settings/test/twilio/", platform_settings_views.platform_settings_test_twilio, name="platform_settings_test_twilio"),
    path("settings/test/models/", platform_settings_views.platform_settings_test_models, name="platform_settings_test_models"),
]
