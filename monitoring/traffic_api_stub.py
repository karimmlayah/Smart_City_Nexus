"""Traffic Nexus API stubs — used on Vercel when OpenCV/YOLO are unavailable."""
from __future__ import annotations

from django.views.decorators.http import require_http_methods

from monitoring.runtime import ml_unavailable_json


def _stub(feature: str):
    @require_http_methods(["GET", "POST"])
    def view(request):
        return ml_unavailable_json(feature)

    view.__name__ = f"traffic_stub_{feature.replace(' ', '_')[:40]}"
    return view


traffic_first_frame = _stub("Traffic first frame")
traffic_analyze = _stub("Traffic analyze")
traffic_models = _stub("Traffic models")
traffic_live_init = _stub("Traffic live init")
traffic_live_tick = _stub("Traffic live tick")
traffic_live_close = _stub("Traffic live close")
traffic_report_pdf = _stub("Traffic report PDF")
