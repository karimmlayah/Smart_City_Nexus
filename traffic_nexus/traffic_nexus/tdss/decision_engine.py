from collections import deque
from dataclasses import dataclass
from statistics import mean
from typing import Deque, Dict, List, Tuple

from .congestion_analysis import ZoneTrafficState


def _signal_reference_seconds() -> Tuple[int, int]:
    """Référence vert / rouge (ex. 45 s / 25 s) — surchargeable via settings Django."""
    try:
        from django.conf import settings

        g = int(getattr(settings, "TRAFFIC_SIGNAL_GREEN_REF_S", 45))
        r = int(getattr(settings, "TRAFFIC_SIGNAL_RED_REF_S", 25))
        return max(10, g), max(6, r)
    except Exception:
        return 45, 25


def proportional_green_red(scale: float, green_ref: int, red_ref: int) -> Tuple[int, int]:
    """
    Applique un facteur d’échelle unique aux deux durées pour conserver le rapport vert/rouge.
    scale ~1.0 → proche des références ; congestion faible → cycles plus courts.
    """
    scale = max(0.35, min(1.35, float(scale)))
    g = max(12, round(green_ref * scale))
    r = max(8, round(red_ref * scale))
    return g, r


@dataclass
class ZoneDecision:
    congestion_level: str
    estimated_waiting_time_min: float
    traffic_criticality_score: int  # 0..100
    signal_action: str
    green_time_s: int
    red_time_s: int
    driver_alert: str
    rerouting_suggestion: str
    entry_limitation_recommendation: str
    final_recommendation: str


class DecisionEngine:
    """Rule-based explainable decision engine with short-term memory."""

    def __init__(self, history_size: int = 20) -> None:
        self.history_size = history_size
        self.history: Dict[int, Deque[ZoneTrafficState]] = {}

    def _get_zone_history(self, zone_idx: int) -> Deque[ZoneTrafficState]:
        if zone_idx not in self.history:
            self.history[zone_idx] = deque(maxlen=self.history_size)
        return self.history[zone_idx]

    def decide_zone(self, zone_idx: int, state: ZoneTrafficState) -> ZoneDecision:
        h = self._get_zone_history(zone_idx)
        h.append(state)
        # Use only recent samples to keep congestion level reactive.
        recent = list(h)[-5:]
        avg_density = mean([s.density for s in recent]) if recent else state.density
        avg_count = mean([s.vehicle_count for s in recent]) if recent else state.vehicle_count
        avg_weighted = mean([s.weighted_score for s in recent]) if recent else state.weighted_score

        # Hybrid congestion decision tuned for visible jams in camera feeds.
        if avg_density >= 0.40 or avg_count >= 8 or avg_weighted >= 10:
            congestion = "high"
        elif avg_density >= 0.20 or avg_count >= 4 or avg_weighted >= 5:
            congestion = "medium"
        else:
            congestion = "low"
        # Instant guardrail using current frame to avoid "LOW" under visible queue spikes.
        if state.vehicle_count >= 12 and congestion == "low":
            congestion = "high"
        elif state.vehicle_count >= 6 and congestion == "low":
            congestion = "medium"

        criticality = int(
            max(
                0,
                min(
                    100,
                    avg_density * 50 + min(avg_count / 12.0, 1.0) * 30 + min(avg_weighted / 14.0, 1.0) * 20,
                ),
            )
        )
        waiting_min = round(0.8 + avg_count * 0.45 + avg_density * 5.5, 1)

        crit_n = criticality / 100.0

        if congestion == "high":
            scale = 1.0 + crit_n * 0.15
            signal_action = "Extend green phase"
            alert = "Heavy congestion detected"
            reroute = "Suggest rerouting to alternative corridor"
            entry_limit = "Recommend limiting incoming traffic"
            final = "Extend green + rerouting + send alert + limit entry"
        elif congestion == "medium":
            scale = 0.72 + crit_n * 0.22
            signal_action = "Keep current cycle with slight adaptation"
            alert = "Moderate congestion, monitor closely"
            reroute = "Optional rerouting for non-priority vehicles"
            entry_limit = "No strict limitation, keep monitoring"
            final = "Adaptive cycle + monitor + optional rerouting"
        else:
            scale = 0.48 + crit_n * 0.26
            signal_action = "Reduce green phase"
            alert = "Traffic flow is stable"
            reroute = "No rerouting needed"
            entry_limit = "No entry limitation required"
            final = "Keep normal operation / reduce unnecessary green"

        g_ref, r_ref = _signal_reference_seconds()
        green_s, red_s = proportional_green_red(scale, g_ref, r_ref)

        return ZoneDecision(
            congestion_level=congestion,
            estimated_waiting_time_min=waiting_min,
            traffic_criticality_score=criticality,
            signal_action=signal_action,
            green_time_s=green_s,
            red_time_s=red_s,
            driver_alert=alert,
            rerouting_suggestion=reroute,
            entry_limitation_recommendation=entry_limit,
            final_recommendation=final,
        )

    def decide_many(self, states: List[ZoneTrafficState]) -> List[ZoneDecision]:
        return [self.decide_zone(i, s) for i, s in enumerate(states)]

    @staticmethod
    def green_wave(decisions: List[ZoneDecision]) -> List[str]:
        order = sorted(range(len(decisions)), key=lambda i: decisions[i].traffic_criticality_score, reverse=True)
        return [f"Z{idx + 1}(+{rank * 1}s)" for rank, idx in enumerate(order)]

