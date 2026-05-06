from dataclasses import dataclass

from .congestion_analysis import ZoneTrafficState
from .decision_engine import ZoneDecision


@dataclass
class XAIExplanation:
    title: str
    message: str
    feature_congestion: float
    feature_vehicle_count: float
    feature_waiting_time: float
    feature_criticality: float


def explain_decision(state: ZoneTrafficState, decision: ZoneDecision) -> XAIExplanation:
    msg = (
        f"Decision: {decision.final_recommendation} because congestion level is "
        f"{decision.congestion_level}, vehicle count is {state.vehicle_count}, "
        f"estimated waiting time is {decision.estimated_waiting_time_min} min, "
        f"and criticality score is {decision.traffic_criticality_score}/100."
    )
    return XAIExplanation(
        title="Why this decision?",
        message=msg,
        feature_congestion=round(state.density * 100, 1),
        feature_vehicle_count=float(state.vehicle_count),
        feature_waiting_time=decision.estimated_waiting_time_min,
        feature_criticality=float(decision.traffic_criticality_score),
    )

