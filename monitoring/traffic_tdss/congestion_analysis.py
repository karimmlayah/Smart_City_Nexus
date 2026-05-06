from collections import Counter
from dataclasses import dataclass
from typing import List


@dataclass
class ZoneTrafficState:
    vehicle_count: int
    density: float
    weighted_score: float
    level: str  # low, medium, high


WEIGHTS = {
    "car": 1.0,
    "motorcycle": 0.6,
    "bicycle": 0.3,
    "bus": 2.2,
    "truck": 2.3,
}


def _normalize_density(score: float, area_px: float) -> float:
    if area_px <= 0:
        return 0.0
    # More aggressive normalization for busy urban roads.
    reference = max(1.0, area_px / 8000.0)
    return min(1.0, score / (reference * 3.2))


def _level_from_metrics(density: float, vehicle_count: int, weighted_score: float) -> str:
    # Hybrid rule: prioritize visible queue volume + heavy vehicles.
    if density >= 0.40 or vehicle_count >= 8 or weighted_score >= 10:
        return "high"
    if density >= 0.20 or vehicle_count >= 4 or weighted_score >= 5:
        return "medium"
    return "low"


def compute_zone_states(
    per_zone_counts: List[Counter],
    zone_areas: List[float],
) -> List[ZoneTrafficState]:
    states: List[ZoneTrafficState] = []
    for i, counts in enumerate(per_zone_counts):
        score = float(sum(WEIGHTS.get(k, 1.0) * v for k, v in counts.items()))
        vehicle_count = int(sum(counts.values()))
        area_px = zone_areas[i] if i < len(zone_areas) else 1.0
        density = _normalize_density(score, area_px)
        level = _level_from_metrics(density, vehicle_count, score)
        states.append(
            ZoneTrafficState(
                vehicle_count=vehicle_count,
                density=density,
                weighted_score=score,
                level=level,
            )
        )
    return states

