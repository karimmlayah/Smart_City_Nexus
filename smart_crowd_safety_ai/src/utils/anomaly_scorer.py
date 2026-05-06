"""Combinaison des signaux en score de risque global [0, 1]."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class AnomalyScorer:
    w_fight: float = 0.45
    w_panic: float = 0.30
    w_abandoned: float = 0.25

    def score(
        self,
        fight_prob: float,
        panic_score: float,
        abandoned_score: float,
    ) -> float:
        s = (
            self.w_fight * fight_prob
            + self.w_panic * panic_score
            + self.w_abandoned * abandoned_score
        )
        return float(min(1.0, max(0.0, s)))

    def components(
        self,
        fight_prob: float,
        panic_score: float,
        abandoned_score: float,
    ) -> Dict[str, float]:
        return {
            "fight_weighted": self.w_fight * fight_prob,
            "panic_weighted": self.w_panic * panic_score,
            "abandoned_weighted": self.w_abandoned * abandoned_score,
            "risk": self.score(fight_prob, panic_score, abandoned_score),
        }
