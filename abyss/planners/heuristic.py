"""Terrain heuristic baseline: a fixed belief that steep terrain is interesting.

Scores each feasible unsurveyed cell by standardised slope per unit energy.
It never learns, so it is wrong by design on worlds that favour flat terrain.
"""

from __future__ import annotations

import numpy as np

from ..contract import HOME, Observation, PublicMap, PublicState, Target
from .base import Planner


class TerrainHeuristicPlanner(Planner):
    name = "heuristic"
    feature = "slope"

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        values = public_map.features[..., public_map.feature_names.index(self.feature)]
        self.preference = values - values.min() + 0.1  # strictly positive
        self.survey_cost = public_map.survey_cost

    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        candidates = public_state.feasible & ~public_state.surveyed
        if not candidates.any():
            return HOME
        score = self.preference / (public_state.travel_cost + self.survey_cost)
        score = np.where(candidates, score, -np.inf)
        row, col = np.unravel_index(int(np.argmax(score)), score.shape)
        return int(row), int(col)
