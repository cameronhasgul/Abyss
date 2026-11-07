"""Terrain heuristic baseline: a fixed preference for steep ground. It never learns."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ..contract import HOME, Observation, PublicMap, PublicState, Target
from .base import Planner


class HeuristicPlanner(Planner):
    name = "heuristic"

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        slope = public_map.slope_deg
        self._steepness = rankdata(slope).reshape(slope.shape) / slope.size
        self._survey_cost = public_map.survey_cost

    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        candidates = public_state.feasible & ~public_state.surveyed
        if not candidates.any():
            return HOME
        score = self._steepness / (public_state.travel_cost + self._survey_cost)
        row, col = np.unravel_index(int(np.argmax(np.where(candidates, score, -np.inf))), score.shape)
        return int(row), int(col)
