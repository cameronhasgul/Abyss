"""Frontier baseline: the cheapest unsurveyed cell bordering the surveyed region.

Before the first survey, or if no frontier cell is feasible, it takes the cheapest
feasible unsurveyed cell. It never learns from observations.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from ..contract import HOME, Observation, PublicMap, PublicState, Target
from .base import Planner

_EIGHT = np.ones((3, 3), bool)


class FrontierPlanner(Planner):
    name = "frontier"

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        pass

    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        candidates = public_state.feasible & ~public_state.surveyed
        if not candidates.any():
            return HOME
        frontier = binary_dilation(public_state.surveyed, _EIGHT) & candidates
        pool = frontier if frontier.any() else candidates
        cost = np.where(pool, public_state.travel_cost, np.inf)
        row, col = np.unravel_index(int(np.argmin(cost)), cost.shape)
        return int(row), int(col)
