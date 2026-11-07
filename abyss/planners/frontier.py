"""Frontier baseline: survey the cheapest boundary of the explored area.

A survey marks its cell and neighbours within ``footprint`` as explored. A frontier cell
is unexplored and touches explored ground. Biology plays no part in the choice.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from ..contract import HOME, Observation, PublicMap, PublicState, Target
from .base import Planner

DEFAULT_FOOTPRINT = 1


class FrontierPlanner(Planner):
    name = "frontier"

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        self.footprint = config.get("footprint", DEFAULT_FOOTPRINT)
        self._home_mask = np.zeros(public_map.shape, dtype=bool)
        self._home_mask[public_map.home] = True
        self._structure = np.ones((3, 3), dtype=bool)

    def describe(self) -> dict:
        return {"footprint": self.footprint}

    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        seeds = public_state.surveyed | self._home_mask
        explored = binary_dilation(seeds, self._structure, iterations=self.footprint)
        frontier = binary_dilation(explored, self._structure) & ~explored
        candidates = frontier & public_state.feasible
        if not candidates.any():
            # Everything near the boundary is too expensive, or the map is fully explored.
            candidates = public_state.feasible & ~public_state.surveyed & ~explored
        if not candidates.any():
            return HOME
        costs = np.where(candidates, public_state.travel_cost, np.inf)
        row, col = np.unravel_index(int(np.argmin(costs)), costs.shape)
        return int(row), int(col)
