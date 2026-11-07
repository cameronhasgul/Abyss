"""Lawnmower baseline: a fixed back-and-forth survey pattern set before the mission."""

from __future__ import annotations

from ..contract import HOME, Cell, Observation, PublicMap, PublicState, Target
from .base import Planner

DEFAULT_SPACING = 3


class LawnmowerPlanner(Planner):
    name = "lawnmower"

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        self.spacing = config.get("spacing", DEFAULT_SPACING)
        height, width = public_map.shape
        self._pattern: list[Cell] = []
        for index, row in enumerate(range(0, height, self.spacing)):
            cols = list(range(0, width, self.spacing))
            self._pattern += [(row, col) for col in (cols if index % 2 == 0 else reversed(cols))]
        self._next = 0

    def describe(self) -> dict:
        return {"spacing": self.spacing}

    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        # The pattern is fixed, so stop at the first waypoint that can't be afforded.
        while self._next < len(self._pattern):
            cell = self._pattern[self._next]
            self._next += 1
            if public_state.surveyed[cell]:
                continue
            return cell if public_state.feasible[cell] else HOME
        return HOME
