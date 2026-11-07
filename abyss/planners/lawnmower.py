"""Lawnmower baseline: a fixed boustrophedon sweep, planned before the mission.

Sweeps every ``lane_spacing``-th row, nearest lanes to home first, alternating
direction. Skips cells it can no longer afford and goes home when none are left.
"""

from __future__ import annotations

from ..contract import HOME, Observation, PublicMap, PublicState, Target
from .base import Planner


class LawnmowerPlanner(Planner):
    name = "lawnmower"
    lane_spacing = 2

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        height, width = public_map.shape
        home_row, home_col = public_map.home
        spacing = int((config or {}).get("lane_spacing", self.lane_spacing))
        lanes = sorted(range(home_row % spacing, height, spacing), key=lambda r: abs(r - home_row))
        # Start each sweep from the side nearest home so the first lane begins there.
        start_left = home_col < width / 2
        order = []
        for i, row in enumerate(lanes):
            cols = range(width) if (i % 2 == 0) == start_left else range(width - 1, -1, -1)
            order.extend((row, col) for col in cols)
        self.order = order
        self.cursor = 0

    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        while self.cursor < len(self.order):
            cell = self.order[self.cursor]
            self.cursor += 1
            if public_state.feasible[cell] and not public_state.surveyed[cell]:
                return cell
        return HOME
