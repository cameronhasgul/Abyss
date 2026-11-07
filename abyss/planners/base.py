"""Interface every exploration policy implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..contract import Observation, PublicMap, PublicState, Target


class Planner(ABC):
    name = "planner"

    @abstractmethod
    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        """Start a mission. Use ``seed`` for all planner randomness."""

    @abstractmethod
    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        """Return a (row, col) from ``public_state.feasible``, or HOME."""

    def observe(self, observation: Observation) -> None:
        """Receive each new survey result. Learning planners update here."""

    def describe(self) -> dict:
        """Settings in effect, recorded in the mission log."""
        return {}

    def diagnostics(self) -> dict:
        """Optional per-step log data: predicted maps, uncertainty, utility terms."""
        return {}
