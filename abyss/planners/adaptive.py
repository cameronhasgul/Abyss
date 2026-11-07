"""ABYSS: pick the survey with the best expected value per unit of energy.

    U(x) = (max(0, predicted richness) + beta * information gain + gamma * coverage)
           / (travel energy + survey energy + epsilon)

Curiosity has no special rules: routes change only because observations change the model.
"""

from __future__ import annotations

import numpy as np

from ..contract import HOME, Observation, PublicMap, PublicState, Target
from ..learning.bayesian import BayesianLinearRegression
from .base import Planner

# Frozen. Chosen by abyss.evaluation.tune on development seeds only, over all world types.
DEFAULT_CONFIG = {
    "beta": 4.0,
    "gamma": 0.5,
    "use_spatial": False,
    "prior_precision": 1.0,
    "noise_var": 1.0,
    "coverage_radius": 5,
}
EPSILON = 1e-6
_MAP_DECIMALS = 3


def build_features(public_map: PublicMap, use_spatial: bool) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return (H, W, D) features: [1, d, s, c, d^2, s^2], optionally with coordinates."""
    depth, slope, bpi = np.moveaxis(public_map.features, -1, 0)
    columns = [np.ones_like(depth), depth, slope, bpi, depth**2, slope**2]
    names = ["intercept", "depth", "slope", "bpi", "depth2", "slope2"]
    if use_spatial:
        height, width = public_map.shape
        rows, cols = np.mgrid[0:height, 0:width].astype(float)
        y = 2.0 * rows / max(height - 1, 1) - 1.0
        x = 2.0 * cols / max(width - 1, 1) - 1.0
        columns += [y, x, y**2, x**2, y * x]
        names += ["y", "x", "y2", "x2", "yx"]
    return np.stack(columns, axis=-1), tuple(names)


class AdaptivePlanner(Planner):
    name = "abyss"
    learns = True

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        self.config = {**DEFAULT_CONFIG, **config}
        self.log_maps = self.config.pop("log_maps", True)
        self._map = public_map
        self._phi, self._feature_names = build_features(public_map, self.config["use_spatial"])
        self.model = BayesianLinearRegression(
            self._phi.shape[-1], self.config["prior_precision"], self.config["noise_var"]
        )
        self._visited = [public_map.home]
        self._last_utility: dict | None = None

    def describe(self) -> dict:
        return dict(self.config)

    def _coverage(self) -> np.ndarray:
        height, width = self._map.shape
        rows, cols = np.mgrid[0:height, 0:width]
        visited = np.array(self._visited)
        distance = np.maximum(
            np.abs(rows[..., None] - visited[:, 0]), np.abs(cols[..., None] - visited[:, 1])
        ).min(axis=-1)
        return np.minimum(1.0, distance / self.config["coverage_radius"])

    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        candidates = public_state.feasible & ~public_state.surveyed
        if not candidates.any():
            self._last_utility = None
            return HOME
        mean, variance = self.model.predict(self._phi)
        information = self.model.information_gain(variance)
        coverage = self._coverage()
        value = (
            np.maximum(0.0, mean) + self.config["beta"] * information + self.config["gamma"] * coverage
        )
        energy = public_state.travel_cost + self._map.survey_cost + EPSILON
        utility = np.where(candidates, value / energy, -np.inf)
        row, col = np.unravel_index(int(np.argmax(utility)), utility.shape)
        self._last_utility = {
            "expected_richness": float(max(0.0, mean[row, col])),
            "information_gain": float(information[row, col]),
            "coverage": float(coverage[row, col]),
            "energy_cost": float(energy[row, col]),
            "utility": float(utility[row, col]),
        }
        return int(row), int(col)

    def observe(self, observation: Observation) -> None:
        self._visited.append(observation.waypoint)
        if self.learns:
            self.model.update(self._phi[observation.waypoint], float(observation.richness))

    def diagnostics(self) -> dict:
        report: dict = {
            "model_weights": dict(zip(self._feature_names, self.model.weights.tolist())),
            "model_observations": self.model.n_observations,
        }
        if self._last_utility:
            report["target_utility"] = self._last_utility
        if self.log_maps:
            mean, variance = self.model.predict(self._phi)
            report["predicted_richness"] = np.round(mean, _MAP_DECIMALS)
            report["uncertainty"] = np.round(np.sqrt(variance), _MAP_DECIMALS)
        return report


class NoLearningPlanner(AdaptivePlanner):
    """Ablation: same objective, but the model never updates."""

    name = "abyss_no_learning"
    learns = False


class NoInformationPlanner(AdaptivePlanner):
    """Ablation: learning stays on, but uncertainty carries no value."""

    name = "abyss_no_info"

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        super().reset(public_map, {**config, "beta": 0.0}, seed)
