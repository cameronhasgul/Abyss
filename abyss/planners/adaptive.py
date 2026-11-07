"""ABYSS adaptive planner: online Bayesian habitat model plus energy-aware utility.

    U(x) = ( max(0, y_hat(x)) + beta * I(x) + gamma * C(x) ) / ( E_travel(x) + E_survey + eps )

y_hat  Bayesian linear prediction of local observed richness (``Observation.richness``)
I      information gain 0.5 * log(1 + phi^T Sigma phi / noise_var)
C      coverage incentive, min(1, distance to nearest surveyed cell / coverage_radius)

Only cells the simulator marks feasible and that have not been surveyed are scored.
There are no scripted reactions: the route changes only because the posterior changes.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..contract import HOME, Observation, PublicMap, PublicState, Target
from ..learning.bayesian import BayesianLinearRegression, quadratic_features
from .base import Planner

DEFAULTS: dict = {
    "beta": 1.0,  # information-gain weight
    "gamma": 0.5,  # coverage weight
    "coverage_radius": 3.0,  # cells
    "noise_var": 1.0,
    "prior_precision": 1.0,
    "bias_prior_precision": 0.1,
    "squared": ("depth", "slope"),
    "learn": True,  # ablation: False freezes the model at its prior
    "eps": 1e-6,
}


class AdaptivePlanner(Planner):
    name = "abyss"
    overrides: dict = {}  # ablation subclasses change defaults here

    def reset(self, public_map: PublicMap, config: dict, seed: int) -> None:
        self.config = {**DEFAULTS, **self.overrides, **(config or {})}
        c = self.config
        self.survey_cost = public_map.survey_cost
        self.phi, self.phi_names = quadratic_features(
            public_map.features, public_map.feature_names, tuple(c["squared"])
        )
        dim = self.phi.shape[-1]
        precision = np.full(dim, c["prior_precision"], float)
        precision[0] = c["bias_prior_precision"]
        self.model = BayesianLinearRegression(dim, precision, c["noise_var"])
        self._decision: dict = {}

    # learning -----------------------------------------------------------
    def observe(self, observation: Observation) -> None:
        if self.config["learn"]:
            row, col = observation.waypoint
            self.model.update(self.phi[row, col], float(observation.richness))

    def belief(self) -> tuple[np.ndarray, np.ndarray]:
        """Predicted richness and epistemic standard deviation, both (H, W)."""
        mean, var = self.model.predict(self.phi)
        return mean, np.sqrt(var)

    # planning -----------------------------------------------------------
    def _coverage(self, surveyed: np.ndarray) -> np.ndarray:
        if not surveyed.any():
            return np.ones(surveyed.shape)
        return np.minimum(1.0, distance_transform_edt(~surveyed) / self.config["coverage_radius"])

    def utility_terms(self, state: PublicState) -> dict[str, np.ndarray]:
        c = self.config
        mean, var = self.model.predict(self.phi)
        info = 0.5 * np.log1p(var / c["noise_var"])
        coverage = self._coverage(state.surveyed)
        cost = state.travel_cost + self.survey_cost
        value = np.maximum(0.0, mean) + c["beta"] * info + c["gamma"] * coverage
        candidates = state.feasible & ~state.surveyed
        utility = np.where(candidates, value / (cost + c["eps"]), -np.inf)
        return {
            "expected_richness": mean,
            "information_gain": info,
            "coverage": coverage,
            "energy_cost": cost,
            "utility": utility,
            "candidates": candidates,
        }

    def choose_target(
        self, public_state: PublicState, observation_history: tuple[Observation, ...]
    ) -> Target:
        terms = self.utility_terms(public_state)
        if not terms["candidates"].any():
            self._decision = {}
            return HOME
        index = int(np.argmax(terms["utility"]))  # deterministic: first max, row-major
        row, col = np.unravel_index(index, terms["utility"].shape)
        self._decision = {
            key: float(terms[key][row, col])
            for key in ("expected_richness", "information_gain", "coverage", "energy_cost", "utility")
        }
        self._decision["candidates"] = int(terms["candidates"].sum())
        return int(row), int(col)

    def diagnostics(self) -> dict:
        mean, std = self.belief()
        return {
            "predicted_richness": np.round(mean, 3),
            "uncertainty": np.round(std, 3),
            "target_utility": self._decision,
            "model": {
                "observations": self.model.n,
                "weights": dict(zip(self.phi_names, np.round(self.model.mean, 4).tolist())),
            },
        }


class NoLearnPlanner(AdaptivePlanner):
    """Ablation: same utility, but the model never updates from observations."""

    name = "abyss-nolearn"
    overrides = {"learn": False}


class NoInfoPlanner(AdaptivePlanner):
    """Ablation: learns, but ignores the information-gain term."""

    name = "abyss-noinfo"
    overrides = {"beta": 0.0}
