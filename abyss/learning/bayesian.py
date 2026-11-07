"""Bayesian linear regression with a zero-mean Gaussian prior.

Richness is a count, so the Gaussian noise model is only an approximation. Weights
describe associations in this simulated world, not causes.
"""

from __future__ import annotations

import numpy as np


class BayesianLinearRegression:
    def __init__(self, n_features: int, prior_precision: float, noise_var: float) -> None:
        self.noise_var = noise_var
        self._precision = prior_precision * np.eye(n_features)
        self._b = np.zeros(n_features)
        self.covariance = np.linalg.inv(self._precision)
        self.weights = np.zeros(n_features)
        self.n_observations = 0

    def update(self, phi: np.ndarray, y: float) -> None:
        self._precision += np.outer(phi, phi) / self.noise_var
        self._b += phi * y / self.noise_var
        self.covariance = np.linalg.inv(self._precision)
        self.weights = self.covariance @ self._b
        self.n_observations += 1

    def predict(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted mean and epistemic variance for rows of ``phi``."""
        return phi @ self.weights, np.sum((phi @ self.covariance) * phi, axis=-1)

    def information_gain(self, variance: np.ndarray) -> np.ndarray:
        """Expected entropy reduction from one more observation, in nats."""
        return 0.5 * np.log1p(variance / self.noise_var)
