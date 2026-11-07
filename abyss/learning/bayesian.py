"""Online Bayesian linear regression for local survey richness.

Model:   y = phi(x)^T w + eps,   eps ~ N(0, noise_var),   w ~ N(0, diag(1/prior_precision))

We keep the posterior in information form (precision matrix A and vector b),
so each survey is an O(D^2) rank-one update and the result is identical to a
batch fit on the same data, regardless of order.

Caveat: richness is a count, so the Gaussian noise model is an approximation.
"""
from __future__ import annotations

import numpy as np


def quadratic_features(features: np.ndarray, feature_names: tuple[str, ...],
                       squared: tuple[str, ...] = ("depth", "slope")) -> tuple[np.ndarray, list[str]]:
    """phi(x) = [1, f_1..f_F, f_j^2 for j in squared].  `features` is (..., F)."""
    parts = [np.ones(features.shape[:-1] + (1,)), features]
    names = ["bias", *feature_names]
    for name in squared:
        if name in feature_names:
            parts.append(features[..., [feature_names.index(name)]] ** 2)
            names.append(f"{name}^2")
    return np.concatenate(parts, axis=-1), names


class BayesianLinearRegression:
    def __init__(self, dim: int, prior_precision: float | np.ndarray = 1.0,
                 noise_var: float = 1.0):
        prec = np.broadcast_to(np.asarray(prior_precision, float), (dim,))
        self.dim = dim
        self.noise_var = float(noise_var)
        self.prior_A = np.diag(prec)
        self.reset()

    def reset(self) -> None:
        self.A = self.prior_A.copy()     # posterior precision
        self.b = np.zeros(self.dim)      # precision-weighted mean
        self.n = 0
        self._refresh()

    def _refresh(self) -> None:
        self.cov = np.linalg.inv(self.A)
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.mean = self.cov @ self.b

    def update(self, phi: np.ndarray, y: float | np.ndarray) -> None:
        """Add one observation (phi shape (D,)) or a batch (phi (N, D), y (N,))."""
        phi = np.atleast_2d(phi)
        y = np.atleast_1d(np.asarray(y, float))
        self.A += phi.T @ phi / self.noise_var
        self.b += phi.T @ y / self.noise_var
        self.n += len(y)
        self._refresh()

    def predict(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior predictive mean and epistemic variance phi^T Sigma phi."""
        mean = phi @ self.mean
        var = np.einsum("...i,ij,...j->...", phi, self.cov, phi)
        return mean, np.maximum(var, 0.0)

    def info_gain(self, phi: np.ndarray) -> np.ndarray:
        """Expected information gain about w from one observation at phi:
        I = 0.5 * log(1 + phi^T Sigma phi / noise_var)."""
        _, var = self.predict(phi)
        return 0.5 * np.log1p(var / self.noise_var)
