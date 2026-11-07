import numpy as np

from abyss.learning.bayesian import BayesianLinearRegression, quadratic_features


def test_recovers_linear_weights():
    rng = np.random.default_rng(0)
    w = np.array([1.0, 2.0, -1.5])
    X = np.c_[np.ones(400), rng.normal(size=(400, 2))]
    y = X @ w + rng.normal(0, 0.3, 400)
    m = BayesianLinearRegression(3, prior_precision=1e-3, noise_var=0.09)
    m.update(X, y)
    assert np.allclose(m.mean, w, atol=0.1)


def test_incremental_equals_batch():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(30, 4)), rng.normal(size=30)
    a, b = BayesianLinearRegression(4), BayesianLinearRegression(4)
    a.update(X, y)
    for xi, yi in zip(X, y):
        b.update(xi, yi)
    assert np.allclose(a.mean, b.mean) and np.allclose(a.cov, b.cov)


def test_uncertainty_and_info_gain_shrink_where_observed():
    m = BayesianLinearRegression(3)
    x = np.array([1.0, 0.5, -0.2])
    _, v0 = m.predict(x); ig0 = m.info_gain(x)
    for _ in range(5):
        m.update(x, 2.0)
    _, v1 = m.predict(x); ig1 = m.info_gain(x)
    assert v1 < v0 and ig1 < ig0


def test_learns_negative_relationship():
    """Must be able to learn 'flat is rich', not just 'steep is rich'."""
    s = np.linspace(-2, 2, 50)
    X = np.c_[np.ones(50), s]
    m = BayesianLinearRegression(2, prior_precision=0.1)
    m.update(X, 2.0 - 1.0 * s)
    assert m.mean[1] < -0.8


def test_quadratic_features_layout():
    f = np.zeros((2, 2, 3)); f[..., 1] = 2.0
    phi, names = quadratic_features(f, ("depth", "slope", "bpi"))
    assert names == ["bias", "depth", "slope", "bpi", "depth^2", "slope^2"]
    assert phi.shape == (2, 2, 6) and np.all(phi[..., 0] == 1) and np.all(phi[..., 5] == 4)
