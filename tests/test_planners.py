"""Planner behaviour, safety and honest learning."""

import numpy as np
import pytest
from helpers import ALL_SCENARIOS

from abyss.config import SimConfig
from abyss.contract import Observation
from abyss.env import Environment
from abyss.evaluation.metrics import mission_metrics, paired_comparison
from abyss.learning.bayesian import BayesianLinearRegression
from abyss.planners import PLANNERS
from abyss.planners.adaptive import AdaptivePlanner, NoInformationPlanner, NoLearningPlanner
from abyss.run_mission import run_mission, validate_mission_log

SEEDS = (0, 1)


@pytest.mark.parametrize("name", sorted(PLANNERS))
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
@pytest.mark.parametrize("seed", SEEDS)
def test_every_planner_completes_safely(name, scenario, seed):
    log, _ = run_mission(SimConfig(scenario=scenario), PLANNERS[name](), seed)
    validate_mission_log(log)
    assert log["outcome"]["safe_return"]
    assert not log["outcome"]["forced_return"]
    assert log["outcome"]["surveys"] > 0


@pytest.mark.parametrize("name", sorted(PLANNERS))
def test_planners_are_reproducible(name):
    first, _ = run_mission(SimConfig(), PLANNERS[name](), 3)
    second, _ = run_mission(SimConfig(), PLANNERS[name](), 3)
    assert first == second


@pytest.mark.parametrize("name", ["frontier", "heuristic", "lawnmower", "abyss_no_learning"])
def test_non_learning_planners_ignore_biology(name):
    routes = {
        scenario: [s["route"] for s in run_mission(SimConfig(scenario=scenario), PLANNERS[name](), 2)[0]["steps"]]
        for scenario in ALL_SCENARIOS
    }
    assert all(route == routes["random"] for route in routes.values())


def test_blr_recovers_a_known_relationship():
    rng = np.random.default_rng(0)
    model = BayesianLinearRegression(2, prior_precision=0.01, noise_var=0.25)
    for _ in range(500):
        x = rng.normal()
        model.update(np.array([1.0, x]), 2.0 - 3.0 * x + rng.normal(0.0, 0.5))
    assert model.weights == pytest.approx([2.0, -3.0], abs=0.15)


def test_blr_uncertainty_falls_with_data():
    model = BayesianLinearRegression(2, 1.0, 1.0)
    phi = np.array([[1.0, 2.0]])
    _, before = model.predict(phi)
    model.update(phi[0], 1.0)
    _, after = model.predict(phi)
    assert after[0] < before[0]
    assert model.information_gain(after)[0] < model.information_gain(before)[0]


def _start(planner):
    env = Environment(SimConfig())
    state = env.reset(0)
    planner.reset(env.public_map, {}, 0)
    return env, state


def _observation(env, cell, n_categories):
    features = tuple(float(v) for v in env.public_map.features[cell])
    return Observation(
        1, cell, env.public_map.feature_names, features,
        tuple(f"category_{k:02d}" for k in range(n_categories)), 2.0,
    )


def test_observations_change_predictions_and_decisions():
    quiet, rich = AdaptivePlanner(), AdaptivePlanner()
    env, state = _start(quiet)
    rich.reset(env.public_map, {}, 0)
    steep = np.unravel_index(int(env.public_map.slope_deg.argmax()), env.public_map.shape)
    flat = np.unravel_index(int(env.public_map.slope_deg.argmin()), env.public_map.shape)
    for planner, steep_count, flat_count in ((quiet, 0, 0), (rich, 8, 0)):
        for cell, count in ((steep, steep_count), (flat, flat_count)):
            for _ in range(4):
                planner.observe(_observation(env, tuple(int(v) for v in cell), count))
    before, after = quiet.diagnostics(), rich.diagnostics()
    assert after["model_weights"]["slope"] > before["model_weights"]["slope"]
    assert not np.array_equal(before["predicted_richness"], after["predicted_richness"])


def test_abyss_routes_change_with_what_it_observes():
    routes = {
        scenario: [s["route"] for s in run_mission(SimConfig(scenario=scenario), AdaptivePlanner(), 2)[0]["steps"]]
        for scenario in ("ridge", "reversed")
    }
    assert routes["ridge"] != routes["reversed"]
    # The first target comes before any observation, so it can't depend on the world.
    assert routes["ridge"][0] == routes["reversed"][0]


def test_no_learning_ablation_never_updates():
    planner = NoLearningPlanner()
    env, _ = _start(planner)
    planner.observe(_observation(env, (3, 3), 6))
    assert planner.model.n_observations == 0
    assert not np.any(planner.model.weights)


def test_no_information_ablation_zeroes_beta():
    planner = NoInformationPlanner()
    _start(planner)
    assert planner.describe()["beta"] == 0.0


def _final_slope_weight(scenario, seed):
    log, _ = run_mission(SimConfig(scenario=scenario, terrain_source="gebco"), AdaptivePlanner(), seed)
    surveys = [s for s in log["steps"] if s["observation"]]
    return surveys[-1]["planner_diagnostics"]["model_weights"]["slope"]


def test_model_learns_opposite_relationships_from_evidence():
    ridge = np.mean([_final_slope_weight("ridge", s) for s in range(1000, 1010)])
    reversed_world = np.mean([_final_slope_weight("reversed", s) for s in range(1000, 1010)])
    assert ridge > 0 > reversed_world


def test_diagnostics_reach_the_log_for_the_ui():
    log, _ = run_mission(SimConfig(), AdaptivePlanner(), 0)
    diagnostics = log["steps"][0]["planner_diagnostics"]
    assert np.array(diagnostics["predicted_richness"]).shape == (30, 30)
    assert np.array(diagnostics["uncertainty"]).shape == (30, 30)
    assert set(diagnostics["target_utility"]) >= {"expected_richness", "information_gain", "utility"}
    assert "beta" in log["planner"]["config"]


def test_metrics_score_truth_and_build_a_monotonic_curve():
    log, reveal = run_mission(SimConfig(), AdaptivePlanner(), 0)
    metrics = mission_metrics(log, reveal)
    assert metrics["distinct_categories"] == reveal["true_distinct_count"]
    curve = metrics["curve"]
    assert curve == sorted(curve)
    assert curve[-1] == metrics["distinct_categories"]
    assert curve[0] == 0


def test_paired_comparison_counts_and_interval():
    result = paired_comparison(np.array([3, 3, 2, 5]), np.array([1, 3, 4, 2]))
    assert (result["wins"], result["ties"], result["losses"]) == (2, 1, 1)
    assert result["mean_difference"] == pytest.approx(0.75)
    assert result["ci95_low"] <= 0.75 <= result["ci95_high"]
