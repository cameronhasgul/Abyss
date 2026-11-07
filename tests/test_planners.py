"""Cam's planners on the real simulator: safety, learning, determinism, no leakage."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from abyss.config import SimConfig
from abyss.env import Environment
from abyss.planners import PLANNERS
from abyss.planners.adaptive import AdaptivePlanner
from abyss.run_mission import run_mission, validate_mission_log
from helpers import ALL_SCENARIOS

CAM_PLANNERS = ("abyss", "abyss-nolearn", "abyss-noinfo", "frontier", "heuristic", "lawnmower")


@pytest.mark.parametrize("name", CAM_PLANNERS)
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_safe_valid_missions(name, scenario):
    log, _ = run_mission(SimConfig(scenario=scenario), PLANNERS[name](), seed=0)
    validate_mission_log(log)
    assert not log["outcome"]["forced_return"], log["outcome"]["rejected_targets"]
    assert log["outcome"]["surveys"] > 0


@pytest.mark.parametrize("name", CAM_PLANNERS)
def test_no_repeat_surveys(name):
    log, _ = run_mission(SimConfig(), PLANNERS[name](), seed=1)
    cells = [tuple(s["observation"]["waypoint"]) for s in log["steps"] if s["observation"]]
    assert len(cells) == len(set(cells))


@pytest.mark.parametrize("name", CAM_PLANNERS)
def test_reproducible(name):
    a, _ = run_mission(SimConfig(scenario="ridge"), PLANNERS[name](), seed=3)
    b, _ = run_mission(SimConfig(scenario="ridge"), PLANNERS[name](), seed=3)
    assert a == b


def test_observations_change_beliefs_and_choices():
    """The core claim: the same state gives a different target once the model has learned."""
    env = Environment(SimConfig(scenario="ridge"))
    state = env.reset(0)
    planner = AdaptivePlanner()
    planner.reset(env.public_map, {}, 0)
    before_mean, before_std = planner.belief()
    first = planner.choose_target(state, ())

    result = env.step(first)
    for _ in range(5):
        planner.observe(result.observation)
        result = env.step(planner.choose_target(result.state, ()))
    planner.observe(result.observation)
    after_mean, after_std = planner.belief()

    assert not np.allclose(before_mean, after_mean)
    assert after_std.mean() < before_std.mean()
    # Feed the same fixed state to a fresh and a trained planner: choices should differ.
    fresh = AdaptivePlanner()
    fresh.reset(env.public_map, {}, 0)
    state = result.state
    assert fresh.utility_terms(state)["utility"].argmax() != planner.utility_terms(state)["utility"].argmax()


def test_nolearn_ablation_never_updates():
    log, _ = run_mission(SimConfig(), PLANNERS["abyss-nolearn"](), seed=0)
    assert all(s["planner_diagnostics"]["model"]["observations"] == 0 for s in log["steps"])


def test_diagnostics_shape_for_ui():
    config = SimConfig()
    log, _ = run_mission(config, AdaptivePlanner(), seed=0)
    diag = log["steps"][0]["planner_diagnostics"]
    assert np.array(diag["predicted_richness"]).shape == (config.height, config.width)
    assert np.array(diag["uncertainty"]).shape == (config.height, config.width)
    assert set(diag["target_utility"]) >= {
        "expected_richness", "information_gain", "coverage", "energy_cost", "utility"}


@pytest.mark.parametrize("name", CAM_PLANNERS)
def test_planner_source_never_touches_hidden_world(name):
    source = inspect.getsource(inspect.getmodule(PLANNERS[name]))
    for forbidden in ("HiddenWorld", "abyss.world", "from ..world", "evaluation_record",
                      "encounters", "probability", "scenario", "rho"):
        assert forbidden not in source, forbidden
