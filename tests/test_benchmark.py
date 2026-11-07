"""The benchmark is honest: fair inputs, separate seeds and a safe, complete record."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from abyss.config import SimConfig
from abyss.evaluation.benchmark import (
    DEVELOPMENT_SEEDS,
    EVALUATION_SEEDS,
    make_jobs,
    run_one,
    summarise,
)
from abyss.evaluation.report import render
from abyss.planners import ABLATIONS, BASELINES
from abyss.planners.adaptive import DEFAULT_CONFIG

PLANNERS = ("abyss", *BASELINES, *ABLATIONS)
COMMITTED = Path("ui/logs/benchmark.json")


@pytest.fixture(scope="module")
def committed():
    return json.loads(COMMITTED.read_text(encoding="utf-8"))


def test_development_and_evaluation_seeds_never_overlap():
    assert not set(DEVELOPMENT_SEEDS) & set(EVALUATION_SEEDS)


def test_every_planner_gets_every_world_and_seed():
    jobs = make_jobs(SimConfig(), PLANNERS, (0, 1), (("ridge", 0.5), ("random", 0.0)), ("gebco",))
    assert len(jobs) == len(PLANNERS) * 2 * 2
    assert {job[1] for job in jobs} == set(PLANNERS)


def test_small_benchmark_runs_end_to_end():
    jobs = make_jobs(SimConfig(), PLANNERS, (0, 1, 2), (("ridge", 0.5),), ("gebco",))
    summary = summarise([run_one(job) for job in jobs], (0, 1, 2), {"simulation": SimConfig().to_dict()})
    world = summary["worlds"][0]
    assert set(world["planners"]) == set(PLANNERS)
    assert set(world["abyss_vs"]) == set(PLANNERS) - {"abyss"}
    assert summary["safety"]["safe_returns"] == summary["safety"]["missions"] == len(jobs)


def test_committed_results_are_complete_and_safe(committed):
    assert committed["seeds"] == list(EVALUATION_SEEDS)
    safety = committed["safety"]
    assert safety["safe_returns"] == safety["missions"] and safety["forced_returns"] == 0
    assert safety["missions"] == len(PLANNERS) * len(committed["worlds"]) * len(EVALUATION_SEEDS)
    assert committed["provenance"]["biology"] == "simulated"


def test_committed_results_use_the_frozen_settings(committed):
    assert committed["frozen_config"]["abyss"] == DEFAULT_CONFIG


def test_committed_results_report_worlds_where_abyss_does_not_win(committed):
    unclear = [
        w for w in committed["worlds"]
        if w["abyss_vs"]["frontier"]["ci95_low"] <= 0 <= w["abyss_vs"]["frontier"]["ci95_high"]
    ]
    assert unclear


def test_results_document_matches_the_committed_benchmark(committed):
    document = Path("docs/results.md").read_text(encoding="utf-8")
    assert document == render(committed)


def test_abyss_mission_replays_identically_in_a_benchmark_job():
    config = replace(SimConfig(), scenario="ridge", rho=0.5, terrain_source="gebco")
    job = (config, "abyss", 0, {})
    assert run_one(job) == run_one(job)
