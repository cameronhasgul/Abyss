"""Run every planner on shared worlds and summarise the results.

    python -m abyss.evaluation.tune       # development seeds only
    python -m abyss.evaluation.benchmark  # frozen settings, evaluation seeds

Development and evaluation seeds never overlap. Every world in the grid below is
reported, including those where ABYSS does not win.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import SimConfig, load_config
from ..planners import ABLATIONS, BASELINES, PLANNERS
from ..run_mission import run_mission
from .metrics import CURVE_STEP, mission_metrics, paired_comparison

EVALUATION_SEEDS = tuple(range(50))
DEVELOPMENT_SEEDS = tuple(range(1000, 1020))
WORLDS = (
    ("random", 0.0),
    ("clustered", 1.0),
    ("ridge", 0.25),
    ("ridge", 0.5),
    ("ridge", 1.0),
    ("reversed", 0.25),
    ("reversed", 0.5),
    ("reversed", 1.0),
    ("nonlinear", 1.0),
)
TERRAINS = ("gebco", "synthetic")


def run_one(job: tuple) -> dict:
    config, planner_name, seed, planner_config = job
    log, reveal = run_mission(
        config, PLANNERS[planner_name](), seed, planner_config={"log_maps": False, **planner_config}
    )
    return {
        "terrain": config.terrain_source,
        "scenario": config.scenario,
        "rho": config.rho,
        "seed": seed,
        "planner": planner_name,
        **mission_metrics(log, reveal),
    }


def run_jobs(jobs: list[tuple], workers: int | None = None) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, jobs, chunksize=16))


def make_jobs(
    base: SimConfig, planners, seeds, worlds=WORLDS, terrains=TERRAINS, planner_config=None
) -> list[tuple]:
    jobs = []
    for terrain in terrains:
        for scenario, rho in worlds:
            config = replace(base, terrain_source=terrain, scenario=scenario, rho=rho)
            for name in planners:
                for seed in seeds:
                    jobs.append((config, name, seed, planner_config or {}))
    return jobs


def summarise(rows: list[dict], seeds, frozen: dict) -> dict:
    frame = pd.DataFrame(rows)
    groups = []
    for (terrain, scenario, rho), world in frame.groupby(["terrain", "scenario", "rho"], sort=False):
        planners = {}
        for name, runs in world.groupby("planner", sort=False):
            runs = runs.sort_values("seed")
            values = runs["distinct_categories"].to_numpy(dtype=float)
            planners[name] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=1)),
                "surveys_mean": float(runs["surveys"].mean()),
                "energy_used_mean": float(runs["energy_used"].mean()),
                "energy_to_first_mean": float(runs["energy_to_first_discovery"].mean()),
                "curve_mean": np.mean(np.stack(runs["curve"].to_numpy()), axis=0).round(4).tolist(),
            }
        abyss = world[world["planner"] == "abyss"].sort_values("seed")
        comparisons = {}
        for other in (*BASELINES, *ABLATIONS):
            theirs = world[world["planner"] == other].sort_values("seed")
            comparisons[other] = paired_comparison(
                abyss["distinct_categories"].to_numpy(), theirs["distinct_categories"].to_numpy()
            )
        groups.append(
            {
                "terrain": terrain,
                "scenario": scenario,
                "rho": rho,
                "planners": planners,
                "abyss_vs": comparisons,
            }
        )
    return {
        "provenance": {
            "biology": "simulated",
            "energy": "abstract units",
            "metric": "distinct categories found within the energy budget, scored on true labels",
            "perception": "ideal: labels come directly from the simulator",
            "interval": "95% paired bootstrap across seeds on shared worlds",
            "note": "GEBCO runs share one terrain crop, so they show no regional generalisation",
        },
        "seeds": list(seeds),
        "frozen_config": frozen,
        "curve_step": CURVE_STEP,
        "safety": {
            "missions": len(frame),
            "safe_returns": int(frame["safe_return"].sum()),
            "forced_returns": int(frame["forced_return"].sum()),
        },
        "worlds": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--ui-out", type=Path, default=Path("ui/logs/benchmark.json"))
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    base = load_config(args.config)
    planners = ("abyss", *BASELINES, *ABLATIONS)
    rows = run_jobs(make_jobs(base, planners, EVALUATION_SEEDS), args.workers)

    summary = summarise(rows, EVALUATION_SEEDS, _frozen_settings(base))
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns="curve").to_csv(args.out / "runs.csv", index=False)
    text = json.dumps(summary, indent=1)
    (args.out / "summary.json").write_text(text, encoding="utf-8")
    args.ui_out.parent.mkdir(parents=True, exist_ok=True)
    args.ui_out.write_text(text, encoding="utf-8")
    print(f"{len(rows)} missions. Safe returns: {summary['safety']['safe_returns']}")


def _frozen_settings(base: SimConfig) -> dict:
    from ..planners.adaptive import DEFAULT_CONFIG

    return {"abyss": DEFAULT_CONFIG, "simulation": base.to_dict()}


if __name__ == "__main__":
    main()
