"""Compare ABYSS settings on development seeds only, across all world types.

    python -m abyss.evaluation.tune

The score averages over every world type, so settings aren't picked to suit one world.
Copy the chosen settings into DEFAULT_CONFIG in abyss/planners/adaptive.py before
running the benchmark.
"""

from __future__ import annotations

import argparse
from itertools import product

import pandas as pd

from ..config import load_config
from .benchmark import DEVELOPMENT_SEEDS, make_jobs, run_jobs

BETAS = (0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0)
GAMMAS = (0.0, 0.25, 0.5, 1.0)
SPATIAL = (False, True)
TUNING_WORLDS = (("random", 0.0), ("clustered", 1.0), ("ridge", 1.0), ("reversed", 1.0), ("nonlinear", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--betas", type=float, nargs="+", default=BETAS)
    args = parser.parse_args()

    base = load_config(args.config)
    results = []
    for beta, gamma, spatial in product(args.betas, GAMMAS, SPATIAL):
        settings = {"beta": beta, "gamma": gamma, "use_spatial": spatial}
        jobs = make_jobs(
            base, ("abyss",), DEVELOPMENT_SEEDS, TUNING_WORLDS, ("gebco",), settings
        )
        frame = pd.DataFrame(run_jobs(jobs, args.workers))
        by_world = frame.groupby("scenario")["distinct_categories"].mean()
        results.append({**settings, "mean": by_world.mean(), "worst_world": by_world.min(), **by_world})

    table = pd.DataFrame(results).sort_values("mean", ascending=False)
    print(table.round(2).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
