"""Paired benchmark: every planner runs on the identical hidden world for each seed.

Seed policy: development seeds (tuning) are 0..999; final evaluation uses 1000+.
Never tune planner weights on evaluation seeds.

    python -m abyss.evaluation.benchmark --scenarios random ridge reversed --seeds 10
    python -m abyss.evaluation.benchmark --seed-offset 1000 --seeds 30   # final run
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..config import SCENARIOS, load_config
from ..planners import PLANNERS
from ..run_mission import run_mission, validate_mission_log
from .metrics import mission_metrics

DEFAULT_PLANNERS = (
    "abyss", "frontier", "heuristic", "lawnmower", "random", "abyss-nolearn", "abyss-noinfo",
)


def run_benchmark(config, scenarios, seeds, planners=DEFAULT_PLANNERS, rho=None) -> list[dict]:
    rows = []
    for scenario in scenarios:
        overrides = {"scenario": scenario}
        if rho is not None:
            overrides["rho"] = rho
        world_config = replace(config, **overrides)
        for seed in seeds:
            for name in planners:
                log, reveal = run_mission(world_config, PLANNERS[name](), seed)
                validate_mission_log(log)  # raises on unsafe return or bad accounting
                rows.append({
                    "scenario": scenario, "rho": world_config.rho, "seed": seed, "planner": name,
                    **mission_metrics(log, reveal),
                })
    return rows


def paired_ci(diff: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean paired difference with a 95% bootstrap interval over seeds."""
    rng = np.random.default_rng(seed)
    boots = rng.choice(diff, (n_boot, len(diff))).mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarise(rows: list[dict], reference: str = "abyss", metric: str = "distinct_categories"):
    """Per scenario and planner: mean, sd, paired diff vs reference, CI, win/tie/loss."""
    out = []
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        sub = [r for r in rows if r["scenario"] == scenario]
        by = {}
        for r in sub:
            by.setdefault(r["planner"], {})[r["seed"]] = r[metric]
        seeds = sorted(by[reference])
        ref = np.array([by[reference][s] for s in seeds], float)
        for planner, values in by.items():
            v = np.array([values[s] for s in seeds], float)
            entry = {"scenario": scenario, "planner": planner, "n": len(v), "mean": v.mean(),
                     "sd": v.std(ddof=1) if len(v) > 1 else 0.0}
            if planner != reference:
                d = ref - v
                entry["diff"], entry["ci_low"], entry["ci_high"] = paired_ci(d)
                entry["wins"], entry["ties"] = int((d > 0).sum()), int((d == 0).sum())
                entry["losses"] = len(d) - entry["wins"] - entry["ties"]
            out.append(entry)
    return out


def report(rows, reference="abyss", metric="distinct_categories") -> str:
    lines = [f"metric: {metric}   paired diff = {reference} - planner, 95% bootstrap CI over seeds"]
    current = None
    for e in summarise(rows, reference, metric):
        if e["scenario"] != current:
            current = e["scenario"]
            lines.append(f"\n[{current}]")
        line = f"  {e['planner']:<14} mean {e['mean']:6.2f}  sd {e['sd']:5.2f}"
        if "diff" in e:
            line += (f"   diff {e['diff']:+5.2f} [{e['ci_low']:+5.2f}, {e['ci_high']:+5.2f}]"
                     f"  W/T/L {e['wins']}/{e['ties']}/{e['losses']}")
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-offset", type=int, default=0, help="use 1000 for final evaluation")
    ap.add_argument("--planners", nargs="+", default=list(DEFAULT_PLANNERS), choices=sorted(PLANNERS))
    ap.add_argument("--rho", type=float)
    ap.add_argument("--out", default="results/benchmark.csv")
    a = ap.parse_args()

    seeds = range(a.seed_offset, a.seed_offset + a.seeds)
    rows = run_benchmark(load_config(a.config), a.scenarios, seeds, a.planners, a.rho)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(report(rows))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
