"""Run real missions and write the logs and manifest the replay UI loads.

    python -m scripts.build_ui_logs

Runs each planner listed below on one world of each type. The first planner listed
fills the left panel and the second fills the right. The seed is fixed in advance and
is not chosen by outcome.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from abyss.config import load_config
from abyss.planners import PLANNERS
from abyss.run_mission import run_mission, validate_mission_log

SCENARIOS = {"ridge": 0.5, "reversed": 0.5, "clustered": 1.0, "nonlinear": 1.0, "random": 0.0}
UI_PLANNERS = ("frontier", "abyss", "lawnmower", "heuristic", "random")
LABELS = {"abyss": "ABYSS"}


def build(config_path: str, out_dir: Path, seed: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        if old.name != "benchmark.json":
            old.unlink()

    base = load_config(config_path)
    entries = []
    for scenario, rho in SCENARIOS.items():
        config = replace(base, scenario=scenario, rho=rho)
        for name in UI_PLANNERS:
            log, reveal = run_mission(config, PLANNERS[name](), seed)
            validate_mission_log(log)
            stem = f"{log['world']['id']}-{name}"
            (out_dir / f"{stem}.json").write_text(
                json.dumps(log, separators=(",", ":")), encoding="utf-8"
            )
            (out_dir / f"{stem}.reveal.json").write_text(
                json.dumps(reveal, separators=(",", ":")), encoding="utf-8"
            )
            entries.append(
                {
                    "world_id": log["world"]["id"],
                    "scenario": scenario,
                    "rho": config.rho,
                    "seed": seed,
                    "planner": name,
                    "label": LABELS.get(name, name.capitalize()),
                    "log": f"logs/{stem}.json",
                    "reveal": f"logs/{stem}.reveal.json",
                }
            )
    manifest = {"entries": entries}
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", type=Path, default=Path("ui/logs"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = build(args.config, args.out, args.seed)
    print(f"Wrote {len(manifest['entries'])} missions to {args.out}")


if __name__ == "__main__":
    main()
