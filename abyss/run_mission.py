"""Run one mission and export its log.

    python -m abyss.run_mission --scenario ridge --seed 0 --planner random

Writes two files. The mission log holds only public data. The ``.reveal.json`` file
holds hidden truth for the evaluator and the UI's hidden-world reveal.
Logs contain no timestamps, so identical inputs produce identical bytes.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import SCENARIOS, SimConfig, load_config
from .contract import (
    HOME,
    SCHEMA_VERSION,
    ImagePool,
    MissionStatus,
    Perception,
    PublicMap,
    StepResult,
)
from .env import Environment
from .fathomnet_perception import load_fathomnet
from .planners import PLANNERS
from .planners.base import Planner
from .stub_perception import StubImagePool, StubPerception
from .world import STREAM_PLANNER

REQUIRED_LOG_KEYS = {
    "schema_version",
    "provenance",
    "world",
    "planner",
    "energy",
    "terrain",
    "steps",
    "outcome",
}


def planner_seed_for(world_seed: int) -> int:
    """Derive a planner seed from its own stream, independent of the world's stream."""
    sequence = np.random.SeedSequence(entropy=world_seed, spawn_key=(STREAM_PLANNER,))
    return int(sequence.generate_state(1)[0])


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _terrain_payload(public_map: PublicMap) -> dict:
    return {
        "source": public_map.terrain_source,
        "shape": public_map.shape,
        "home": public_map.home,
        "cell_size_m": public_map.cell_size_m,
        "depth_m": np.round(public_map.depth_m, 1),
        "slope_deg": np.round(public_map.slope_deg, 3),
        "bpi_m": np.round(public_map.bpi_m, 1),
    }


def _step_payload(result: StepResult, planned: object, diagnostics: dict, surveys: int) -> dict:
    observation = result.observation
    state = result.state
    return {
        "step": state.step,
        "planned_target": planned,
        "route": result.route,
        "position": state.position,
        "energy_remaining": state.energy_remaining,
        "energy_travel": result.energy_travel,
        "energy_survey": result.energy_survey,
        "observation": None
        if observation is None
        else {
            "waypoint": observation.waypoint,
            "features": dict(zip(observation.feature_names, observation.features)),
            "categories": observation.categories,
            "richness": observation.richness,
            "image_refs": observation.image_refs,
            "novelty_scores": observation.novelty_scores,
            "unknown_flags": observation.unknown_flags,
            "image_labels": observation.image_labels,
            "perception_mode": observation.perception_mode,
        },
        "new_discoveries": result.new_discoveries,
        "cumulative": {
            "distinct_categories": len(state.discovered),
            "unknown_encounters": state.unknown_encounters,
            "surveys": surveys,
            "energy_used": state.energy_used,
        },
        "planner_diagnostics": diagnostics,
    }


def run_mission(
    config: SimConfig,
    planner: Planner,
    seed: int,
    planner_config: dict | None = None,
    planner_seed: int | None = None,
    perception: Perception | None = None,
    image_pool: ImagePool | None = None,
) -> tuple[dict, dict]:
    """Run a mission to completion and return (public mission log, reveal payload)."""
    planner_config = planner_config or {}
    planner_seed = planner_seed_for(seed) if planner_seed is None else planner_seed

    env = Environment(config, perception=perception, image_pool=image_pool)
    state = env.reset(seed)
    public_map = env.public_map
    planner.reset(public_map, planner_config, planner_seed)

    history: list = []
    steps: list[dict] = []
    rejected: list[dict] = []
    forced_return = False

    while state.status is MissionStatus.RUNNING:
        planned = planner.choose_target(state, tuple(history))
        result = env.step(planned)
        if not result.accepted:
            # The simulator overrules a bad target: the robot goes home and the
            # violation is recorded, so a faulty planner cannot strand it or hang.
            rejected.append({"step": state.step, "target": planned, "reason": result.reason})
            forced_return = True
            planned = HOME
            result = env.step(HOME)
        if result.observation is not None:
            history.append(result.observation)
            planner.observe(result.observation)
        steps.append(_step_payload(result, planned, planner.diagnostics(), len(history)))
        state = result.state

    energy_ok = state.energy_remaining >= config.energy_reserve - 1e-9
    log = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "terrain": public_map.terrain_source,
            "biology": "simulated",
            "perception": perception.name if perception else "ideal",
        },
        "world": {
            "id": f"{config.scenario}-rho{config.rho:g}-seed{seed}",
            "scenario": config.scenario,
            "rho": config.rho,
            "seed": seed,
        },
        "planner": {
            "name": planner.name,
            "seed": planner_seed,
            "config": {**planner_config, **planner.describe()},
        },
        "energy": {
            "budget": config.energy_budget,
            "survey_cost": config.survey_cost,
            "reserve": config.energy_reserve,
            "terrain_cost_factor": config.terrain_cost_factor,
        },
        "terrain": _terrain_payload(public_map),
        "steps": steps,
        "outcome": {
            "status": state.status.value,
            "safe_return": state.position == public_map.home and energy_ok,
            "final_position": state.position,
            "energy_remaining": state.energy_remaining,
            "energy_used": state.energy_used,
            "surveys": len(history),
            "waypoints_surveyed": int(state.surveyed.sum()),
            "distinct_categories": len(state.discovered),
            "discovered": state.discovered,
            "unknown_encounters": state.unknown_encounters,
            "forced_return": forced_return,
            "rejected_targets": rejected,
        },
    }
    return _jsonable(log), _jsonable(env.evaluation_record())


def validate_mission_log(log: dict) -> None:
    """Raise ValueError if a mission log breaks the contract."""
    missing = REQUIRED_LOG_KEYS - log.keys()
    if missing:
        raise ValueError(f"Mission log is missing keys: {sorted(missing)}")

    tolerance = 1e-5
    energy = log["energy"]["budget"]
    position = log["terrain"]["home"]
    for step in log["steps"]:
        route = step["route"]
        if route[0] != position:
            raise ValueError(f"Step {step['step']}: route does not start at the robot's position")
        for start, end in zip(route, route[1:]):
            if max(abs(start[0] - end[0]), abs(start[1] - end[1])) != 1:
                raise ValueError(f"Step {step['step']}: route is not eight-connected")
        position = route[-1]
        if step["position"] != position:
            raise ValueError(f"Step {step['step']}: position does not match the route")
        expected = energy - step["energy_travel"] - step["energy_survey"]
        if abs(step["energy_remaining"] - expected) > tolerance:
            raise ValueError(f"Step {step['step']}: energy accounting does not add up")
        energy = step["energy_remaining"]

    outcome = log["outcome"]
    if outcome["status"] != MissionStatus.COMPLETE.value:
        raise ValueError("Mission did not complete")
    if position != log["terrain"]["home"] or not outcome["safe_return"]:
        raise ValueError("Mission did not return home safely")
    if outcome["energy_remaining"] < log["energy"]["reserve"] - tolerance:
        raise ValueError("Mission finished below the energy reserve")


def write_logs(log: dict, reveal: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{log['world']['id']}-{log['planner']['name']}"
    if log["provenance"]["perception"] != "ideal":
        # Model names can hold characters that aren't valid in filenames.
        stem += "-" + re.sub(r"[^A-Za-z0-9.]+", "_", log["provenance"]["perception"])[:40]
    log_path = out_dir / f"{stem}.json"
    reveal_path = out_dir / f"{stem}.reveal.json"
    log_path.write_text(json.dumps(log, separators=(",", ":")), encoding="utf-8")
    reveal_path.write_text(json.dumps(reveal, separators=(",", ":")), encoding="utf-8")
    return log_path, reveal_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--rho", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--planner", choices=sorted(PLANNERS), default="random")
    parser.add_argument("--out", default="results/missions")
    parser.add_argument(
        "--perception", choices=("ideal", "stub", "fathomnet"), default="ideal",
        help="stub is a fake model for testing integration, not computer vision",
    )
    parser.add_argument("--manifest", help="split manifest, for --perception fathomnet")
    parser.add_argument("--artifact", default="data/perception", help="saved perception model")
    args = parser.parse_args()
    if args.perception == "fathomnet" and not args.manifest:
        parser.error("--perception fathomnet needs --manifest")

    config = load_config(args.config)
    overrides = {k: v for k, v in (("scenario", args.scenario), ("rho", args.rho)) if v is not None}
    perception, image_pool = None, None
    if args.perception == "stub":
        overrides["perception_mode"] = "real"
        image_pool = StubImagePool(config.n_categories)
        perception = StubPerception(image_pool)
    elif args.perception == "fathomnet":
        kit = load_fathomnet(args.manifest, args.artifact)
        perception, image_pool = kit["perception"], kit["image_pool"]
        overrides["perception_mode"] = "real"
        overrides["n_categories"] = len(image_pool.categories)
    config = replace(config, **overrides)

    log, reveal = run_mission(
        config, PLANNERS[args.planner](), args.seed, perception=perception, image_pool=image_pool
    )
    validate_mission_log(log)
    log_path, reveal_path = write_logs(log, reveal, args.out)

    outcome = log["outcome"]
    print(f"world:      {log['world']['id']}  ({log['provenance']['terrain']} terrain)")
    print(f"planner:    {log['planner']['name']}")
    print(f"surveys:    {outcome['surveys']}")
    print(
        f"discovered: {reveal['true_distinct_count']} of "
        f"{reveal['categories_available_count']} simulated categories"
    )
    if perception is not None:
        print(f"perceived:  {outcome['distinct_categories']} categories, "
              f"{outcome['unknown_encounters']} unfamiliar images ({perception.name} model)")
    print(f"energy:     {outcome['energy_used']:.1f} used, {outcome['energy_remaining']:.1f} left")
    print(f"safe home:  {outcome['safe_return']}")
    print(f"log:        {log_path}")
    print(f"reveal:     {reveal_path}")


if __name__ == "__main__":
    main()
