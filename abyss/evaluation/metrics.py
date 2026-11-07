"""Per-mission metrics and paired comparisons, computed from executed missions."""

from __future__ import annotations

import numpy as np

CURVE_STEP = 10.0
N_BOOTSTRAP = 10_000


def mission_metrics(log: dict, reveal: dict) -> dict:
    """Score one mission against true labels, including its discovery curve over energy."""
    truth = {obs["step"]: obs["categories"] for obs in reveal["true_observations"]}
    seen: set[str] = set()
    events = [(0.0, 0)]
    first_discovery = None
    for step in log["steps"]:
        categories = truth.get(step["step"], [])
        if set(categories) - seen:
            seen.update(categories)
            used = step["cumulative"]["energy_used"]
            events.append((used, len(seen)))
            if first_discovery is None:
                first_discovery = used

    budget = log["energy"]["budget"]
    grid = np.arange(0.0, budget + CURVE_STEP, CURVE_STEP)
    energies = np.array([e for e, _ in events])
    counts = np.array([c for _, c in events])
    curve = counts[np.searchsorted(energies, grid, side="right") - 1]

    outcome = log["outcome"]
    return {
        "distinct_categories": reveal["true_distinct_count"],
        "categories_available": reveal["categories_available_count"],
        "surveys": outcome["surveys"],
        "waypoints_surveyed": outcome["waypoints_surveyed"],
        "energy_used": outcome["energy_used"],
        "discoveries_per_energy": reveal["true_distinct_count"] / max(outcome["energy_used"], 1e-9),
        "energy_to_first_discovery": first_discovery,
        "safe_return": outcome["safe_return"],
        "forced_return": outcome["forced_return"],
        "curve": curve.tolist(),
    }


def paired_comparison(ours: np.ndarray, theirs: np.ndarray, seed: int = 0) -> dict:
    """Paired difference (ours - theirs) on shared worlds, with a 95% bootstrap interval."""
    differences = np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(len(differences), size=(N_BOOTSTRAP, len(differences)))
    means = differences[samples].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_difference": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
    }
