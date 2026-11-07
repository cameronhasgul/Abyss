"""Metrics computed from a mission log and its reveal payload (the trusted evaluator).

The primary metric is ``distinct_categories``: true simulated categories the robot
encountered, taken from the reveal file so real-perception errors cannot inflate it.
"""

from __future__ import annotations


def mission_metrics(log: dict, reveal: dict) -> dict:
    outcome = log["outcome"]
    distinct = reveal["true_distinct_count"]
    used = outcome["energy_used"]
    height, width = log["terrain"]["shape"]

    seen: set[str] = set()
    first_discovery = None
    curve = []  # (energy used, cumulative true distinct) after each survey
    for obs in reveal["true_observations"]:
        seen.update(obs["categories"])
        curve.append(len(seen))
    surveys = [s for s in log["steps"] if s["observation"] is not None]
    for step, count in zip(surveys, curve):
        if count and first_discovery is None:
            first_discovery = step["cumulative"]["energy_used"]

    return {
        "distinct_categories": distinct,
        "available_categories": reveal["categories_available_count"],
        "perceived_categories": outcome["distinct_categories"],
        "surveys": outcome["surveys"],
        "energy_used": used,
        "distinct_per_energy": distinct / used if used > 0 else 0.0,
        "energy_to_first_discovery": first_discovery,
        "coverage": outcome["waypoints_surveyed"] / (height * width),
        "mean_survey_richness": (
            sum(s["observation"]["richness"] for s in surveys) / len(surveys) if surveys else 0.0
        ),
        "safe_return": outcome["safe_return"],
        "forced_return": outcome["forced_return"],
    }
