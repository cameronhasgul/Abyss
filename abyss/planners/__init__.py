"""Planner registry. Add new policies here so the mission runner can find them."""

from .adaptive import AdaptivePlanner, NoInformationPlanner, NoLearningPlanner
from .base import Planner
from .frontier import FrontierPlanner
from .heuristic import HeuristicPlanner
from .lawnmower import LawnmowerPlanner
from .random import RandomPlanner

PLANNERS: dict[str, type[Planner]] = {
    planner.name: planner
    for planner in (
        AdaptivePlanner,
        FrontierPlanner,
        HeuristicPlanner,
        LawnmowerPlanner,
        NoInformationPlanner,
        NoLearningPlanner,
        RandomPlanner,
    )
}
BASELINES = ("frontier", "heuristic", "lawnmower", "random")
ABLATIONS = ("abyss_no_info", "abyss_no_learning")
