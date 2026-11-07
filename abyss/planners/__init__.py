"""Planner registry. Add new policies here so the mission runner can find them."""

from .adaptive import AdaptivePlanner, NoInfoPlanner, NoLearnPlanner
from .base import Planner
from .frontier import FrontierPlanner
from .heuristic import TerrainHeuristicPlanner
from .lawnmower import LawnmowerPlanner
from .random import RandomPlanner

PLANNERS: dict[str, type[Planner]] = {
    cls.name: cls
    for cls in (
        RandomPlanner,
        AdaptivePlanner,
        NoLearnPlanner,
        NoInfoPlanner,
        FrontierPlanner,
        TerrainHeuristicPlanner,
        LawnmowerPlanner,
    )
}
