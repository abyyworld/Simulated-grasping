"""Grasp policies: what to grasp, given an observation.

``LearnedPolicy`` is intentionally **not** imported here. It pulls in torch,
and the data-collection and baseline workers -- which never touch it -- would
otherwise pay a ~2 s import per spawned process. Import it directly, or go
through ``simgrasp.evaluation.build_policy("cnn")``.
"""

from .base import Policy
from .heuristic import HeuristicPolicy
from .oracle import GraspSampler, OraclePolicy

__all__ = ["Policy", "OraclePolicy", "GraspSampler", "HeuristicPolicy"]
