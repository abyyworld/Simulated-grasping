"""Policy interface.

Every policy maps an :class:`~simgrasp.env.Observation` to a single
:class:`~simgrasp.grasp.Grasp` in world coordinates. Policies that need
privileged state (the oracle) take the environment; policies that only see
pixels (heuristic, learned) take the observation alone. Keeping that distinction
explicit in the signature is the point: it is what makes the comparison honest.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..env import Observation, PandaGraspEnv
from ..grasp import Grasp


class Policy(Protocol):
    name: str

    def __call__(self, obs: Observation, env: PandaGraspEnv,
                 rng: np.random.Generator) -> Grasp: ...
