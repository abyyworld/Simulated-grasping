"""Shared fixtures. Compiling the Panda takes ~0.5 s, so models are session-scoped."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session")
def template_model():
    from simgrasp.scene import build_template_model
    return build_template_model()


@pytest.fixture(scope="session")
def env():
    from simgrasp.env import PandaGraspEnv
    e = PandaGraspEnv(image_size=128, base_seed=0)
    yield e
    e.close()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
