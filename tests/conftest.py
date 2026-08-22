"""Shared fixtures.

Two things happen at import time, before any test runs, and both matter.

**Backend selection.** The suite uses the same probe the scripts use rather than
hard-coding one, so CI and a developer laptop exercise the same path.

**Importing torch and triton first**, via the same helper the scripts use. See
``_bootstrap.preimport_torch_if_needed``: MuJoCo's OSMesa renderer and Triton
each load their own LLVM, and whichever loads second segfaults the process.
Without this the suite crashed with no traceback while every file passed
individually.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _bootstrap import choose_gl_backend, preimport_torch_if_needed  # noqa: E402

preimport_torch_if_needed(choose_gl_backend())

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
