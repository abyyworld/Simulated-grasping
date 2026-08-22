"""Canonical filesystem locations.

Everything is derived from the repository root so scripts work regardless of
the directory they are launched from.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root() -> Path:
    """Locate the checkout by walking up for a marker file.

    ``parents[2]`` is right for the normal layout (src/simgrasp/paths.py) and for
    an editable install, but wrong if the package is ever copied into
    site-packages. Looking for pyproject.toml is correct in both cases, and the
    env var gives an explicit escape hatch.
    """
    override = os.environ.get("SIMGRASP_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    return here.parents[2]


REPO_ROOT = _find_repo_root()

ASSETS_DIR = Path(os.environ.get("SIMGRASP_ASSETS_DIR", REPO_ROOT / "assets"))
PANDA_DIR = ASSETS_DIR / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"

DATA_DIR = Path(os.environ.get("SIMGRASP_DATA_DIR", REPO_ROOT / "data"))
RUNS_DIR = Path(os.environ.get("SIMGRASP_RUNS_DIR", REPO_ROOT / "runs"))
MEDIA_DIR = REPO_ROOT / "media"


def require_panda_assets() -> Path:
    """Return the Panda MJCF path, with an actionable error if it is missing."""
    if not PANDA_XML.exists():
        raise FileNotFoundError(
            f"Franka Panda MJCF not found at {PANDA_XML}.\n"
            "The robot model is vendored from MuJoCo Menagerie and is not committed "
            "to this repository. Fetch it with:\n\n"
            "    make assets\n"
            "    # or: python scripts/fetch_assets.py\n"
        )
    return PANDA_XML
