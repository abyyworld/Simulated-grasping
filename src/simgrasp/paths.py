"""Canonical filesystem locations.

Everything is derived from the repository root so scripts work regardless of
the directory they are launched from.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/simgrasp/paths.py -> src/simgrasp -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

ASSETS_DIR = REPO_ROOT / "assets"
PANDA_DIR = ASSETS_DIR / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"

CONFIG_DIR = REPO_ROOT / "configs"
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
