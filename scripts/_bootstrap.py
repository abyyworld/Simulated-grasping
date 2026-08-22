"""Make ``src/`` importable and pick a sane MuJoCo render backend.

Importing this module has side effects by design, so every script can start with
``import _bootstrap  # noqa`` and then import ``simgrasp``.
"""

from __future__ import annotations

import ctypes.util
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def choose_gl_backend() -> str:
    """Pick a headless-capable MuJoCo GL backend unless the user set one.

    macOS always has CGL, so MuJoCo's default is right there. On Linux we prefer
    EGL (hardware, needs a GPU driver) and fall back to OSMesa (software, works
    anywhere but is ~50x slower per frame).
    """
    if "MUJOCO_GL" in os.environ:
        return os.environ["MUJOCO_GL"]
    if platform.system() == "Darwin":
        return ""
    if os.environ.get("DISPLAY"):
        return ""
    backend = "osmesa"
    if ctypes.util.find_library("EGL"):
        backend = "egl"
    os.environ["MUJOCO_GL"] = backend
    return backend


choose_gl_backend()
