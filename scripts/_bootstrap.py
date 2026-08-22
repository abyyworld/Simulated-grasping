"""Make ``src/`` importable and pick a MuJoCo render backend that actually works.

Importing this module has side effects by design, so every script can start with
``import _bootstrap  # noqa`` and then import ``simgrasp``.
"""

from __future__ import annotations

import ctypes.util
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Probing costs ~1 s, so the answer is cached per machine. Delete the file to
# re-probe (after installing a GPU driver, say).
_CACHE = ROOT / ".simgrasp_gl_backend"

_PROBE = (
    "import mujoco;"
    "c = mujoco.GLContext(64, 64); c.make_current();"
    "m = mujoco.MjModel.from_xml_string("
    "'<mujoco><worldbody><geom type=\"plane\" size=\"1 1 .1\"/></worldbody></mujoco>');"
    "r = mujoco.Renderer(m, 32, 32); r.update_scene(mujoco.MjData(m)); r.render();"
    "print('ok')"
)


def _backend_works(name: str) -> bool:
    """Render one frame in a subprocess with ``MUJOCO_GL=name``.

    A subprocess, because MuJoCo binds its GL backend at import time: once
    ``mujoco`` has been imported with a broken backend, this process cannot try
    another one.
    """
    env = {**os.environ, "MUJOCO_GL": name}
    try:
        done = subprocess.run([sys.executable, "-c", _PROBE], env=env, timeout=90,
                              capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0 and "ok" in done.stdout


def choose_gl_backend(verbose: bool = False) -> str:
    """Select a working MuJoCo GL backend unless the user already set one.

    Per platform:

    * **Windows** -- MuJoCo uses WGL, always present. Leave ``MUJOCO_GL`` unset;
      forcing "egl" or "osmesa" would break an otherwise fine install.
    * **macOS** -- CGL, same reasoning.
    * **Linux with a display** -- GLX via the running X or Wayland session.
    * **Linux headless** -- EGL when it works (hardware accelerated), otherwise
      OSMesa (software: works anywhere, ~50x slower per frame).

    The EGL case is why this probes rather than just checking for the library.
    Cloud VMs and CI runners routinely ship ``libEGL.so`` with no usable driver
    behind it, so "the library exists" says nothing about whether a context can
    be created. The probe renders an actual frame.

    Returns the backend name, or "" when MuJoCo's own default is used.
    """
    if "MUJOCO_GL" in os.environ:
        return os.environ["MUJOCO_GL"]
    if platform.system() in ("Windows", "Darwin"):
        return ""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return ""

    if _CACHE.exists():
        cached = _CACHE.read_text().strip()
        if cached:
            os.environ["MUJOCO_GL"] = cached
            return cached

    candidates = ["egl", "osmesa"] if ctypes.util.find_library("EGL") else ["osmesa"]
    for name in candidates:
        if verbose:
            print(f"[bootstrap] probing MUJOCO_GL={name} ...", flush=True)
        if _backend_works(name):
            os.environ["MUJOCO_GL"] = name
            try:
                _CACHE.write_text(name + "\n")
            except OSError:
                pass  # read-only checkout: just re-probe next time
            if verbose:
                print(f"[bootstrap] using MUJOCO_GL={name}")
            return name

    raise RuntimeError(
        "No working MuJoCo render backend found on this machine.\n"
        "On headless Linux install one of:\n"
        "    sudo apt-get install -y libosmesa6      # software, works anywhere\n"
        "    sudo apt-get install -y libegl1         # needs a GPU driver\n"
        "Then delete .simgrasp_gl_backend and retry, or set MUJOCO_GL yourself."
    )


choose_gl_backend()
