#!/usr/bin/env python3
"""Open the interactive MuJoCo viewer on a randomly generated scene.

Needs a display. On macOS run it with ``mjpython scripts/view_scene.py``.
Headless machines should use ``scripts/make_media.py`` instead.
"""

from __future__ import annotations

import argparse
import os

# The interactive viewer needs a real window, so do not let _bootstrap force a
# headless backend.
os.environ.setdefault("MUJOCO_GL", "glfw")

import _bootstrap  # noqa: E402, F401


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--category", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import mujoco.viewer

    from simgrasp.env import PandaGraspEnv

    env = PandaGraspEnv(base_seed=args.seed)
    env.reset(args.episode, category=args.category)
    print(f"episode {args.episode}: {env.state.category} at "
          f"({env.state.object_xy[0]:.3f}, {env.state.object_xy[1]:.3f})")
    print("close the window to exit")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            env.controller.step(10)
            viewer.sync()
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
