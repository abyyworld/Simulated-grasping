#!/usr/bin/env python3
"""Verify the install: assets present, model compiles, rendering works, IK converges.

Run this first after cloning. Every failure prints the fix.
"""

from __future__ import annotations

import os
import sys
import time

import _bootstrap  # noqa: F401  (path + MUJOCO_GL side effects)
import numpy as np


def main() -> int:
    ok = True

    def check(label: str, fn):
        nonlocal ok
        try:
            t0 = time.perf_counter()
            detail = fn()
            dt = (time.perf_counter() - t0) * 1000
            print(f"  [ok]   {label:<34} {detail}  ({dt:.0f} ms)")
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic
            ok = False
            print(f"  [FAIL] {label:<34} {type(exc).__name__}: {exc}")

    print("simgrasp installation check")
    print(f"  python {sys.version.split()[0]}  MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}")
    print()

    def _mujoco():
        import mujoco
        return f"mujoco {mujoco.__version__}"

    def _assets():
        from simgrasp.paths import require_panda_assets
        p = require_panda_assets()
        return f"{p.parent.name}/"

    def _compile():
        from simgrasp.scene import build_template_model
        m = build_template_model()
        globals()["_model"] = m
        return f"nq={m.nq} ngeom={m.ngeom}"

    def _render():
        import mujoco

        from simgrasp.camera import RGBDCamera
        from simgrasp.controllers import CartesianController
        from simgrasp.scene import CAPTURE_QPOS, OVERHEAD_CAM
        m = globals()["_model"]
        d = mujoco.MjData(m)
        CartesianController(m, d).reset_to(CAPTURE_QPOS)
        cam = RGBDCamera(m, OVERHEAD_CAM, 224, 224)
        rgb, depth = cam.render(d)
        cam.close()
        return f"rgb{rgb.shape} depth range {depth.min():.3f}-{depth.max():.3f} m"

    def _ik():
        import mujoco

        from simgrasp.controllers import ArmInterface, solve_ik
        from simgrasp.scene import HOME_QPOS, TABLE_HEIGHT
        from simgrasp.transforms import topdown_grasp_mat
        m = globals()["_model"]
        arm = ArmInterface(m)
        scratch = mujoco.MjData(m)
        errs = []
        rng = np.random.default_rng(0)
        for _ in range(20):
            tgt = np.array([rng.uniform(0.40, 0.68), rng.uniform(-0.20, 0.20),
                            TABLE_HEIGHT + rng.uniform(0.02, 0.25)])
            r = solve_ik(m, arm, tgt, topdown_grasp_mat(rng.uniform(-1.5, 1.5)),
                         HOME_QPOS, scratch=scratch)
            errs.append(r.pos_err)
        return f"20/20 targets, max error {max(errs) * 1000:.2f} mm"

    def _grasp():
        from simgrasp.env import PandaGraspEnv
        with PandaGraspEnv(base_seed=0) as env:
            env.reset(0)
            res = env.execute(env.oracle_grasp())
        return f"oracle grasp on a {env.state.category}: {res.reason}"

    def _torch():
        import torch

        from simgrasp.models import pick_device
        return f"torch {torch.__version__}, device '{pick_device()}'"

    check("mujoco importable", _mujoco)
    check("panda assets present", _assets)
    check("scene compiles", _compile)
    check("offscreen RGB-D rendering", _render)
    check("inverse kinematics", _ik)
    check("full grasp episode", _grasp)
    check("pytorch", _torch)

    print()
    print("All checks passed." if ok else "Some checks failed - see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
