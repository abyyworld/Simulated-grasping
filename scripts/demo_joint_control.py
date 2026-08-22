#!/usr/bin/env python3
"""Week 1-2 milestone: drive the arm to commanded joint positions and measure error.

Commands a sequence of joint targets, waits for the servos to settle, and reports
the steady-state error per joint. Writes a plot of the tracking to media/.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import mujoco
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=int, default=6, help="number of random joint targets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true", help="save a tracking plot to media/")
    args = ap.parse_args()

    from simgrasp.controllers import ArmInterface, CartesianController
    from simgrasp.paths import MEDIA_DIR
    from simgrasp.scene import ARM_JOINTS, HOME_QPOS, build_template_model

    model = build_template_model()
    data = mujoco.MjData(model)
    arm = ArmInterface(model)
    ctrl = CartesianController(model, data, arm)
    ctrl.reset_to(HOME_QPOS)

    rng = np.random.default_rng(args.seed)
    lo, hi = arm.joint_range[:, 0], arm.joint_range[:, 1]
    history: list[tuple[float, np.ndarray, np.ndarray]] = []

    def record(_m, d):
        history.append((d.time, arm.get_arm_qpos(d), d.ctrl[arm.arm_act].copy()))

    print(f"{'target':>7}  " + "".join(f"{j:>9}" for j in ARM_JOINTS) + f"{'max err':>11}")
    errors = []
    for k in range(args.targets):
        # Stay inside 70% of each joint's range so targets are comfortably reachable.
        mid, half = 0.5 * (lo + hi), 0.35 * (hi - lo)
        target = mid + rng.uniform(-1.0, 1.0, size=7) * half
        ctrl.move_to_joint(target, duration=1.5, hook=record)
        ctrl.settle(0.5, hook=record)
        err = arm.get_arm_qpos(data) - target
        errors.append(np.abs(err))
        print(f"{k:>7}  " + "".join(f"{e * 1000:>9.3f}" for e in err) + f"{np.abs(err).max() * 1000:>10.3f}")

    errors = np.array(errors)
    print()
    print(f"steady-state joint error: mean {errors.mean() * 1000:.3f} mrad, "
          f"max {errors.max() * 1000:.3f} mrad over {args.targets} targets")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.array([h[0] for h in history])
        q = np.array([h[1] for h in history])
        c = np.array([h[2] for h in history])
        fig, axes = plt.subplots(7, 1, figsize=(9, 12), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(t, c[:, i], "--", lw=1.0, label="command")
            ax.plot(t, q[:, i], lw=1.2, label="measured")
            ax.set_ylabel(f"joint{i + 1}\n[rad]", fontsize=8)
            if i == 0:
                ax.legend(fontsize=8, ncol=2)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle("Panda joint servo tracking")
        fig.tight_layout()
        MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        out = MEDIA_DIR / "joint_tracking.png"
        fig.savefig(out, dpi=110)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
