#!/usr/bin/env python3
"""Week 3-4 milestone: Cartesian control, IK accuracy and gripper open/close.

Drives the TCP through a square in the plane above the table, then to a set of
random top-down poses, reporting the achieved position and orientation error at
each. Also exercises the gripper across its full travel.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import mujoco
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from simgrasp.controllers import ArmInterface, CartesianController
    from simgrasp.scene import HOME_QPOS, TABLE_HEIGHT, build_template_model
    from simgrasp.transforms import rotation_error, topdown_grasp_mat

    model = build_template_model()
    data = mujoco.MjData(model)
    arm = ArmInterface(model)
    ctrl = CartesianController(model, data, arm)
    ctrl.reset_to(HOME_QPOS)

    print("1. Cartesian square, 0.10 m above the table")
    corners = [(0.45, -0.12), (0.45, 0.12), (0.63, 0.12), (0.63, -0.12), (0.45, -0.12)]
    mat = topdown_grasp_mat(0.0)
    for x, y in corners:
        target = np.array([x, y, TABLE_HEIGHT + 0.10])
        ctrl.move_to_pose(target, mat, duration=0.8, cartesian=True, waypoints=8)
        pos, _ = arm.tcp_pose(data)
        print(f"   target ({x:.3f}, {y:.3f})  reached ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
              f"  error {np.linalg.norm(pos - target) * 1000:.2f} mm")

    print("\n2. Random top-down poses")
    rng = np.random.default_rng(args.seed)
    pos_errs, rot_errs = [], []
    print(f"   {'x':>7}{'y':>8}{'z':>8}{'yaw':>9}{'pos err':>11}{'rot err':>11}")
    for _ in range(args.poses):
        x = rng.uniform(0.42, 0.66)
        y = rng.uniform(-0.18, 0.18)
        z = TABLE_HEIGHT + rng.uniform(0.03, 0.25)
        yaw = rng.uniform(-np.pi / 2, np.pi / 2)
        target = np.array([x, y, z])
        tmat = topdown_grasp_mat(yaw)
        ctrl.move_to_pose(target, tmat, duration=1.0, cartesian=False)
        pos, rot = arm.tcp_pose(data)
        pe = np.linalg.norm(pos - target)
        re = np.degrees(np.linalg.norm(rotation_error(rot, tmat)))
        pos_errs.append(pe)
        rot_errs.append(re)
        print(f"   {x:>7.3f}{y:>8.3f}{z:>8.3f}{np.degrees(yaw):>8.1f} deg"
              f"{pe * 1000:>9.2f} mm{re:>9.2f} deg")

    print(f"\n   mean position error {np.mean(pos_errs) * 1000:.2f} mm, "
          f"max {np.max(pos_errs) * 1000:.2f} mm")
    print(f"   mean orientation error {np.mean(rot_errs):.2f} deg, max {np.max(rot_errs):.2f} deg")

    print("\n3. Gripper")
    for width in (0.08, 0.04, 0.0, 0.08):
        ctrl.set_gripper(width, duration=0.6)
        print(f"   commanded {width * 1000:>5.1f} mm  ->  measured "
              f"{arm.gripper_width(data) * 1000:>5.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
