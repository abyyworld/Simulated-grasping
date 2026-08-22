from __future__ import annotations

import mujoco
import numpy as np
import pytest

from simgrasp.controllers import (
    GRIPPER_MAX_WIDTH,
    ArmInterface,
    CartesianController,
    minimum_jerk,
    solve_ik,
)
from simgrasp.scene import HOME_QPOS, TABLE_HEIGHT
from simgrasp.transforms import rotation_error, topdown_grasp_mat


@pytest.fixture(scope="module")
def arm(template_model):
    return ArmInterface(template_model)


def test_ik_converges_across_the_workspace(template_model, arm):
    scratch = mujoco.MjData(template_model)
    rng = np.random.default_rng(0)
    for _ in range(80):
        target = np.array([rng.uniform(0.40, 0.68), rng.uniform(-0.20, 0.20),
                           TABLE_HEIGHT + rng.uniform(0.01, 0.30)])
        mat = topdown_grasp_mat(rng.uniform(-np.pi / 2, np.pi / 2))
        res = solve_ik(template_model, arm, target, mat, HOME_QPOS, scratch=scratch)
        assert res.success, f"IK failed at {target}, error {res.pos_err}"
        assert res.pos_err < 3e-4
        assert res.rot_err < 5e-3


def test_ik_solution_reproduces_the_target_pose(template_model, arm):
    """The returned joint angles must actually put the TCP where asked."""
    data = mujoco.MjData(template_model)
    target = np.array([0.55, 0.05, TABLE_HEIGHT + 0.12])
    mat = topdown_grasp_mat(0.4)
    res = solve_ik(template_model, arm, target, mat, HOME_QPOS)
    arm.set_arm_qpos(data, res.qpos)
    mujoco.mj_forward(template_model, data)
    pos, rot = arm.tcp_pose(data)
    assert np.allclose(pos, target, atol=5e-4)
    assert np.linalg.norm(rotation_error(rot, mat)) < 5e-3


def test_ik_respects_joint_limits(template_model, arm):
    rng = np.random.default_rng(1)
    scratch = mujoco.MjData(template_model)
    for _ in range(30):
        target = np.array([rng.uniform(0.40, 0.68), rng.uniform(-0.20, 0.20),
                           TABLE_HEIGHT + rng.uniform(0.01, 0.30)])
        res = solve_ik(template_model, arm, target, topdown_grasp_mat(0.0), HOME_QPOS,
                       scratch=scratch)
        assert np.all(res.qpos >= arm.joint_range[:, 0] - 1e-9)
        assert np.all(res.qpos <= arm.joint_range[:, 1] + 1e-9)


def test_servo_reaches_commanded_joint_positions(template_model, arm):
    """With gravity compensation the steady-state joint error must be tiny."""
    data = mujoco.MjData(template_model)
    ctrl = CartesianController(template_model, data, arm)
    ctrl.reset_to(HOME_QPOS)
    rng = np.random.default_rng(2)
    lo, hi = arm.joint_range[:, 0], arm.joint_range[:, 1]
    for _ in range(3):
        target = 0.5 * (lo + hi) + rng.uniform(-0.3, 0.3, 7) * (hi - lo)
        ctrl.move_to_joint(target, duration=1.5)
        ctrl.settle(0.6)
        assert np.abs(arm.get_arm_qpos(data) - target).max() < 2e-3


def test_cartesian_move_reaches_the_target(template_model, arm):
    data = mujoco.MjData(template_model)
    ctrl = CartesianController(template_model, data, arm)
    ctrl.reset_to(HOME_QPOS)
    target = np.array([0.60, -0.10, TABLE_HEIGHT + 0.08])
    mat = topdown_grasp_mat(-0.5)
    ctrl.move_to_pose(target, mat, duration=1.2, cartesian=True, waypoints=8)
    pos, rot = arm.tcp_pose(data)
    assert np.linalg.norm(pos - target) < 1e-3
    assert np.degrees(np.linalg.norm(rotation_error(rot, mat))) < 2.0


def test_gripper_tracks_commanded_width(template_model, arm):
    data = mujoco.MjData(template_model)
    ctrl = CartesianController(template_model, data, arm)
    ctrl.reset_to(HOME_QPOS)
    for width in (0.08, 0.05, 0.02, 0.0):
        ctrl.set_gripper(width, duration=0.8)
        assert arm.gripper_width(data) == pytest.approx(width, abs=1.5e-3)


def test_gripper_width_is_clamped(template_model, arm):
    data = mujoco.MjData(template_model)
    ctrl = CartesianController(template_model, data, arm)
    ctrl.reset_to(HOME_QPOS)
    ctrl.set_gripper(10.0, duration=0.6)
    assert arm.gripper_width(data) <= GRIPPER_MAX_WIDTH + 1e-3


def test_minimum_jerk_profile():
    assert minimum_jerk(0.0) == 0.0
    assert minimum_jerk(1.0) == pytest.approx(1.0)
    assert minimum_jerk(0.5) == pytest.approx(0.5)
    # Monotone, and clamped outside [0, 1].
    xs = np.linspace(0, 1, 50)
    assert np.all(np.diff(minimum_jerk(xs)) >= -1e-12)
    assert minimum_jerk(-1.0) == 0.0 and minimum_jerk(2.0) == pytest.approx(1.0)
