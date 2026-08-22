from __future__ import annotations

import numpy as np
import pytest

from simgrasp.transforms import (
    euler_to_mat,
    mat_to_quat,
    quat_to_mat,
    rotation_error,
    rotz,
    topdown_grasp_mat,
    wrap_grasp_angle,
    wrap_to_pi,
)


@pytest.mark.parametrize("theta", np.linspace(-np.pi, np.pi, 13))
def test_topdown_grasp_mat_is_a_rotation(theta):
    m = topdown_grasp_mat(theta)
    assert np.allclose(m.T @ m, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(m), 1.0)


@pytest.mark.parametrize("theta", np.linspace(-np.pi, np.pi, 13))
def test_topdown_grasp_axes(theta):
    """z_hand points down (the approach) and y_hand is the closing direction."""
    m = topdown_grasp_mat(theta)
    assert np.allclose(m[:, 2], [0, 0, -1])
    assert np.allclose(m[:, 1], [np.cos(theta), np.sin(theta), 0])


def test_quat_mat_round_trip(rng):
    for _ in range(200):
        m = euler_to_mat(*rng.uniform(-np.pi, np.pi, 3))
        assert np.allclose(quat_to_mat(mat_to_quat(m)), m, atol=1e-10)


def test_rotation_error_recovers_the_rotation(rng):
    for _ in range(100):
        angle = rng.uniform(-np.pi + 1e-3, np.pi - 1e-3)
        err = rotation_error(np.eye(3), rotz(angle))
        assert np.allclose(err, [0, 0, angle], atol=1e-9)


def test_rotation_error_is_zero_for_identical_frames(rng):
    m = euler_to_mat(*rng.uniform(-2, 2, 3))
    assert np.allclose(rotation_error(m, m), 0, atol=1e-9)


def test_rotation_error_handles_180_degrees():
    err = rotation_error(np.eye(3), rotz(np.pi))
    assert np.isclose(np.linalg.norm(err), np.pi, atol=1e-6)


def test_wrap_grasp_angle_is_half_turn_symmetric(rng):
    a = rng.uniform(-10, 10, 500)
    assert np.allclose(wrap_grasp_angle(a), wrap_grasp_angle(a + np.pi), atol=1e-9)
    w = wrap_grasp_angle(a)
    assert np.all(w >= -np.pi / 2 - 1e-12) and np.all(w < np.pi / 2 + 1e-12)


def test_wrap_to_pi():
    assert np.isclose(wrap_to_pi(3 * np.pi), np.pi) or np.isclose(wrap_to_pi(3 * np.pi), -np.pi)
