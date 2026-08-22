"""Rigid-body transform helpers.

Conventions used throughout this repository
-------------------------------------------
* Quaternions are MuJoCo-ordered: ``[w, x, y, z]``.
* Rotation matrices are 3x3, column-major in the sense that ``R[:, i]`` is the
  world-frame direction of the i-th body axis.
* "Top-down grasp yaw" ``theta`` is a rotation about the **world z** axis. It
  defines the direction along which the two Panda fingers close.

Panda gripper frame
-------------------
In the Menagerie Panda MJCF the ``hand`` body has:
* +z pointing *out of the flange*, i.e. the approach direction, and
* the two prismatic finger joints sliding along +/- y.

So a top-down grasp with yaw ``theta`` needs the hand rotated such that
``z_hand = -z_world`` and ``y_hand = (cos theta, sin theta, 0)``.
"""

from __future__ import annotations

import numpy as np


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> MuJoCo quaternion [w,x,y,z] (Shepperd's method)."""
    m = np.asarray(mat, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_to_mat(rx: float, ry: float, rz: float) -> np.ndarray:
    """Intrinsic XYZ Euler angles -> rotation matrix. Used for object spawn poses."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def topdown_grasp_mat(theta: float) -> np.ndarray:
    """Hand rotation matrix for a top-down grasp with yaw ``theta``.

    Columns are the world-frame hand axes:
        x_hand = (-sin t,  cos t, 0)
        y_hand = ( cos t,  sin t, 0)   <- finger closing direction
        z_hand = (     0,      0, -1)  <- approach direction (downwards)
    """
    c, s = np.cos(theta), np.sin(theta)
    x_hand = np.array([-s, c, 0.0])
    y_hand = np.array([c, s, 0.0])
    z_hand = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x_hand, y_hand, z_hand])


def rotation_error(mat_current: np.ndarray, mat_target: np.ndarray) -> np.ndarray:
    """Axis-angle rotation error (world frame) taking ``current`` to ``target``.

    Returned vector has magnitude equal to the rotation angle in radians, which
    is what the IK solver stacks under the translational error.
    """
    err = mat_target @ np.asarray(mat_current, dtype=np.float64).T
    # Convert the residual rotation matrix to axis-angle.
    cos_angle = np.clip((np.trace(err) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3)
    if angle > np.pi - 1e-6:
        # Near-180 degree rotation: use the eigenvector of err for eigenvalue +1.
        eigvals, eigvecs = np.linalg.eig(err)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        axis = np.real(eigvecs[:, idx])
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array([err[2, 1] - err[1, 2], err[0, 2] - err[2, 0], err[1, 0] - err[0, 1]])
    return axis * (angle / (2.0 * np.sin(angle)))


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def wrap_grasp_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap to [-pi/2, pi/2).

    A parallel-jaw grasp is symmetric under a 180 degree rotation, so ``theta``
    and ``theta + pi`` are the same grasp. Every angle stored in the dataset is
    wrapped into this half-open interval.
    """
    return (np.asarray(angle) + np.pi / 2.0) % np.pi - np.pi / 2.0
