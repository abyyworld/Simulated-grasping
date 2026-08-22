"""Camera geometry. Every grasp label depends on these conversions being exact."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from simgrasp.camera import (
    RGBDCamera,
    camera_pose,
    deproject_pixels,
    depth_to_pointcloud,
    height_map,
    intrinsics_from_model,
    project_points,
)
from simgrasp.controllers import CartesianController
from simgrasp.scene import CAPTURE_QPOS, OVERHEAD_CAM, TABLE_HEIGHT


@pytest.fixture(scope="module")
def rendered(template_model):
    data = mujoco.MjData(template_model)
    CartesianController(template_model, data).reset_to(CAPTURE_QPOS)
    mujoco.mj_forward(template_model, data)
    cam = RGBDCamera(template_model, OVERHEAD_CAM, 128, 128)
    rgb, depth = cam.render(data)
    pos, mat = cam.pose(data)
    out = (rgb, depth, pos, mat, cam.intrinsics)
    cam.close()
    return out


def test_depth_is_perpendicular_distance_not_ray_length(rendered):
    """A flat table must render as a *constant* depth.

    If MuJoCo returned ray length this would fall off as 1/cos(angle) toward the
    corners, and deprojecting with the wrong assumption would bow the table into
    a bowl and put every off-centre grasp at the wrong height.
    """
    _rgb, depth, pos, _mat, _intr = rendered
    expected = float(pos[2] - TABLE_HEIGHT)
    assert np.allclose(depth, expected, atol=1e-5)


def test_projection_and_deprojection_are_inverses(rendered, rng):
    _rgb, _depth, pos, mat, intr = rendered
    pts = np.stack([
        rng.uniform(0.35, 0.75, 200),
        rng.uniform(-0.22, 0.22, 200),
        rng.uniform(TABLE_HEIGHT, TABLE_HEIGHT + 0.2, 200),
    ], axis=1)
    uv, depth = project_points(pts, pos, mat, intr)
    back = deproject_pixels(uv, depth, pos, mat, intr)
    assert np.allclose(back, pts, atol=1e-9)


def test_rendered_depth_matches_analytic_projection(rendered):
    """Cross-check the renderer against the pinhole model at the image centre."""
    _rgb, depth, pos, mat, intr = rendered
    centre = np.array([pos[0], pos[1], TABLE_HEIGHT])
    uv, d = project_points(centre, pos, mat, intr)
    assert np.allclose(uv, [intr.cx, intr.cy], atol=1e-6)
    assert np.isclose(depth[int(round(uv[1])), int(round(uv[0]))], d, atol=1e-5)


def test_height_map_is_zero_on_an_empty_table(rendered):
    _rgb, depth, pos, mat, intr = rendered
    h = height_map(depth, pos, mat, intr, TABLE_HEIGHT)
    assert np.abs(h).max() < 1e-5


def test_pointcloud_matches_per_pixel_deprojection(rendered, rng):
    _rgb, depth, pos, mat, intr = rendered
    cloud = depth_to_pointcloud(depth, pos, mat, intr)
    for _ in range(20):
        v = int(rng.integers(depth.shape[0]))
        u = int(rng.integers(depth.shape[1]))
        one = deproject_pixels(np.array([[u, v]]), np.array([depth[v, u]]), pos, mat, intr)[0]
        assert np.allclose(cloud[v, u], one, atol=1e-9)


def test_intrinsics_have_square_pixels(template_model):
    intr = intrinsics_from_model(template_model, OVERHEAD_CAM, 224, 224)
    assert np.isclose(intr.fx, intr.fy)
    assert np.isclose(intr.cx, (224 - 1) / 2)


def test_camera_looks_straight_down(template_model):
    data = mujoco.MjData(template_model)
    mujoco.mj_forward(template_model, data)
    _pos, mat = camera_pose(data, template_model, OVERHEAD_CAM)
    # MuJoCo cameras look along their own -z; here that must be world -z.
    assert np.allclose(mat[:, 2], [0, 0, 1], atol=1e-9)


def test_object_appears_at_its_projected_pixel(env):
    """End-to-end: an object's true top-centre must land on the right pixel."""
    obs = env.reset(0)
    st = env.state
    top = np.array([st.object_xy[0], st.object_xy[1], st.object_z0 + st.spec.top_z])
    uv, d = project_points(top, obs.cam_pos, obs.cam_mat, obs.intrinsics)
    u, v = int(round(uv[0])), int(round(uv[1]))
    assert 0 <= u < obs.depth.shape[1] and 0 <= v < obs.depth.shape[0]
    assert np.isclose(obs.depth[v, u], d, atol=2e-3)
    assert obs.height[v, u] == pytest.approx(st.spec.top_z, abs=2e-3)
