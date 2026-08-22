from __future__ import annotations

import numpy as np
import pytest

from simgrasp.camera import CameraIntrinsics
from simgrasp.grasp import (
    GRASP_DEPTH,
    MAX_APPROACH_DEPTH,
    MIN_TCP_HEIGHT,
    Grasp,
    ImageGrasp,
    grasp_to_image,
    grasp_z_from_surface,
    image_to_grasp,
)
from simgrasp.transforms import wrap_grasp_angle

INTR = CameraIntrinsics(fx=251.6, fy=251.6, cx=111.5, cy=111.5, width=224, height=224)
CAM_POS = np.array([0.54, 0.0, 0.95])
CAM_MAT = np.eye(3)


def test_image_world_round_trip(rng):
    for _ in range(300):
        g = Grasp(x=float(rng.uniform(0.42, 0.66)), y=float(rng.uniform(-0.18, 0.18)),
                  z=float(rng.uniform(0.41, 0.50)),
                  yaw=float(wrap_grasp_angle(rng.uniform(-np.pi, np.pi))),
                  width=float(rng.uniform(0.01, 0.07)))
        img = grasp_to_image(g, CAM_POS, CAM_MAT, INTR)
        back = image_to_grasp(img, CAM_POS, CAM_MAT, INTR)
        assert back.x == pytest.approx(g.x, abs=1e-9)
        assert back.y == pytest.approx(g.y, abs=1e-9)
        assert back.z == pytest.approx(g.z, abs=1e-9)
        assert back.width == pytest.approx(g.width, abs=1e-9)
        assert wrap_grasp_angle(back.yaw - g.yaw) == pytest.approx(0.0, abs=1e-9)


def test_image_angles_stay_wrapped(rng):
    for _ in range(200):
        g = Grasp(0.54, 0.0, 0.44, float(rng.uniform(-10, 10)), 0.04)
        img = grasp_to_image(g, CAM_POS, CAM_MAT, INTR)
        assert -np.pi / 2 - 1e-9 <= img.angle < np.pi / 2 + 1e-9


def test_overhead_camera_flips_the_angle_sign():
    """Image +v runs along world -y, so an overhead view mirrors the yaw."""
    for yaw in (0.3, -0.7, 1.2):
        img = grasp_to_image(Grasp(0.54, 0.0, 0.44, yaw, 0.04), CAM_POS, CAM_MAT, INTR)
        assert img.angle == pytest.approx(float(wrap_grasp_angle(-yaw)), abs=1e-9)


def test_grasp_height_respects_the_table():
    """A flat object must not drive the fingertips through the table."""
    z = grasp_z_from_surface(0.40 + 0.004, table_z=0.40)
    assert z >= 0.40 + MIN_TCP_HEIGHT - 1e-12


def test_grasp_height_bites_the_side_of_a_tall_object():
    top = 0.40 + 0.10
    z = grasp_z_from_surface(top, table_z=0.40, mid_z_world=0.40 + 0.05)
    assert z <= top - GRASP_DEPTH
    assert z >= top - MAX_APPROACH_DEPTH - 1e-12


def test_grasp_height_targets_below_mid_height():
    """The calibrated bias must lower the grasp relative to mid-height."""
    # A 40 mm tall object: none of the three limits bind, so the bias applies.
    top, table, mid = 0.44, 0.40, 0.42
    z = grasp_z_from_surface(top, table, mid_z_world=mid, bias=0.005)
    assert z == pytest.approx(mid - 0.005)


def test_hand_clearance_overrides_the_bias_on_tall_objects():
    """A 100 mm object cannot be gripped at mid-height: the hand would hit its top."""
    top, table, mid = 0.50, 0.40, 0.45
    z = grasp_z_from_surface(top, table, mid_z_world=mid, bias=0.005)
    assert z == pytest.approx(top - MAX_APPROACH_DEPTH)
    assert z > mid, "the hand-clearance limit must win here"


def test_grasp_height_default_mid_is_half_the_height():
    a = grasp_z_from_surface(0.48, 0.40)
    b = grasp_z_from_surface(0.48, 0.40, mid_z_world=0.44)
    assert a == pytest.approx(b)


def test_preshape_opens_wider_than_the_object():
    g = Grasp(0.5, 0.0, 0.44, 0.0, 0.03)
    assert g.preshape_width() > g.width
    assert g.preshape_width() <= 0.08


def test_grasp_array_round_trip(rng):
    g = Grasp(*rng.uniform(-1, 1, 5))
    assert Grasp.from_array(g.as_array()).as_array() == pytest.approx(g.as_array(), abs=1e-6)


def test_image_grasp_is_serialisable():
    img = ImageGrasp(u=10.0, v=20.0, angle=0.3, width_px=12.0, depth=0.5)
    assert img.as_array().shape == (5,)
