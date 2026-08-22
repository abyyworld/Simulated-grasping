from __future__ import annotations

import numpy as np
import pytest

from simgrasp.camera import CameraIntrinsics
from simgrasp.heightmap import (
    estimate_width_px,
    metres_per_pixel,
    object_mask,
    principal_axis,
    surface_height,
)

INTR = CameraIntrinsics(fx=251.6, fy=251.6, cx=111.5, cy=111.5, width=224, height=224)


def _bar(height=224, width=224, half_w=8, half_h=30, value=0.03):
    """A rectangular bar centred in the image, long along v (image y)."""
    h = np.zeros((height, width), np.float32)
    cy, cx = height // 2, width // 2
    h[cy - half_h:cy + half_h, cx - half_w:cx + half_w] = value
    return h, (cx, cy)


def test_object_mask_thresholds_the_table():
    h, _ = _bar()
    m = object_mask(h)
    assert m.sum() == 16 * 60
    assert not m[0, 0]


def test_surface_height_takes_the_local_max():
    h, (cx, cy) = _bar()
    assert surface_height(h, cx, cy) == pytest.approx(0.03)
    # Just outside the bar, the patch max still sees it.
    assert surface_height(h, cx + 9, cy, patch=3) == pytest.approx(0.03)
    assert surface_height(h, cx + 40, cy, patch=1) == pytest.approx(0.0)


def test_estimate_width_measures_across_the_bar():
    h, (cx, cy) = _bar(half_w=8)
    # Angle 0 runs along image +u, i.e. across the bar's short axis.
    assert estimate_width_px(h, cx, cy, angle=0.0) == pytest.approx(16.0, abs=1.5)
    # Along the bar it should measure the long axis instead.
    assert estimate_width_px(h, cx, cy, angle=np.pi / 2) == pytest.approx(60.0, abs=2.0)


def test_estimate_width_is_zero_off_the_object():
    h, _ = _bar()
    assert estimate_width_px(h, 5, 5, angle=0.0) == 0.0


def test_principal_axis_finds_the_long_direction():
    h, (cx, cy) = _bar(half_w=6, half_h=40)
    centroid, major, minor = principal_axis(object_mask(h))
    assert centroid[0] == pytest.approx(cx, abs=1.0)
    assert centroid[1] == pytest.approx(cy, abs=1.0)
    # The bar is long along v, so the major axis is +/- 90 degrees.
    assert abs(abs(np.degrees(major)) - 90) < 2
    # The jaw should close along the minor axis, i.e. across the bar.
    assert abs(np.degrees(minor)) % 180 < 2


def test_principal_axis_handles_an_empty_mask():
    centroid, _major, _minor = principal_axis(np.zeros((32, 32), bool))
    assert centroid.shape == (2,)


def test_metres_per_pixel_matches_the_camera():
    mpp = metres_per_pixel(INTR, 0.55)
    assert mpp == pytest.approx(0.55 / 251.6)
    # 224 px across should span the designed ~0.49 m field of view.
    assert 224 * mpp == pytest.approx(0.49, abs=0.01)
