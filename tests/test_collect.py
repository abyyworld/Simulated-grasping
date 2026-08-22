"""Dataset generation: state snapshots and the multi-angle grasp variants.

These two together are what make orientation learnable. If snapshots are not
exact, the K grasps in a scene are not really comparable; if the variants do not
spread over the half turn, they carry no more angle information than one label.
"""

from __future__ import annotations

import numpy as np
import pytest

from simgrasp.collect import _angle_variants
from simgrasp.grasp import Grasp
from simgrasp.transforms import wrap_grasp_angle


def test_snapshot_restore_is_bit_exact(env):
    """Replaying a grasp from a snapshot must reproduce it exactly."""
    env.reset(0, category="box")
    snap = env.snapshot()
    grasp = env.oracle_grasp()

    first = env.execute(grasp)
    env.restore(snap)
    second = env.execute(grasp)

    assert first.reason == second.reason
    assert first.lift_height == pytest.approx(second.lift_height, abs=1e-12)
    assert first.success == second.success


def test_restore_returns_the_object_to_its_settled_pose(env):
    env.reset(0, category="cylinder")
    snap = env.snapshot()
    before, _ = env.randomizer.object_pose(env.data)

    env.execute(env.oracle_grasp())
    moved, _ = env.randomizer.object_pose(env.data)
    assert np.linalg.norm(moved - before) > 0.01, "the grasp should have moved the object"

    env.restore(snap)
    after, _ = env.randomizer.object_pose(env.data)
    assert np.allclose(after, before, atol=1e-12)


def test_angle_variants_share_a_point_and_spread_over_the_half_turn(rng):
    base = Grasp(0.55, 0.02, 0.44, 0.3, 0.03)
    for count in (2, 3, 4, 6):
        variants = _angle_variants(base, count, rng)
        assert len(variants) == count
        # Same grasp point and width: only the orientation varies.
        for g in variants:
            assert (g.x, g.y, g.z, g.width) == (base.x, base.y, base.z, base.width)
        # Distinct orientations, reasonably spread.
        angles = sorted(float(wrap_grasp_angle(g.yaw)) for g in variants)
        gaps = np.diff(angles + [angles[0] + np.pi])
        assert gaps.min() > 0.4 * (np.pi / count), f"variants bunched together: {angles}"


def test_single_variant_is_the_samplers_own_proposal(rng):
    base = Grasp(0.5, 0.0, 0.44, -0.7, 0.04)
    assert _angle_variants(base, 1, rng) == [base]
    # And the first of a multi-variant set is unchanged, so K=1 reproduces the
    # original single-grasp collection exactly.
    assert _angle_variants(base, 4, rng)[0] == base


def test_angle_variants_stay_wrapped(rng):
    base = Grasp(0.5, 0.0, 0.44, 1.5, 0.04)
    for g in _angle_variants(base, 6, rng):
        assert -np.pi / 2 - 1e-9 <= g.yaw < np.pi / 2 + 1e-9
