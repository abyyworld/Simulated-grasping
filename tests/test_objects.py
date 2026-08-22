from __future__ import annotations

import numpy as np
import pytest

from simgrasp.objects import (
    ALL_CATEGORIES,
    SAFE_GRASP_WIDTH,
    SEEN_CATEGORIES,
    UNSEEN_CATEGORIES,
    sample_category,
    sample_object,
)


def test_splits_are_disjoint_and_cover_everything():
    assert set(SEEN_CATEGORIES).isdisjoint(UNSEEN_CATEGORIES)
    assert set(ALL_CATEGORIES) == set(SEEN_CATEGORIES) | set(UNSEEN_CATEGORIES)


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_sampled_objects_are_graspable_and_sit_on_the_table(category, rng):
    for _ in range(150):
        spec = sample_object(rng, category)
        assert spec.geoms, "object must have at least one geom"
        for hint in spec.grasp_hints:
            assert 0.0 < hint.width <= SAFE_GRASP_WIDTH
            # The grasp height must lie inside the object's vertical extent.
            assert 0.0 < hint.grasp_z <= hint.surface_z + 1e-9
        assert spec.top_z > 0
        assert 0 < spec.density
        # Geoms are defined with z = 0 at the table surface and must not sink
        # below it, since spawning places the body origin on the table.
        for g in spec.geoms:
            assert g.pos[2] >= -1e-9


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_sampling_is_deterministic_given_a_seed(category):
    a = sample_object(np.random.default_rng(3), category)
    b = sample_object(np.random.default_rng(3), category)
    assert a == b


def test_sample_category_respects_the_split(rng):
    for _ in range(200):
        assert sample_category(rng, "seen") in SEEN_CATEGORIES
        assert sample_category(rng, "unseen") in UNSEEN_CATEGORIES
        assert sample_category(rng, "all") in ALL_CATEGORIES


def test_ellipsoids_rest_on_their_smallest_axis(rng):
    """Standing an ellipsoid on its longest axis is an unstable equilibrium."""
    for _ in range(200):
        spec = sample_object(rng, "ellipsoid")
        a, b, c = spec.geoms[0].size
        assert c <= a and c <= b
