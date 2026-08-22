from __future__ import annotations

import numpy as np
import pytest

from simgrasp.grasp import Grasp
from simgrasp.objects import ALL_CATEGORIES, SEEN_CATEGORIES, UNSEEN_CATEGORIES
from simgrasp.scene import WORKSPACE_X, WORKSPACE_Y


def test_reset_is_deterministic(env):
    a = env.reset(42)
    state_a = env.state
    b = env.reset(42)
    state_b = env.state
    assert state_a.category == state_b.category
    assert state_a.object_xy == pytest.approx(state_b.object_xy)
    assert np.array_equal(a.rgb, b.rgb)
    assert np.allclose(a.depth, b.depth)


def test_different_episodes_differ(env):
    env.reset(1)
    first = (env.state.category, env.state.object_xy)
    env.reset(2)
    assert (env.state.category, env.state.object_xy) != first


def test_observation_is_well_formed(env):
    obs = env.reset(0)
    assert obs.rgb.dtype == np.uint8 and obs.rgb.shape[-1] == 3
    assert obs.depth.shape == obs.rgb.shape[:2]
    assert obs.height.shape == obs.depth.shape
    assert np.isfinite(obs.depth).all()
    assert obs.object_mask().sum() > 20, "the object should be visible"


def test_object_lands_inside_the_workspace(env):
    for ep in range(12):
        env.reset(ep)
        x, y = env.state.object_xy
        assert WORKSPACE_X[0] - 0.08 <= x <= WORKSPACE_X[1] + 0.08
        assert WORKSPACE_Y[0] - 0.08 <= y <= WORKSPACE_Y[1] + 0.08


def test_objects_settle_before_the_image_is_captured(env):
    for ep in range(12):
        env.reset(ep)
        assert env.state.settle_displacement < 0.01


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_oracle_grasp_is_on_the_object(env, category):
    env.reset(0, category=category)
    st = env.state
    g = env.oracle_grasp()
    # The grasp point must be within the object's footprint and above the table.
    assert np.hypot(g.x - st.object_xy[0], g.y - st.object_xy[1]) <= st.spec.footprint_radius + 1e-6
    assert env.table_z < g.z <= st.object_z0 + st.spec.top_z
    assert 0 < g.width <= 0.07


def test_split_selection(env):
    for ep in range(6):
        env.reset(ep, split="seen")
        assert env.state.category in SEEN_CATEGORIES
        env.reset(ep, split="unseen")
        assert env.state.category in UNSEEN_CATEGORIES


def test_unreachable_grasp_fails_fast_without_moving(env):
    env.reset(0)
    far = Grasp(x=1.6, y=0.0, z=env.table_z + 0.05, yaw=0.0, width=0.04)
    result = env.execute(far)
    assert not result.success
    assert result.reason == "ik_failed"


def test_grasping_bare_table_fails(env):
    """A grasp on empty table must close on nothing -- this is the free-negative premise."""
    env.reset(0)
    st = env.state
    # Pick a spot well away from the object but inside the workspace.
    x = WORKSPACE_X[0] if st.object_xy[0] > 0.54 else WORKSPACE_X[1]
    result = env.execute(Grasp(x=x, y=-st.object_xy[1], z=env.table_z + 0.014,
                               yaw=0.0, width=0.04))
    assert not result.success


def test_oracle_lifts_a_simple_object(env):
    env.reset(0, category="box")
    result = env.execute(env.oracle_grasp())
    assert result.success
    assert result.lift_height > 0.08


def test_unstable_episodes_are_reported_not_scored(env, monkeypatch):
    """A diverged solve must be flagged, never turned into a training label."""
    import mujoco

    env.reset(0, category="box")
    grasp = env.oracle_grasp()
    monkeypatch.setattr(type(env), "_is_unstable", lambda self: True)
    result = env.execute(grasp)
    assert result.reason == "unstable"
    assert not result.success
    assert result.lift_height == 0.0
    del mujoco


def test_stability_check_is_clean_on_a_normal_episode(env):
    env.reset(0, category="box")
    env.execute(env.oracle_grasp())
    assert not env._is_unstable()


def test_execute_requires_reset():
    from simgrasp.env import PandaGraspEnv
    e = PandaGraspEnv(image_size=64)
    try:
        with pytest.raises(RuntimeError):
            e.oracle_grasp()
    finally:
        e.close()
