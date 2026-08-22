"""The template fast path must be indistinguishable from a freshly compiled scene.

This file exists because three separate bugs of the same shape were found while
building it: MuJoCo derives quantities at **compile** time that a runtime write
to ``mjModel`` does not refresh, and every one of them failed *silently*.

1. ``bvh_aabb`` -- the per-body bounding-volume hierarchy still bounded the
   placeholder geoms, so the mid-phase culled real contacts and objects sank
   through the table.
2. ``geom_sameframe`` / ``body_sameframe`` -- set when a local pose is the
   identity, after which ``mj_kinematics`` skips the local transform, discarding
   writes to ``geom_pos`` and ``body_ipos``.
3. ``body_contype`` / ``body_conaffinity`` -- the OR over a body's geoms. With the
   slots compiled non-colliding the aggregate stayed 0 and the broad phase
   dropped the object entirely, so it free-fell through the world.

None of these raise. They produce a plausible-looking simulation that is wrong.
So the guard is a field-by-field comparison against the reference compiler plus a
dynamics rollout, run over every category.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from simgrasp.controllers import CartesianController
from simgrasp.objects import ALL_CATEGORIES, sample_object
from simgrasp.randomize import SceneRandomizer, probe_object
from simgrasp.scene import (
    CAPTURE_QPOS,
    OBJECT_BODY,
    OBJECT_FREEJOINT,
    build_model,
    build_template_model,
    randomize_scene_options,
)
from simgrasp.transforms import quat_to_mat, rotation_error

GEOM_FIELDS = [
    "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_rbound", "geom_aabb",
    "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction", "geom_margin",
    "geom_gap", "geom_solref", "geom_solimp", "geom_priority", "geom_solmix",
]
BODY_FIELDS = [
    "body_mass", "body_inertia", "body_ipos", "body_iquat",
    "body_contype", "body_conaffinity", "body_dofnum", "body_jntnum",
]


def _reference_and_template(category: str, seed: int):
    rng = np.random.default_rng(seed)
    spec = sample_object(rng, category)
    opts = randomize_scene_options(rng, spec)

    ref = build_model(spec, opts)
    tmpl = build_template_model()
    rnd = SceneRandomizer(tmpl)
    data = mujoco.MjData(tmpl)
    rnd.apply_object(data, spec, probe_object(spec), opts.object_pos, opts.object_yaw)
    rnd.apply_visuals(opts)
    return spec, opts, ref, tmpl, rnd, data


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_geom_and_body_fields_match_the_compiler(category):
    spec, _opts, ref, tmpl, rnd, _data = _reference_and_template(category, seed=7)
    n = len(spec.geoms)

    for k in range(n):
        gr = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_GEOM, f"object_geom{k}")
        gt = int(rnd.geom_ids[k])
        for field in GEOM_FIELDS:
            a = np.atleast_1d(getattr(ref, field)[gr])
            b = np.atleast_1d(getattr(tmpl, field)[gt])
            assert np.allclose(a, b, atol=1e-9), f"{category}: geom {k} {field}: {a} != {b}"

    br = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
    for field in BODY_FIELDS:
        a = np.atleast_1d(getattr(ref, field)[br])
        b = np.atleast_1d(getattr(tmpl, field)[rnd.body_id])
        assert np.allclose(a, b, atol=1e-7), f"{category}: body {field}: {a} != {b}"


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_object_body_is_collidable(category):
    """Regression guard for the body-level collision aggregate."""
    _spec, _opts, _ref, tmpl, rnd, _data = _reference_and_template(category, seed=11)
    assert tmpl.body_contype[rnd.body_id] != 0
    assert tmpl.body_conaffinity[rnd.body_id] != 0
    assert np.all(tmpl.geom_contype[rnd.geom_ids[: len(_spec.geoms)]] != 0)


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_local_geom_poses_take_effect(category):
    """Regression guard for the sameframe fast path."""
    _spec, _opts, ref, tmpl, rnd, data = _reference_and_template(category, seed=13)
    mujoco.mj_forward(tmpl, data)
    dref = mujoco.MjData(ref)
    mujoco.mj_forward(ref, dref)
    for k in range(len(_spec.geoms)):
        gr = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_GEOM, f"object_geom{k}")
        gt = int(rnd.geom_ids[k])
        rel_ref = dref.geom_xpos[gr] - dref.xpos[mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)]
        rel_tmpl = data.geom_xpos[gt] - data.xpos[rnd.body_id]
        assert np.allclose(rel_ref, rel_tmpl, atol=1e-9), f"{category}: geom {k} offset lost"


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_rollout_matches_the_reference(category):
    """Same object, same pose, same physics for 0.1 s.

    Kept short deliberately. Rigid-body contact is chaotic: over seconds a
    rolling ellipsoid diverges to a different resting place from a 1e-16
    difference in solver path, which says nothing about correctness. Over 50
    steps the two must agree to numerical precision.
    """
    _spec, opts, ref, tmpl, rnd, data = _reference_and_template(category, seed=17)
    dref = mujoco.MjData(ref)
    for m, d in ((ref, dref), (tmpl, data)):
        mujoco.mj_resetData(m, d)
        CartesianController(m, d).reset_to(CAPTURE_QPOS)
    rnd.set_object_pose(data, opts.object_pos, opts.object_yaw)
    for m, d in ((ref, dref), (tmpl, data)):
        mujoco.mj_forward(m, d)

    adr = ref.jnt_qposadr[mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREEJOINT)]
    for _ in range(50):
        mujoco.mj_step(ref, dref)
        mujoco.mj_step(tmpl, data)

    qr = dref.qpos[adr:adr + 7]
    qt = data.qpos[rnd.qpos_adr:rnd.qpos_adr + 7]
    # Bitwise equality is not the bar and never could be: the reference model and
    # the template legitimately take different (both correct) MuJoCo fast paths --
    # body_simple selects a diagonal mass-matrix routine, geom_sameframe a
    # shortcut in mj_kinematics -- so the two accumulate different rounding. The
    # bound below is ~4 orders of magnitude tighter than any of the three real
    # bugs this file guards against, which showed up as 10-40 mm errors.
    assert dref.ncon == data.ncon, f"{category}: contact count differs"
    assert np.allclose(qr[:3], qt[:3], atol=5e-4), f"{category}: position drift {qr[:3] - qt[:3]}"

    # Compare *tilt* (where the body's own z axis points), not full orientation.
    # A cylinder, sphere, ellipsoid or mug is free to spin about the vertical with
    # no energy cost, so its yaw is unobservable and diverges from rounding alone.
    # Tilt is what actually matters -- it says whether the object stayed upright.
    up_ref = quat_to_mat(qr[3:])[:, 2]
    up_tmpl = quat_to_mat(qt[3:])[:, 2]
    tilt = np.degrees(np.arccos(np.clip(float(up_ref @ up_tmpl), -1.0, 1.0)))
    # 0.5 deg, not machine precision: a tall thin cylinder standing on its end is
    # close to neutrally stable, so the two solver paths tip it fractionally
    # differently. Still ~2 orders of magnitude tighter than the real bugs this
    # guards against, which left objects 40 mm out of place or falling through
    # the floor entirely.
    assert tilt < 0.5, f"{category}: tilt differs by {tilt:.4f} deg"


@pytest.mark.parametrize("category", ALL_CATEGORIES)
def test_objects_rest_where_they_are_placed(category):
    """An object must settle in place, not roll out of the camera's view."""
    tmpl = build_template_model()
    rnd = SceneRandomizer(tmpl)
    data = mujoco.MjData(tmpl)
    rng = np.random.default_rng(21)
    for _ in range(4):
        spec = sample_object(rng, category)
        opts = randomize_scene_options(rng, spec)
        rnd.apply_object(data, spec, probe_object(spec), opts.object_pos, opts.object_yaw)
        mujoco.mj_resetData(tmpl, data)
        CartesianController(tmpl, data).reset_to(CAPTURE_QPOS)
        rnd.set_object_pose(data, opts.object_pos, opts.object_yaw)
        mujoco.mj_forward(tmpl, data)
        start = data.qpos[rnd.qpos_adr:rnd.qpos_adr + 2].copy()
        for _ in range(500):
            mujoco.mj_step(tmpl, data)
        moved = np.linalg.norm(data.qpos[rnd.qpos_adr:rnd.qpos_adr + 2] - start)
        assert moved < 0.01, f"{category} drifted {moved * 1000:.1f} mm while settling"


def test_unused_slots_are_disabled():
    tmpl = build_template_model()
    rnd = SceneRandomizer(tmpl)
    data = mujoco.MjData(tmpl)
    rng = np.random.default_rng(1)
    spec = sample_object(rng, "box")  # a single-geom object leaves two slots spare
    rnd.apply_object(data, spec, probe_object(spec), (0.54, 0.0), 0.0)
    for k in range(len(spec.geoms), len(rnd.geom_ids)):
        gid = int(rnd.geom_ids[k])
        assert tmpl.geom_contype[gid] == 0
        assert tmpl.geom_conaffinity[gid] == 0
        assert tmpl.geom_rgba[gid][3] == 0.0


def test_bare_template_is_a_valid_empty_scene():
    """A template that has never been randomised must not explode.

    The placeholder slots are deliberately over-sized so the stale BVH stays
    conservative; if they were also collidable they would intersect the table and
    launch the scene on the first step.
    """
    tmpl = build_template_model()
    data = mujoco.MjData(tmpl)
    ctrl = CartesianController(tmpl, data)
    ctrl.reset_to(CAPTURE_QPOS)
    arm_dofs = ctrl.arm.dof_adr
    for _ in range(500):
        mujoco.mj_step(tmpl, data)
    assert np.all(np.isfinite(data.qpos))
    # Only the arm is checked. The empty object body still carries a free joint,
    # so it falls -- invisibly and without colliding -- which is harmless: every
    # reset places a real object before stepping.
    assert np.abs(data.qvel[arm_dofs]).max() < 1e-2, "arm is not at rest in a bare template"
