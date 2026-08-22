"""Rewrite a compiled scene template in place, once per episode.

Motivation
----------
Recompiling the MJCF for every episode costs ~120 ms, and rebuilding the OpenGL
``MjrContext`` that a new model requires costs another ~250 ms and leaks memory.
Over 10k episodes that is ~1 hour of pure overhead per worker and enough leaked
memory to matter on a 16 GB laptop. Everything that varies between episodes --
object shape, size, pose, colour, friction, mass, table tint, lighting -- can
instead be written directly into ``mjModel``/``mjData`` fields of a template
compiled once per process.

The one genuinely hard part is inertia. ``mjModel.body_mass``,
``body_inertia``, ``body_ipos`` and ``body_iquat`` are derived by the *compiler*
from geom geometry and density; changing ``geom_size`` at runtime does not update
them, and a stale inertia tensor silently produces wrong dynamics.

Rather than hand-deriving inertia formulas for boxes, cylinders, capsules,
spheres and ellipsoids (and getting the capsule wrong, as everyone does), this
module compiles a **throwaway probe model containing only the object's geoms**
and reads MuJoCo's own answer back out. That costs ~0.6 ms, is correct by
construction, and also hands us ``geom_rbound`` and ``geom_aabb``, which the
broad-phase collision filter needs and which are likewise compile-time derived.

``tests/test_randomize.py`` asserts that a randomised template is numerically
identical to a freshly compiled scene -- same mass, inertia, contact set, and
the same state after a 1 s rollout.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import mujoco
import numpy as np

from .objects import ObjectSpec
from .scene import (
    MAX_OBJECT_GEOMS,
    OBJECT_BODY,
    OBJECT_FREEJOINT,
    TABLE_HEIGHT,
    TABLE_MATERIAL,
    SceneOptions,
    yaw_to_quat,
)

# Geoms in unused slots are shrunk, made non-colliding and fully transparent.
_DISABLED_SIZE = 1e-5

_MJ_GEOM_TYPE = {
    "plane": mujoco.mjtGeom.mjGEOM_PLANE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
}


@dataclass(frozen=True)
class ObjectProbe:
    """Compile-time-derived quantities for one object, read out of MuJoCo."""

    n_geoms: int
    geom_type: np.ndarray  # (n, )   int
    geom_size: np.ndarray  # (n, 3)
    geom_pos: np.ndarray  # (n, 3)
    geom_quat: np.ndarray  # (n, 4)
    geom_rbound: np.ndarray  # (n, )
    geom_aabb: np.ndarray  # (n, 6)
    mass: float
    inertia: np.ndarray  # (3, ) principal moments
    ipos: np.ndarray  # (3, )
    iquat: np.ndarray  # (4, )


def _probe_xml(spec: ObjectSpec) -> str:
    root = ET.Element("mujoco")
    # MJCF defaults to DEGREES. The object catalogue stores euler angles in
    # radians (matching panda.xml, which sets angle="radian"), so without this
    # line a capsule's pi/2 tilt is read as 1.57 degrees and the probe returns a
    # near-upright capsule -- wrong orientation *and* wrong rotational inertia,
    # with no error raised anywhere.
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", name="probe")
    for g in spec.geoms:
        attrs = {
            "type": g.type,
            "size": " ".join(f"{v:.9g}" for v in g.size),
            "pos": " ".join(f"{v:.9g}" for v in g.pos),
            "density": f"{spec.density:.9g}",
        }
        if any(abs(e) > 1e-9 for e in g.euler):
            attrs["euler"] = " ".join(f"{v:.9g}" for v in g.euler)
        ET.SubElement(body, "geom", **attrs)
    return ET.tostring(root, encoding="unicode")


def probe_object(spec: ObjectSpec) -> ObjectProbe:
    """Compile a geoms-only model and read MuJoCo's derived mass/inertia/bounds."""
    if len(spec.geoms) > MAX_OBJECT_GEOMS:
        raise ValueError(
            f"{spec.category} has {len(spec.geoms)} geoms but the scene template only has "
            f"{MAX_OBJECT_GEOMS} slots; raise scene.MAX_OBJECT_GEOMS."
        )
    m = mujoco.MjModel.from_xml_string(_probe_xml(spec))
    n = len(spec.geoms)
    body = 1  # the single non-world body
    return ObjectProbe(
        n_geoms=n,
        geom_type=m.geom_type[:n].copy(),
        geom_size=m.geom_size[:n].copy(),
        geom_pos=m.geom_pos[:n].copy(),
        geom_quat=m.geom_quat[:n].copy(),
        geom_rbound=m.geom_rbound[:n].copy(),
        geom_aabb=m.geom_aabb[:n].copy(),
        mass=float(m.body_mass[body]),
        inertia=m.body_inertia[body].copy(),
        ipos=m.body_ipos[body].copy(),
        iquat=m.body_iquat[body].copy(),
    )


class SceneRandomizer:
    """Applies per-episode randomisation to a template model.

    Holds the name->id lookups so the per-episode path does no string work.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
        if self.body_id < 0:
            raise KeyError(f"template has no {OBJECT_BODY!r} body")
        self.geom_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"object_geom{i}")
            for i in range(MAX_OBJECT_GEOMS)
        ])
        if np.any(self.geom_ids < 0):
            raise KeyError("template is missing object geom slots")
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREEJOINT)
        self.qpos_adr = int(model.jnt_qposadr[joint_id])
        self.dof_adr = int(model.jnt_dofadr[joint_id])
        self.table_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, TABLE_MATERIAL)
        self.key_light_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "key_light")
        # Saved so a disabled slot can be restored exactly.
        self._contype = model.geom_contype[self.geom_ids].copy()
        self._conaffinity = model.geom_conaffinity[self.geom_ids].copy()
        self._clear_sameframe_fast_paths()

    def _clear_sameframe_fast_paths(self) -> None:
        """Force MuJoCo onto the general kinematics path for the object body.

        The compiler sets ``geom_sameframe``/``body_sameframe`` when a geom's
        local pose is the identity and a body's inertial frame coincides with its
        body frame. ``mj_kinematics`` then *skips the local transform entirely*,
        so a runtime write to ``geom_pos``, ``geom_quat``, ``body_ipos`` or
        ``body_iquat`` is silently ignored -- the object collides as if every geom
        sat at the body origin. It fails quietly: contacts still appear, the
        object just rests at the wrong height.

        Clearing the flags costs a few extra multiplies per step and makes the
        writes take effect. Every object in the catalogue has a non-trivial geom
        pose anyway, so a freshly compiled scene has these flags clear too --
        this only restores what the placeholder slots optimised away.
        """
        m = self.model
        m.geom_sameframe[self.geom_ids] = 0
        m.body_sameframe[self.body_id] = 0
        # body_simple selects a diagonal-mass-matrix fast path that assumes the
        # compile-time inertia; the object's inertia changes every episode.
        m.body_simple[self.body_id] = 0

    # -- object ------------------------------------------------------------- #
    def apply_object(self, data: mujoco.MjData, spec: ObjectSpec, probe: ObjectProbe,
                     pos_xy: tuple[float, float], yaw: float,
                     table_z: float = TABLE_HEIGHT) -> None:
        m = self.model
        n = probe.n_geoms

        for k in range(n):
            gid = self.geom_ids[k]
            m.geom_type[gid] = probe.geom_type[k]
            m.geom_size[gid] = probe.geom_size[k]
            m.geom_pos[gid] = probe.geom_pos[k]
            m.geom_quat[gid] = probe.geom_quat[k]
            m.geom_rbound[gid] = probe.geom_rbound[k]
            m.geom_aabb[gid] = probe.geom_aabb[k]
            m.geom_rgba[gid] = spec.rgba
            m.geom_friction[gid] = spec.friction
            m.geom_contype[gid] = self._contype[k]
            m.geom_conaffinity[gid] = self._conaffinity[k]

        for k in range(n, MAX_OBJECT_GEOMS):
            gid = self.geom_ids[k]
            m.geom_type[gid] = _MJ_GEOM_TYPE["sphere"]
            m.geom_size[gid] = (_DISABLED_SIZE, 0.0, 0.0)
            m.geom_pos[gid] = (0.0, 0.0, 0.0)
            m.geom_quat[gid] = (1.0, 0.0, 0.0, 0.0)
            m.geom_rbound[gid] = _DISABLED_SIZE
            m.geom_aabb[gid] = (0.0, 0.0, 0.0, _DISABLED_SIZE, _DISABLED_SIZE, _DISABLED_SIZE)
            m.geom_rgba[gid] = (0.0, 0.0, 0.0, 0.0)
            m.geom_contype[gid] = 0
            m.geom_conaffinity[gid] = 0

        # Compile-time inertial properties, taken from MuJoCo's own compiler.
        m.body_mass[self.body_id] = probe.mass
        m.body_inertia[self.body_id] = probe.inertia
        m.body_ipos[self.body_id] = probe.ipos
        m.body_iquat[self.body_id] = probe.iquat

        self.set_object_pose(data, pos_xy, yaw, table_z)
        # Recomputes body_subtreemass, dof_M0 and friends from the new masses.
        mujoco.mj_setConst(m, data)

    def set_object_pose(self, data: mujoco.MjData, pos_xy: tuple[float, float], yaw: float,
                        table_z: float = TABLE_HEIGHT) -> None:
        a = self.qpos_adr
        data.qpos[a:a + 3] = (pos_xy[0], pos_xy[1], table_z)
        data.qpos[a + 3:a + 7] = yaw_to_quat(yaw)
        data.qvel[self.dof_adr:self.dof_adr + 6] = 0.0

    def object_pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        a = self.qpos_adr
        return data.qpos[a:a + 3].copy(), data.qpos[a + 3:a + 7].copy()

    # -- appearance --------------------------------------------------------- #
    def apply_visuals(self, opts: SceneOptions) -> None:
        m = self.model
        if self.table_mat_id >= 0:
            m.mat_rgba[self.table_mat_id] = (*opts.table_rgb, 1.0)
        if self.key_light_id >= 0:
            m.light_pos[self.key_light_id] = opts.light_pos
            m.light_diffuse[self.key_light_id] = opts.light_diffuse
        m.vis.headlight.diffuse = opts.light_diffuse
        m.vis.headlight.ambient = opts.light_ambient
