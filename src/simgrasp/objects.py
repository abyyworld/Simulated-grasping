"""Procedural object catalogue.

Why procedural primitives instead of a mesh dataset (YCB / ShapeNet)?
---------------------------------------------------------------------
1. **A controllable generalisation split.** The headline result of this project
   is "how much does grasp success drop on object *categories* the network
   never saw?". With procedural shapes I decide exactly what novelty means:
   the held-out set contains non-convex and multi-part shapes (L, T, mug,
   dumbbell) that share *no* topology with the convex training shapes.
   With a mesh dataset the split is whatever the folder structure happens to be.
2. **No download, no licence friction, reproducible from a seed.** The entire
   dataset is regenerable from one integer.
3. **Cheap collision geometry.** Primitives need no convex decomposition, so a
   scene compiles in ~120 ms and steps at ~38k steps/s on a laptop CPU.

The cost is honest and should be stated in any write-up: primitives are less
visually and geometrically realistic than scanned meshes, so absolute success
rates here are optimistic relative to a real robot. The *relative* seen-vs-unseen
gap is the number that carries meaning.

Frame convention
----------------
Every object is defined in a local frame whose **z = 0 plane is the table
surface** and whose geoms all sit at z >= 0. Spawning therefore just places the
body origin on the table top and applies a yaw about z; no per-category resting
height calculation is needed.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import numpy as np

# Franka Panda hand: each finger travels 0 -> 0.04 m, so the maximum opening is
# 0.08 m. Leave clearance so the fingers can descend around the object.
MAX_GRIPPER_WIDTH = 0.08
SAFE_GRASP_WIDTH = 0.068


@dataclass(frozen=True)
class GeomSpec:
    """A single MuJoCo primitive in the object's local frame."""

    type: str
    size: tuple[float, ...]
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class GraspHint:
    """A ground-truth top-down grasp expressed in the object's local frame.

    Attributes
    ----------
    xy:        contact-point projection on the table plane, local frame.
    yaw:       angle of the **finger closing direction**, local frame.
    width:     object width along the closing direction (metres).
    surface_z: height of the object's top surface at ``xy`` (metres above table).
    """

    xy: tuple[float, float]
    yaw: float
    width: float
    surface_z: float


@dataclass(frozen=True)
class ObjectSpec:
    category: str
    geoms: tuple[GeomSpec, ...]
    grasp_hints: tuple[GraspHint, ...]
    rgba: tuple[float, float, float, float]
    density: float
    friction: tuple[float, float, float]
    top_z: float
    footprint_radius: float
    centroid_xy: tuple[float, float] = (0.0, 0.0)
    params: dict = field(default_factory=dict)

    @property
    def primary_hint(self) -> GraspHint:
        return self.grasp_hints[0]


SEEN_CATEGORIES = ("box", "cylinder", "capsule", "sphere")
UNSEEN_CATEGORIES = ("ellipsoid", "l_shape", "t_shape", "mug", "dumbbell")
ALL_CATEGORIES = SEEN_CATEGORIES + UNSEEN_CATEGORIES


def _random_rgba(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Saturated, well-lit colours so RGB carries usable signal without being trivial."""
    h = float(rng.uniform(0.0, 1.0))
    s = float(rng.uniform(0.55, 0.95))
    v = float(rng.uniform(0.55, 0.95))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b, 1.0)


def _random_material(rng: np.random.Generator) -> tuple[float, tuple[float, float, float]]:
    """Sample (density, friction) where friction is MuJoCo's (slide, torsion, roll).

    Torsional and rolling friction are only active with ``condim=6`` geoms (see
    ``scene.py``). Rolling friction matters here: without it a curved object that
    gets nudged during settling rolls indefinitely on a frictionless-to-rotation
    table, which no real object does.
    """
    density = float(rng.uniform(300.0, 1200.0))
    friction = (float(rng.uniform(0.6, 1.2)), 0.005, 0.005)
    return density, friction


def _box_hint(half_x: float, half_y: float, top: float) -> GraspHint:
    """Close the fingers along whichever horizontal axis is narrower."""
    if half_x <= half_y:
        return GraspHint(xy=(0.0, 0.0), yaw=0.0, width=2.0 * half_x, surface_z=top)
    return GraspHint(xy=(0.0, 0.0), yaw=np.pi / 2.0, width=2.0 * half_y, surface_z=top)


# --------------------------------------------------------------------------- #
# Seen categories: convex, single-primitive
# --------------------------------------------------------------------------- #


def _make_box(rng: np.random.Generator) -> tuple[tuple[GeomSpec, ...], tuple[GraspHint, ...], float, float, dict]:
    hx = float(rng.uniform(0.013, 0.033))
    hy = float(rng.uniform(0.013, 0.033))
    hz = float(rng.uniform(0.015, 0.045))
    geoms = (GeomSpec("box", (hx, hy, hz), pos=(0.0, 0.0, hz)),)
    hints = (_box_hint(hx, hy, 2.0 * hz),)
    return geoms, hints, 2.0 * hz, float(np.hypot(hx, hy)), {"half_extents": (hx, hy, hz)}


def _make_cylinder(rng: np.random.Generator):
    r = float(rng.uniform(0.013, 0.033))
    hh = float(rng.uniform(0.020, 0.050))
    geoms = (GeomSpec("cylinder", (r, hh), pos=(0.0, 0.0, hh)),)
    # Rotationally symmetric: any yaw is equally good, 0.0 is as valid as any.
    hints = (GraspHint(xy=(0.0, 0.0), yaw=0.0, width=2.0 * r, surface_z=2.0 * hh),)
    return geoms, hints, 2.0 * hh, r, {"radius": r, "half_height": hh}


def _make_capsule(rng: np.random.Generator):
    r = float(rng.uniform(0.012, 0.026))
    hl = float(rng.uniform(0.020, 0.050))
    # Lie the capsule down along local x (rotate its z axis onto x).
    geoms = (GeomSpec("capsule", (r, hl), pos=(0.0, 0.0, r), euler=(0.0, np.pi / 2.0, 0.0)),)
    # Grasp across the shaft: fingers close along local y.
    hints = (GraspHint(xy=(0.0, 0.0), yaw=np.pi / 2.0, width=2.0 * r, surface_z=2.0 * r),)
    return geoms, hints, 2.0 * r, hl + r, {"radius": r, "half_length": hl}


def _make_sphere(rng: np.random.Generator):
    r = float(rng.uniform(0.018, 0.032))
    geoms = (GeomSpec("sphere", (r,), pos=(0.0, 0.0, r)),)
    hints = (GraspHint(xy=(0.0, 0.0), yaw=0.0, width=2.0 * r, surface_z=2.0 * r),)
    return geoms, hints, 2.0 * r, r, {"radius": r}


# --------------------------------------------------------------------------- #
# Unseen categories: curved, non-convex, and multi-part
# --------------------------------------------------------------------------- #


def _make_ellipsoid(rng: np.random.Generator):
    # An ellipsoid resting on a plane has stable equilibria only when the
    # *smallest* semi-axis is vertical; standing on the longest axis is an egg
    # balanced on its tip, and the middle axis is a saddle. Sampling freely and
    # spawning upright produced ellipsoids that rolled 360 mm off the table
    # before settling, taking them out of the camera view entirely. So sample
    # three radii and put the smallest on z.
    radii = sorted(float(v) for v in rng.uniform(0.014, 0.033, size=3))
    c = radii[0]
    a, b = (radii[1], radii[2]) if rng.random() < 0.5 else (radii[2], radii[1])
    geoms = (GeomSpec("ellipsoid", (a, b, c), pos=(0.0, 0.0, c)),)
    hints = (_box_hint(a, b, 2.0 * c),)
    return geoms, hints, 2.0 * c, float(np.hypot(a, b)), {"radii": (a, b, c)}


def _make_l_shape(rng: np.random.Generator):
    la = float(rng.uniform(0.030, 0.050))  # half-length of the long arm
    lb = float(rng.uniform(0.022, 0.040))  # half-length of the short arm
    w = float(rng.uniform(0.011, 0.017))  # half-width of both arms
    t = float(rng.uniform(0.010, 0.018))  # half-thickness
    geoms = (
        GeomSpec("box", (la, w, t), pos=(0.0, 0.0, t)),
        GeomSpec("box", (w, lb, t), pos=(la - w, lb + w, t)),
    )
    hints = (
        GraspHint(xy=(-la / 2.0, 0.0), yaw=np.pi / 2.0, width=2.0 * w, surface_z=2.0 * t),
        GraspHint(xy=(la - w, lb + w), yaw=0.0, width=2.0 * w, surface_z=2.0 * t),
    )
    return geoms, hints, 2.0 * t, float(np.hypot(la, lb)), {"arms": (la, lb, w, t)}


def _make_t_shape(rng: np.random.Generator):
    la = float(rng.uniform(0.028, 0.048))  # half-length of the crossbar
    ls = float(rng.uniform(0.020, 0.038))  # half-length of the stem
    w = float(rng.uniform(0.011, 0.017))
    t = float(rng.uniform(0.010, 0.018))
    geoms = (
        GeomSpec("box", (la, w, t), pos=(0.0, 0.0, t)),
        GeomSpec("box", (w, ls, t), pos=(0.0, -(w + ls), t)),
    )
    hints = (
        GraspHint(xy=(la / 2.0, 0.0), yaw=np.pi / 2.0, width=2.0 * w, surface_z=2.0 * t),
        GraspHint(xy=(0.0, -(w + ls)), yaw=0.0, width=2.0 * w, surface_z=2.0 * t),
    )
    return geoms, hints, 2.0 * t, float(np.hypot(la, ls)), {"cross": (la, ls, w, t)}


def _make_mug(rng: np.random.Generator):
    r = float(rng.uniform(0.020, 0.031))
    hh = float(rng.uniform(0.028, 0.048))
    hw = float(rng.uniform(0.006, 0.010))  # handle half-thickness (radial)
    ht = float(rng.uniform(0.004, 0.007))  # handle half-thickness (tangential)
    hz = float(rng.uniform(0.012, 0.020))  # handle half-height
    geoms = (
        GeomSpec("cylinder", (r, hh), pos=(0.0, 0.0, hh)),
        GeomSpec("box", (hw, ht, hz), pos=(r + hw, 0.0, hh)),
    )
    # Close perpendicular to the handle so the fingers miss it.
    hints = (GraspHint(xy=(0.0, 0.0), yaw=np.pi / 2.0, width=2.0 * r, surface_z=2.0 * hh),)
    return geoms, hints, 2.0 * hh, r + 2.0 * hw, {"radius": r, "half_height": hh}


def _make_dumbbell(rng: np.random.Generator):
    R = float(rng.uniform(0.016, 0.024))  # end-ball radius
    rr = float(rng.uniform(0.007, 0.011))  # shaft radius
    d = R + float(rng.uniform(0.020, 0.032))  # half-distance between the ball centres
    geoms = (
        GeomSpec("sphere", (R,), pos=(-d, 0.0, R)),
        GeomSpec("sphere", (R,), pos=(d, 0.0, R)),
        GeomSpec("capsule", (rr, d - R), pos=(0.0, 0.0, R), euler=(0.0, np.pi / 2.0, 0.0)),
    )
    # Grasp the shaft: fingers close along local y, descending between the balls.
    hints = (GraspHint(xy=(0.0, 0.0), yaw=np.pi / 2.0, width=2.0 * rr, surface_z=R + rr),)
    return geoms, hints, 2.0 * R, d + R, {"ball_radius": R, "shaft_radius": rr, "half_span": d}


_FACTORIES = {
    "box": _make_box,
    "cylinder": _make_cylinder,
    "capsule": _make_capsule,
    "sphere": _make_sphere,
    "ellipsoid": _make_ellipsoid,
    "l_shape": _make_l_shape,
    "t_shape": _make_t_shape,
    "mug": _make_mug,
    "dumbbell": _make_dumbbell,
}


def _centroid_xy(geoms: tuple[GeomSpec, ...]) -> tuple[float, float]:
    """Axis-aligned bounding-box centre in xy, used to centre the object at spawn."""
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for g in geoms:
        # Conservative per-geom half-extent; exact for boxes, an over-estimate for
        # rotated capsules, which is fine for centring and camera framing.
        if g.type == "box":
            ext = np.array(g.size[:2])
        elif g.type == "sphere":
            ext = np.array([g.size[0], g.size[0]])
        elif g.type == "cylinder":
            ext = np.array([g.size[0], g.size[0]])
        elif g.type == "ellipsoid":
            ext = np.array(g.size[:2])
        elif g.type == "capsule":
            reach = g.size[1] + g.size[0]
            # Capsules in this catalogue lie along local x after the euler rotation.
            ext = np.array([reach, g.size[0]]) if abs(g.euler[1]) > 1e-6 else np.array([g.size[0], g.size[0]])
        else:  # pragma: no cover - catalogue is closed
            raise ValueError(f"unknown geom type {g.type}")
        centre = np.array(g.pos[:2])
        lo = np.minimum(lo, centre - ext)
        hi = np.maximum(hi, centre + ext)
    mid = 0.5 * (lo + hi)
    return (float(mid[0]), float(mid[1]))


def sample_object(rng: np.random.Generator, category: str) -> ObjectSpec:
    """Sample one randomly-dimensioned instance of ``category``."""
    if category not in _FACTORIES:
        raise KeyError(f"unknown category {category!r}; known: {sorted(_FACTORIES)}")
    geoms, hints, top_z, footprint, params = _FACTORIES[category](rng)
    density, friction = _random_material(rng)

    for h in hints:
        if h.width > SAFE_GRASP_WIDTH:
            raise AssertionError(
                f"{category} sampled a grasp width of {h.width:.3f} m, which exceeds the "
                f"Panda's usable opening ({SAFE_GRASP_WIDTH} m). Tighten the size ranges."
            )

    return ObjectSpec(
        category=category,
        geoms=geoms,
        grasp_hints=hints,
        rgba=_random_rgba(rng),
        density=density,
        friction=friction,
        top_z=top_z,
        footprint_radius=footprint,
        centroid_xy=_centroid_xy(geoms),
        params=params,
    )


def sample_category(rng: np.random.Generator, split: str) -> str:
    """Draw a category uniformly from ``"seen"``, ``"unseen"`` or ``"all"``."""
    pool = {"seen": SEEN_CATEGORIES, "unseen": UNSEEN_CATEGORIES, "all": ALL_CATEGORIES}[split]
    return str(rng.choice(pool))
