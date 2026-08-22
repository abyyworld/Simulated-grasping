"""Build the tabletop grasping scene as MJCF.

The Menagerie ``panda.xml`` is loaded with ElementTree and *edited* rather than
``<include>``-ed, because the two things this project needs most cannot be done
from an including file:

1. attaching a ``tcp`` site inside the ``hand`` body (the IK target), and
2. lifting the robot base onto a plinth so the table top and the base sit at the
   same height, which is how a Panda is actually mounted.

The result is emitted as an XML string with an **absolute** ``meshdir`` so the
model can be compiled from a string without depending on the process CWD.

Two build paths
---------------
``build_model(spec)``
    Compiles a scene containing one specific object. Simple and obviously
    correct; used by the demo scripts and as the *reference* that the fast path
    is tested against.

``build_template_model()``
    Compiles a scene with ``MAX_OBJECT_GEOMS`` empty object slots, which
    :mod:`simgrasp.randomize` then rewrites in place. This is the path data
    collection uses. Profiling motivated it: per episode the reference path cost
    ~120 ms to compile the MJCF and ~250 ms to rebuild the OpenGL ``MjrContext``
    (which re-uploads all 67 Panda meshes), and repeatedly creating render
    contexts leaked ~25 MB per episode -- a real out-of-memory risk over a 10k
    episode run on a 16 GB machine. Compiling once per worker removes both.

Scene layout (metres)
---------------------
    floor plane .................... z = 0
    plinth  x in [-.13,.13] ........ top z = 0.40
    panda base ..................... z = 0.40
    table   x in [ .20,.90] ........ top z = 0.40
    object spawn region ............ x in [.40,.68], y in [-.20,.20]
    overhead RGB-D camera .......... z = 0.95, looking straight down
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import mujoco
import numpy as np

from .objects import ObjectSpec
from .paths import require_panda_assets

TABLE_HEIGHT = 0.40
TABLE_CENTER_X = 0.55
TABLE_HALF = (0.35, 0.45, TABLE_HEIGHT / 2.0)

PLINTH_HALF = (0.13, 0.13, TABLE_HEIGHT / 2.0)

CAMERA_HEIGHT = 0.55
CAMERA_X = 0.54
CAMERA_FOVY_DEG = 48.0

WORKSPACE_X = (0.40, 0.68)
WORKSPACE_Y = (-0.20, 0.20)

OBJECT_BODY = "object"
OBJECT_FREEJOINT = "object_free"
TABLE_MATERIAL = "table_mat"
TCP_SITE = "tcp"
OVERHEAD_CAM = "overhead"
SCENE_CAM = "scene"

# Largest geom count over the object catalogue (the dumbbell: two balls + shaft).
MAX_OBJECT_GEOMS = 3

# Half-extent of the placeholder geoms in the template's object slots.
#
# This is not cosmetic. MuJoCo builds a per-body bounding-volume hierarchy
# (``mjModel.bvh_aabb``) at *compile* time and uses it in the mid-phase to cull
# geom pairs. Those AABBs live in an internal, rotated, undocumented frame and
# are NOT refreshed when ``geom_size`` is written at runtime, so a template whose
# slots were compiled small silently culls real contacts: the object sinks into
# the table with ``ncon == 0``. Compiling the slots *larger* than any object in
# the catalogue makes the stale hierarchy conservative instead of wrong -- an
# over-sized AABB can only admit extra narrow-phase tests, never miss a contact.
# The exact per-geom ``geom_rbound`` used by the broad phase *is* refreshed from
# the probe model, so the cheap filter stays tight.
#
# Largest reach of any single catalogue geom from the body origin is ~0.10 m
# (the L-shape's far arm); 0.15 leaves a comfortable margin.
SLOT_BOUND = 0.15

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")

HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
# Retracted pose used when the RGB-D observation is captured, so the arm never
# occludes the object in the overhead view. Asserted in tests/test_scene.py.
CAPTURE_QPOS = np.array([0.0, -1.35, 0.0, -2.60, 0.0, 1.40, -0.7853])

# TCP offset along the hand's +z axis. The Panda fingertip pads span roughly
# z in [0.096, 0.108] in hand coordinates; 0.1034 puts the site at the pad centre.
TCP_OFFSET_Z = 0.1034


@dataclass
class SceneOptions:
    """Per-episode randomisation and rendering knobs."""

    table_rgb: tuple[float, float, float] = (0.55, 0.50, 0.45)
    light_pos: tuple[float, float, float] = (0.4, 0.0, 1.6)
    light_diffuse: tuple[float, float, float] = (0.75, 0.75, 0.75)
    light_ambient: tuple[float, float, float] = (0.35, 0.35, 0.35)
    object_pos: tuple[float, float] = (0.54, 0.0)
    object_yaw: float = 0.0
    add_object: bool = True
    camera_height: float = CAMERA_HEIGHT
    camera_fovy_deg: float = CAMERA_FOVY_DEG
    extra: dict = field(default_factory=dict)


def randomize_scene_options(rng: np.random.Generator, spec: ObjectSpec | None = None) -> SceneOptions:
    """Sample the visual/placement randomisation for one episode.

    Object placement accounts for the object's bounding-box centre so that a
    shape whose body origin is off-centre (L, T, mug) still lands inside the
    workspace rather than half off the table.
    """
    margin = 0.0 if spec is None else min(spec.footprint_radius, 0.06)
    x = float(rng.uniform(WORKSPACE_X[0] + margin, WORKSPACE_X[1] - margin))
    y = float(rng.uniform(WORKSPACE_Y[0] + margin, WORKSPACE_Y[1] - margin))
    yaw = float(rng.uniform(-np.pi, np.pi))

    if spec is not None:
        cx, cy = spec.centroid_xy
        c, s = np.cos(yaw), np.sin(yaw)
        x -= c * cx - s * cy
        y -= s * cx + c * cy

    shade = float(rng.uniform(0.35, 0.70))
    tint = rng.uniform(-0.06, 0.06, size=3)
    table_rgb = tuple(float(np.clip(shade + t, 0.15, 0.90)) for t in tint)
    diffuse = float(rng.uniform(0.55, 0.95))
    ambient = float(rng.uniform(0.25, 0.45))

    return SceneOptions(
        table_rgb=table_rgb,
        light_pos=(float(rng.uniform(-0.2, 0.9)), float(rng.uniform(-0.6, 0.6)),
                   float(rng.uniform(1.3, 2.0))),
        light_diffuse=(diffuse, diffuse, diffuse),
        light_ambient=(ambient, ambient, ambient),
        object_pos=(x, y),
        object_yaw=yaw,
    )


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.atleast_1d(values))


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    raise KeyError(f"body {name!r} not found in MJCF")


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _add_visual_and_assets(root: ET.Element, opts: SceneOptions) -> None:
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", diffuse=_fmt(opts.light_diffuse),
                  ambient=_fmt(opts.light_ambient), specular="0 0 0")
    ET.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
    ET.SubElement(visual, "global", azimuth="140", elevation="-25", offwidth="1280", offheight="960")
    ET.SubElement(visual, "map", znear="0.02", zfar="6")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="256", height="1536")
    ET.SubElement(asset, "texture", type="2d", name="floor_tex", builtin="checker", mark="edge",
                  rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8",
                  width="300", height="300")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex",
                  texuniform="true", texrepeat="4 4", reflectance="0.1")
    # The table texture is deliberately *greyscale*: MuJoCo multiplies material
    # rgba by the texture, so the table's colour can then be re-tinted at runtime
    # via model.mat_rgba without rebuilding the OpenGL context. A uniform grey
    # table would also make the object mask trivially separable in RGB, which is
    # not the generalisation test we want -- hence a checker rather than a flat fill.
    ET.SubElement(asset, "texture", type="2d", name="table_tex", builtin="checker",
                  rgb1="1 1 1", rgb2="0.86 0.86 0.86", width="200", height="200")
    ET.SubElement(asset, "material", name=TABLE_MATERIAL, texture="table_tex",
                  texuniform="true", texrepeat="12 14", specular="0.1", shininess="0.1",
                  rgba=_fmt((*opts.table_rgb, 1.0)))
    ET.SubElement(asset, "material", name="plinth_mat", rgba="0.25 0.26 0.28 1")


def _add_worldbody(root: ET.Element, opts: SceneOptions) -> ET.Element:
    world = root.find("worldbody")
    if world is None:  # pragma: no cover - Menagerie always ships one
        world = ET.SubElement(root, "worldbody")

    ET.SubElement(world, "light", name="key_light", pos=_fmt(opts.light_pos), dir="0 0 -1",
                  directional="true", diffuse=_fmt(opts.light_diffuse))
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05", material="floor_mat")

    plinth = ET.SubElement(world, "body", name="plinth", pos=f"0 0 {PLINTH_HALF[2]:.6g}")
    ET.SubElement(plinth, "geom", name="plinth_geom", type="box", size=_fmt(PLINTH_HALF),
                  material="plinth_mat")

    table = ET.SubElement(world, "body", name="table", pos=f"{TABLE_CENTER_X} 0 {TABLE_HALF[2]:.6g}")
    ET.SubElement(table, "geom", name="table_geom", type="box", size=_fmt(TABLE_HALF),
                  material=TABLE_MATERIAL, friction="1.0 0.005 0.005", condim="6")

    cam_z = TABLE_HEIGHT + opts.camera_height
    # xyaxes puts camera +x on world +x and camera +y on world +y, so the camera
    # -z (its viewing direction) points straight down. Image column -> world +x,
    # image row -> world -y.
    ET.SubElement(world, "camera", name=OVERHEAD_CAM, mode="fixed",
                  pos=f"{CAMERA_X} 0 {cam_z:.6g}", xyaxes="1 0 0 0 1 0",
                  fovy=f"{opts.camera_fovy_deg:.6g}")
    ET.SubElement(world, "camera", name=SCENE_CAM, mode="fixed",
                  pos="1.32 -0.98 1.12", xyaxes="0.595 0.804 0 -0.333 0.246 0.910", fovy="45")
    return world


def object_geom_attrs(index: int, geom_type: str, size, pos, euler=None, rgba=(0.7, 0.3, 0.3, 1.0),
                      density: float = 700.0, friction=(1.0, 0.005, 0.0001)) -> dict:
    attrs = {
        "name": f"object_geom{index}",
        "type": geom_type,
        "size": _fmt(size),
        "pos": _fmt(pos),
        "rgba": _fmt(rgba),
        "density": f"{float(density):.6g}",
        "friction": _fmt(friction),
        # condim 6 enables torsional *and* rolling friction. Torsional friction
        # stops small objects spinning out from between the fingers during the
        # lift; rolling friction stops curved objects rolling away during
        # settling. Both geoms in a contact pair need it, so the table is
        # condim 6 too.
        "condim": "6",
        "solref": "0.005 1",
        "group": "2",
    }
    if euler is not None and any(abs(e) > 1e-9 for e in euler):
        attrs["euler"] = _fmt(euler)
    return attrs


def _add_object(world: ET.Element, spec: ObjectSpec, opts: SceneOptions) -> None:
    x, y = opts.object_pos
    body = ET.SubElement(world, "body", name=OBJECT_BODY,
                         pos=f"{x:.6g} {y:.6g} {TABLE_HEIGHT:.6g}",
                         quat=_fmt(yaw_to_quat(opts.object_yaw)))
    ET.SubElement(body, "freejoint", name=OBJECT_FREEJOINT)
    for i, g in enumerate(spec.geoms):
        ET.SubElement(world.find(f"body[@name='{OBJECT_BODY}']"), "geom",
                      **object_geom_attrs(i, g.type, g.size, g.pos, g.euler, spec.rgba,
                                          spec.density, spec.friction))


def _add_object_slots(world: ET.Element, opts: SceneOptions, n_slots: int) -> None:
    """Placeholder object body whose geoms are rewritten at runtime."""
    x, y = opts.object_pos
    body = ET.SubElement(world, "body", name=OBJECT_BODY,
                         pos=f"{x:.6g} {y:.6g} {TABLE_HEIGHT:.6g}",
                         quat=_fmt(yaw_to_quat(opts.object_yaw)))
    ET.SubElement(body, "freejoint", name=OBJECT_FREEJOINT)
    for i in range(n_slots):
        # See SLOT_BOUND: deliberately over-sized so the compile-time BVH is
        # conservative. density is tiny only to keep the placeholder mass sane;
        # the real mass is written from the probe model before any stepping.
        ET.SubElement(body, "geom",
                      **object_geom_attrs(i, "box", (SLOT_BOUND,) * 3, (0.0, 0.0, 0.0), density=1.0))


def _base_tree(opts: SceneOptions) -> tuple[ET.Element, ET.Element]:
    panda_xml = require_panda_assets()
    root = ET.parse(panda_xml).getroot()
    root.set("model", "panda_tabletop_grasp")

    # meshdir is resolved relative to the XML file on disk; we compile from a
    # string, so make it absolute.
    compiler = root.find("compiler")
    if compiler is None:  # pragma: no cover - Menagerie always ships one
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str((panda_xml.parent / compiler.get("meshdir", "assets")).resolve()))

    # The shipped keyframe has nq == 9; adding a free-jointed object changes nq
    # and MuJoCo would reject it. Home/capture poses are applied from Python.
    for key in list(root.findall("keyframe")):
        root.remove(key)

    _add_visual_and_assets(root, opts)

    # Mount the arm on the plinth so its base is level with the table top.
    _find_body(root, "link0").set("pos", f"0 0 {TABLE_HEIGHT:.6g}")

    # IK target: a site rigidly attached to the hand at the fingertip-pad centre.
    hand = _find_body(root, "hand")
    ET.SubElement(hand, "site", name=TCP_SITE, pos=f"0 0 {TCP_OFFSET_Z:.6g}",
                  size="0.005", rgba="1 0 0 0.6", group="4")

    return root, _add_worldbody(root, opts)


def build_scene_xml(spec: ObjectSpec | None, opts: SceneOptions | None = None) -> str:
    """MJCF for a scene containing exactly ``spec`` (reference path)."""
    opts = opts or SceneOptions()
    root, world = _base_tree(opts)
    if spec is not None and opts.add_object:
        _add_object(world, spec, opts)
    return ET.tostring(root, encoding="unicode")


def build_template_xml(opts: SceneOptions | None = None, n_slots: int = MAX_OBJECT_GEOMS) -> str:
    """MJCF for a scene with ``n_slots`` rewritable object geoms (fast path)."""
    opts = opts or SceneOptions()
    root, world = _base_tree(opts)
    _add_object_slots(world, opts, n_slots)
    return ET.tostring(root, encoding="unicode")


def build_model(spec: ObjectSpec | None, opts: SceneOptions | None = None) -> mujoco.MjModel:
    """Compile a scene containing one specific object (~120 ms)."""
    return mujoco.MjModel.from_xml_string(build_scene_xml(spec, opts))


def build_template_model(opts: SceneOptions | None = None,
                         n_slots: int = MAX_OBJECT_GEOMS) -> mujoco.MjModel:
    """Compile the reusable scene template. Call once per process."""
    return mujoco.MjModel.from_xml_string(build_template_xml(opts, n_slots))
