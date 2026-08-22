"""The grasping environment: reset -> observe -> execute one grasp -> score it.

An "episode" here is a single open-loop grasp attempt, not a control loop. That
is deliberate: the learning problem in Project 1 is *where to grasp*, given one
RGB-D image, and the scripted controller is the fixed executor. Framing it that
way keeps the label space one grasp per image, which is exactly what a
fully-convolutional grasp network consumes, and makes the success rate directly
comparable to the bin-picking literature.

Episode timeline
----------------
    reset:   place a random object, settle 0.6 s, retract the arm, render RGB-D
    execute: pre-shape gripper -> approach above the grasp -> descend straight
             down -> close -> lift 0.20 m -> hold
    score:   the object counts as grasped if it rose more than 0.08 m and is
             still off the table when the hold ends

The observation is captured with the arm in ``CAPTURE_QPOS`` so the robot never
occludes the object in the overhead view.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .camera import CameraIntrinsics, RGBDCamera, height_map
from .controllers import ArmInterface, CartesianController
from .grasp import Grasp, grasp_z_from_surface
from .objects import ObjectSpec, sample_category, sample_object
from .randomize import SceneRandomizer, probe_object
from .scene import (
    CAPTURE_QPOS,
    HOME_QPOS,
    OVERHEAD_CAM,
    TABLE_HEIGHT,
    WORKSPACE_X,
    WORKSPACE_Y,
    build_template_model,
    randomize_scene_options,
)
from .seeding import rng_for_episode
from .transforms import quat_to_mat, topdown_grasp_mat, wrap_grasp_angle

# Object counts as lifted if it rises this far above its resting height.
LIFT_SUCCESS_THRESHOLD = 0.08
# How far the TCP travels upward during the lift phase.
LIFT_DISTANCE = 0.20
# Height above the grasp point at which the approach starts.
APPROACH_HEIGHT = 0.12

SETTLE_TIME = 0.6
# An object that ends up outside this padded workspace after settling has rolled
# somewhere the camera or the arm cannot deal with, and the episode is re-placed.
SETTLE_BOUNDS_PAD = 0.06


@dataclass
class Observation:
    """One RGB-D view of the scene plus everything needed to un-project it."""

    rgb: np.ndarray
    depth: np.ndarray
    height: np.ndarray
    cam_pos: np.ndarray
    cam_mat: np.ndarray
    intrinsics: CameraIntrinsics
    table_z: float

    def object_mask(self, threshold: float = 0.005) -> np.ndarray:
        """Pixels standing more than ``threshold`` above the table."""
        return self.height > threshold


@dataclass
class EpisodeState:
    index: int
    seed: int
    category: str
    spec: ObjectSpec
    object_xy: tuple[float, float]
    object_yaw: float
    object_z0: float
    spawn_xy: tuple[float, float]
    settle_displacement: float
    replaced: bool = False


@dataclass
class GraspResult:
    success: bool
    reason: str
    grasp: Grasp
    lift_height: float = 0.0
    final_gripper_width: float = 0.0
    object_displacement: float = 0.0
    ik_failures: int = 0
    info: dict[str, Any] = field(default_factory=dict)


FrameHook = Callable[[np.ndarray], None]


class PandaGraspEnv:
    """A Franka Panda grasping single procedurally-generated objects on a table.

    The MuJoCo model is compiled **once**; every episode rewrites the object in
    place (see :mod:`simgrasp.randomize`). One instance is therefore cheap to
    reuse for thousands of episodes and is what the data collector runs per
    worker process.
    """

    def __init__(
        self,
        image_size: int = 224,
        base_seed: int = 0,
        camera: str = OVERHEAD_CAM,
        render_camera: str | None = None,
        render_size: tuple[int, int] = (360, 480),
    ):
        self.base_seed = int(base_seed)
        self.image_size = int(image_size)
        self.model = build_template_model()
        self.data = mujoco.MjData(self.model)
        self.arm = ArmInterface(self.model)
        self.controller = CartesianController(self.model, self.data, self.arm)
        self.randomizer = SceneRandomizer(self.model)
        self.camera = RGBDCamera(self.model, camera, image_size, image_size)
        self.table_z = TABLE_HEIGHT

        self._render_cam = render_camera
        self._renderer = (
            mujoco.Renderer(self.model, render_size[0], render_size[1])
            if render_camera is not None else None
        )
        self.state: EpisodeState | None = None

    # -- lifecycle ---------------------------------------------------------- #
    def close(self) -> None:
        self.camera.close()
        if self._renderer is not None:
            self._renderer.close()

    def __enter__(self) -> PandaGraspEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reset -------------------------------------------------------------- #
    def reset(self, episode_index: int, split: str = "all", category: str | None = None,
              max_placement_tries: int = 4) -> Observation:
        rng = rng_for_episode(self.base_seed, episode_index)
        cat = category or sample_category(rng, split)
        spec = sample_object(rng, cat)
        opts = randomize_scene_options(rng, spec)
        probe = probe_object(spec)

        self.randomizer.apply_visuals(opts)
        self.randomizer.apply_object(self.data, spec, probe, opts.object_pos, opts.object_yaw,
                                     self.table_z)

        spawn_xy = opts.object_pos
        replaced = False
        for attempt in range(max_placement_tries):
            mujoco.mj_resetData(self.model, self.data)
            self.controller.reset_to(CAPTURE_QPOS)
            self.randomizer.set_object_pose(self.data, spawn_xy, opts.object_yaw, self.table_z)
            mujoco.mj_forward(self.model, self.data)
            self.controller.settle(SETTLE_TIME)

            pos, quat = self.randomizer.object_pose(self.data)
            if self._within_workspace(pos) or attempt == max_placement_tries - 1:
                break
            # Rolled out of reach: re-place nearer the middle of the workspace.
            replaced = True
            spawn_xy = (
                float(np.clip(spawn_xy[0], *self._padded(WORKSPACE_X))),
                float(np.clip(spawn_xy[1], *self._padded(WORKSPACE_Y))),
            )
            spawn_xy = (0.5 * (spawn_xy[0] + np.mean(WORKSPACE_X)),
                        0.5 * (spawn_xy[1] + np.mean(WORKSPACE_Y)))

        pos, quat = self.randomizer.object_pose(self.data)
        mat = quat_to_mat(quat)
        settled_yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        self.state = EpisodeState(
            index=int(episode_index),
            seed=self.base_seed,
            category=cat,
            spec=spec,
            object_xy=(float(pos[0]), float(pos[1])),
            object_yaw=settled_yaw,
            object_z0=float(pos[2]),
            spawn_xy=(float(opts.object_pos[0]), float(opts.object_pos[1])),
            settle_displacement=float(np.linalg.norm(pos[:2] - np.array(opts.object_pos))),
            replaced=replaced,
        )
        return self.observe()

    @staticmethod
    def _padded(bounds: tuple[float, float]) -> tuple[float, float]:
        return (bounds[0] + SETTLE_BOUNDS_PAD, bounds[1] - SETTLE_BOUNDS_PAD)

    def _within_workspace(self, pos: np.ndarray) -> bool:
        lo_x, hi_x = WORKSPACE_X[0] - SETTLE_BOUNDS_PAD, WORKSPACE_X[1] + SETTLE_BOUNDS_PAD
        lo_y, hi_y = WORKSPACE_Y[0] - SETTLE_BOUNDS_PAD, WORKSPACE_Y[1] + SETTLE_BOUNDS_PAD
        return bool(lo_x <= pos[0] <= hi_x and lo_y <= pos[1] <= hi_y)

    # -- observation -------------------------------------------------------- #
    def observe(self) -> Observation:
        rgb, depth = self.camera.render(self.data)
        cam_pos, cam_mat = self.camera.pose(self.data)
        intr = self.camera.intrinsics
        return Observation(
            rgb=rgb,
            depth=depth,
            height=height_map(depth, cam_pos, cam_mat, intr, self.table_z),
            cam_pos=cam_pos,
            cam_mat=cam_mat,
            intrinsics=intr,
            table_z=self.table_z,
        )

    def render_frame(self) -> np.ndarray | None:
        """Render the third-person view used for demo videos."""
        if self._renderer is None:
            return None
        self._renderer.update_scene(self.data, camera=self._render_cam)
        return self._renderer.render().copy()

    # -- ground-truth grasp -------------------------------------------------- #
    def oracle_grasp(self, hint_index: int = 0) -> Grasp:
        """The scripted grasp derived from the object's true settled pose.

        This is the upper bound the learned policy is measured against, and the
        centre of the distribution the data collector samples around.
        """
        st = self._require_state()
        hint = st.spec.grasp_hints[hint_index % len(st.spec.grasp_hints)]
        c, s = np.cos(st.object_yaw), np.sin(st.object_yaw)
        hx, hy = hint.xy
        x = st.object_xy[0] + c * hx - s * hy
        y = st.object_xy[1] + s * hx + c * hy
        z = grasp_z_from_surface(st.object_z0 + hint.surface_z, self.table_z,
                                 mid_z_world=st.object_z0 + hint.grasp_z)
        yaw = float(wrap_grasp_angle(st.object_yaw + hint.yaw))
        return Grasp(x=float(x), y=float(y), z=float(z), yaw=yaw, width=float(hint.width))

    # -- execution ---------------------------------------------------------- #
    def execute(self, grasp: Grasp, frame_hook: FrameHook | None = None,
                frame_every: int = 12) -> GraspResult:
        """Run the scripted approach-grasp-lift sequence for ``grasp``."""
        st = self._require_state()
        ctrl = self.controller
        ctrl.ik_failures = 0

        hook = self._make_step_hook(frame_hook, frame_every)
        mat = topdown_grasp_mat(grasp.yaw)
        approach = np.array([grasp.x, grasp.y, grasp.z + APPROACH_HEIGHT])
        target = grasp.position

        # Reachability is checked before moving so an unreachable proposal costs
        # ~5 ms instead of a full 4 s rollout. It is still recorded as a failure:
        # proposing an unreachable grasp is a policy error, not a free pass.
        pre = ctrl.solve_ik(approach, mat, q_init=HOME_QPOS)
        at = ctrl.solve_ik(target, mat, q_init=pre.qpos)
        if not (pre.success and at.success):
            return GraspResult(False, "ik_failed", grasp, ik_failures=2,
                               info={"pos_err": max(pre.pos_err, at.pos_err)})

        ctrl.set_gripper(grasp.preshape_width(), duration=0.3, hook=hook)
        ctrl.move_to_joint(pre.qpos, duration=1.3, hook=hook)
        approach_err = ctrl.hold_until_converged(approach, mat, tol=1e-3, max_time=0.5, hook=hook)
        # Descend, then wait for the servo to actually arrive before closing --
        # see CartesianController.hold_until_converged for why this matters.
        ctrl.move_to_pose(target, mat, duration=0.9, cartesian=True, waypoints=8, hook=hook,
                          converge_tol=6e-4, converge_time=0.8)
        descend_err = ctrl.last_convergence_error
        ctrl.close_gripper(duration=0.8, hook=hook)

        lift_target = target + np.array([0.0, 0.0, LIFT_DISTANCE])
        # A slower lift: at 0.22 m/s the dynamic torque on an off-centre grasp was
        # enough on its own to rotate flat objects out of the jaws.
        ctrl.move_to_pose(lift_target, mat, duration=1.3, cartesian=True, waypoints=10, hook=hook,
                          converge=False)
        ctrl.settle(0.4, hook=hook)

        pos, _ = self.randomizer.object_pose(self.data)
        lift = float(pos[2] - st.object_z0)
        width = self.arm.gripper_width(self.data)
        displacement = float(np.linalg.norm(pos[:2] - np.array(st.object_xy)))
        success = lift > LIFT_SUCCESS_THRESHOLD

        if success:
            reason = "success"
        elif width < 1e-3:
            # Fingers fully closed: they met nothing, so the grasp missed.
            reason = "closed_empty"
        elif lift > 0.005:
            reason = "dropped"
        else:
            reason = "no_lift"

        return GraspResult(
            success=success,
            reason=reason,
            grasp=grasp,
            lift_height=lift,
            final_gripper_width=width,
            object_displacement=displacement,
            ik_failures=ctrl.ik_failures,
            info={"approach_err": float(approach_err), "descend_err": float(descend_err)},
        )

    def _make_step_hook(self, frame_hook: FrameHook | None, frame_every: int):
        if frame_hook is None:
            return None
        counter = {"n": 0}

        def hook(_model, _data):
            counter["n"] += 1
            if counter["n"] % frame_every == 0:
                frame = self.render_frame()
                if frame is not None:
                    frame_hook(frame)

        return hook

    def _require_state(self) -> EpisodeState:
        if self.state is None:
            raise RuntimeError("call reset() before using the environment")
        return self.state
