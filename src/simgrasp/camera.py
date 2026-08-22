"""RGB-D rendering, intrinsics, projection and deprojection.

MuJoCo depth semantics (verified in ``tests/test_camera.py``)
------------------------------------------------------------
``mujoco.Renderer`` with depth enabled returns the **perpendicular** distance
along the camera's optical axis in metres -- not the ray length. A flat plane
parallel to the image sensor therefore renders as a constant depth value. This
matters: deprojecting with a ray-length assumption would bow the reconstructed
table into a bowl and put every off-centre grasp at the wrong height.

Camera frame convention
-----------------------
MuJoCo cameras look down their own **-z** axis, with +x to the image right and
+y to the image *up*. So for a point ``P_cam = (X, Y, Z)`` in front of the
camera (``Z < 0``) with ``depth = -Z``::

    u = cx + fx * X / depth
    v = cy - fy * Y / depth
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])


def intrinsics_from_model(model: mujoco.MjModel, camera: str | int, height: int, width: int) -> CameraIntrinsics:
    """Pinhole intrinsics matching MuJoCo's OpenGL projection for a fixed camera.

    MuJoCo builds the frustum from ``fovy`` and ``aspect = width / height``, which
    makes pixels square, so ``fx == fy``. NDC 0 maps to pixel index ``(N-1)/2``.
    """
    cam_id = camera if isinstance(camera, int) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise KeyError(f"camera {camera!r} not found")
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    f = (height / 2.0) / np.tan(fovy / 2.0)
    return CameraIntrinsics(fx=f, fy=f, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                            width=width, height=height)


def camera_pose(data: mujoco.MjData, model: mujoco.MjModel, camera: str | int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position, rotation)`` of the camera frame in world coordinates."""
    cam_id = camera if isinstance(camera, int) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    return data.cam_xpos[cam_id].copy(), data.cam_xmat[cam_id].reshape(3, 3).copy()


def project_points(points_world: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
                   intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (pixel uv as float, perpendicular depth).

    Returns ``uv`` with shape ``(..., 2)`` ordered ``(u=column, v=row)``.
    """
    pts = np.atleast_2d(np.asarray(points_world, dtype=np.float64))
    p_cam = (pts - cam_pos[None, :]) @ cam_mat  # cam_mat columns are world axes -> inverse rotation
    depth = -p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.cx + intr.fx * p_cam[:, 0] / depth
        v = intr.cy - intr.fy * p_cam[:, 1] / depth
    uv = np.stack([u, v], axis=-1)
    if np.ndim(points_world) == 1:
        return uv[0], depth[0]
    return uv, depth


def deproject_pixels(uv: np.ndarray, depth: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
                     intr: CameraIntrinsics) -> np.ndarray:
    """(pixel uv, perpendicular depth) -> world points. Inverse of :func:`project_points`."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    depth = np.atleast_1d(np.asarray(depth, dtype=np.float64))
    x = (uv[:, 0] - intr.cx) / intr.fx * depth
    y = -(uv[:, 1] - intr.cy) / intr.fy * depth
    z = -depth
    p_cam = np.stack([x, y, z], axis=-1)
    world = p_cam @ cam_mat.T + cam_pos[None, :]
    if np.ndim(uv) == 1 or world.shape[0] == 1 and np.ndim(depth) == 0:
        return world[0]
    return world


def depth_to_pointcloud(depth: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray,
                        intr: CameraIntrinsics) -> np.ndarray:
    """Full-image depth map -> ``(H, W, 3)`` world-frame point cloud."""
    h, w = depth.shape
    vv, uu = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    x = (uu - intr.cx) / intr.fx * depth
    y = -(vv - intr.cy) / intr.fy * depth
    z = -depth
    p_cam = np.stack([x, y, z], axis=-1)
    return p_cam @ cam_mat.T + cam_pos[None, None, :]


class RGBDCamera:
    """A fixed MuJoCo camera that renders an aligned RGB + depth pair.

    One ``mujoco.Renderer`` is reused and toggled between colour and depth mode:
    two renderers would double the offscreen framebuffer allocation for no gain,
    and on the OSMesa software path each extra context is expensive.
    """

    def __init__(self, model: mujoco.MjModel, camera: str = "overhead", height: int = 224, width: int = 224):
        self.model = model
        self.camera = camera
        self.height = int(height)
        self.width = int(width)
        self._renderer = mujoco.Renderer(model, self.height, self.width)
        self.intrinsics = intrinsics_from_model(model, camera, self.height, self.width)

    def render(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(rgb uint8 [H,W,3], depth float32 [H,W])`` for the current state."""
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=self.camera)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self.camera)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth.astype(np.float32)

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return camera_pose(data, self.model, self.camera)

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> RGBDCamera:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def height_map(depth: np.ndarray, cam_pos: np.ndarray, cam_mat: np.ndarray, intr: CameraIntrinsics,
               table_z: float) -> np.ndarray:
    """Depth map -> per-pixel height above the table surface, in metres.

    This is the representation the network actually consumes: it is invariant to
    camera height, which is what lets a policy trained at one camera distance
    transfer if the rig moves.
    """
    cloud = depth_to_pointcloud(depth, cam_pos, cam_mat, intr)
    return (cloud[..., 2] - table_z).astype(np.float32)
