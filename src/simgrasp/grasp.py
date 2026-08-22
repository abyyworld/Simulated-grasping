"""Top-down parallel-jaw grasp representation and image<->world conversions.

A grasp is 4-DOF: position ``(x, y, z)`` of the TCP plus a yaw ``theta`` about
world z giving the finger closing direction. The 5th number, ``width``, is the
object's extent along the closing direction; it is used to pre-shape the gripper
before descending and is a regression target for the network.

Why 4-DOF and not 6-DOF? A top-down grasp is the standard restriction for
bin-picking work (Dex-Net 2.0, GG-CNN, GR-ConvNet) because it makes the grasp
representable as an *image-aligned* quantity: one pixel, one angle, one width.
That is what lets a fully-convolutional network predict grasps densely in a
single forward pass. Full 6-DOF grasping needs point-cloud methods and is out of
scope for this project; it is the natural extension and is noted in the README.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import CameraIntrinsics, deproject_pixels, project_points
from .transforms import wrap_grasp_angle

# Distance below the object's top surface at which the fingertip pads are centred.
GRASP_DEPTH = 0.012
# Lowest TCP height above the table. The Panda fingertips reach ~9 mm below the
# TCP site, so this keeps 5 mm of clearance between the tips and the table.
MIN_TCP_HEIGHT = 0.014
# Extra opening added to the object width when pre-shaping, so the fingers clear
# the object on the way down.
PRESHAPE_CLEARANCE = 0.022


@dataclass(frozen=True)
class Grasp:
    """A 4-DOF top-down grasp in world coordinates."""

    x: float
    y: float
    z: float
    yaw: float
    width: float = 0.04

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def with_yaw_wrapped(self) -> Grasp:
        return Grasp(self.x, self.y, self.z, float(wrap_grasp_angle(self.yaw)), self.width)

    def preshape_width(self, max_width: float = 0.08) -> float:
        return float(np.clip(self.width + PRESHAPE_CLEARANCE, 0.02, max_width))

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.yaw, self.width], dtype=np.float32)

    @staticmethod
    def from_array(a) -> Grasp:
        a = np.asarray(a, dtype=np.float64)
        return Grasp(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]))


def grasp_z_from_surface(surface_z_world: float, table_z: float, depth: float = GRASP_DEPTH) -> float:
    """TCP height for a grasp whose local top surface is at ``surface_z_world``."""
    return float(max(surface_z_world - depth, table_z + MIN_TCP_HEIGHT))


@dataclass(frozen=True)
class ImageGrasp:
    """The same grasp expressed in image coordinates -- the network's label space.

    ``angle`` is the finger closing direction in the image plane, wrapped to
    ``[-pi/2, pi/2)`` because a parallel jaw is symmetric under a half turn.
    ``width_px`` is the object width along that direction, in pixels.
    """

    u: float
    v: float
    angle: float
    width_px: float
    depth: float

    def as_array(self) -> np.ndarray:
        return np.array([self.u, self.v, self.angle, self.width_px, self.depth], dtype=np.float32)


def grasp_to_image(grasp: Grasp, cam_pos: np.ndarray, cam_mat: np.ndarray,
                   intr: CameraIntrinsics) -> ImageGrasp:
    """Project a world grasp into image space.

    The angle and pixel width are obtained by projecting the two jaw contact
    points and measuring the resulting image vector, rather than assuming the
    camera is axis-aligned. That keeps this correct for the wrist camera added in
    Project 2.
    """
    centre = grasp.position
    half = 0.5 * grasp.width * np.array([np.cos(grasp.yaw), np.sin(grasp.yaw), 0.0])
    pts = np.stack([centre, centre - half, centre + half])
    uv, depth = project_points(pts, cam_pos, cam_mat, intr)
    d_uv = uv[2] - uv[1]
    angle = float(wrap_grasp_angle(np.arctan2(d_uv[1], d_uv[0])))
    return ImageGrasp(u=float(uv[0, 0]), v=float(uv[0, 1]), angle=angle,
                      width_px=float(np.linalg.norm(d_uv)), depth=float(depth[0]))


def image_to_grasp(img_grasp: ImageGrasp, cam_pos: np.ndarray, cam_mat: np.ndarray,
                   intr: CameraIntrinsics) -> Grasp:
    """Inverse of :func:`grasp_to_image` (exact for a camera whose optical axis is vertical)."""
    centre = deproject_pixels(np.array([[img_grasp.u, img_grasp.v]]),
                              np.array([img_grasp.depth]), cam_pos, cam_mat, intr)[0]
    half_uv = 0.5 * img_grasp.width_px * np.array([np.cos(img_grasp.angle), np.sin(img_grasp.angle)])
    ends = deproject_pixels(
        np.stack([[img_grasp.u - half_uv[0], img_grasp.v - half_uv[1]],
                  [img_grasp.u + half_uv[0], img_grasp.v + half_uv[1]]]),
        np.array([img_grasp.depth, img_grasp.depth]), cam_pos, cam_mat, intr,
    )
    axis = ends[1] - ends[0]
    yaw = float(wrap_grasp_angle(np.arctan2(axis[1], axis[0])))
    return Grasp(x=float(centre[0]), y=float(centre[1]), z=float(centre[2]),
                 yaw=yaw, width=float(np.linalg.norm(axis[:2])))
