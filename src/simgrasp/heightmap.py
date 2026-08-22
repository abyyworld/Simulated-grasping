"""Utilities over the orthographic-ish height map produced by the overhead camera.

The overhead camera looks straight down, so ``Observation.height`` is very close
to a true height field: pixel (v, u) holds the height above the table of the
topmost surface at that point. Perspective still applies -- a tall object is
slightly magnified -- but over a 0.49 m field of view at 0.55 m the effect is
small, and every conversion below goes through the real camera intrinsics rather
than assuming orthography.

These helpers are shared by the grasp sampler, the heuristic baseline and the
network's label generation, so they are written once and tested once.
"""

from __future__ import annotations

import numpy as np

from .camera import CameraIntrinsics, deproject_pixels

# Height above the table at which a pixel is considered part of an object.
OBJECT_HEIGHT_THRESHOLD = 0.005


def object_mask(height: np.ndarray, threshold: float = OBJECT_HEIGHT_THRESHOLD) -> np.ndarray:
    return height > threshold


def metres_per_pixel(intr: CameraIntrinsics, distance: float) -> float:
    """Ground sampling distance of the camera at a given perpendicular distance."""
    return float(distance / intr.fx)


def surface_height(height: np.ndarray, u: float, v: float, patch: int = 2) -> float:
    """Max height in a small patch around (u, v).

    The maximum rather than the value at the exact pixel: a grasp should clear
    the tallest thing under the fingers, and single-pixel depth is noisy at
    object edges.
    """
    h, w = height.shape
    ui, vi = int(round(u)), int(round(v))
    u0, u1 = max(0, ui - patch), min(w, ui + patch + 1)
    v0, v1 = max(0, vi - patch), min(h, vi + patch + 1)
    if u0 >= u1 or v0 >= v1:
        return 0.0
    return float(np.max(height[v0:v1, u0:u1]))


def sample_line(height: np.ndarray, u: float, v: float, angle: float,
                max_px: int, step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Sample the height map outward from (u, v) along +/- ``angle``.

    Returns ``(offsets_px, heights)`` where offsets run from ``-max_px`` to
    ``+max_px``. Out-of-image samples come back as 0 height (i.e. table level).
    """
    h, w = height.shape
    offsets = np.arange(-max_px, max_px + step, step)
    du, dv = np.cos(angle), np.sin(angle)
    us = np.clip(np.round(u + offsets * du).astype(int), 0, w - 1)
    vs = np.clip(np.round(v + offsets * dv).astype(int), 0, h - 1)
    inside = (u + offsets * du >= 0) & (u + offsets * du < w) & \
             (v + offsets * dv >= 0) & (v + offsets * dv < h)
    vals = height[vs, us]
    return offsets, np.where(inside, vals, 0.0)


def estimate_width_px(height: np.ndarray, u: float, v: float, angle: float,
                      max_px: int = 60, drop: float = 0.6) -> float:
    """Width of the object at (u, v) measured along ``angle``, in pixels.

    Walks outward in both directions until the surface falls below ``drop`` times
    the height at the grasp point, and returns the distance between the two
    crossings. This is what the gripper needs in order to pre-shape, and it is
    the quantity the network's width head regresses.
    """
    centre = surface_height(height, u, v)
    if centre <= OBJECT_HEIGHT_THRESHOLD:
        return 0.0
    cutoff = max(centre * drop, OBJECT_HEIGHT_THRESHOLD)
    offsets, vals = sample_line(height, u, v, angle, max_px)
    mid = len(offsets) // 2

    def edge(indices) -> float:
        for i in indices:
            if vals[i] < cutoff:
                return abs(offsets[i])
        return float(max_px)

    return float(edge(range(mid, len(offsets))) + edge(range(mid, -1, -1)))


def pixel_to_world(u: float, v: float, height_at: float, obs_cam_pos: np.ndarray,
                   obs_cam_mat: np.ndarray, intr: CameraIntrinsics, table_z: float) -> np.ndarray:
    """Convert a pixel plus a known height-above-table into a world point."""
    surface_world_z = table_z + height_at
    # Perpendicular distance from a downward-looking camera to that height.
    depth = float(obs_cam_pos[2] - surface_world_z)
    return deproject_pixels(np.array([[u, v]]), np.array([depth]), obs_cam_pos, obs_cam_mat, intr)[0]


def principal_axis(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Centroid and orientation of a binary mask via PCA.

    Returns ``(centroid_uv, major_angle, minor_angle)`` with angles in image
    coordinates. A parallel jaw should close along the **minor** axis.
    """
    vs, us = np.nonzero(mask)
    if us.size < 3:
        return np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0]), 0.0, np.pi / 2.0
    pts = np.stack([us, vs], axis=1).astype(np.float64)
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    minor = eigvecs[:, int(np.argmin(eigvals))]
    return centroid, float(np.arctan2(major[1], major[0])), float(np.arctan2(minor[1], minor[0]))
