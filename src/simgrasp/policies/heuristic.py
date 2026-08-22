"""Depth-only heuristic baseline: grasp the centroid across the minor axis.

This is the floor the learned policy must beat. It uses no learning and no
ground truth -- just the height map: segment everything standing above the
table, take the centroid, and close the fingers along the silhouette's minor
principal axis.

It is a genuinely reasonable heuristic (it is roughly what a hand-written
bin-picking script does) and it is strong on convex, roughly-symmetric objects.
Its weakness is exactly the interesting one: on non-convex shapes the centroid
of the silhouette need not lie on the object at all -- consider an L -- so this
baseline should degrade sharply on the held-out categories while a network that
has learned local graspable geometry should not.
"""

from __future__ import annotations

import numpy as np

from ..env import Observation, PandaGraspEnv
from ..grasp import Grasp, ImageGrasp, grasp_z_from_surface, image_to_grasp
from ..heightmap import estimate_width_px, object_mask, principal_axis, surface_height
from ..objects import SAFE_GRASP_WIDTH


class HeuristicPolicy:
    name = "heuristic"

    def __init__(self, snap_to_object: bool = True):
        # When the centroid falls off the object (non-convex shapes), optionally
        # snap to the nearest object pixel. Off by default in the reported
        # baseline so the failure mode stays visible in the numbers.
        self.snap_to_object = snap_to_object

    def __call__(self, obs: Observation, env: PandaGraspEnv,
                 rng: np.random.Generator) -> Grasp:
        mask = object_mask(obs.height)
        if not mask.any():
            centre = (obs.intrinsics.cx, obs.intrinsics.cy)
            return self._grasp_at(obs, centre[0], centre[1], 0.0)

        centroid, _major, minor = principal_axis(mask)
        u, v = float(centroid[0]), float(centroid[1])

        if self.snap_to_object and not mask[int(np.clip(round(v), 0, mask.shape[0] - 1)),
                                            int(np.clip(round(u), 0, mask.shape[1] - 1))]:
            vs, us = np.nonzero(mask)
            d = (us - u) ** 2 + (vs - v) ** 2
            i = int(np.argmin(d))
            u, v = float(us[i]), float(vs[i])

        return self._grasp_at(obs, u, v, minor)

    @staticmethod
    def _grasp_at(obs: Observation, u: float, v: float, angle: float) -> Grasp:
        h = surface_height(obs.height, u, v)
        depth = float(obs.cam_pos[2] - (obs.table_z + h))
        width_px = estimate_width_px(obs.height, u, v, angle)
        img = ImageGrasp(u=u, v=v, angle=angle, width_px=max(width_px, 1.0), depth=depth)
        g = image_to_grasp(img, obs.cam_pos, obs.cam_mat, obs.intrinsics)
        return Grasp(x=g.x, y=g.y,
                     z=grasp_z_from_surface(obs.table_z + h, obs.table_z),
                     yaw=g.yaw, width=float(np.clip(g.width, 0.008, SAFE_GRASP_WIDTH)))
