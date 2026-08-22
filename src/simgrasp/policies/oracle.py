"""Ground-truth grasp policy and the data-collection grasp sampler."""

from __future__ import annotations

import cv2
import numpy as np

from ..env import Observation, PandaGraspEnv
from ..grasp import Grasp, ImageGrasp, grasp_z_from_surface, image_to_grasp
from ..heightmap import estimate_width_px, object_mask, surface_height
from ..objects import SAFE_GRASP_WIDTH
from ..scene import WORKSPACE_X, WORKSPACE_Y
from ..transforms import wrap_grasp_angle


class OraclePolicy:
    """Grasps using the object's true settled pose and its catalogue grasp hint.

    This is the ceiling for the whole project: it is what a perfect perception
    system feeding this controller would achieve. Its failures are controller and
    physics failures, not perception failures, which makes it the right thing to
    subtract when reading the learned policy's numbers.
    """

    name = "oracle"

    def __call__(self, obs: Observation, env: PandaGraspEnv,
                 rng: np.random.Generator) -> Grasp:
        st = env.state
        assert st is not None
        idx = int(rng.integers(len(st.spec.grasp_hints))) if len(st.spec.grasp_hints) > 1 else 0
        return env.oracle_grasp(idx)


class GraspSampler:
    """The distribution grasps are drawn from during dataset collection.

    Executing only the oracle grasp would give a dataset that is ~90% positive
    and contains no information about *where not to grasp*, which is precisely
    what a grasp-quality network has to learn. So episodes are drawn from a
    mixture:

    ``near_oracle``  perturbed ground truth -- mostly positives, and the
                     near-miss cases that teach precision.
    ``on_object``    a random point on the object mask at a random angle --
                     a mix, and the bulk of the interesting signal.
    ``edge``         a point near the silhouette boundary -- hard cases where a
                     few millimetres decide the outcome.
    ``off_object``   a random point on bare table -- unambiguous negatives that
                     stop the network predicting high quality everywhere.

    The weights are a design choice, not a tuned hyper-parameter; they are
    recorded in the dataset metadata so a rerun is reproducible.
    """

    name = "sampler"

    def __init__(self, weights: dict[str, float] | None = None,
                 pos_noise: float = 0.012, yaw_noise: float = 0.30, z_noise: float = 0.004):
        self.weights = weights or {
            "near_oracle": 0.45,
            "on_object": 0.25,
            "edge": 0.15,
            "off_object": 0.15,
        }
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.pos_noise = pos_noise
        self.yaw_noise = yaw_noise
        self.z_noise = z_noise
        self.last_mode = "near_oracle"

    def __call__(self, obs: Observation, env: PandaGraspEnv,
                 rng: np.random.Generator) -> Grasp:
        modes = list(self.weights)
        probs = [self.weights[m] for m in modes]
        mode = str(rng.choice(modes, p=probs))
        self.last_mode = mode

        if mode == "near_oracle":
            return self._near_oracle(obs, env, rng)
        if mode == "off_object":
            return self._off_object(obs, env, rng)
        return self._from_mask(obs, env, rng, edge=(mode == "edge"))

    # -- modes -------------------------------------------------------------- #
    def _near_oracle(self, obs: Observation, env: PandaGraspEnv, rng) -> Grasp:
        st = env.state
        assert st is not None
        idx = int(rng.integers(len(st.spec.grasp_hints)))
        g = env.oracle_grasp(idx)
        return Grasp(
            x=g.x + float(rng.normal(0.0, self.pos_noise)),
            y=g.y + float(rng.normal(0.0, self.pos_noise)),
            z=max(g.z + float(rng.normal(0.0, self.z_noise)), obs.table_z + 0.014),
            yaw=float(wrap_grasp_angle(g.yaw + rng.normal(0.0, self.yaw_noise))),
            width=g.width,
        )

    def _from_mask(self, obs: Observation, env: PandaGraspEnv, rng, edge: bool) -> Grasp:
        mask = object_mask(obs.height)
        if edge:
            # Silhouette band: object pixels within ~4 px of the boundary. These
            # are the grasps where a couple of millimetres decide the outcome.
            eroded = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1)
            band = mask & (eroded == 0)
            if band.any():
                mask = band
        vs, us = np.nonzero(mask)
        if us.size == 0:
            return self._off_object(obs, env, rng)
        i = int(rng.integers(us.size))
        return self._grasp_at_pixel(obs, float(us[i]), float(vs[i]),
                                    float(rng.uniform(-np.pi / 2, np.pi / 2)))

    def _off_object(self, obs: Observation, env: PandaGraspEnv, rng) -> Grasp:
        x = float(rng.uniform(*WORKSPACE_X))
        y = float(rng.uniform(*WORKSPACE_Y))
        return Grasp(x=x, y=y, z=obs.table_z + 0.014,
                     yaw=float(rng.uniform(-np.pi / 2, np.pi / 2)), width=0.05)

    @staticmethod
    def _grasp_at_pixel(obs: Observation, u: float, v: float, angle: float) -> Grasp:
        """Turn an (image pixel, image angle) proposal into a world-frame grasp.

        Routed through :func:`~simgrasp.grasp.image_to_grasp` rather than
        re-deriving the axis convention here, so there is exactly one place in
        the codebase that knows how image angles map to world yaw.
        """
        h = surface_height(obs.height, u, v)
        depth = float(obs.cam_pos[2] - (obs.table_z + h))
        width_px = estimate_width_px(obs.height, u, v, angle)
        img = ImageGrasp(u=u, v=v, angle=angle, width_px=max(width_px, 1.0), depth=depth)
        g = image_to_grasp(img, obs.cam_pos, obs.cam_mat, obs.intrinsics)
        return Grasp(x=g.x, y=g.y,
                     z=grasp_z_from_surface(obs.table_z + h, obs.table_z),
                     yaw=g.yaw, width=float(np.clip(g.width, 0.008, SAFE_GRASP_WIDTH)))
