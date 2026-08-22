"""The learned policy: one forward pass, argmax over (pixel, angle), execute.

Inference is deliberately trivial -- no candidate sampling, no CEM, no
refinement. The network predicts grasp quality densely for every pixel and every
gripper angle, so choosing a grasp is an argmax over that volume. Two masks are
applied first:

* **reachability** -- pixels whose deprojection lies outside the arm's workspace
  are excluded, because a grasp the robot cannot reach is not a useful
  prediction and would otherwise show up as an IK failure rather than a
  perception error.
* **smoothing** -- the quality map is blurred before the argmax. A dense
  prediction has isolated high-value pixels that are not supported by their
  neighbourhood; picking one puts the gripper a millimetre from a cliff edge.
  Blurring selects the centre of a broad high-quality region instead, which is
  what GG-CNN does for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from ..camera import deproject_pixels
from ..env import Observation, PandaGraspEnv
from ..grasp import Grasp, ImageGrasp, grasp_z_from_surface, image_to_grasp
from ..heightmap import surface_height
from ..models import bin_to_angle
from ..models.device import pick_device
from ..objects import SAFE_GRASP_WIDTH
from ..scene import WORKSPACE_X, WORKSPACE_Y


class LearnedPolicy:
    name = "cnn"

    def __init__(self, checkpoint: str | Path = "runs/grasp_cnn/best.pt",
                 device: str | None = None, smooth_sigma: float = 2.0,
                 workspace_margin: float = 0.03, min_quality: float = 0.0):
        from ..training import load_checkpoint

        self.device = pick_device(device)
        self.model, self.cfg = load_checkpoint(checkpoint, self.device)
        self.smooth_sigma = float(smooth_sigma)
        self.workspace_margin = float(workspace_margin)
        self.min_quality = float(min_quality)
        self._mask_cache: dict[tuple, np.ndarray] = {}
        self.last_quality: np.ndarray | None = None
        self.last_confidence: float = 0.0

    # -- input --------------------------------------------------------------- #
    def _tensor(self, obs: Observation) -> tuple[torch.Tensor, float]:
        """Build the network input, resized to whatever the model was trained at.

        Returns the tensor and the scale from *model* pixels back to *observation*
        pixels, so predictions can be mapped onto the real image.
        """
        from ..data.dataset import normalise_height, normalise_rgb

        height, rgb = obs.height, obs.rgb
        scale = 1.0
        target = self.cfg.input_size
        if target and target != height.shape[0]:
            interp = cv2.INTER_AREA if target < height.shape[0] else cv2.INTER_LINEAR
            scale = height.shape[0] / target
            height = cv2.resize(height, (target, target), interpolation=interp)
            rgb = cv2.resize(rgb, (target, target), interpolation=interp)

        channels = [normalise_height(height)]
        if self.cfg.use_rgb:
            channels.extend(normalise_rgb(rgb))
        x = np.stack(channels, axis=0)[None]
        return torch.from_numpy(x).to(self.device), scale

    def _workspace_mask(self, obs: Observation, shape: tuple[int, int]) -> np.ndarray:
        key = (shape, float(obs.cam_pos[2]))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        h, w = shape
        sy = obs.height.shape[0] / h
        sx = obs.height.shape[1] / w
        vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        uv = np.stack([uu.ravel() * sx, vv.ravel() * sy], axis=1).astype(np.float64)
        depth = np.full(uv.shape[0], float(obs.cam_pos[2] - obs.table_z))
        world = deproject_pixels(uv, depth, obs.cam_pos, obs.cam_mat, obs.intrinsics)
        m = self.workspace_margin
        ok = ((world[:, 0] >= WORKSPACE_X[0] - m) & (world[:, 0] <= WORKSPACE_X[1] + m) &
              (world[:, 1] >= WORKSPACE_Y[0] - m) & (world[:, 1] <= WORKSPACE_Y[1] + m))
        mask = ok.reshape(h, w)
        self._mask_cache[key] = mask
        return mask

    # -- inference ------------------------------------------------------------ #
    @torch.no_grad()
    def predict_maps(self, obs: Observation) -> tuple[np.ndarray, np.ndarray, float]:
        """Dense (quality, width) maps plus the model-to-observation pixel scale."""
        x, scale = self._tensor(obs)
        out = self.model(x)
        quality = torch.sigmoid(out["quality"])[0].float().cpu().numpy()
        width = out["width"][0].float().cpu().numpy()
        if self.smooth_sigma > 0:
            sigma = self.smooth_sigma / scale  # blur a fixed *physical* radius
            k = max(3, int(2 * round(3 * sigma) + 1))
            quality = np.stack([cv2.GaussianBlur(q, (k, k), sigma) for q in quality])
        return quality, width, scale

    def __call__(self, obs: Observation, env: PandaGraspEnv,
                 rng: np.random.Generator) -> Grasp:
        quality, width, scale = self.predict_maps(obs)
        mask = self._workspace_mask(obs, quality.shape[1:])
        quality = np.where(mask[None], quality, -1.0)
        self.last_quality = quality

        flat = int(np.argmax(quality))
        b, v, u = np.unravel_index(flat, quality.shape)
        self.last_confidence = float(quality[b, v, u])

        # Map the prediction back onto the observation's pixel grid.
        u_obs, v_obs = float(u) * scale, float(v) * scale
        angle = float(bin_to_angle(int(b)))
        h = surface_height(obs.height, u_obs, v_obs)
        depth = float(obs.cam_pos[2] - (obs.table_z + h))
        width_px = float(width[b, v, u]) * quality.shape[-1] * scale
        width_px = float(np.clip(width_px, 2.0, 40.0))

        img = ImageGrasp(u=u_obs, v=v_obs, angle=angle, width_px=width_px, depth=depth)
        g = image_to_grasp(img, obs.cam_pos, obs.cam_mat, obs.intrinsics)
        return Grasp(x=g.x, y=g.y,
                     z=grasp_z_from_surface(obs.table_z + h, obs.table_z),
                     yaw=g.yaw, width=float(np.clip(g.width, 0.008, SAFE_GRASP_WIDTH)))
