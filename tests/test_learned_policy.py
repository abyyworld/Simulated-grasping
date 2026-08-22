"""End-to-end wiring of the learned policy.

An untrained network makes useless grasps, which is fine: what is under test is
that a checkpoint round-trips, that inference runs at whatever resolution the
model was trained at, and that the predicted pixel maps back onto the real image.
That last one is the easy thing to get silently wrong when `--input-size` rescales
the inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simgrasp.models import ANGLE_BINS, GraspNet
from simgrasp.scene import WORKSPACE_X, WORKSPACE_Y
from simgrasp.training import TrainConfig, load_checkpoint, save_checkpoint


@pytest.fixture(params=[None, 112], ids=["native", "resized"])
def checkpoint(tmp_path_factory, request):
    """A tiny saved checkpoint, at native resolution and at half resolution."""
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    cfg = TrainConfig(use_rgb=True, input_size=request.param)
    save_checkpoint(path, GraspNet(in_channels=4, angle_bins=ANGLE_BINS), cfg, {})
    return path


def test_checkpoint_round_trips(checkpoint):
    model, cfg = load_checkpoint(checkpoint, torch.device("cpu"))
    assert cfg.use_rgb
    assert isinstance(model, GraspNet)
    assert not model.training, "a loaded model must be in eval mode"


def test_policy_produces_a_reachable_grasp(env, checkpoint, rng):
    from simgrasp.policies.learned import LearnedPolicy

    policy = LearnedPolicy(checkpoint=checkpoint, device="cpu")
    obs = env.reset(0)
    grasp = policy(obs, env, rng)

    margin = policy.workspace_margin + 1e-6
    assert WORKSPACE_X[0] - margin <= grasp.x <= WORKSPACE_X[1] + margin
    assert WORKSPACE_Y[0] - margin <= grasp.y <= WORKSPACE_Y[1] + margin
    assert grasp.z > env.table_z
    assert -np.pi / 2 - 1e-9 <= grasp.yaw < np.pi / 2 + 1e-9
    assert 0.008 <= grasp.width <= 0.07


def test_predicted_maps_have_one_channel_per_angle(env, checkpoint):
    from simgrasp.policies.learned import LearnedPolicy

    policy = LearnedPolicy(checkpoint=checkpoint, device="cpu")
    obs = env.reset(0)
    quality, width, scale = policy.predict_maps(obs)
    assert quality.shape[0] == ANGLE_BINS
    assert width.shape == quality.shape
    assert quality.min() >= 0.0 and quality.max() <= 1.0, "quality must be a probability"
    # scale maps model pixels back onto observation pixels.
    assert quality.shape[-1] * scale == pytest.approx(obs.height.shape[-1], abs=1.0)


def test_workspace_mask_excludes_unreachable_pixels(env, checkpoint):
    from simgrasp.policies.learned import LearnedPolicy

    policy = LearnedPolicy(checkpoint=checkpoint, device="cpu")
    obs = env.reset(0)
    quality, _width, _scale = policy.predict_maps(obs)
    mask = policy._workspace_mask(obs, quality.shape[1:])
    assert mask.any(), "some of the image must be reachable"
    assert not mask.all(), "the camera sees beyond the workspace, so some must be masked"


def test_grasp_is_executable(env, checkpoint, rng):
    """Whatever the policy proposes must at least be a runnable command."""
    from simgrasp.policies.learned import LearnedPolicy

    policy = LearnedPolicy(checkpoint=checkpoint, device="cpu")
    obs = env.reset(0)
    result = env.execute(policy(obs, env, rng))
    assert result.reason in {"success", "closed_empty", "dropped", "no_lift",
                             "ik_failed", "unstable"}


def test_policy_is_deterministic(env, checkpoint, rng):
    from simgrasp.policies.learned import LearnedPolicy

    policy = LearnedPolicy(checkpoint=checkpoint, device="cpu")
    obs = env.reset(3)
    a = policy(obs, env, rng)
    b = policy(obs, env, rng)
    assert a.as_array() == pytest.approx(b.as_array())
