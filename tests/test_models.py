from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from simgrasp.models import ANGLE_BINS, GraspNet, angle_to_bin, bin_to_angle, pick_device
from simgrasp.models.grasp_net import sample_at
from simgrasp.training import average_precision


@pytest.mark.parametrize("in_channels", [1, 4])
def test_forward_shapes(in_channels):
    net = GraspNet(in_channels=in_channels)
    out = net(torch.randn(2, in_channels, 96, 96))
    assert out["quality"].shape == (2, ANGLE_BINS, 96, 96)
    assert out["width"].shape == (2, ANGLE_BINS, 96, 96)


def test_output_matches_input_resolution_for_odd_sizes():
    net = GraspNet(in_channels=1)
    out = net(torch.randn(1, 1, 100, 76))
    assert out["quality"].shape[-2:] == (100, 76)


def test_quality_starts_pessimistic():
    """A uniform 0.5 prior wastes the first epochs unlearning itself."""
    net = GraspNet(in_channels=4)
    with torch.no_grad():
        p = torch.sigmoid(net(torch.zeros(1, 4, 64, 64))["quality"])
    assert p.mean() < 0.35


def test_angle_bins_tile_a_half_turn():
    edges = torch.tensor([-math.pi / 2, -math.pi / 2 + 1e-6, 0.0, math.pi / 2 - 1e-6])
    bins = angle_to_bin(edges)
    assert bins.min() >= 0 and bins.max() < ANGLE_BINS
    assert angle_to_bin(torch.tensor(-math.pi / 2)) == 0
    assert angle_to_bin(torch.tensor(math.pi / 2 - 1e-6)) == ANGLE_BINS - 1


def test_bin_centres_round_trip():
    for i in range(ANGLE_BINS):
        assert int(angle_to_bin(bin_to_angle(i))) == i


def test_bin_to_angle_stays_in_range():
    a = bin_to_angle(torch.arange(ANGLE_BINS))
    assert torch.all(a >= -math.pi / 2) and torch.all(a < math.pi / 2)


def test_sample_at_is_bilinear_and_exact_on_a_ramp():
    h = w = 8
    ramp = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w).expand(2, 3, h, w).contiguous()
    u = torch.tensor([0.0, 3.5])
    v = torch.tensor([2.0, 5.0])
    out = sample_at(ramp, u, v)
    assert out.shape == (2, 3)
    assert out[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert out[1, 0] == pytest.approx(3.5, abs=1e-5)


def test_sample_at_multi_point():
    h = w = 8
    ramp = torch.arange(w, dtype=torch.float32).view(1, 1, 1, w).expand(2, 3, h, w).contiguous()
    u = torch.tensor([[0.0, 7.0], [1.0, 2.0]])
    v = torch.zeros_like(u)
    out = sample_at(ramp, u, v)
    assert out.shape == (2, 2, 3)
    assert out[0, 1, 0] == pytest.approx(7.0, abs=1e-5)


def test_sample_at_clamps_outside_the_image():
    ramp = torch.arange(8, dtype=torch.float32).view(1, 1, 1, 8).expand(1, 1, 8, 8).contiguous()
    out = sample_at(ramp, torch.tensor([100.0]), torch.tensor([100.0]))
    assert torch.isfinite(out).all()


def test_gradients_flow_to_the_sampled_pixel():
    net = GraspNet(in_channels=1)
    x = torch.randn(1, 1, 64, 64)
    q = net(x)["quality"]
    sample_at(q, torch.tensor([12.3]), torch.tensor([40.7])).sum().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_average_precision():
    assert average_precision(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == 1.0
    assert np.isnan(average_precision(np.array([0.5, 0.5]), np.array([1, 1])))
    mixed = average_precision(np.array([0.9, 0.2, 0.8, 0.1]), np.array([1, 1, 0, 0]))
    assert 0.0 < mixed < 1.0


def test_pick_device_returns_something_usable():
    assert pick_device().type in {"cuda", "mps", "cpu"}
    assert pick_device("cpu").type == "cpu"
