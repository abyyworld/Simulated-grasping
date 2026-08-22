"""Dense grasp-quality network: ResNet-18 encoder, U-Net decoder, per-angle heads.

Output
------
For an input image of shape ``(C, H, W)`` the network predicts, at every pixel
and for each of ``ANGLE_BINS`` gripper orientations:

* ``quality``  logit for "a grasp centred here at this angle succeeds"
* ``width``    the object's width along that angle, in pixels / ``H``

Why per-angle quality instead of GG-CNN's (quality, cos2t, sin2t)?
-----------------------------------------------------------------
GG-CNN regresses a single best angle per pixel and is trained from *annotated*
grasp rectangles, where every label is positive. Our labels come from executing
a sampled grasp, so a failure carries no information about whether the position
or the angle was wrong -- regressing the angle from failures is meaningless, and
throwing failures away discards more than half the dataset.

Binning the angle turns the problem into classification over
``(u, v, angle_bin)``, and then a failure is a clean negative for exactly the
cell that was tried. Inference is an argmax over the whole (pixel, angle) volume,
which is a single forward pass. This is the discretised-orientation formulation
used by Zeng et al.'s pick-and-place work, adapted to a dense decoder.

Angles are wrapped to ``[-pi/2, pi/2)`` because a parallel jaw is symmetric under
a half turn, so ``ANGLE_BINS`` bins tile 180 degrees, not 360.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

ANGLE_BINS = 12
BIN_WIDTH = math.pi / ANGLE_BINS


def angle_to_bin(angle: torch.Tensor | float) -> torch.Tensor:
    """Wrapped angle in [-pi/2, pi/2) -> bin index in [0, ANGLE_BINS)."""
    a = torch.as_tensor(angle, dtype=torch.float32)
    shifted = (a + math.pi / 2.0) / math.pi  # -> [0, 1)
    idx = torch.floor(shifted * ANGLE_BINS)
    return idx.clamp_(0, ANGLE_BINS - 1).long()


def bin_to_angle(index: torch.Tensor | int) -> torch.Tensor:
    """Bin index -> the angle at the centre of that bin."""
    i = torch.as_tensor(index, dtype=torch.float32)
    return (i + 0.5) * BIN_WIDTH - math.pi / 2.0


class UpBlock(nn.Module):
    """Bilinear upsample, concatenate the encoder skip, then two 3x3 convs.

    Bilinear + conv rather than a transposed convolution: transposed convs put
    a checkerboard pattern into dense predictions, which for a grasp-quality map
    means a grid of spurious local maxima that the argmax happily picks.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        size = skip.shape[-2:] if skip is not None else (x.shape[-2] * 2, x.shape[-1] * 2)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.block(x)


class GraspNet(nn.Module):
    """ResNet-18 encoder + U-Net decoder + per-angle quality and width heads."""

    def __init__(self, in_channels: int = 4, angle_bins: int = ANGLE_BINS,
                 pretrained: bool = False, decoder_channels=(256, 128, 64, 32, 32)):
        super().__init__()
        self.in_channels = in_channels
        self.angle_bins = angle_bins

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.stem = self._make_stem(net, in_channels, pretrained)
        self.maxpool = net.maxpool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4

        c = decoder_channels
        self.up4 = UpBlock(512, 256, c[0])
        self.up3 = UpBlock(c[0], 128, c[1])
        self.up2 = UpBlock(c[1], 64, c[2])
        self.up1 = UpBlock(c[2], 64, c[3])
        self.up0 = UpBlock(c[3], 0, c[4])

        self.quality_head = nn.Conv2d(c[4], angle_bins, 1)
        self.width_head = nn.Conv2d(c[4], angle_bins, 1)
        # Start pessimistic: most (pixel, angle) cells are not graspable, so bias
        # the quality logits negative and let evidence push them up. Without this
        # the first epochs are dominated by unlearning a uniform 0.5 prior.
        nn.init.constant_(self.quality_head.bias, -2.0)

    @staticmethod
    def _make_stem(net, in_channels: int, pretrained: bool) -> nn.Sequential:
        conv = net.conv1
        if in_channels != 3:
            new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    w = conv.weight  # (64, 3, 7, 7)
                    if in_channels >= 3:
                        # Channel 0 is height: seed it with the mean RGB filter so
                        # it starts as a generic edge detector rather than noise.
                        new.weight[:, 0] = w.mean(dim=1)
                        new.weight[:, 1:4] = w[:, : min(3, in_channels - 1)]
                    else:
                        new.weight[:] = w.mean(dim=1, keepdim=True)
            conv = new
        return nn.Sequential(conv, net.bn1, net.relu)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        c1 = self.stem(x)          # stride 2,  64
        c2 = self.layer1(self.maxpool(c1))  # stride 4,  64
        c3 = self.layer2(c2)       # stride 8,  128
        c4 = self.layer3(c3)       # stride 16, 256
        c5 = self.layer4(c4)       # stride 32, 512

        d = self.up4(c5, c4)
        d = self.up3(d, c3)
        d = self.up2(d, c2)
        d = self.up1(d, c1)
        d = self.up0(d, None)
        if d.shape[-2:] != x.shape[-2:]:
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"quality": self.quality_head(d), "width": self.width_head(d)}


def sample_at(maps: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample ``maps`` (B, C, H, W) at float pixel coordinates.

    ``u``/``v`` may be shape ``(B,)`` -- one point per image, returning ``(B, C)``
    -- or ``(B, K)`` -- K points per image, returning ``(B, K, C)``.

    Grasp labels have sub-pixel positions, so nearest-neighbour indexing would
    quantise every label to the pixel grid and throw away up to 1.1 mm of the
    2.19 mm ground sampling distance. ``grid_sample`` keeps the supervision
    exactly where the grasp happened.
    """
    b, c, h, w = maps.shape
    squeeze = u.dim() == 1
    uu = u.view(b, -1)
    vv = v.view(b, -1)
    k = uu.shape[1]
    # grid_sample expects normalised coordinates in [-1, 1] on pixel *centres*.
    gx = (2.0 * uu / max(w - 1, 1)) - 1.0
    gy = (2.0 * vv / max(h - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(b, 1, k, 2)
    out = F.grid_sample(maps, grid, mode="bilinear", padding_mode="border", align_corners=True)
    out = out.view(b, c, k).permute(0, 2, 1)  # (B, K, C)
    return out[:, 0, :] if squeeze else out
