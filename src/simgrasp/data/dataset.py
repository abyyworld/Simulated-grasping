"""Streaming ``Dataset`` over the shards written by :mod:`simgrasp.data.writer`.

Three things happen here beyond reading arrays off disk, and each exists because
of a measured failure:

**Memory mapping.** Shards are opened with ``mmap_mode="r"`` and never fully
read. A 10k-episode dataset at 224 px is ~2.5 GB; the machine this was built on
has 16 GB and also has to hold the model, the optimiser and four dataloader
workers.

**Free negatives.** Each episode supervises one cell of a
``224 x 224 x 12`` output volume. That is a gradient on 1 cell in 600k, and the
network's shortest path to a low loss is to predict "graspable" nowhere or
everywhere. Table pixels far from any object are certain failures at every
angle, they are free to label from the height map, and they give the loss a
dense negative signal. They are down-weighted in the loss because they are
plentiful and trivial (see :mod:`simgrasp.training`).

**Rotation augmentation.** A top-down grasp dataset has an accidental
orientation prior: objects settle in whatever poses the sampler produced. The
augmentation rotates image and label together. The sign of the angle transform
is the whole game, and it fails *silently* if inverted (the loss still falls,
the network just learns the perpendicular grasp), so it is pinned by
``tests/test_augmentation.py`` against measured image geometry rather than
against the derivation below.

Angle convention
----------------
``cv2.getRotationMatrix2D(centre, phi_deg, 1.0)`` maps a point offset
``(du, dv)`` from the centre to ``(cos phi du + sin phi dv, -sin phi du +
cos phi dv)``. A direction ``(cos theta, sin theta)`` therefore becomes
``(cos(theta - phi), sin(theta - phi))``: the label angle transforms as
``theta -> theta - phi``, *not* ``theta + phi``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..heightmap import OBJECT_HEIGHT_THRESHOLD
from ..transforms import wrap_grasp_angle
from .writer import decode_height

# Height at which the input channel saturates, in metres. The tallest object in
# the catalogue tops out at 0.10 m, so this keeps the whole catalogue inside the
# linear range while still using most of it.
HEIGHT_CLIP = 0.12
# Free negatives are mined at least this far (in pixels) from any object pixel,
# so a "negative" is never a near miss that might actually have succeeded.
FREE_NEGATIVE_MARGIN_PX = 12


def normalise_height(height: np.ndarray) -> np.ndarray:
    """Height map in metres -> a single float32 channel in [0, 1]."""
    h = np.asarray(height, dtype=np.float32)
    return np.clip(h, 0.0, HEIGHT_CLIP) / np.float32(HEIGHT_CLIP)


def normalise_rgb(rgb: np.ndarray) -> list[np.ndarray]:
    """uint8 RGB -> three float32 channels in [0, 1]."""
    arr = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    return [arr[..., i] for i in range(3)]


class GraspDataset(Dataset):
    """Every executed grasp in ``root``, filtered and returned as tensors.

    Filters compose; each one is applied to the label index at construction time,
    so a filtered dataset costs no more per sample than an unfiltered one.

    Parameters
    ----------
    split, categories, successes_only, episode_filter:
        Label filters. ``episode_filter`` is what keeps train and validation
        disjoint: an episode is one *image*, so splitting on anything finer would
        put the same image in both sets.
    use_rgb:
        4-channel (height + RGB) input when true, height only when false.
    free_negatives:
        How many table pixels to mine per sample. 0 disables mining.
    input_size:
        Resize images to this resolution and scale the labels with them.
    rotate:
        Rotation augmentation. Training only; rotating the validation set makes
        the metric noisy and incomparable between runs.
    """

    def __init__(
        self,
        root: Path | str,
        split: str | None = None,
        categories: list[str] | None = None,
        successes_only: bool = False,
        episode_filter: list[int] | None = None,
        use_rgb: bool = True,
        free_negatives: int = 32,
        free_negative_margin_px: int = FREE_NEGATIVE_MARGIN_PX,
        input_size: int | None = None,
        rotate: bool = False,
        seed: int = 0,
    ):
        self.root = Path(root)
        self.use_rgb = bool(use_rgb)
        self.free_negatives = int(free_negatives)
        self.free_negative_margin_px = int(free_negative_margin_px)
        self.input_size = int(input_size) if input_size else None
        self.rotate = bool(rotate)
        self._rng = np.random.default_rng(seed)

        # ``dataset_meta.json`` also matches ``*_meta.json``; excluding it by name
        # rather than by glob keeps the shard discovery robust to new sidecars.
        shard_metas = sorted(p for p in self.root.glob("*_meta.json")
                             if p.name != "dataset_meta.json")
        if not shard_metas:
            raise FileNotFoundError(f"no shards found in {self.root}")

        wanted = set(categories) if categories else None
        episodes = set(episode_filter) if episode_filter is not None else None

        self.labels: list[dict[str, Any]] = []
        self._shard_files: dict[str, tuple[Path, Path | None]] = {}
        self.image_size: int | None = None

        for meta_path in shard_metas:
            meta = json.loads(meta_path.read_text())
            stem = meta.get("shard", meta_path.name[: -len("_meta.json")])
            height_path = self.root / f"{stem}_height.npy"
            if not height_path.exists():
                continue
            rgb_path = self.root / f"{stem}_rgb.npy"
            self._shard_files[stem] = (height_path, rgb_path if rgb_path.exists() else None)
            self.image_size = self.image_size or int(meta.get("image_size", 0)) or None

            for row, label in enumerate(meta["labels"]):
                if split and label.get("split") != split:
                    continue
                if wanted and label.get("category") not in wanted:
                    continue
                if successes_only and not label.get("success"):
                    continue
                if episodes is not None and label.get("episode") not in episodes:
                    continue
                self.labels.append({**label, "_shard": stem, "_row": row})

        if self.image_size is None:
            dataset_meta = self.root / "dataset_meta.json"
            if dataset_meta.exists():
                self.image_size = int(json.loads(dataset_meta.read_text())["image_size"])

        # Opened lazily, and per process: a memmap cannot be inherited across a
        # dataloader worker fork safely, so the handle cache must fill after the
        # fork rather than before it.
        self._height_maps: dict[str, np.memmap] = {}
        self._rgb_maps: dict[str, np.memmap | None] = {}

    # -- indexing -------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.labels)

    def _height_map(self, stem: str) -> np.memmap:
        handle = self._height_maps.get(stem)
        if handle is None:
            handle = np.load(self._shard_files[stem][0], mmap_mode="r")
            self._height_maps[stem] = handle
        return handle

    def _rgb_map(self, stem: str) -> np.memmap | None:
        if stem not in self._rgb_maps:
            path = self._shard_files[stem][1]
            self._rgb_maps[stem] = np.load(path, mmap_mode="r") if path else None
        return self._rgb_maps[stem]

    def raw(self, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Decoded ``(height_metres, rgb_uint8, label)`` for one sample."""
        label = self.labels[index]
        stem, row = label["_shard"], label["_row"]
        height = decode_height(np.asarray(self._height_map(stem)[row]))
        rgb_map = self._rgb_map(stem)
        if rgb_map is not None:
            rgb = np.asarray(rgb_map[row], dtype=np.uint8)
        else:
            # Height-only datasets still need a placeholder of the right shape so
            # the augmentation and resize paths do not need a special case.
            rgb = np.zeros((*height.shape, 3), dtype=np.uint8)
        return height, rgb, dict(label)

    # -- augmentation ---------------------------------------------------------- #
    def _rotate_sample(self, height: np.ndarray, rgb: np.ndarray, label: dict[str, Any]):
        """Rotate image and label together by a uniform random angle.

        Returns ``(height, rgb, label, valid)`` where ``valid`` marks pixels that
        came from the original image rather than from border fill. Padding is not
        observed table and must never be mined as a free negative.
        """
        phi_deg = float(self._rng.uniform(-180.0, 180.0))
        h, w = height.shape
        centre = ((w - 1) / 2.0, (h - 1) / 2.0)
        mat = cv2.getRotationMatrix2D(centre, phi_deg, 1.0)

        # Border fill: table level for height, and the image's median colour for
        # RGB. A black fill would be a hard artificial edge for the network to
        # latch onto instead of the object.
        table_rgb = np.median(rgb.reshape(-1, 3), axis=0)
        rot_height = cv2.warpAffine(height, mat, (w, h), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        rot_rgb = cv2.warpAffine(rgb, mat, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=tuple(float(c) for c in table_rgb))
        valid = cv2.warpAffine(np.ones((h, w), np.uint8), mat, (w, h),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0).astype(bool)

        u, v = float(label["u"]), float(label["v"])
        u_rot = mat[0, 0] * u + mat[0, 1] * v + mat[0, 2]
        v_rot = mat[1, 0] * u + mat[1, 1] * v + mat[1, 2]
        # theta -> theta - phi. See the module docstring; the sign is load-bearing.
        angle = float(wrap_grasp_angle(float(label["angle"]) - np.deg2rad(phi_deg)))

        out = dict(label)
        out["u"], out["v"], out["angle"] = float(u_rot), float(v_rot), angle
        return rot_height, rot_rgb, out, valid

    def _mine_free_negatives(self, height: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Sample table pixels at least ``margin`` px clear of every object pixel.

        Returns ``(free_negatives, 2)`` of ``(u, v)``, padded with ``-1`` when the
        image does not contain enough qualifying pixels. The sentinel matters: an
        impossible margin must degrade to "no negatives", not to negatives mined
        from somewhere wrong.
        """
        k = self.free_negatives
        if k <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        free = (height <= OBJECT_HEIGHT_THRESHOLD) & valid
        # Distance from every pixel to the nearest object pixel. Padding counts as
        # object here so negatives are never mined up against the border fill.
        obstacle = (~free).astype(np.uint8)
        distance = cv2.distanceTransform((1 - obstacle).astype(np.uint8), cv2.DIST_L2, 5)
        eligible = distance > self.free_negative_margin_px

        vs, us = np.nonzero(eligible)
        out = np.full((k, 2), -1.0, dtype=np.float32)
        if us.size:
            pick = self._rng.integers(0, us.size, size=min(k, us.size))
            out[: len(pick), 0] = us[pick]
            out[: len(pick), 1] = vs[pick]
        return out

    # -- tensors --------------------------------------------------------------- #
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        height, rgb, label = self.raw(index)
        valid = np.ones(height.shape, dtype=bool)
        if self.rotate:
            height, rgb, label, valid = self._rotate_sample(height, rgb, label)

        u, v = float(label["u"]), float(label["v"])
        width_px = float(label["width_px"])

        target = self.input_size
        if target and target != height.shape[0]:
            interp = cv2.INTER_AREA if target < height.shape[0] else cv2.INTER_LINEAR
            scale = target / height.shape[0]
            height = cv2.resize(height, (target, target), interpolation=interp)
            rgb = cv2.resize(rgb, (target, target), interpolation=interp)
            valid = cv2.resize(valid.astype(np.uint8), (target, target),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            u, v, width_px = u * scale, v * scale, width_px * scale

        neg_uv = self._mine_free_negatives(height, valid)

        channels = [normalise_height(height)]
        if self.use_rgb:
            channels.extend(normalise_rgb(rgb))
        image = torch.from_numpy(np.stack(channels, axis=0).astype(np.float32))

        def scalar(x: float) -> torch.Tensor:
            return torch.tensor(float(x), dtype=torch.float32)

        return {
            "image": image,
            "u": scalar(u),
            "v": scalar(v),
            "angle": scalar(label["angle"]),
            "width_px": scalar(width_px),
            "success": scalar(bool(label["success"])),
            "neg_uv": torch.from_numpy(neg_uv),
        }

    # -- reporting -------------------------------------------------------------- #
    def summary(self) -> dict[str, Any]:
        """Counts and positive rates, overall and per category."""
        by_category: dict[str, dict[str, Any]] = {}
        for label in self.labels:
            row = by_category.setdefault(label["category"], {"n": 0, "positives": 0})
            row["n"] += 1
            row["positives"] += int(bool(label["success"]))
        for row in by_category.values():
            row["positive_rate"] = row["positives"] / row["n"] if row["n"] else 0.0

        n = len(self.labels)
        positives = sum(int(bool(lab["success"])) for lab in self.labels)
        return {
            "n": n,
            "positives": positives,
            "positive_rate": positives / n if n else 0.0,
            "by_category": by_category,
            "image_size": self.image_size,
            "input_size": self.input_size,
        }
