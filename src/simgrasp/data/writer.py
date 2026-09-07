"""Sharded, memory-mappable dataset writer.

Storage format
--------------
A dataset directory holds a set of shards, each written by one worker::

    w00_00000_height.npy   (N, S, S) uint16   height above the table
    w00_00000_rgb.npy      (N, S, S, 3) uint8 colour, optional
    w00_00000_meta.json    N label records
    dataset_meta.json      one per dataset, written by the collector

Three decisions worth stating:

**Plain ``.npy``, one array per shard.** ``np.load(..., mmap_mode="r")`` gives a
memory map straight out of the file, so a training run touches only the samples
in the current batch. The dataset is larger than RAM on the hardware this was
built for, and a format that has to be decoded per sample (HDF5, a tar of PNGs)
would either lose that or cost a decode on every access.

**Height stored as uint16 in tenths of a millimetre.** Heights are physically
bounded (0 to ~6.5 m at this scale) and the camera's own resolution is far
coarser than 0.1 mm, so float32 spends two bytes per pixel carrying noise.
Halving the height channel halves the dataset.

**Labels in JSON next to the arrays, not inside them.** Labels are read in full
at load time to build the index; the images are not. Keeping them apart means
constructing a ``GraspDataset`` costs a few JSON reads rather than opening every
array.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Heights are stored as uint16 counts of 1/HEIGHT_SCALE metres, i.e. 0.1 mm.
# The camera's ground sampling distance is 2.2 mm at 224 px, so this quantisation
# is two orders of magnitude finer than the measurement it stores.
HEIGHT_SCALE = 10_000.0
# uint16 tops out at 65535 counts = 6.5535 m above the table.
MAX_HEIGHT = 65_535 / HEIGHT_SCALE


def encode_height(height: np.ndarray) -> np.ndarray:
    """Height in metres -> uint16 counts of 0.1 mm.

    Negative heights are clipped to the table. They occur only as depth noise a
    fraction of a millimetre below the plane, and "below the table" is not a
    state the height map is meant to represent.
    """
    clipped = np.clip(np.asarray(height, dtype=np.float64), 0.0, MAX_HEIGHT)
    return np.round(clipped * HEIGHT_SCALE).astype(np.uint16)


def decode_height(encoded: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_height`."""
    return (np.asarray(encoded, dtype=np.float32) / HEIGHT_SCALE).astype(np.float32)


def _as_dict(label: Any) -> dict:
    if is_dataclass(label) and not isinstance(label, type):
        return asdict(label)
    if isinstance(label, dict):
        return dict(label)
    raise TypeError(f"label must be a dataclass or dict, got {type(label).__name__}")


class ShardWriter:
    """Buffer samples in memory and flush a shard once ``shard_size`` is reached.

    Used as a context manager so the trailing partial shard is always written::

        with ShardWriter(out, image_size=224, prefix="w00") as w:
            w.add(rgb, height, label)

    ``prefix`` separates workers: each collector process owns its own shard
    series and they never contend for a file.
    """

    def __init__(self, out_dir: Path | str, image_size: int, shard_size: int = 256,
                 prefix: str = "w00", store_rgb: bool = True):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.image_size = int(image_size)
        self.shard_size = int(shard_size)
        self.prefix = str(prefix)
        self.store_rgb = bool(store_rgb)

        self._heights: list[np.ndarray] = []
        self._rgbs: list[np.ndarray] = []
        self._labels: list[dict] = []
        self._shard_index = 0
        self.n_written = 0

    # -- writing -------------------------------------------------------------- #
    def add(self, rgb: np.ndarray, height: np.ndarray, label: Any) -> None:
        """Append one sample. Shape is validated here, not at read time."""
        height = np.asarray(height)
        if height.shape != (self.image_size, self.image_size):
            raise ValueError(
                f"height map is {height.shape}, expected "
                f"({self.image_size}, {self.image_size})")
        self._heights.append(encode_height(height))

        if self.store_rgb:
            rgb = np.asarray(rgb, dtype=np.uint8)
            if rgb.shape != (self.image_size, self.image_size, 3):
                raise ValueError(
                    f"rgb is {rgb.shape}, expected "
                    f"({self.image_size}, {self.image_size}, 3)")
            self._rgbs.append(rgb)

        self._labels.append(_as_dict(label))
        if len(self._labels) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        """Write the buffered samples as one shard. A no-op when empty."""
        if not self._labels:
            return
        stem = f"{self.prefix}_{self._shard_index:05d}"
        np.save(self.out_dir / f"{stem}_height.npy", np.stack(self._heights))
        if self.store_rgb:
            np.save(self.out_dir / f"{stem}_rgb.npy", np.stack(self._rgbs))
        (self.out_dir / f"{stem}_meta.json").write_text(
            json.dumps({"shard": stem, "image_size": self.image_size,
                        "store_rgb": self.store_rgb, "labels": self._labels},
                       default=float))

        self.n_written += len(self._labels)
        self._shard_index += 1
        self._heights.clear()
        self._rgbs.clear()
        self._labels.clear()

    # -- lifecycle ------------------------------------------------------------ #
    def close(self) -> None:
        self.flush()

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
