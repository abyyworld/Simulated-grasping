from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from simgrasp.data import GraspDataset, ShardWriter
from simgrasp.data.writer import HEIGHT_SCALE, decode_height, encode_height


@dataclass
class _Label:
    episode: int
    category: str
    split: str
    u: float
    v: float
    angle: float
    width_px: float
    success: bool
    sampler_mode: str


def _fake_dataset(tmp_path, n=40, size=32, shard=16):
    rng = np.random.default_rng(0)
    with ShardWriter(tmp_path, image_size=size, shard_size=shard, prefix="w00") as w:
        for i in range(n):
            height = np.zeros((size, size), np.float32)
            height[8:20, 10:18] = 0.03  # a bar to give free negatives something to avoid
            rgb = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            w.add(rgb, height, _Label(
                episode=i, category="box" if i % 2 else "l_shape",
                split="seen" if i % 2 else "unseen",
                u=float(rng.uniform(0, size)), v=float(rng.uniform(0, size)),
                angle=float(rng.uniform(-1.5, 1.5)), width_px=float(rng.uniform(2, 20)),
                success=bool(i % 3), sampler_mode="on_object"))
    (tmp_path / "dataset_meta.json").write_text(json.dumps({"image_size": size}))
    return tmp_path


def test_height_encoding_is_lossless_to_a_tenth_of_a_millimetre():
    h = np.array([[-0.002, 0.0, 0.0123, 0.09999, 0.5]], np.float32)
    back = decode_height(encode_height(h))
    assert np.allclose(back, np.clip(h, 0, None), atol=1.0 / HEIGHT_SCALE)
    assert encode_height(h).dtype == np.uint16


def test_negative_heights_clip_to_the_table():
    assert encode_height(np.array([-1.0]))[0] == 0


def test_writer_shards_and_dataset_reads_them_back(tmp_path):
    root = _fake_dataset(tmp_path, n=40, shard=16)
    files = sorted(p.name for p in root.glob("w00_*_height.npy"))
    assert len(files) == 3, "40 episodes at 16 per shard is 3 shards"
    ds = GraspDataset(root, free_negatives=0)
    assert len(ds) == 40


def test_dataset_meta_is_not_mistaken_for_a_shard(tmp_path):
    """dataset_meta.json also matches '*_meta.json' and must be excluded."""
    root = _fake_dataset(tmp_path, n=8, shard=8)
    assert len(GraspDataset(root, free_negatives=0)) == 8


def test_split_and_category_filters(tmp_path):
    root = _fake_dataset(tmp_path, n=40, shard=16)
    assert len(GraspDataset(root, split="seen", free_negatives=0)) == 20
    assert len(GraspDataset(root, split="unseen", free_negatives=0)) == 20
    assert len(GraspDataset(root, categories=["box"], free_negatives=0)) == 20
    assert all(lab["success"] for lab in
               GraspDataset(root, successes_only=True, free_negatives=0).labels)


def test_episode_filter_keeps_train_and_val_disjoint(tmp_path):
    root = _fake_dataset(tmp_path, n=40, shard=16)
    a = GraspDataset(root, episode_filter=list(range(0, 20)), free_negatives=0)
    b = GraspDataset(root, episode_filter=list(range(20, 40)), free_negatives=0)
    assert len(a) == 20 and len(b) == 20
    assert not ({lab["episode"] for lab in a.labels} & {lab["episode"] for lab in b.labels})


def test_getitem_returns_a_batchable_tensor(tmp_path):
    root = _fake_dataset(tmp_path, n=8, shard=8)
    item = GraspDataset(root, free_negatives=4)[0]
    assert item["image"].shape == (4, 32, 32)
    assert item["image"].dtype == torch.float32
    assert item["neg_uv"].shape == (4, 2)
    for key in ("u", "v", "angle", "width_px", "success"):
        assert item[key].dtype == torch.float32


def test_height_only_mode(tmp_path):
    root = _fake_dataset(tmp_path, n=8, shard=8)
    assert GraspDataset(root, use_rgb=False, free_negatives=0)[0]["image"].shape == (1, 32, 32)


def test_free_negatives_avoid_the_object(tmp_path):
    """Mined negatives must be clear of the object by the margin, or they are wrong."""
    root = _fake_dataset(tmp_path, n=8, shard=8)
    ds = GraspDataset(root, free_negatives=16, free_negative_margin_px=5)
    for i in range(len(ds)):
        height, _rgb, _lab = ds.raw(i)
        for u, v in ds[i]["neg_uv"].numpy():
            if u < 0:
                continue
            assert height[int(v), int(u)] <= 0.005


def test_free_negatives_degrade_gracefully_when_none_exist(tmp_path):
    root = _fake_dataset(tmp_path, n=4, shard=4)
    ds = GraspDataset(root, free_negatives=8, free_negative_margin_px=10_000)
    assert (ds[0]["neg_uv"][:, 0] < 0).all(), "impossible margin must yield -1 sentinels"


def test_dataset_is_streamed_not_loaded(tmp_path):
    """The shards must be memory-mapped: this is the 16 GB-RAM constraint."""
    root = _fake_dataset(tmp_path, n=32, shard=16)
    ds = GraspDataset(root, free_negatives=0)
    ds.raw(0)
    handle = next(iter(ds._height_maps.values()))
    assert isinstance(handle, np.memmap), f"expected a memmap, got {type(handle)}"


def test_summary_counts_positives(tmp_path):
    root = _fake_dataset(tmp_path, n=30, shard=10)
    s = GraspDataset(root, free_negatives=0).summary()
    assert s["n"] == 30
    assert 0 <= s["positive_rate"] <= 1
    assert set(s["by_category"]) == {"box", "l_shape"}
