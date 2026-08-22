"""Rotation augmentation must move the label with the image, exactly.

The sign of the angle transform is the whole game. If it is inverted the network
is trained to associate every image with the *perpendicular* grasp, which is
worse than not augmenting at all -- and it fails silently, because the loss still
goes down. So these tests check the transform against measurable image geometry
rather than against the derivation that produced it.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from simgrasp.data import GraspDataset, ShardWriter
from simgrasp.heightmap import estimate_width_px, principal_axis, surface_height
from simgrasp.transforms import wrap_grasp_angle

SIZE = 96


def _bar_dataset(tmp_path, half_w=6, half_h=26):
    """One image containing a bar, long along image y, centred."""
    height = np.zeros((SIZE, SIZE), np.float32)
    c = SIZE // 2
    height[c - half_h:c + half_h, c - half_w:c + half_w] = 0.04
    rgb = np.full((SIZE, SIZE, 3), 90, np.uint8)
    rgb[c - half_h:c + half_h, c - half_w:c + half_w] = (200, 40, 40)

    label = {
        "episode": 0, "category": "box", "split": "seen",
        # The jaws close across the bar's short axis, which is image angle 0.
        "u": float(c), "v": float(c), "angle": 0.0,
        "width_px": float(2 * half_w), "success": True, "sampler_mode": "near_oracle",
    }
    with ShardWriter(tmp_path, image_size=SIZE, shard_size=4, prefix="w00") as w:
        w.add(rgb, height, label)
    (tmp_path / "dataset_meta.json").write_text(json.dumps({"image_size": SIZE}))
    return tmp_path


def test_rotation_preserves_the_measured_grasp_width(tmp_path):
    """The decisive test.

    Measuring the object's extent along the transformed angle must give the same
    answer as before the rotation. With the sign inverted this measures the bar's
    long axis instead and the width jumps by ~4x.
    """
    ds = GraspDataset(_bar_dataset(tmp_path), rotate=True, free_negatives=0)
    height0, rgb0, lab0 = ds.raw(0)
    before = estimate_width_px(height0, lab0["u"], lab0["v"], lab0["angle"])

    for _ in range(40):
        h, _rgb, lab, _valid = ds._rotate_sample(height0, rgb0, lab0)
        after = estimate_width_px(h, lab["u"], lab["v"], lab["angle"])
        assert after == pytest.approx(before, abs=4.0), (
            f"width {before:.1f} px -> {after:.1f} px after rotation; the angle "
            "transform has the wrong sign"
        )


def test_rotated_label_still_lands_on_the_object(tmp_path):
    ds = GraspDataset(_bar_dataset(tmp_path), rotate=True, free_negatives=0)
    height0, rgb0, lab0 = ds.raw(0)
    for _ in range(40):
        h, _rgb, lab, _valid = ds._rotate_sample(height0, rgb0, lab0)
        assert surface_height(h, lab["u"], lab["v"]) > 0.02, "label fell off the object"


def test_rotated_angle_tracks_the_principal_axis(tmp_path):
    """The grasp angle must stay perpendicular to the bar's long axis."""
    ds = GraspDataset(_bar_dataset(tmp_path), rotate=True, free_negatives=0)
    height0, rgb0, lab0 = ds.raw(0)
    for _ in range(30):
        h, _rgb, lab, _valid = ds._rotate_sample(height0, rgb0, lab0)
        _centroid, major, _minor = principal_axis(h > 0.005)
        # Grasp angle and major axis must differ by ~90 degrees, modulo 180.
        delta = abs(float(wrap_grasp_angle(lab["angle"] - major)))
        assert delta > np.deg2rad(75), (
            f"grasp angle is {np.degrees(delta):.1f} deg from the long axis; "
            "expected ~90"
        )


def test_rotated_angle_stays_wrapped(tmp_path):
    ds = GraspDataset(_bar_dataset(tmp_path), rotate=True, free_negatives=0)
    height0, rgb0, lab0 = ds.raw(0)
    for _ in range(50):
        _h, _rgb, lab, _valid = ds._rotate_sample(height0, rgb0, lab0)
        assert -np.pi / 2 - 1e-9 <= lab["angle"] < np.pi / 2 + 1e-9


def test_free_negatives_are_not_mined_from_rotation_padding(tmp_path):
    """Padding is not observed table and must never become a training label."""
    ds = GraspDataset(_bar_dataset(tmp_path), rotate=True, free_negatives=24,
                      free_negative_margin_px=3)
    height0, rgb0, lab0 = ds.raw(0)
    for _ in range(20):
        h, _rgb, _lab, valid = ds._rotate_sample(height0, rgb0, lab0)
        neg = ds._mine_free_negatives(h, valid)
        for u, v in neg:
            if u < 0:
                continue
            assert valid[int(v), int(u)], "a free negative came from rotation padding"


def test_rotation_is_off_by_default(tmp_path):
    ds = GraspDataset(_bar_dataset(tmp_path), free_negatives=0)
    assert not ds.rotate
    a = ds[0]["image"].numpy()
    b = ds[0]["image"].numpy()
    assert np.array_equal(a, b), "without rotation, samples must be deterministic"


def test_rotation_actually_varies_the_sample(tmp_path):
    ds = GraspDataset(_bar_dataset(tmp_path), rotate=True, free_negatives=0)
    angles = {round(float(ds[0]["angle"]), 4) for _ in range(15)}
    assert len(angles) > 5, "rotation augmentation is not varying the angle"


def test_border_fill_matches_the_table_colour(tmp_path):
    """A black fill would be a hard artificial edge for the network to latch onto."""
    ds = GraspDataset(_bar_dataset(tmp_path), rotate=True, free_negatives=0)
    height0, rgb0, lab0 = ds.raw(0)
    _h, rgb, _lab, valid = ds._rotate_sample(height0, rgb0, lab0)
    filled = rgb[~valid]
    if filled.size:
        table = np.median(rgb0.reshape(-1, 3), axis=0)
        assert np.abs(filled.mean(axis=0) - table).max() < 40
