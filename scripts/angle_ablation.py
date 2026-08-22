#!/usr/bin/env python3
"""Measure how well a checkpoint predicts grasp *orientation*.

Validation AP mostly measures *where* to grasp, and stays high even when the
angle head has collapsed -- which is exactly what happened without rotation
augmentation. This script isolates orientation with two numbers per model:

``angle_error_deg``
    Mean absolute difference between the predicted grasp angle and the oracle's,
    over elongated objects only (a cylinder or sphere has no meaningful angle).
    Wrapped to a half turn, so **random guessing scores 45 degrees**.

``bin_spread``
    Range of predicted quality across the twelve angle bins *at the pixel the
    model chose*. A collapsed angle head gives ~0: it predicts the same success
    probability whichever way the gripper is turned.

    python scripts/angle_ablation.py --checkpoints runs/grasp_cnn_norot/best.pt runs/grasp_cnn/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

# An angle error is only meaningful when the correct angle is *determinate*.
#
# A cylinder or a sphere grasps equally well at every angle, so scoring a
# prediction against "the" oracle angle measures nothing. A nearly square box is
# the same problem in weaker form: when its two horizontal half-extents are
# within a few percent, either axis is a fine grasp and the oracle's choice is
# arbitrary. Averaging those in swamps the signal -- an earlier version of this
# script did exactly that and reported no effect where there was a large one.
#
# So objects are filtered by *shape*, per episode, not by category: keep only
# those whose horizontal extents differ by at least this ratio.
ASPECT_THRESHOLD = 1.3

ALL_CATEGORIES = ("box", "cylinder", "capsule", "sphere",
                  "ellipsoid", "l_shape", "t_shape", "mug", "dumbbell")


def angle_is_determinate(spec) -> bool:
    """True when the object has a clearly narrower horizontal axis to grasp across."""
    geoms = spec.geoms
    if len(geoms) > 1:
        return True  # L, T, mug, dumbbell: multi-part, orientation always matters
    g = geoms[0]
    if g.type in ("sphere", "cylinder"):
        return False  # rotationally symmetric about the vertical
    if g.type == "capsule":
        return True  # lying down: the shaft defines the grasp axis
    hx, hy = g.size[0], g.size[1]
    lo, hi = min(hx, hy), max(hx, hy)
    return lo > 0 and hi / lo >= ASPECT_THRESHOLD


def measure(checkpoint: str, episodes: int, offset: int) -> dict:
    from simgrasp.env import PandaGraspEnv
    from simgrasp.objects import SEEN_CATEGORIES
    from simgrasp.policies.learned import LearnedPolicy
    from simgrasp.transforms import wrap_grasp_angle

    policy = LearnedPolicy(checkpoint=checkpoint, device="cpu")
    errors: dict[str, list[float]] = {}
    spreads: dict[str, list[float]] = {}
    skipped = 0

    with PandaGraspEnv(base_seed=0) as env:
        for cat in ALL_CATEGORIES:
            for k in range(episodes):
                obs = env.reset(offset + k, category=cat)
                if not angle_is_determinate(env.state.spec):
                    skipped += 1
                    continue
                quality, _width, _scale = policy.predict_maps(obs)
                mask = policy._workspace_mask(obs, quality.shape[1:])
                masked = np.where(mask[None], quality, -1.0)
                b, v, u = np.unravel_index(int(np.argmax(masked)), masked.shape)
                column = masked[:, v, u]
                spreads.setdefault(cat, []).append(float(column.max() - column.min()))

                rng = np.random.default_rng(0)
                predicted = policy(obs, env, rng).yaw
                oracle = env.oracle_grasp().yaw
                delta = abs(float(wrap_grasp_angle(predicted - oracle)))
                errors.setdefault(cat, []).append(np.degrees(delta))

    def mean_over(cats) -> tuple[float, float, int]:
        e = [x for c in cats for x in errors.get(c, [])]
        s = [x for c in cats for x in spreads.get(c, [])]
        return (float(np.mean(e)) if e else float("nan"),
                float(np.mean(s)) if s else float("nan"), len(e))

    seen = [c for c in ALL_CATEGORIES if c in SEEN_CATEGORIES]
    held = [c for c in ALL_CATEGORIES if c not in SEEN_CATEGORIES]
    err_all, spread_all, n_all = mean_over(ALL_CATEGORIES)
    err_seen, spread_seen, n_seen = mean_over(seen)
    err_held, spread_held, n_held = mean_over(held)
    return {
        "checkpoint": checkpoint,
        "angle_error_deg": err_all,
        "angle_error_deg_seen": err_seen,
        "angle_error_deg_heldout": err_held,
        "bin_spread": spread_all,
        "n_scored": n_all, "n_seen": n_seen, "n_heldout": n_held,
        "n_skipped_indeterminate": skipped,
        "per_category_error_deg": {c: float(np.mean(v)) for c, v in errors.items()},
        "per_category_bin_spread": {c: float(np.mean(v)) for c, v in spreads.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+",
                    default=["runs/grasp_cnn_norot/best.pt", "runs/grasp_cnn/best.pt"],
                    help="no-rotation checkpoint first, rotation second")
    ap.add_argument("--episodes", type=int, default=10, help="episodes per category")
    ap.add_argument("--offset", type=int, default=2_000_000)
    ap.add_argument("--out", default="results/angle_ablation.json")
    args = ap.parse_args()

    results = []
    for ck in args.checkpoints:
        if not Path(ck).exists():
            print(f"skipping missing checkpoint {ck}")
            continue
        print(f"measuring {ck} ...", flush=True)
        results.append(measure(ck, args.episodes, args.offset))

    print()
    print(f"{'checkpoint':<32}{'all':>10}{'seen':>10}{'held-out':>11}{'n':>6}")
    print("-" * 69)
    for r in results:
        print(f"{r['checkpoint']:<32}{r['angle_error_deg']:>7.1f} deg"
              f"{r['angle_error_deg_seen']:>7.1f} deg{r['angle_error_deg_heldout']:>8.1f} deg"
              f"{r['n_scored']:>6}")
    print("-" * 69)
    print("mean |grasp angle - oracle angle|, degrees; random guessing scores 45")
    if results:
        print(f"scored only shapes with a determinate grasp axis "
              f"({results[0]['n_skipped_indeterminate']} symmetric/near-square episodes skipped)")

    payload: dict = {"runs": results}
    if len(results) == 2:
        # Keys the report generator consumes.
        payload.update({
            "norot_deg": results[0]["angle_error_deg"],
            "rot_deg": results[1]["angle_error_deg"],
            "norot_spread": results[0]["bin_spread"],
            "rot_spread": results[1]["bin_spread"],
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
