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

# Rotationally symmetric objects are excluded: every angle is equally correct for
# them, so an "error" against the oracle angle is meaningless.
ELONGATED = ("capsule", "box", "l_shape", "t_shape", "dumbbell")
SYMMETRIC = ("cylinder", "sphere", "ellipsoid", "mug")


def measure(checkpoint: str, episodes: int, offset: int) -> dict:
    from simgrasp.env import PandaGraspEnv
    from simgrasp.policies.learned import LearnedPolicy
    from simgrasp.transforms import wrap_grasp_angle

    policy = LearnedPolicy(checkpoint=checkpoint, device="cpu")
    errors: dict[str, list[float]] = {}
    spreads: dict[str, list[float]] = {}

    with PandaGraspEnv(base_seed=0) as env:
        for cat in ELONGATED + SYMMETRIC:
            for k in range(episodes):
                obs = env.reset(offset + k, category=cat)
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

    def mean_over(cats) -> tuple[float, float]:
        e = [x for c in cats for x in errors.get(c, [])]
        s = [x for c in cats for x in spreads.get(c, [])]
        return float(np.mean(e)) if e else float("nan"), float(np.mean(s)) if s else float("nan")

    err_elong, spread_elong = mean_over(ELONGATED)
    err_sym, spread_sym = mean_over(SYMMETRIC)
    return {
        "checkpoint": checkpoint,
        "angle_error_deg_elongated": err_elong,
        "angle_error_deg_symmetric": err_sym,
        "bin_spread_elongated": spread_elong,
        "bin_spread_symmetric": spread_sym,
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
    print(f"{'checkpoint':<34}{'angle err (elong)':>19}{'bin spread':>13}")
    print("-" * 66)
    for r in results:
        print(f"{r['checkpoint']:<34}{r['angle_error_deg_elongated']:>16.1f} deg"
              f"{r['bin_spread_elongated']:>13.3f}")
    print("-" * 66)
    print("random guessing over a half-turn-symmetric grasp scores 45 deg")

    payload: dict = {"runs": results}
    if len(results) == 2:
        # Keys the report generator consumes.
        payload.update({
            "norot_deg": results[0]["angle_error_deg_elongated"],
            "rot_deg": results[1]["angle_error_deg_elongated"],
            "norot_spread": results[0]["bin_spread_elongated"],
            "rot_spread": results[1]["bin_spread_elongated"],
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
