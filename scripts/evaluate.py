#!/usr/bin/env python3
"""Week 10-11 milestone: evaluate policies on seen vs held-out object categories.

    python scripts/evaluate.py --checkpoint runs/grasp_cnn/best.pt --episodes 200

Runs each policy separately on the seen and the held-out category pools, over
identical scenes, and prints the generalisation gap. Episode indices start well
past the training range so no evaluated scene was ever collected into the
training set.
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200, help="episodes per split")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=None,
                    help="starting episode index (default: the held-out evaluation range)")
    ap.add_argument("--policies", nargs="+", default=["oracle", "heuristic", "cnn"])
    ap.add_argument("--checkpoint", default="runs/grasp_cnn/best.pt")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()

    from pathlib import Path

    from simgrasp.evaluation import (
        EVAL_EPISODE_OFFSET,
        evaluate_policy,
        format_summary,
        save_summary,
    )

    offset = EVAL_EPISODE_OFFSET if args.offset is None else args.offset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table: dict[str, dict[str, float]] = {}

    for name in args.policies:
        kwargs = {"checkpoint": args.checkpoint} if name == "cnn" else None
        table[name] = {}
        for split in ("seen", "unseen"):
            print(f"\n=== {name} / {split} ===", flush=True)
            summary = evaluate_policy(name, n_episodes=args.episodes, split=split,
                                      workers=args.workers, base_seed=args.seed,
                                      episode_offset=offset, policy_kwargs=kwargs)
            print(format_summary(summary))
            save_summary(summary, out / f"{name}_{split}.json", keep_records=False)
            table[name][split] = summary["success_rate"]
            table[name][f"{split}_ci"] = summary["ci95"]

    print("\n\n" + "=" * 66)
    print(f"{'policy':<14}{'seen':>12}{'held-out':>12}{'gap':>10}")
    print("-" * 66)
    for name, row in table.items():
        gap = row["seen"] - row["unseen"]
        print(f"{name:<14}{row['seen']:>11.1%}{row['unseen']:>12.1%}{gap:>10.1f}pp")
    print("=" * 66)
    print(f"n = {args.episodes} episodes per cell")

    (out / "summary_table.json").write_text(json.dumps(table, indent=2, default=float))
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
