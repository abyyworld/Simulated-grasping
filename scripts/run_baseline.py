#!/usr/bin/env python3
"""Week 5-6 milestone: measure scripted-grasp success over N trials.

    python scripts/run_baseline.py --episodes 200 --workers 6

Runs the ground-truth oracle and the depth-only heuristic over identical scenes
and writes results/JSON+CSV so the numbers can be regenerated and diffed.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--policies", nargs="+", default=["oracle", "heuristic"])
    ap.add_argument("--checkpoint", default="runs/grasp_cnn/best.pt",
                    help="used when 'cnn' is among --policies")
    ap.add_argument("--out", default="results/baseline")
    args = ap.parse_args()

    from pathlib import Path

    from simgrasp.evaluation import (
        evaluate_policy,
        format_summary,
        save_records_csv,
        save_summary,
    )

    out = Path(args.out)
    for name in args.policies:
        kwargs = {"checkpoint": args.checkpoint} if name == "cnn" else None
        print(f"\n=== {name} ===", flush=True)
        summary = evaluate_policy(name, n_episodes=args.episodes, workers=args.workers,
                                  base_seed=args.seed, episode_offset=args.offset,
                                  policy_kwargs=kwargs)
        print(format_summary(summary))
        save_summary(summary, out / f"{name}.json", keep_records=False)
        save_records_csv(summary["records"], out / f"{name}_trials.csv")
    print(f"\nwrote {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
