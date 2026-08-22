#!/usr/bin/env python3
"""Collect the grasp dataset (Week 7-9 milestone).

Examples
--------
    # quick smoke test
    python scripts/collect_dataset.py --episodes 200 --workers 4 --out data/smoke

    # the real thing
    python scripts/collect_dataset.py --episodes 10000 --workers 6 --out data/grasp10k
"""

from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--out", default="data/grasp10k")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--shard-size", type=int, default=256)
    ap.add_argument("--split", default="all", choices=["all", "seen", "unseen"],
                    help="'seen' collects only training categories; the default 'all' "
                         "collects everything and the loader filters at train time")
    ap.add_argument("--offset", type=int, default=0, help="starting episode index")
    ap.add_argument("--no-rgb", action="store_true", help="store height only (40%% smaller)")
    ap.add_argument("--angles-per-scene", type=int, default=1,
                    help="execute this many grasps at the same point per scene, spread "
                         "over the half turn. >1 gives the network contrastive evidence "
                         "about orientation, which one label per image cannot")
    args = ap.parse_args()

    from simgrasp.collect import collect_dataset

    meta = collect_dataset(
        out_dir=args.out, n_episodes=args.episodes, workers=args.workers,
        base_seed=args.seed, image_size=args.image_size, split=args.split,
        shard_size=args.shard_size, store_rgb=not args.no_rgb,
        angles_per_scene=args.angles_per_scene, episode_offset=args.offset,
    )
    s = meta["stats"]
    print()
    print(f"collected {s['n']} episodes in {meta['elapsed_seconds']:.1f}s "
          f"({meta['episodes_per_second']:.1f} ep/s) -> {args.out}")
    print(f"positive rate {s['positive_rate']:.1%}")
    print()
    print(f"  {'category':<12}{'n':>7}{'positive':>11}")
    for cat, v in sorted(s["by_category"].items()):
        print(f"  {cat:<12}{v['n']:>7}{v['positive_rate']:>10.1%}")
    print()
    print(f"  {'sampler mode':<12}{'n':>7}{'positive':>11}")
    for mode, v in sorted(s["by_mode"].items()):
        print(f"  {mode:<12}{v['n']:>7}{v['positive_rate']:>10.1%}")
    print()
    print(json.dumps({k: meta[k] for k in ("image_size", "table_z")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
