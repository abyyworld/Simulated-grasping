#!/usr/bin/env python3
"""Train the grasp-prediction network (Week 7-9 milestone).

    python scripts/train.py --data data/grasp10k --out runs/grasp_cnn --epochs 30

Trains on the *seen* categories only. The held-out categories are never shown to
the network and exist solely for the generalisation evaluation in evaluate.py.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/grasp10k")
    ap.add_argument("--out", default="runs/grasp_cnn")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-rgb", action="store_true", help="train on the height channel only")
    ap.add_argument("--pretrained", action="store_true",
                    help="ImageNet-initialised ResNet-18 (downloads ~45 MB once)")
    ap.add_argument("--free-negatives", type=int, default=32)
    ap.add_argument("--free-negative-weight", type=float, default=0.2)
    ap.add_argument("--width-weight", type=float, default=0.5)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--train-split", default="seen", choices=["seen", "unseen", None])
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap training episodes (for smoke tests)")
    ap.add_argument("--input-size", type=int, default=None,
                    help="resize inputs to NxN (labels are scaled too). Halving the "
                         "resolution is ~4x cheaper per step; useful without a GPU")
    args = ap.parse_args()

    from simgrasp.training import TrainConfig, train

    cfg = TrainConfig(
        data=args.data, out=args.out, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, workers=args.workers, use_rgb=not args.no_rgb, pretrained=args.pretrained,
        free_negatives=args.free_negatives, free_negative_weight=args.free_negative_weight,
        width_weight=args.width_weight, val_fraction=args.val_fraction,
        train_split=args.train_split, device=args.device, amp=not args.no_amp,
        seed=args.seed, limit=args.limit, input_size=args.input_size,
    )
    summary = train(cfg)
    print()
    print(f"best val AP {summary['best']['ap']:.4f} at epoch {summary['best']['epoch']}")
    print(f"checkpoints in {args.out}/  ({summary['elapsed_seconds'] / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
