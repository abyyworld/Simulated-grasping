"""Training loop for :class:`~simgrasp.models.GraspNet`.

Supervision has two sources:

* **Executed labels** -- one per episode: the sampled grasp was run in the
  simulator and either lifted the object or did not. Applied at the exact
  sub-pixel location and the angle bin that was tried.
* **Free negatives** -- table pixels far from any object, mined from the height
  map at load time (see ``GraspDataset._mine_free_negatives``). These are
  certain failures at every angle and cost nothing to label, and they are what
  stops the network from predicting high quality everywhere: without them the
  entire gradient signal is one cell out of 224*224*12 per image.

Free negatives are down-weighted because they are both plentiful and trivial;
the executed labels carry the information that actually distinguishes a good
grasp from a bad one.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import GraspDataset
from .models import ANGLE_BINS, GraspNet, angle_to_bin
from .models.device import autocast_dtype, pick_device
from .models.grasp_net import sample_at


@dataclass
class TrainConfig:
    data: str = "data/grasp10k"
    out: str = "runs/grasp_cnn"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 4
    use_rgb: bool = True
    pretrained: bool = False
    angle_bins: int = ANGLE_BINS
    free_negatives: int = 32
    free_negative_weight: float = 0.2
    width_weight: float = 0.5
    val_fraction: float = 0.1
    train_split: str = "seen"
    seed: int = 0
    device: str | None = None
    amp: bool = True
    limit: int | None = None
    extra: dict = field(default_factory=dict)


def split_episodes(dataset_root: str | Path, split: str, val_fraction: float,
                   seed: int) -> tuple[list[int], list[int]]:
    """Partition episode indices into train/val.

    Split by *episode*, not by sample: one episode is one image, so splitting any
    other way would put the same image in both sets.
    """
    probe = GraspDataset(dataset_root, split=split, free_negatives=0)
    episodes = sorted({lab["episode"] for lab in probe.labels})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    n_val = max(1, int(round(len(episodes) * val_fraction)))
    val = {episodes[i] for i in order[:n_val]}
    train = [e for e in episodes if e not in val]
    return train, sorted(val)


def build_loaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, GraspDataset]:
    train_eps, val_eps = split_episodes(cfg.data, cfg.train_split, cfg.val_fraction, cfg.seed)
    if cfg.limit:
        train_eps = train_eps[: cfg.limit]

    train_ds = GraspDataset(cfg.data, split=cfg.train_split, use_rgb=cfg.use_rgb,
                            episode_filter=train_eps, free_negatives=cfg.free_negatives)
    val_ds = GraspDataset(cfg.data, split=cfg.train_split, use_rgb=cfg.use_rgb,
                          episode_filter=val_eps, free_negatives=cfg.free_negatives)

    common = dict(batch_size=cfg.batch_size, num_workers=cfg.workers,
                  pin_memory=False, persistent_workers=cfg.workers > 0)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > cfg.batch_size,
                              **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, train_ds


def compute_losses(model: nn.Module, batch: dict, cfg: TrainConfig,
                   device: torch.device) -> tuple[torch.Tensor, dict[str, float], dict[str, np.ndarray]]:
    image = batch["image"].to(device, non_blocking=True)
    u = batch["u"].to(device)
    v = batch["v"].to(device)
    success = batch["success"].to(device)
    width_px = batch["width_px"].to(device)
    bins = angle_to_bin(batch["angle"]).to(device)

    out = model(image)
    quality, width = out["quality"], out["width"]
    b = image.shape[0]
    idx = torch.arange(b, device=device)

    # --- executed labels ---------------------------------------------------- #
    q_at = sample_at(quality, u, v)                     # (B, A)
    logit = q_at[idx, bins]
    loss_exec = F.binary_cross_entropy_with_logits(logit, success)

    # --- free negatives ----------------------------------------------------- #
    neg = batch["neg_uv"].to(device)                    # (B, K, 2)
    valid = neg[..., 0] >= 0
    if valid.any():
        nu = neg[..., 0].clamp(min=0)
        nv = neg[..., 1].clamp(min=0)
        q_neg = sample_at(quality, nu, nv)              # (B, K, A)
        target = torch.zeros_like(q_neg)
        per = F.binary_cross_entropy_with_logits(q_neg, target, reduction="none")
        mask = valid.unsqueeze(-1).float()
        loss_neg = (per * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        loss_neg = torch.zeros((), device=device)

    # --- width regression, positives only ------------------------------------ #
    w_at = sample_at(width, u, v)[idx, bins]
    target_w = width_px / image.shape[-1]
    n_pos = success.sum()
    if n_pos > 0:
        loss_width = (F.smooth_l1_loss(w_at, target_w, reduction="none", beta=0.02)
                      * success).sum() / n_pos
    else:
        loss_width = torch.zeros((), device=device)

    total = loss_exec + cfg.free_negative_weight * loss_neg + cfg.width_weight * loss_width
    logs = {"loss": float(total.detach()), "loss_exec": float(loss_exec.detach()),
            "loss_neg": float(loss_neg.detach()), "loss_width": float(loss_width.detach())}
    preds = {"prob": torch.sigmoid(logit).detach().float().cpu().numpy(),
             "target": success.detach().cpu().numpy()}
    return total, logs, preds


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (the 'AP' of the PR curve)."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return float((precision * labels).sum() / labels.sum())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig,
             device: torch.device) -> dict[str, float]:
    model.eval()
    agg: dict[str, list[float]] = {}
    probs, targets = [], []
    for batch in loader:
        _, logs, preds = compute_losses(model, batch, cfg, device)
        for k, val in logs.items():
            agg.setdefault(k, []).append(val)
        probs.append(preds["prob"])
        targets.append(preds["target"])
    model.train()
    if not probs:
        return {}
    p = np.concatenate(probs)
    t = np.concatenate(targets)
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    out["accuracy"] = float(((p > 0.5) == (t > 0.5)).mean())
    out["ap"] = average_precision(p, t)
    out["positive_rate"] = float(t.mean())
    return out


def train(cfg: TrainConfig) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, train_ds = build_loaders(cfg)
    in_channels = 4 if cfg.use_rgb else 1
    model = GraspNet(in_channels=in_channels, angle_bins=cfg.angle_bins,
                     pretrained=cfg.pretrained).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=max(1, cfg.epochs * max(1, len(train_loader))),
        pct_start=0.25)
    amp_dtype = autocast_dtype(device) if cfg.amp else None
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)

    print(f"device={device}  train={len(train_ds)} samples / {len(train_loader)} batches  "
          f"val={len(val_loader.dataset)}  params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    history: list[dict[str, Any]] = []
    best = {"ap": -np.inf, "epoch": -1}
    t_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        t0 = time.perf_counter()
        agg: dict[str, list[float]] = {}
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device.type, dtype=amp_dtype):
                    loss, logs, _ = compute_losses(model, batch, cfg, device)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss, logs, _ = compute_losses(model, batch, cfg, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
            for k, val in logs.items():
                agg.setdefault(k, []).append(val)

        train_logs = {f"train_{k}": float(np.mean(v)) for k, v in agg.items()}
        val_logs = {f"val_{k}": v for k, v in evaluate(model, val_loader, cfg, device).items()}
        row = {"epoch": epoch, "lr": sched.get_last_lr()[0],
               "seconds": time.perf_counter() - t0, **train_logs, **val_logs}
        history.append(row)
        print(f"epoch {epoch:>3}  train_loss {row['train_loss']:.4f}  "
              f"val_loss {row.get('val_loss', float('nan')):.4f}  "
              f"val_acc {row.get('val_accuracy', float('nan')):.3f}  "
              f"val_AP {row.get('val_ap', float('nan')):.3f}  ({row['seconds']:.1f}s)", flush=True)

        ap = row.get("val_ap", float("nan"))
        if not np.isnan(ap) and ap > best["ap"]:
            best = {"ap": float(ap), "epoch": epoch}
            save_checkpoint(out_dir / "best.pt", model, cfg, row)
        save_checkpoint(out_dir / "last.pt", model, cfg, row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=float))

    summary = {"config": asdict(cfg), "best": best, "history": history,
               "elapsed_seconds": time.perf_counter() - t_start,
               "device": str(device), "n_train": len(train_ds)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def save_checkpoint(path: Path, model: nn.Module, cfg: TrainConfig, row: dict) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "row": row}, path)


def load_checkpoint(path: str | Path, device: torch.device | None = None) -> tuple[GraspNet, TrainConfig]:
    device = device or pick_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = GraspNet(in_channels=4 if cfg.use_rgb else 1, angle_bins=cfg.angle_bins)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg
