"""Run a policy for N episodes and report success rates, optionally in parallel.

Everything here is deterministic given ``base_seed``: episode *i* always spawns
the same object in the same pose, whichever worker executes it and however many
workers there are. That is what makes two policies comparable -- the oracle and
the network see literally the same 500 scenes, so a difference in success rate
is a difference in the policies and not in the luck of the draw.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .env import PandaGraspEnv
from .objects import SEEN_CATEGORIES, UNSEEN_CATEGORIES
from .seeding import rng_for_episode


@dataclass
class TrialRecord:
    episode: int
    category: str
    split: str
    policy: str
    success: bool
    reason: str
    lift_height: float
    grasp_x: float
    grasp_y: float
    grasp_z: float
    grasp_yaw: float
    grasp_width: float
    final_gripper_width: float
    object_displacement: float
    sampler_mode: str = ""
    settle_displacement: float = 0.0


def build_policy(name: str, **kwargs):
    """Construct a policy by name. Used so workers can be spawned picklably."""
    from .policies import GraspSampler, HeuristicPolicy, OraclePolicy

    if name == "oracle":
        return OraclePolicy()
    if name == "heuristic":
        return HeuristicPolicy(**kwargs)
    if name == "sampler":
        return GraspSampler(**kwargs)
    if name == "cnn":
        from .policies.learned import LearnedPolicy

        return LearnedPolicy(**kwargs)
    raise KeyError(f"unknown policy {name!r}")


def run_episodes(
    policy_name: str,
    episodes: Sequence[int],
    split: str = "all",
    base_seed: int = 0,
    image_size: int = 224,
    policy_kwargs: dict[str, Any] | None = None,
    progress: bool = False,
) -> list[TrialRecord]:
    """Execute ``episodes`` in this process and return one record per episode."""
    policy = build_policy(policy_name, **(policy_kwargs or {}))
    records: list[TrialRecord] = []
    with PandaGraspEnv(image_size=image_size, base_seed=base_seed) as env:
        for n, ep in enumerate(episodes):
            obs = env.reset(ep, split=split)
            # A separate stream from the scene's, so changing the policy never
            # changes which object is spawned.
            rng = rng_for_episode(base_seed + 1_000_003, ep)
            grasp = policy(obs, env, rng)
            result = env.execute(grasp)
            st = env.state
            assert st is not None
            records.append(TrialRecord(
                episode=int(ep),
                category=st.category,
                split="seen" if st.category in SEEN_CATEGORIES else "unseen",
                policy=policy_name,
                success=bool(result.success),
                reason=result.reason,
                lift_height=float(result.lift_height),
                grasp_x=float(grasp.x), grasp_y=float(grasp.y), grasp_z=float(grasp.z),
                grasp_yaw=float(grasp.yaw), grasp_width=float(grasp.width),
                final_gripper_width=float(result.final_gripper_width),
                object_displacement=float(result.object_displacement),
                sampler_mode=getattr(policy, "last_mode", ""),
                settle_displacement=float(st.settle_displacement),
            ))
            if progress and (n + 1) % 25 == 0:
                rate = np.mean([r.success for r in records])
                print(f"    {n + 1}/{len(episodes)} episodes, running success {rate:.1%}", flush=True)
    return records


def _worker(args) -> list[dict]:
    policy_name, episodes, split, base_seed, image_size, policy_kwargs = args
    recs = run_episodes(policy_name, episodes, split, base_seed, image_size, policy_kwargs)
    return [asdict(r) for r in recs]


def evaluate_policy(
    policy_name: str,
    n_episodes: int = 200,
    split: str = "all",
    base_seed: int = 0,
    workers: int = 1,
    image_size: int = 224,
    policy_kwargs: dict[str, Any] | None = None,
    episode_offset: int = 0,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate ``policy_name`` over ``n_episodes`` and return a summary dict."""
    episodes = list(range(episode_offset, episode_offset + n_episodes))
    t0 = time.perf_counter()

    if workers <= 1:
        records = [asdict(r) for r in run_episodes(
            policy_name, episodes, split, base_seed, image_size, policy_kwargs, progress)]
    else:
        chunks = [episodes[i::workers] for i in range(workers)]
        args = [(policy_name, c, split, base_seed, image_size, policy_kwargs) for c in chunks if c]
        # "spawn": each worker needs its own OpenGL context, and forking a
        # process that already holds one is unreliable across platforms.
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(args)) as pool:
            records = [r for part in pool.map(_worker, args) for r in part]
        records.sort(key=lambda r: r["episode"])

    elapsed = time.perf_counter() - t0
    return summarise(records, policy=policy_name, elapsed=elapsed, split=split,
                     base_seed=base_seed, n_episodes=n_episodes)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval.

    Reported instead of a bare percentage because at n=200 the 95% interval is
    about +/-7 points: quoting "84%" without it invites over-reading small gaps.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def summarise(records: Iterable[dict], **meta) -> dict[str, Any]:
    records = list(records)
    n = len(records)
    successes = sum(r["success"] for r in records)

    by_category: dict[str, dict[str, Any]] = {}
    for cat in SEEN_CATEGORIES + UNSEEN_CATEGORIES:
        rows = [r for r in records if r["category"] == cat]
        if not rows:
            continue
        s = sum(r["success"] for r in rows)
        lo, hi = wilson_interval(s, len(rows))
        by_category[cat] = {"n": len(rows), "successes": s, "rate": s / len(rows),
                            "ci95": [lo, hi],
                            "split": "seen" if cat in SEEN_CATEGORIES else "unseen"}

    by_split: dict[str, dict[str, Any]] = {}
    for sp in ("seen", "unseen"):
        rows = [r for r in records if r["split"] == sp]
        if not rows:
            continue
        s = sum(r["success"] for r in rows)
        lo, hi = wilson_interval(s, len(rows))
        by_split[sp] = {"n": len(rows), "successes": s, "rate": s / len(rows), "ci95": [lo, hi]}

    reasons: dict[str, int] = {}
    for r in records:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1

    lo, hi = wilson_interval(successes, n)
    return {
        **meta,
        "n": n,
        "successes": successes,
        "success_rate": successes / n if n else 0.0,
        "ci95": [lo, hi],
        "by_category": by_category,
        "by_split": by_split,
        "failure_reasons": reasons,
        "records": records,
    }


def format_summary(summary: dict[str, Any]) -> str:
    lines = []
    pol = summary.get("policy", "?")
    n, rate = summary["n"], summary["success_rate"]
    lo, hi = summary["ci95"]
    lines.append(f"policy={pol}  n={n}  success={rate:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
    if "elapsed" in summary:
        lines.append(f"elapsed {summary['elapsed']:.1f}s "
                     f"({summary['elapsed'] / max(n, 1) * 1000:.0f} ms/episode)")
    lines.append("")
    lines.append(f"  {'category':<12}{'split':<8}{'n':>5}{'success':>10}{'95% CI':>18}")
    lines.append("  " + "-" * 53)
    for cat, s in summary["by_category"].items():
        ci = f"[{s['ci95'][0]:.0%}, {s['ci95'][1]:.0%}]"
        lines.append(f"  {cat:<12}{s['split']:<8}{s['n']:>5}{s['rate']:>9.1%}{ci:>18}")
    lines.append("  " + "-" * 53)
    for sp, s in summary["by_split"].items():
        ci = f"[{s['ci95'][0]:.0%}, {s['ci95'][1]:.0%}]"
        lines.append(f"  {'ALL ' + sp:<20}{s['n']:>5}{s['rate']:>9.1%}{ci:>18}")
    if summary["failure_reasons"]:
        lines.append("")
        order = sorted(summary["failure_reasons"].items(), key=lambda kv: -kv[1])
        lines.append("  outcomes: " + ", ".join(f"{k}={v}" for k, v in order))
    return "\n".join(lines)


def save_summary(summary: dict[str, Any], path: Path, keep_records: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    if not keep_records:
        payload.pop("records", None)
    path.write_text(json.dumps(payload, indent=2, default=float))


def save_records_csv(records: Sequence[dict], path: Path) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
