"""Generate the grasp dataset: sample a grasp, execute it, store image + label.

One episode produces exactly one supervised example: an RGB-D image of a scene,
one grasp expressed in image coordinates, and the binary outcome of executing it.
That is the "self-supervised grasping" setup -- labels come from the simulator
attempting the grasp, not from human annotation -- and it is why the grasp
*sampler* (see :class:`~simgrasp.policies.oracle.GraspSampler`) matters as much
as the network: the sampler defines the label distribution.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data.writer import ShardWriter
from .env import PandaGraspEnv
from .grasp import grasp_to_image
from .objects import SEEN_CATEGORIES
from .policies import GraspSampler
from .seeding import rng_for_episode


@dataclass
class EpisodeLabel:
    episode: int
    category: str
    split: str
    # Grasp in image coordinates -- the network's label space.
    u: float
    v: float
    angle: float
    width_px: float
    depth: float
    # The same grasp in world coordinates, for replay and debugging.
    world_x: float
    world_y: float
    world_z: float
    world_yaw: float
    world_width: float
    # Outcome.
    success: bool
    reason: str
    lift_height: float
    sampler_mode: str
    object_top_z: float
    settle_displacement: float


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def collect_worker(
    episodes: list[int],
    out_dir: str,
    worker_id: int = 0,
    base_seed: int = 0,
    image_size: int = 224,
    split: str = "all",
    shard_size: int = 256,
    store_rgb: bool = True,
    sampler_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run ``episodes`` in this process, writing shards under ``w{worker_id}``."""
    sampler = GraspSampler(**(sampler_kwargs or {}))
    stats = {"n": 0, "successes": 0, "by_category": {}, "by_mode": {}}
    t0 = time.perf_counter()

    with PandaGraspEnv(image_size=image_size, base_seed=base_seed) as env, \
            ShardWriter(out_dir, image_size=image_size, shard_size=shard_size,
                        prefix=f"w{worker_id:02d}", store_rgb=store_rgb) as writer:
        for n, ep in enumerate(episodes):
            obs = env.reset(ep, split=split)
            rng = rng_for_episode(base_seed + 7_777_777, ep)
            grasp = sampler(obs, env, rng)
            result = env.execute(grasp)
            img = grasp_to_image(grasp, obs.cam_pos, obs.cam_mat, obs.intrinsics)
            st = env.state
            assert st is not None

            label = EpisodeLabel(
                episode=int(ep), category=st.category,
                split="seen" if st.category in SEEN_CATEGORIES else "unseen",
                u=img.u, v=img.v, angle=img.angle, width_px=img.width_px, depth=img.depth,
                world_x=grasp.x, world_y=grasp.y, world_z=grasp.z,
                world_yaw=grasp.yaw, world_width=grasp.width,
                success=bool(result.success), reason=result.reason,
                lift_height=float(result.lift_height), sampler_mode=sampler.last_mode,
                object_top_z=float(st.spec.top_z),
                settle_displacement=float(st.settle_displacement),
            )
            writer.add(obs.rgb, obs.height, label)

            stats["n"] += 1
            stats["successes"] += int(result.success)
            c = stats["by_category"].setdefault(st.category, [0, 0])
            c[0] += 1
            c[1] += int(result.success)
            m = stats["by_mode"].setdefault(sampler.last_mode, [0, 0])
            m[0] += 1
            m[1] += int(result.success)
            if verbose and (n + 1) % 50 == 0:
                rate = stats["successes"] / stats["n"]
                eps = (n + 1) / (time.perf_counter() - t0)
                print(f"  [w{worker_id:02d}] {n + 1}/{len(episodes)}  "
                      f"positive rate {rate:.1%}  {eps:.1f} ep/s", flush=True)

    stats["elapsed"] = time.perf_counter() - t0
    return stats


def _worker_entry(kwargs) -> dict[str, Any]:
    return collect_worker(**kwargs)


def collect_dataset(
    out_dir: Path | str,
    n_episodes: int = 10_000,
    workers: int = 1,
    base_seed: int = 0,
    image_size: int = 224,
    split: str = "all",
    shard_size: int = 256,
    store_rgb: bool = True,
    sampler_kwargs: dict[str, Any] | None = None,
    episode_offset: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(range(episode_offset, episode_offset + n_episodes))

    common = dict(out_dir=str(out_dir), base_seed=base_seed, image_size=image_size,
                  split=split, shard_size=shard_size, store_rgb=store_rgb,
                  sampler_kwargs=sampler_kwargs, verbose=verbose)
    t0 = time.perf_counter()

    if workers <= 1:
        parts = [collect_worker(episodes=episodes, worker_id=0, **common)]
    else:
        # Round-robin so every worker sees the same category mix, and so a
        # partial run still covers the whole episode range.
        chunks = [episodes[i::workers] for i in range(workers)]
        jobs = [dict(episodes=c, worker_id=i, **common) for i, c in enumerate(chunks) if c]
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(jobs)) as pool:
            parts = pool.map(_worker_entry, jobs)

    elapsed = time.perf_counter() - t0
    merged = _merge_stats(parts)
    meta = _build_meta(out_dir, n_episodes, base_seed, image_size, split, workers,
                       shard_size, store_rgb, sampler_kwargs, merged, elapsed)
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return meta


def _merge_stats(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"n": 0, "successes": 0, "by_category": {}, "by_mode": {}}
    for p in parts:
        out["n"] += p["n"]
        out["successes"] += p["successes"]
        for key in ("by_category", "by_mode"):
            for k, (n, s) in p[key].items():
                cur = out[key].setdefault(k, [0, 0])
                cur[0] += n
                cur[1] += s
    for key in ("by_category", "by_mode"):
        out[key] = {k: {"n": n, "positives": s, "positive_rate": s / n if n else 0.0}
                    for k, (n, s) in out[key].items()}
    out["positive_rate"] = out["successes"] / out["n"] if out["n"] else 0.0
    return out


def _build_meta(out_dir, n_episodes, base_seed, image_size, split, workers, shard_size,
                store_rgb, sampler_kwargs, stats, elapsed) -> dict[str, Any]:
    # Camera geometry is identical for every episode, so record it once here
    # rather than per sample.
    with PandaGraspEnv(image_size=image_size, base_seed=base_seed) as env:
        env.reset(0)
        obs = env.observe()
        intr = obs.intrinsics
        cam_pos, cam_mat = obs.cam_pos, obs.cam_mat
        table_z = obs.table_z

    return {
        "n_episodes": n_episodes,
        "image_size": image_size,
        "base_seed": base_seed,
        "split": split,
        "workers": workers,
        "shard_size": shard_size,
        "store_rgb": store_rgb,
        "sampler_kwargs": sampler_kwargs or {},
        "sampler_weights": GraspSampler(**(sampler_kwargs or {})).weights,
        "camera": {
            "fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
            "width": intr.width, "height": intr.height,
            "pos": np.asarray(cam_pos).tolist(),
            "mat": np.asarray(cam_mat).tolist(),
        },
        "table_z": float(table_z),
        "stats": stats,
        "elapsed_seconds": elapsed,
        "episodes_per_second": n_episodes / elapsed if elapsed else 0.0,
        "provenance": {
            "git_commit": _git_commit(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
