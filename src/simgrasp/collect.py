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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data.writer import ShardWriter
from .env import PandaGraspEnv
from .grasp import grasp_to_image
from .objects import SEEN_CATEGORIES
from .policies import GraspSampler
from .seeding import rng_for_episode
from .transforms import wrap_grasp_angle


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
    # Which of the angles-per-scene variants this is; 0 is the sampler's own.
    angle_index: int = 0


def _angle_variants(base, count: int, rng) -> list:
    """``count`` grasps at the same point, spread over the half turn.

    The first is always the sampler's own proposal, so a run with
    ``count == 1`` is byte-identical to the original behaviour. The rest are
    offset by multiples of pi/count with a little jitter, which covers the
    gripper's full range of distinct orientations (a parallel jaw is symmetric
    under a half turn) without landing every sample on a bin centre.
    """
    from .grasp import Grasp

    out = [base]
    if count <= 1:
        return out
    step = np.pi / count
    for j in range(1, count):
        jitter = float(rng.normal(0.0, 0.25 * step))
        yaw = float(wrap_grasp_angle(base.yaw + j * step + jitter))
        out.append(Grasp(base.x, base.y, base.z, yaw, base.width))
    return out


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
    angles_per_scene: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run ``episodes`` in this process, writing shards under ``w{worker_id}``.

    ``angles_per_scene`` > 1 executes several grasps at the *same* point in the
    same settled scene, at orientations spread over the half turn, restoring the
    simulator state between them. This is the fix for the failure documented in
    docs/design.md section 6: with one label per image the network can explain
    every label with a function of the pixel alone, and orientation collapses.
    Several labels at one pixel with different outcomes make that impossible.

    It is also cheap. The settle and the render are shared, so K grasps per scene
    cost far less than K independent episodes.
    """
    sampler = GraspSampler(**(sampler_kwargs or {}))
    stats = {"n": 0, "successes": 0, "unstable": 0, "by_category": {}, "by_mode": {}}
    t0 = time.perf_counter()

    with PandaGraspEnv(image_size=image_size, base_seed=base_seed) as env, \
            ShardWriter(out_dir, image_size=image_size, shard_size=shard_size,
                        prefix=f"w{worker_id:02d}", store_rgb=store_rgb) as writer:
        for n, ep in enumerate(episodes):
            obs = env.reset(ep, split=split)
            snapshot = env.snapshot()
            rng = rng_for_episode(base_seed + 7_777_777, ep)
            base = sampler(obs, env, rng)
            mode = sampler.last_mode
            st = env.state
            assert st is not None

            for k, grasp in enumerate(_angle_variants(base, angles_per_scene, rng)):
                if k:
                    env.restore(snapshot)
                result = env.execute(grasp)
                if result.reason == "unstable":
                    # The physics diverged; the outcome is not a real label.
                    stats["unstable"] = stats.get("unstable", 0) + 1
                    continue
                img = grasp_to_image(grasp, obs.cam_pos, obs.cam_mat, obs.intrinsics)

                label = EpisodeLabel(
                    episode=int(ep), category=st.category,
                    split="seen" if st.category in SEEN_CATEGORIES else "unseen",
                    u=img.u, v=img.v, angle=img.angle, width_px=img.width_px, depth=img.depth,
                    world_x=grasp.x, world_y=grasp.y, world_z=grasp.z,
                    world_yaw=grasp.yaw, world_width=grasp.width,
                    success=bool(result.success), reason=result.reason,
                    lift_height=float(result.lift_height), sampler_mode=mode,
                    object_top_z=float(st.spec.top_z),
                    settle_displacement=float(st.settle_displacement),
                    angle_index=k,
                )
                writer.add(obs.rgb, obs.height, label)

                stats["n"] += 1
                stats["successes"] += int(result.success)
                c = stats["by_category"].setdefault(st.category, [0, 0])
                c[0] += 1
                c[1] += int(result.success)
                m = stats["by_mode"].setdefault(mode, [0, 0])
                m[0] += 1
                m[1] += int(result.success)

            if verbose and (n + 1) % 50 == 0:
                rate = stats["successes"] / max(stats["n"], 1)
                eps = (n + 1) / (time.perf_counter() - t0)
                print(f"  [w{worker_id:02d}] scene {n + 1}/{len(episodes)}  "
                      f"{stats['n']} samples  positive rate {rate:.1%}  "
                      f"{eps:.1f} scenes/s", flush=True)

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
    angles_per_scene: int = 1,
    episode_offset: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = list(range(episode_offset, episode_offset + n_episodes))

    common = dict(out_dir=str(out_dir), base_seed=base_seed, image_size=image_size,
                  split=split, shard_size=shard_size, store_rgb=store_rgb,
                  sampler_kwargs=sampler_kwargs, angles_per_scene=angles_per_scene,
                  verbose=verbose)
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
    meta["angles_per_scene"] = angles_per_scene
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return meta


def _merge_stats(parts: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"n": 0, "successes": 0, "unstable": 0, "by_category": {}, "by_mode": {}}
    for p in parts:
        out["n"] += p["n"]
        out["successes"] += p["successes"]
        out["unstable"] += p.get("unstable", 0)
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
