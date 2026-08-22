#!/usr/bin/env python3
"""Render the README media: a grasp GIF and a prediction figure.

    python scripts/make_media.py --gif --episodes 3
    python scripts/make_media.py --figure --checkpoint runs/grasp_cnn/best.pt
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np


def make_gif(args) -> None:
    import imageio.v2 as imageio

    from simgrasp.env import PandaGraspEnv
    from simgrasp.evaluation import build_policy
    from simgrasp.paths import MEDIA_DIR
    from simgrasp.scene import SCENE_CAM
    from simgrasp.seeding import rng_for_episode

    kwargs = {"checkpoint": args.checkpoint} if args.policy == "cnn" else None
    policy = build_policy(args.policy, **(kwargs or {}))
    frames: list[np.ndarray] = []

    env = PandaGraspEnv(base_seed=args.seed, render_camera=SCENE_CAM,
                        render_size=(args.height, args.width))
    try:
        shown = 0
        ep = args.offset
        while shown < args.episodes and ep < args.offset + 200:
            obs = env.reset(ep)
            rng = rng_for_episode(args.seed + 1_000_003, ep)
            grasp = policy(obs, env, rng)
            start = len(frames)
            result = env.execute(grasp, frame_hook=frames.append, frame_every=args.frame_every)
            if args.successes_only and not result.success:
                del frames[start:]
                ep += 1
                continue
            print(f"  episode {ep}: {env.state.category:<10} {result.reason}")
            frames.extend([frames[-1]] * 6)  # brief pause on the lifted object
            shown += 1
            ep += 1
    finally:
        env.close()

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out = MEDIA_DIR / args.name
    imageio.mimsave(out, frames, duration=args.frame_every * 0.002 * 1000, loop=0)
    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out}  ({len(frames)} frames, {size_mb:.1f} MB)")


def make_figure(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import cv2

    from simgrasp.camera import project_points
    from simgrasp.env import PandaGraspEnv
    from simgrasp.heightmap import metres_per_pixel
    from simgrasp.objects import SEEN_CATEGORIES
    from simgrasp.paths import MEDIA_DIR
    from simgrasp.policies.learned import LearnedPolicy
    from simgrasp.seeding import rng_for_episode

    policy = LearnedPolicy(checkpoint=args.checkpoint)
    env = PandaGraspEnv(base_seed=args.seed)
    n = args.columns
    fig, axes = plt.subplots(3, n, figsize=(3.1 * n, 9.2))
    axes = np.atleast_2d(axes)

    try:
        for col in range(n):
            ep = args.offset + col
            obs = env.reset(ep)
            rng = rng_for_episode(args.seed + 1_000_003, ep)
            grasp = policy(obs, env, rng)
            # The network may run at a lower resolution than the observation
            # (--input-size), so put its map back on the image grid before
            # overlaying anything in image pixels.
            best = policy.last_quality.max(axis=0)
            if best.shape != obs.height.shape:
                best = cv2.resize(best, obs.height.shape[::-1], interpolation=cv2.INTER_LINEAR)
            uv, _ = project_points(grasp.position, obs.cam_pos, obs.cam_mat, obs.intrinsics)
            result = env.execute(grasp)

            split = "seen" if env.state.category in SEEN_CATEGORIES else "HELD-OUT"
            axes[0, col].imshow(obs.rgb)
            axes[0, col].set_title(f"{env.state.category}  ({split})", fontsize=10)
            axes[1, col].imshow(obs.height, cmap="viridis")
            axes[1, col].set_title("height map", fontsize=9)
            im = axes[2, col].imshow(np.clip(best, 0, 1), cmap="inferno", vmin=0, vmax=1)
            axes[2, col].set_title(f"grasp quality\n{result.reason}", fontsize=9)

            # Convert the jaw width to pixels through the real intrinsics rather
            # than a hard-coded ground sampling distance.
            mpp = metres_per_pixel(obs.intrinsics, float(obs.cam_pos[2] - grasp.z))
            half = 0.5 * grasp.width / mpp
            dx, dy = np.cos(-grasp.yaw) * half, np.sin(-grasp.yaw) * half
            for ax in axes[:, col]:
                ax.plot([uv[0] - dx, uv[0] + dx], [uv[1] - dy, uv[1] + dy],
                        color="lime" if result.success else "red", lw=2.2)
                ax.plot(uv[0], uv[1], "o", color="white", ms=3)
                ax.set_xticks([])
                ax.set_yticks([])
    finally:
        env.close()

    fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.02)
    fig.suptitle("Predicted grasp quality (max over angle bins); green = lifted, red = failed")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    out = MEDIA_DIR / "predictions.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--policy", default="oracle")
    ap.add_argument("--checkpoint", default="runs/grasp_cnn/best.pt")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--columns", type=int, default=4)
    ap.add_argument("--offset", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--frame-every", type=int, default=20)
    ap.add_argument("--name", default="grasp_demo.gif")
    ap.add_argument("--successes-only", action="store_true")
    args = ap.parse_args()

    if not (args.gif or args.figure):
        args.gif = True
    if args.gif:
        make_gif(args)
    if args.figure:
        make_figure(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
