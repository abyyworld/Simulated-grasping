"""Deterministic seeding helpers.

Every episode in this project is reproducible from a single integer. Data
collection derives a per-episode seed from (base_seed, episode_index) so that
workers running in parallel never collide and a re-run reproduces the dataset
byte-for-byte.
"""

from __future__ import annotations

import numpy as np


def episode_seed(base_seed: int, episode_index: int) -> int:
    """Map (base_seed, episode_index) to a well-mixed 32-bit seed.

    Uses SeedSequence rather than `base_seed + index` so that nearby indices do
    not produce correlated random streams.
    """
    seq = np.random.SeedSequence(entropy=int(base_seed), spawn_key=(int(episode_index),))
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def rng_for_episode(base_seed: int, episode_index: int) -> np.random.Generator:
    return np.random.default_rng(episode_seed(base_seed, episode_index))


def seed_everything(seed: int) -> None:
    """Seed numpy + torch (torch imported lazily so sim-only code stays light)."""
    np.random.seed(seed % (2**32))
    try:
        import random

        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is optional for sim-only use
        pass
