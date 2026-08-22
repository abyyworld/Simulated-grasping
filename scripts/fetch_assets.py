#!/usr/bin/env python3
"""Fetch the Franka Panda MJCF from MuJoCo Menagerie into ``assets/``.

The robot meshes are ~34 MB, so they are *not* committed to this repository.
They are pulled from a pinned Menagerie commit so the geometry is identical for
every user of this repo. Run once after cloning:

    python scripts/fetch_assets.py       # or: make assets
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simgrasp.paths import ASSETS_DIR, PANDA_DIR, PANDA_XML  # noqa: E402

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
# Pinned so everyone gets byte-identical robot geometry.
MENAGERIE_COMMIT = "da76818e269b82289eba39808e2fb91d679d6994"
MODEL_SUBDIR = "franka_emika_panda"

# Files we assert exist after the copy, as a cheap integrity check.
REQUIRED_FILES = [
    "panda.xml",
    "LICENSE",
    "assets/link0.stl",
    "assets/hand.stl",
    "assets/finger_0.obj",
]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _sparse_clone(dest: Path) -> None:
    """Clone only ``franka_emika_panda`` at the pinned commit."""
    _run(["git", "init", "-q", str(dest)])
    _run(["git", "remote", "add", "origin", MENAGERIE_URL], cwd=dest)
    _run(["git", "config", "core.sparseCheckout", "true"], cwd=dest)
    (dest / ".git" / "info" / "sparse-checkout").write_text(f"{MODEL_SUBDIR}/*\nLICENSE\n")
    _run(["git", "fetch", "--depth", "1", "origin", MENAGERIE_COMMIT], cwd=dest)
    _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)


def fetch(force: bool = False, retries: int = 4) -> Path:
    if PANDA_XML.exists() and not force:
        print(f"[fetch_assets] Panda model already present at {PANDA_DIR} (use --force to refetch).")
        return PANDA_DIR

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(retries):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "menagerie"
            try:
                print(f"[fetch_assets] Cloning {MODEL_SUBDIR} @ {MENAGERIE_COMMIT[:8]} "
                      f"(attempt {attempt + 1}/{retries})...")
                _sparse_clone(work)
                src = work / MODEL_SUBDIR
                if not src.is_dir():
                    raise RuntimeError(f"{MODEL_SUBDIR} missing from checkout")
                if PANDA_DIR.exists():
                    shutil.rmtree(PANDA_DIR)
                shutil.copytree(src, PANDA_DIR)
                # The 2 MB preview render is not needed at runtime.
                (PANDA_DIR / "panda.png").unlink(missing_ok=True)
                break
            except Exception as exc:  # noqa: BLE001 - retry any network/git failure
                last_error = exc
                detail = getattr(exc, "stderr", b"")
                if detail:
                    print(f"    {detail.decode(errors='replace').strip().splitlines()[-1]}")
                if attempt < retries - 1:
                    backoff = 2 ** (attempt + 1)
                    print(f"[fetch_assets] Failed; retrying in {backoff}s...")
                    time.sleep(backoff)
    else:
        raise RuntimeError(f"Could not fetch Menagerie assets after {retries} attempts") from last_error

    missing = [f for f in REQUIRED_FILES if not (PANDA_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"Fetch incomplete, missing: {missing}")

    n_meshes = len(list((PANDA_DIR / "assets").glob("*")))
    print(f"[fetch_assets] OK -> {PANDA_DIR} ({n_meshes} mesh files)")
    print("[fetch_assets] Model is Apache-2.0 licensed by Franka Emika / Google DeepMind; "
          f"see {PANDA_DIR / 'LICENSE'}")
    return PANDA_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="refetch even if assets exist")
    args = parser.parse_args()
    fetch(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
