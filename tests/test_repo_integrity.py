"""Guards against a class of bug that made the published repository unusable.

``.gitignore`` contained an unanchored ``data/`` pattern, intended for the
generated dataset directory at the repository root. Git applies an unanchored
pattern at *every* level, so it also matched ``src/simgrasp/data/`` and silently
excluded that package from all 31 commits. Every test passed locally, CI was
green, and a fresh clone could not import ``simgrasp`` at all: ``collect.py``
does ``from .data.writer import ShardWriter`` and ``training.py`` does
``from .data import GraspDataset``.

Nothing in a normal test suite catches that, because the suite runs against the
working tree rather than against what was committed. These tests check the
repository itself.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Directories whose entire contents must be committed for the package to work.
SOURCE_DIRS = ("src", "scripts", "tests")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=True).stdout


def _is_git_repo() -> bool:
    try:
        _git("rev-parse", "--git-dir")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


requires_git = pytest.mark.skipif(not _is_git_repo(), reason="not a git checkout")


@requires_git
def test_no_source_file_is_gitignored():
    """The regression test for the bug above."""
    candidates = [
        p for d in SOURCE_DIRS for p in (ROOT / d).rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    assert candidates, "found no source files to check"

    # check-ignore exits 1 when nothing matches, which is the outcome we want.
    result = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", *map(str, candidates)],
        capture_output=True, text=True)
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, (
        "these source files are excluded by .gitignore and would not be committed:\n  "
        + "\n  ".join(ignored)
        + "\nAnchor the offending pattern with a leading slash."
    )


@requires_git
def test_every_package_directory_is_tracked():
    """A package whose ``__init__.py`` is untracked is missing from a fresh clone."""
    tracked = set(_git("ls-files").splitlines())
    for init in (ROOT / "src").rglob("__init__.py"):
        rel = init.relative_to(ROOT).as_posix()
        assert rel in tracked, f"{rel} is not tracked; a fresh clone cannot import it"


@pytest.mark.parametrize("module", [
    "simgrasp.data",
    "simgrasp.data.writer",
    "simgrasp.data.dataset",
    "simgrasp.collect",
    "simgrasp.training",
])
def test_module_imports(module):
    """Importing is the cheapest possible check that a module actually ships."""
    assert importlib.import_module(module) is not None
