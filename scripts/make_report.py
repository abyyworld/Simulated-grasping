#!/usr/bin/env python3
"""Generate docs/results.md from the JSON written by run_baseline.py / evaluate.py.

The results tables are generated rather than hand-written so they cannot drift
away from the runs that produced them. Every number here traces to a file in
results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def _pct(x) -> str:
    return f"{x * 100:.1f}%"


def _ci(lo, hi) -> str:
    return f"{lo * 100:.0f}–{hi * 100:.0f}%"


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def baseline_section(results: Path) -> list[str]:
    from simgrasp.objects import SEEN_CATEGORIES

    names = [("oracle", "Oracle (ground-truth pose)"),
             ("heuristic", "Heuristic (depth centroid + PCA)"),
             ("cnn", "CNN (ours)")]
    loaded = [(key, label, load(results / "baseline" / f"{key}.json")) for key, label in names]
    loaded = [(k, lbl, s) for k, lbl, s in loaded if s]
    if not loaded:
        return []

    lines = ["## Scripted-grasp baseline", "",
             f"{loaded[0][2]['n']} trials, identical scenes for every policy "
             f"(seed {loaded[0][2].get('base_seed', 0)}).", "",
             "| policy | overall | 95% CI | seen | held-out |",
             "|---|---|---|---|---|"]
    for _key, label, s in loaded:
        seen = s["by_split"].get("seen", {}).get("rate", float("nan"))
        unseen = s["by_split"].get("unseen", {}).get("rate", float("nan"))
        lines.append(f"| {label} | **{_pct(s['success_rate'])}** | "
                     f"{_ci(*s['ci95'])} | {_pct(seen)} | {_pct(unseen)} |")

    lines += ["", "### Per category", "",
              "| category | split | " + " | ".join(lbl for _k, lbl, _s in loaded) + " |",
              "|---|---|" + "---|" * len(loaded)]
    cats = list(loaded[0][2]["by_category"])
    for cat in cats:
        split = "seen" if cat in SEEN_CATEGORIES else "**held-out**"
        cells = []
        for _k, _lbl, s in loaded:
            row = s["by_category"].get(cat)
            cells.append(f"{_pct(row['rate'])} ({row['n']})" if row else "–")
        lines.append(f"| `{cat}` | {split} | " + " | ".join(cells) + " |")

    lines += ["", "### Failure modes", "",
              "| policy | " + " | ".join(
                  sorted({r for _k, _l, s in loaded for r in s["failure_reasons"]})) + " |",
              "|---|" + "---|" * len({r for _k, _l, s in loaded for r in s["failure_reasons"]})]
    all_reasons = sorted({r for _k, _l, s in loaded for r in s["failure_reasons"]})
    for _k, label, s in loaded:
        lines.append(f"| {label} | " +
                     " | ".join(str(s["failure_reasons"].get(r, 0)) for r in all_reasons) + " |")
    return lines


def generalisation_section(results: Path) -> list[str]:
    table = load(results / "eval" / "summary_table.json")
    if not table:
        return []
    lines = ["## Generalisation to held-out categories", "",
             "Separate evaluation on disjoint episode indices, `--split seen` versus "
             "`--split unseen`, so each cell is a full run on that category pool.", "",
             "| policy | seen | held-out | gap |", "|---|---|---|---|"]
    for name, row in table.items():
        gap = (row["seen"] - row["unseen"]) * 100
        lines.append(f"| {name} | {_pct(row['seen'])} | {_pct(row['unseen'])} | "
                     f"{gap:+.1f} pp |")
    return lines


def ablation_section(run_dir: Path, ablation_dir: Path) -> list[str]:
    """Rotation-augmentation ablation: two runs identical but for the augmentation."""
    main = load(run_dir / "summary.json")
    abl = load(ablation_dir / "summary.json")
    if not (main and abl):
        return []
    angles = load(Path("results") / "angle_ablation.json")

    lines = ["## Ablation: rotation augmentation", "",
             "Two runs with identical data, architecture, schedule and seed; the only "
             "difference is whether training images are randomly rotated (with the grasp "
             "label carried along).", "",
             "| | no rotation | with rotation |", "|---|---|---|",
             f"| best validation AP | {abl['best']['ap']:.3f} | {main['best']['ap']:.3f} |",
             f"| best epoch | {abl['best']['epoch']} | {main['best']['epoch']} |"]
    if angles:
        lines += [
            f"| mean grasp-angle error, elongated objects | {angles['norot_deg']:.0f}° | "
            f"**{angles['rot_deg']:.0f}°** |",
            f"| per-angle quality spread at the chosen pixel | {angles['norot_spread']:.3f} | "
            f"**{angles['rot_spread']:.3f}** |",
        ]
    lines += ["", "Validation AP measures *where* to grasp and is barely affected. The "
              "angle metrics are the point: without augmentation each episode supervises "
              "one of twelve angle bins at one pixel, so the network predicts the "
              "angle-marginal success rate and orientation collapses. Random guessing over "
              "a half-turn-symmetric grasp scores 45°.", ""]
    return lines


def training_section(run_dir: Path) -> list[str]:
    summary = load(run_dir / "summary.json")
    if not summary:
        return []
    cfg = summary["config"]
    best = summary["best"]
    hist = summary["history"]
    row = next((h for h in hist if h["epoch"] == best["epoch"]), hist[-1] if hist else {})
    lines = ["## Training", "",
             "| setting | value |", "|---|---|",
             f"| dataset | `{cfg['data']}` ({summary.get('n_train', '?')} training samples) |",
             f"| input | {'height + RGB (4 ch)' if cfg['use_rgb'] else 'height only (1 ch)'} |",
             f"| epochs | {cfg['epochs']} |",
             f"| batch size | {cfg['batch_size']} |",
             f"| optimiser | AdamW, one-cycle, max lr {cfg['lr']} |",
             f"| device | {summary.get('device', '?')} |",
             f"| wall clock | {summary['elapsed_seconds'] / 60:.1f} min |",
             "",
             f"Best validation average precision **{best['ap']:.3f}** at epoch {best['epoch']}"
             + (f", validation accuracy {row.get('val_accuracy', float('nan')):.3f}"
                if row.get("val_accuracy") is not None else "") + ".", ""]
    return lines


MARKER = "<!-- RESULTS_TABLE -->"


def headline_table(results: Path) -> list[str]:
    """The compact table injected into the README between RESULTS_TABLE markers."""
    from simgrasp.objects import SEEN_CATEGORIES  # noqa: F401  (kept for symmetry)

    names = [("oracle", "Oracle (ground-truth pose)", "upper bound: perfect perception"),
             ("heuristic", "Heuristic (depth centroid + PCA)", "no learning, depth only"),
             ("cnn", "**CNN (ours)**", "one RGB-D image, one forward pass")]
    rows = []
    for key, label, note in names:
        s = load(results / "baseline" / f"{key}.json")
        if not s:
            continue
        seen = s["by_split"].get("seen", {}).get("rate", float("nan"))
        unseen = s["by_split"].get("unseen", {}).get("rate", float("nan"))
        drop = (seen - unseen) * 100
        rows.append(f"| {label} | {_pct(seen)} | {_pct(unseen)} | {drop:+.1f} pp | {note} |")
    if not rows:
        return []
    n = load(results / "baseline" / "oracle.json")["n"]
    return (["| policy | seen categories | held-out categories | drop | |",
             "|---|---|---|---|---|"] + rows +
            ["", f"*n = {n} trials, identical scenes for every policy.*"])


def update_readme(readme: Path, table: list[str]) -> bool:
    if not (readme.exists() and table):
        return False
    text = readme.read_text()
    if text.count(MARKER) == 1:
        text = text.replace(MARKER, MARKER + "\n\n" + "\n".join(table) + "\n\n" + MARKER)
    elif text.count(MARKER) == 2:
        head, _mid, tail = text.split(MARKER)
        text = head + MARKER + "\n\n" + "\n".join(table) + "\n\n" + MARKER + tail
    else:
        return False
    readme.write_text(text)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--run", default="runs/grasp_cnn")
    ap.add_argument("--ablation", default="runs/grasp_cnn_norot")
    ap.add_argument("--out", default="docs/results.md")
    ap.add_argument("--readme", default="README.md")
    args = ap.parse_args()

    results = Path(args.results)
    parts = ["# Results", "",
             "Generated by `python scripts/make_report.py`. Do not edit by hand.", ""]
    for section in (baseline_section(results), generalisation_section(results),
                    training_section(Path(args.run)),
                    ablation_section(Path(args.run), Path(args.ablation))):
        if section:
            parts += section + [""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    print(f"wrote {out} ({len(parts)} lines)")

    if update_readme(Path(args.readme), headline_table(results)):
        print(f"updated the results table in {args.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
