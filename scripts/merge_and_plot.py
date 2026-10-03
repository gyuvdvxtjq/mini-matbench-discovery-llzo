#!/usr/bin/env python3
"""Merge a split audit into cross-model figures and a README-ready table.

The audit is run in several passes (one per environment, see
cloud/run_split.sh); all of them append to the same
``data/runs/<run>/checkpoints.jsonl``. This script reads that file and writes:

* ``figures/hull_mae_by_chemsys_multimodel.png`` -- hull MAE per chemical
  system, one bar group per model
* ``figures/hull_parity_multimodel.png`` -- model vs MP hull distance
* ``figures/model_summary.md`` -- the table pasted into README.md

It degrades gracefully: a run with only some of the configured models still
produces figures and a table, annotated with which models are missing.

Usage
-----
python3 scripts/merge_and_plot.py --run-dir data/runs/llzo
python3 scripts/merge_and_plot.py --config configs/llzo.yaml   # expected models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlip_audit.config import load_config  # noqa: E402
from mlip_audit.merge import (  # noqa: E402
    hull_mae_by_chemsys,
    load_records,
    markdown_summary,
    metrics_table,
    model_frames,
    plot_hull_mae_by_chemsys,
    plot_hull_parity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="data/runs/llzo", help="Audit output dir")
    parser.add_argument(
        "--config",
        default=None,
        help="Audit config; used only to learn which models should be present",
    )
    parser.add_argument(
        "--protocol",
        default="single_point",
        choices=("single_point", "relaxation"),
        help="Protocol to compare across models",
    )
    parser.add_argument("--fig-dir", default="figures", help="Figure output dir")
    parser.add_argument(
        "--candidate-window", type=float, default=None, help="Override, eV/atom"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    fig_dir = Path(args.fig_dir)
    if not fig_dir.is_absolute():
        fig_dir = REPO_ROOT / fig_dir

    try:
        records = load_records(run_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    frames = model_frames(records, protocol=args.protocol)
    if not frames:
        print(
            f"error: no usable {args.protocol} records in {run_dir/'checkpoints.jsonl'}",
            file=sys.stderr,
        )
        return 1

    expected: list[str] = []
    window = 0.05
    resamples = 2000
    if args.config:
        config = load_config(args.config)
        expected = [m.name for m in config.models if m.enabled]
        window = args.candidate_window or config.candidate_window_ev
        resamples = config.bootstrap_resamples
    if args.candidate_window is not None:
        window = args.candidate_window

    table = hull_mae_by_chemsys(frames)
    metrics = metrics_table(frames, candidate_window=window, bootstrap_resamples=resamples)

    print(f"run dir       : {run_dir}")
    print(f"models merged : {', '.join(frames) or 'none'}")
    if expected:
        missing = [m for m in expected if m not in frames]
        if missing:
            print(f"models absent : {', '.join(missing)}  (figures cover the rest)")

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = [
        plot_hull_mae_by_chemsys(table, fig_dir / "hull_mae_by_chemsys_multimodel.png"),
        plot_hull_parity(frames, fig_dir / "hull_parity_multimodel.png"),
    ]
    summary = markdown_summary(frames, metrics, protocol=args.protocol, expected=expected)
    summary_path = fig_dir / "model_summary.md"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    written.append(summary_path)

    print()
    print(summary)
    print()
    for path in written:
        if path:
            shown = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
