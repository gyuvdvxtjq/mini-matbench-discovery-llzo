"""Merge the checkpoint into cross-model comparisons.

The audit runs in passes (see cloud/run_split.sh), so a single report can be
assembled from several environments. Everything here is derived from
``checkpoints.jsonl`` -- the one artefact both passes write -- and it
reconstructs the convex hull per model with the same
:func:`mlip_audit.analysis.build_hull` the runner uses, so the figures cannot
drift from the recorded numbers.

Missing models are not an error: the report is generated from whatever is in
the checkpoint and says which models it covers.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis import build_hull, evaluate_stability

# Protocol -> (energy column, hull column prefix) in the checkpoint records.
PROTOCOLS: dict[str, tuple[str, str]] = {
    "single_point": ("energy_per_atom", "sp"),
    "relaxation": ("relaxed_e_per_atom", "relax"),
}


def load_records(run_dir: str | Path) -> list[dict[str, Any]]:
    """All checkpoint records, newest write per key wins."""
    path = Path(run_dir) / "checkpoints.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}")
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["_key"]] = record
    return list(records.values())


def split_by_model(records: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group checkpoint records by model name, keyed from ``_key``."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        model = str(record["_key"]).split("|", 1)[0]
        if model == "":  # defensive: a malformed key is not a model result
            continue
        grouped.setdefault(model, []).append(record)
    return grouped


def model_frames(
    records: Iterable[dict[str, Any]], protocol: str = "single_point"
) -> dict[str, pd.DataFrame]:
    """Per-model DataFrame with the model hull columns filled in."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    energy_col, prefix = PROTOCOLS[protocol]
    frames: dict[str, pd.DataFrame] = {}
    for model, rows in split_by_model(records).items():
        rows = [r for r in rows if r.get("status") == "ok" and energy_col in r]
        if len(rows) < 2:
            continue  # a hull needs at least a couple of entries
        frame = build_hull(pd.DataFrame(rows), energy_col, prefix)
        frames[model] = frame[frame["status"] == "ok"].copy()
    return frames


def hull_mae_by_chemsys(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """|model hull error| per chemical system, models as columns."""
    per_model: dict[str, pd.Series] = {}
    for model, frame in frames.items():
        if frame.empty:
            continue
        error = (frame["sp_energy_above_hull"] - frame["mp_energy_above_hull"]).abs()
        per_model[model] = error.groupby(frame["chemsys"]).mean()
    if not per_model:
        return pd.DataFrame()
    return pd.DataFrame(per_model).sort_index()


def metrics_table(
    frames: dict[str, pd.DataFrame],
    *,
    candidate_window: float = 0.05,
    bootstrap_resamples: int = 2000,
) -> dict[str, dict[str, Any]]:
    """Headline metrics per model, scored exactly as run_audit scores them."""
    table: dict[str, dict[str, Any]] = {}
    for model, frame in frames.items():
        if len(frame) < 5:
            table[model] = {
                "n_success": int(len(frame)),
                "note": f"too few successful records (n={len(frame)} < 5)",
            }
            continue
        table[model] = evaluate_stability(
            frame, "sp", candidate_window=candidate_window,
            bootstrap_resamples=bootstrap_resamples,
        )
    return table


def _fmt(metric: Any, digits: int = 4) -> str:
    if not isinstance(metric, dict) or metric.get("point") is None:
        return "n/a"
    point = float(metric["point"])
    if point != point:  # NaN
        return "n/a"
    low, high = metric.get("ci_low"), metric.get("ci_high")
    if low is None or high is None or low != low or high != high:
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def markdown_summary(
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, Any]],
    *,
    protocol: str = "single_point",
    expected: Iterable[str] = (),
) -> str:
    """A Markdown block that can be pasted straight into README.md."""
    expected = list(expected)
    lines = [
        f"### Cross-model summary ({protocol.replace('_', '-')}, "
        "hull distance vs Materials Project DFT)",
        "",
    ]
    if expected:
        covered = [m for m in expected if m in frames]
        missing = [m for m in expected if m not in frames]
        lines.append(
            f"Covered: {', '.join(covered) if covered else 'none'}"
            + (f". Missing: {', '.join(missing)}." if missing else ".")
        )
    else:
        lines.append("Covered: " + (", ".join(frames) if frames else "none") + ".")
    lines += [
        "",
        "| Model | n | Hull MAE (eV/atom) | Hull Spearman | Formation-energy MAE | "
        "Stable F1 | Candidate F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, row in metrics.items():
        lines.append(
            f"| {model} | {row.get('n_success', 0)} "
            f"| {_fmt(row.get('hull_mae_eV_per_atom'))} "
            f"| {_fmt(row.get('hull_spearman'), 3)} "
            f"| {_fmt(row.get('formation_energy_mae_eV_per_atom'))} "
            f"| {_fmt(row.get('stable_classification_f1'), 3)} "
            f"| {_fmt(row.get('candidate_classification_f1'), 3)} |"
        )
    lines += [
        "",
        "Values are point estimates with 95% percentile-bootstrap intervals over "
        "structures. Hull MAE is |model E-hull - MP E-hull| in eV/atom; the hull is "
        "rebuilt per model from the single-point energies in `checkpoints.jsonl`.",
    ]
    return "\n".join(lines)


def plot_hull_mae_by_chemsys(table: pd.DataFrame, path: str | Path) -> Path | None:
    """Grouped bar chart of per-chemsys hull MAE, one bar group per model."""
    if table.empty:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(table)), 6.0))
    table.plot.bar(ax=ax, width=0.8)
    ax.set(
        xlabel="Chemical system",
        ylabel="Hull MAE (eV/atom)",
        title="Hull-distance MAE by chemical system",
    )
    ax.tick_params(axis="x", rotation=60)
    ax.legend(title="Model", fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_hull_parity(frames: dict[str, pd.DataFrame], path: str | Path) -> Path | None:
    """Model vs MP hull distance parity, one marker set per model."""
    usable = {m: f for m, f in frames.items() if not f.empty}
    if not usable:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for model, frame in usable.items():
        ax.scatter(
            frame["mp_energy_above_hull"],
            frame["sp_energy_above_hull"],
            s=16,
            alpha=0.6,
            label=model,
        )
    high = max(
        float(f["mp_energy_above_hull"].max()) for f in usable.values()
    )
    ax.plot([0, high], [0, high], "--", color="gray", linewidth=1)
    ax.set(
        xlabel="MP energy above hull (eV/atom)",
        ylabel="Model energy above hull (eV/atom)",
        title="Hull parity vs Materials Project",
    )
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
