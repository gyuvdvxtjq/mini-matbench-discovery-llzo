"""Analysis: rebuild the hull from model energies, score it against MP,
quantify confidence signals, and attribute failures across models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .stats import auroc, bootstrap_metric, safe_spearman


def build_hull(frame: pd.DataFrame, energy_col: str, prefix: str) -> pd.DataFrame:
    """Rebuild the convex hull from a column of per-atom energies."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    ok = frame[frame["status"] == "ok"].copy()
    entries = [
        ComputedEntry(
            Composition(row["formula"]),
            row[energy_col] * Composition(row["formula"]).num_atoms,
            entry_id=row["material_id"],
            data={"frame_index": int(idx)},
        )
        for idx, row in ok.iterrows()
    ]
    diagram = PhaseDiagram(entries)
    for entry in entries:
        idx = entry.data["frame_index"]
        frame.loc[idx, f"{prefix}_formation_energy_per_atom"] = (
            diagram.get_form_energy_per_atom(entry)
        )
        frame.loc[idx, f"{prefix}_energy_above_hull"] = diagram.get_e_above_hull(entry)
    return frame


def _f1(frame: pd.DataFrame, truth_col: str, pred_col: str) -> float:
    truth = frame[truth_col].astype(bool)
    pred = frame[pred_col].astype(bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    if tp == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def evaluate_stability(
    frame: pd.DataFrame,
    prefix: str,
    *,
    candidate_window: float,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Metrics for one model/protocol against MP DFT references."""
    ok = frame[frame["status"] == "ok"].copy()
    eah = f"{prefix}_energy_above_hull"
    ok["hull_error"] = ok[eah] - ok["mp_energy_above_hull"]
    ok["pred_is_stable"] = ok[eah] <= 1e-7
    ok["pred_candidate"] = ok[eah] <= candidate_window
    ok["mp_candidate"] = ok["mp_energy_above_hull"] <= candidate_window
    n = len(ok)

    def ci(metric):
        return bootstrap_metric(ok, metric, n_resamples=bootstrap_resamples)

    metrics: dict[str, Any] = {
        "n_total": int(len(frame)),
        "n_success": n,
        "n_failed": int(len(frame) - n),
        "hull_mae_eV_per_atom": ci(lambda d: d["hull_error"].abs().mean()),
        "hull_rmse_eV_per_atom": ci(
            lambda d: float(np.sqrt(np.mean(np.square(d["hull_error"]))))
        ),
        "hull_spearman": ci(
            lambda d: (
                np.nan
                if (v := safe_spearman(d[eah], d["mp_energy_above_hull"])) is None
                else v
            )
        ),
        "formation_energy_mae_eV_per_atom": ci(
            lambda d: (
                d[f"{prefix}_formation_energy_per_atom"]
                - d["mp_formation_energy_per_atom"]
            )
            .abs()
            .mean()
        ),
        "stable_classification_f1": ci(
            lambda d: _f1(d, "mp_is_stable", "pred_is_stable")
        ),
        "candidate_classification_f1": ci(
            lambda d: _f1(d, "mp_candidate", "pred_candidate")
        ),
        "candidate_window_eV": candidate_window,
        "n_mp_stable": int(ok["mp_is_stable"].sum()),
        "n_mp_candidates": int(ok["mp_candidate"].sum()),
    }
    if "initial_max_force" in ok:
        flagged = ok["hull_error"].abs() > candidate_window
        metrics["confidence_signal"] = {
            "description": (
                "initial max force at the MP geometry as a cheap OOD proxy; "
                f"positive class = |hull error| > {candidate_window} eV/atom"
            ),
            "auroc": auroc(ok["initial_max_force"], flagged),
            "spearman_force_vs_abs_error": safe_spearman(
                ok["initial_max_force"], ok["hull_error"].abs()
            ),
            "n_flagged": int(flagged.sum()),
        }
    return metrics


def cross_model_attribution(
    frames: dict[str, pd.DataFrame], *, top_k: int = 20
) -> dict[str, Any]:
    """Which failures are shared (data-coverage) vs model-specific (architecture)?"""
    errors: dict[str, pd.Series] = {}
    for model, frame in frames.items():
        ok = frame[frame["status"] == "ok"].copy()
        # The MP reference column also ends in "_energy_above_hull"; the model
        # column is the one NOT prefixed with "mp_".
        eah_cols = [
            c
            for c in ok.columns
            if c.endswith("_energy_above_hull") and not c.startswith("mp_")
        ]
        if len(eah_cols) != 1:
            raise ValueError(f"{model}: expected 1 model hull column, got {eah_cols}")
        ok["abs_hull_error"] = (ok[eah_cols[0]] - ok["mp_energy_above_hull"]).abs()
        errors[model] = ok.set_index("material_id")["abs_hull_error"]
    merged = pd.DataFrame(errors).dropna()
    models = list(frames)
    report: dict[str, Any] = {"n_shared_structures": int(len(merged)), "models": models}
    report["pairwise_error_spearman"] = {
        f"{a} vs {b}": safe_spearman(merged[a], merged[b])
        for i, a in enumerate(models)
        for b in models[i + 1 :]
    }
    worst = {m: set(merged[m].nlargest(min(top_k, len(merged))).index) for m in models}
    if len(models) >= 2:
        shared = set.intersection(*worst.values())
        union = set.union(*worst.values())
        report["top_error_overlap"] = {
            "top_k": top_k,
            "n_shared_worst": len(shared),
            "jaccard": len(shared) / len(union) if union else None,
            "shared_worst_material_ids": sorted(shared),
        }
    return report


def write_metrics(out_dir: Path, name: str, metrics: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"metrics_{name}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
