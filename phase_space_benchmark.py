"""Benchmark CHGNet energies and convex-hull stability in Li-La-Zr-O phase space."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ELEMENTS = ("Li", "La", "Zr", "O")
DATA_DIR = Path("data/phase_space")
RAW_PATH = Path("data/raw/mp_phase_space.json")
RESULT_PATH = DATA_DIR / "chgnet_phase_space.csv"
METRICS_PATH = DATA_DIR / "metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-cache", action="store_true")
    return parser.parse_args()


def require_api_key() -> str:
    load_dotenv()
    key = os.getenv("MP_API_KEY", "").strip()
    if not key or key == "replace_with_your_key":
        raise RuntimeError("MP_API_KEY is missing from .env.")
    return key


def chemical_systems() -> list[str]:
    return [
        "-".join(sorted(combo))
        for size in range(1, len(ELEMENTS) + 1)
        for combo in itertools.combinations(ELEMENTS, size)
    ]


def fetch_phase_space(api_key: str) -> list[dict[str, Any]]:
    from mp_api.client import MPRester

    fields = [
        "material_id",
        "formula_pretty",
        "chemsys",
        "structure",
        "nsites",
        "energy_per_atom",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
    ]
    records: list[dict[str, Any]] = []
    with MPRester(api_key) as mpr:
        for chemsys in chemical_systems():
            docs = mpr.materials.summary.search(
                chemsys=[chemsys], fields=fields, chunk_size=1000
            )
            print(f"{chemsys}: {len(docs)}")
            records.extend(
                {
                    "material_id": str(doc.material_id),
                    "formula": doc.formula_pretty,
                    "chemsys": doc.chemsys,
                    "nsites": int(doc.nsites),
                    "mp_energy_per_atom": float(doc.energy_per_atom),
                    "mp_formation_energy_per_atom": float(
                        doc.formation_energy_per_atom
                    ),
                    "mp_energy_above_hull": float(doc.energy_above_hull),
                    "mp_is_stable": bool(doc.is_stable),
                    "structure": doc.structure.as_dict(),
                }
                for doc in docs
            )
    records.sort(key=lambda row: row["material_id"])
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def load_or_fetch(api_key: str | None, use_cache: bool) -> list[dict[str, Any]]:
    if use_cache and RAW_PATH.exists():
        return json.loads(RAW_PATH.read_text(encoding="utf-8"))
    if not api_key:
        raise RuntimeError("MP_API_KEY is missing from .env.")
    return fetch_phase_space(api_key)


def scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def predict_energies(
    records: list[dict[str, Any]], device: str, batch_size: int
) -> list[dict[str, Any]]:
    from chgnet.model import CHGNet
    from pymatgen.core import Structure

    model = CHGNet.load(use_device=device)
    structures = [Structure.from_dict(row["structure"]) for row in records]
    predictions: list[dict[str, Any]] = []
    for start in range(0, len(records), batch_size):
        stop = min(start + batch_size, len(records))
        batch_records = records[start:stop]
        batch_structures = structures[start:stop]
        try:
            values = model.predict_structure(
                batch_structures, task="e", batch_size=batch_size
            )
            if isinstance(values, dict):
                values = [values]
            for source, value in zip(batch_records, values, strict=True):
                predictions.append(
                    {
                        **{key: val for key, val in source.items() if key != "structure"},
                        "status": "ok",
                        "chgnet_energy_per_atom": scalar(value["e"]),
                        "error": "",
                    }
                )
        except Exception:
            # Preserve individual failures without discarding the rest of a batch.
            for source, structure in zip(batch_records, batch_structures, strict=True):
                try:
                    value = model.predict_structure(structure, task="e")
                    predictions.append(
                        {
                            **{
                                key: val
                                for key, val in source.items()
                                if key != "structure"
                            },
                            "status": "ok",
                            "chgnet_energy_per_atom": scalar(value["e"]),
                            "error": "",
                        }
                    )
                except Exception as exc:
                    predictions.append(
                        {
                            **{
                                key: val
                                for key, val in source.items()
                                if key != "structure"
                            },
                            "status": "error",
                            "chgnet_energy_per_atom": np.nan,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
        print(f"Predicted {stop}/{len(records)}")
    return predictions


def add_phase_diagram_values(frame: pd.DataFrame) -> pd.DataFrame:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    ok = frame[frame["status"] == "ok"].copy()
    entries = []
    for index, row in ok.iterrows():
        composition = Composition(row["formula"])
        entry = ComputedEntry(
            composition,
            row["chgnet_energy_per_atom"] * composition.num_atoms,
            entry_id=row["material_id"],
            data={"frame_index": int(index)},
        )
        entries.append(entry)
    phase_diagram = PhaseDiagram(entries)
    for entry in entries:
        index = entry.data["frame_index"]
        frame.loc[index, "chgnet_formation_energy_per_atom"] = (
            phase_diagram.get_form_energy_per_atom(entry)
        )
        frame.loc[index, "chgnet_energy_above_hull"] = phase_diagram.get_e_above_hull(
            entry
        )
    return frame


def safe_corr(left: pd.Series, right: pd.Series) -> float | None:
    value = left.corr(right, method="spearman")
    return None if pd.isna(value) else float(value)


def classification_metrics(truth: pd.Series, prediction: pd.Series) -> dict[str, Any]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    tn = int((~truth & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_phase_diagram_values(pd.DataFrame(records))
    ok = frame[frame["status"] == "ok"].copy()
    ok["energy_error"] = ok["chgnet_energy_per_atom"] - ok["mp_energy_per_atom"]
    ok["formation_energy_error"] = (
        ok["chgnet_formation_energy_per_atom"]
        - ok["mp_formation_energy_per_atom"]
    )
    ok["hull_error"] = (
        ok["chgnet_energy_above_hull"] - ok["mp_energy_above_hull"]
    )
    ok["chgnet_is_stable"] = ok["chgnet_energy_above_hull"] <= 1e-7
    ok["mp_within_50meV"] = ok["mp_energy_above_hull"] <= 0.05
    ok["chgnet_within_50meV"] = ok["chgnet_energy_above_hull"] <= 0.05
    for column in (
        "energy_error",
        "formation_energy_error",
        "hull_error",
        "chgnet_is_stable",
        "mp_within_50meV",
        "chgnet_within_50meV",
    ):
        frame.loc[ok.index, column] = ok[column]
    frame.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    metrics = {
        "n_total": int(len(frame)),
        "n_success": int(len(ok)),
        "n_failed": int(len(frame) - len(ok)),
        "n_exact_llzo": int((ok["chemsys"] == "La-Li-O-Zr").sum()),
        "energy_mae_eV_per_atom": float(ok["energy_error"].abs().mean()),
        "energy_rmse_eV_per_atom": float(np.sqrt(np.mean(ok["energy_error"] ** 2))),
        "formation_energy_mae_eV_per_atom": float(
            ok["formation_energy_error"].abs().mean()
        ),
        "hull_mae_eV_per_atom": float(ok["hull_error"].abs().mean()),
        "hull_rmse_eV_per_atom": float(np.sqrt(np.mean(ok["hull_error"] ** 2))),
        "energy_spearman": safe_corr(
            ok["chgnet_energy_per_atom"], ok["mp_energy_per_atom"]
        ),
        "hull_spearman": safe_corr(
            ok["chgnet_energy_above_hull"], ok["mp_energy_above_hull"]
        ),
        "stable_classification": classification_metrics(
            ok["mp_is_stable"], ok["chgnet_is_stable"]
        ),
        "within_50meV_classification": classification_metrics(
            ok["mp_within_50meV"], ok["chgnet_within_50meV"]
        ),
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ok.nlargest(10, "hull_error", keep="all").to_csv(
        DATA_DIR / "largest_hull_overestimates.csv", index=False, encoding="utf-8-sig"
    )
    ok.nsmallest(10, "hull_error", keep="all").to_csv(
        DATA_DIR / "largest_hull_underestimates.csv", index=False, encoding="utf-8-sig"
    )
    make_plots(ok)
    return metrics


def make_plots(frame: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = DATA_DIR / "plots"
    plots.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(frame["mp_energy_above_hull"], frame["chgnet_energy_above_hull"], s=18, alpha=0.7)
    high = max(frame["mp_energy_above_hull"].max(), frame["chgnet_energy_above_hull"].max())
    ax.plot([0, high], [0, high], "--", color="gray", linewidth=1)
    ax.set(xlabel="MP energy above hull (eV/atom)", ylabel="CHGNet energy above hull (eV/atom)")
    fig.tight_layout()
    fig.savefig(plots / "hull_parity.png", dpi=180)
    plt.close(fig)

    grouped = frame.groupby("chemsys")["hull_error"].apply(lambda x: x.abs().mean()).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    grouped.plot.barh(ax=ax)
    ax.set(xlabel="Hull MAE (eV/atom)", ylabel="Chemical system")
    fig.tight_layout()
    fig.savefig(plots / "hull_mae_by_chemsys.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    # Cached runs are intentionally offline: no API key should be needed when
    # the raw Materials Project snapshot is already present.
    api_key = None if args.use_cache and RAW_PATH.exists() else require_api_key()
    source = load_or_fetch(api_key, args.use_cache)
    print(f"Total phase-space structures: {len(source)}")
    predicted = predict_energies(source, args.device, args.batch_size)
    metrics = analyze(predicted)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
