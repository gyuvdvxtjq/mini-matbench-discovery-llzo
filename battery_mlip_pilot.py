"""Evaluate CHGNet relaxation energies on Materials Project LLZO structures."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

CHEMSYS = "Li-La-Zr-O"
DEFAULT_LIMIT = 20
DATA_DIR = Path("data")
RAW_PATH = DATA_DIR / "raw" / "mp_llzo.json"
CHECKPOINT_PATH = DATA_DIR / "checkpoints" / "relaxations.jsonl"
RESULT_PATH = DATA_DIR / "chgnet_vs_mp_llzo.csv"
METRICS_PATH = DATA_DIR / "metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-sites", type=int, default=120)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--fmax", type=float, default=0.1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--no-relax-cell", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Ignore prior checkpoints.")
    return parser.parse_args()


def require_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("MP_API_KEY", "").strip()
    if not api_key or api_key == "replace_with_your_key":
        raise RuntimeError("MP_API_KEY is missing. Copy .env.example to .env and set it.")
    return api_key


def fetch_mp_structures(api_key: str, limit: int, max_sites: int) -> list[dict[str, Any]]:
    """Fetch exact four-element LLZO records and persist a reproducible local cache."""
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
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            chemsys=[CHEMSYS],
            num_sites=(1, max_sites),
            fields=fields,
            num_chunks=1,
            chunk_size=max(limit, 100),
        )

    docs = sorted(docs, key=lambda doc: (doc.nsites, str(doc.material_id)))[:limit]
    records = [
        {
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "chemsys": doc.chemsys,
            "nsites": int(doc.nsites),
            "mp_energy_per_atom": float(doc.energy_per_atom),
            "mp_formation_energy_per_atom": float(doc.formation_energy_per_atom),
            "mp_energy_above_hull": float(doc.energy_above_hull),
            "mp_is_stable": bool(doc.is_stable),
            "structure": doc.structure.as_dict(),
        }
        for doc in docs
    ]
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def load_completed() -> dict[str, dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    for line in CHECKPOINT_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            completed[record["material_id"]] = record
    return completed


def append_checkpoint(record: dict[str, Any]) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def relax_records(
    records: list[dict[str, Any]],
    *,
    device: str,
    steps: int,
    fmax: float,
    relax_cell: bool,
    fresh: bool,
) -> list[dict[str, Any]]:
    from chgnet.model import CHGNet, StructOptimizer
    from pymatgen.core import Structure

    if fresh and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    completed = {} if fresh else load_completed()
    model = CHGNet.load()
    optimizer = StructOptimizer(model=model, use_device=device)

    output: list[dict[str, Any]] = []
    for index, source in enumerate(records, start=1):
        material_id = source["material_id"]
        if material_id in completed and completed[material_id].get("status") == "ok":
            print(f"[{index}/{len(records)}] {material_id} resumed")
            output.append(completed[material_id])
            continue

        started = time.perf_counter()
        base = {key: value for key, value in source.items() if key != "structure"}
        base.update(
            {
                "model": "CHGNet-0.3.0",
                "device": device,
                "max_steps": steps,
                "fmax_target": fmax,
                "relax_cell": relax_cell,
            }
        )
        try:
            structure = Structure.from_dict(source["structure"])
            initial = model.predict_structure(structure, task="e")
            result = optimizer.relax(
                structure,
                fmax=fmax,
                steps=steps,
                relax_cell=relax_cell,
                verbose=False,
            )
            final_structure = result["final_structure"]
            trajectory = result["trajectory"]
            final_forces = np.asarray(trajectory.forces[-1], dtype=float)
            max_force = float(np.linalg.norm(final_forces, axis=1).max())
            record = {
                **base,
                "status": "ok",
                "chgnet_initial_e_per_atom": scalar(initial["e"]),
                "chgnet_relaxed_e_per_atom": float(trajectory.energies[-1])
                / len(final_structure),
                "ionic_steps": len(trajectory.energies) - 1,
                "final_max_force": max_force,
                "converged": max_force <= fmax,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": "",
            }
        except Exception as exc:  # retain failures as benchmark evidence
            record = {
                **base,
                "status": "error",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        append_checkpoint(record)
        output.append(record)
        print(
            f"[{index}/{len(records)}] {material_id} {record['status']} "
            f"({record['elapsed_seconds']:.1f}s)"
        )
    return output


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    DATA_DIR.mkdir(exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No successful relaxation is available for analysis.")

    ok["energy_error_eV_per_atom"] = (
        ok["chgnet_relaxed_e_per_atom"] - ok["mp_energy_per_atom"]
    )
    # All rows share one chemsys (La-Li-O-Zr), so elemental reference offsets
    # cancel to first order; this is a relative comparison, not an absolute
    # cross-chemsys energy accuracy claim.
    ok["abs_energy_error_eV_per_atom"] = ok["energy_error_eV_per_atom"].abs()
    for column in ("energy_error_eV_per_atom", "abs_energy_error_eV_per_atom"):
        frame.loc[ok.index, column] = ok[column]
    frame.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    # Rank correlation is meaningless on tiny samples; require enough points
    # before reporting it (the exact LLZO query currently returns only 3).
    if len(ok) >= 5:
        spearman = ok["chgnet_relaxed_e_per_atom"].corr(
            ok["mp_energy_per_atom"], method="spearman"
        )
        spearman_block: dict[str, Any] = {
            "spearman_energy": None if pd.isna(spearman) else float(spearman),
        }
    else:
        spearman_block = {
            "spearman_energy": None,
            "spearman_note": f"not computed: n={len(ok)} < 5",
        }
    metrics = {
        "chemical_system": CHEMSYS,
        "requested_limit": DEFAULT_LIMIT,
        "n_total": int(len(frame)),
        "n_success": int(len(ok)),
        "n_failed": int(len(frame) - len(ok)),
        "n_converged": int(ok["converged"].sum()),
        "mae_eV_per_atom": float(ok["abs_energy_error_eV_per_atom"].mean()),
        "rmse_eV_per_atom": float(
            np.sqrt(np.mean(np.square(ok["energy_error_eV_per_atom"])))
        ),
        "spearman_energy": spearman_block["spearman_energy"],
        "spearman_note": spearman_block.get("spearman_note", ""),
        # Raw total energies are only compared within the single La-Li-O-Zr
        # elemental space, where both energies share the same elemental
        # references; cross-chemsys conclusions use formation energies and
        # hull distances from the phase-space benchmark instead.
        "energy_comparison_note": (
            "same-chemsys relative comparison; "
            "stability conclusions rely on phase_space E-hull"
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ok.nlargest(3, "abs_energy_error_eV_per_atom").to_csv(
        DATA_DIR / "largest_errors.csv", index=False, encoding="utf-8-sig"
    )
    make_plots(ok)
    return metrics


def make_plots(frame: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = DATA_DIR / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(frame["mp_energy_per_atom"], frame["chgnet_relaxed_e_per_atom"], s=55)
    low = min(frame["mp_energy_per_atom"].min(), frame["chgnet_relaxed_e_per_atom"].min())
    high = max(frame["mp_energy_per_atom"].max(), frame["chgnet_relaxed_e_per_atom"].max())
    ax.plot([low, high], [low, high], "--", color="gray", linewidth=1)
    ax.set(xlabel="MP DFT energy (eV/atom)", ylabel="CHGNet relaxed energy (eV/atom)")
    fig.tight_layout()
    fig.savefig(plots / "energy_parity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(frame["material_id"], frame["abs_energy_error_eV_per_atom"])
    ax.set(ylabel="Absolute error (eV/atom)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots / "absolute_errors.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    records = fetch_mp_structures(require_api_key(), args.limit, args.max_sites)
    print(f"Materials Project returned {len(records)} exact {CHEMSYS} structures.")
    if not records:
        raise RuntimeError("No matching Materials Project structures were returned.")
    if args.fetch_only:
        return
    relaxed = relax_records(
        records,
        device=args.device,
        steps=args.steps,
        fmax=args.fmax,
        relax_cell=not args.no_relax_cell,
        fresh=args.fresh,
    )
    metrics = analyze(relaxed)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
