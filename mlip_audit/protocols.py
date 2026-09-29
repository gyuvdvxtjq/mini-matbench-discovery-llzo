"""Protocols: single-point and relaxation evaluation over a structure set.

Both protocols iterate structures with per-record checkpointing, so an
interrupted run resumes exactly where it stopped and adding a model only
computes that model. Failures are recorded, never dropped.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from .checkpoint import CheckpointStore, make_key
from .models import to_atoms

MetaFn = Callable[[dict[str, Any]], dict[str, Any]]


def _max_force(atoms) -> float:
    return float(np.linalg.norm(atoms.get_forces(), axis=1).max())


def _meta(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "structure"}


def run_single_point(
    records: list[dict[str, Any]],
    calculator,
    store: CheckpointStore,
    model_name: str,
) -> list[dict[str, Any]]:
    """Energy/force/stress at the reference (MP-relaxed) geometry."""
    params = {"kind": "single_point"}
    output = []
    for index, record in enumerate(records, start=1):
        key = make_key(model_name, "single_point", params, record["material_id"])
        cached = store.get_ok(key)
        if cached:
            output.append(cached)
            continue
        started = time.perf_counter()
        try:
            atoms = to_atoms(record["structure"])
            atoms.calc = calculator
            result = {
                **_meta(record),
                "status": "ok",
                "energy_per_atom": float(atoms.get_potential_energy()) / len(atoms),
                "max_force": _max_force(atoms),
                "rms_force": float(
                    np.sqrt(np.mean(np.square(atoms.get_forces())))
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": "",
            }
        except Exception as exc:  # failures are benchmark evidence
            result = {
                **_meta(record),
                "status": "error",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        store.append(key, result)
        output.append(result)
        if index % 10 == 0 or index == len(records):
            print(f"  single_point {model_name}: {index}/{len(records)}", flush=True)
    return output


def run_relaxation(
    records: list[dict[str, Any]],
    calculator_factory: Callable[[], Any],
    store: CheckpointStore,
    model_name: str,
    *,
    fmax: float,
    steps: int,
    relax_cell: bool,
) -> list[dict[str, Any]]:
    """Relax each structure on the model's PES (FIRE + FrechetCellFilter)."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    params = {"kind": "relax", "fmax": fmax, "steps": steps, "relax_cell": relax_cell}
    calculator = calculator_factory()  # one calculator reused across structures
    output = []
    for index, record in enumerate(records, start=1):
        key = make_key(model_name, "relaxation", params, record["material_id"])
        cached = store.get_ok(key)
        if cached:
            output.append(cached)
            continue
        started = time.perf_counter()
        try:
            atoms = to_atoms(record["structure"])
            atoms.calc = calculator
            target = FrechetCellFilter(atoms) if relax_cell else atoms
            history: list[tuple[float, float]] = [
                (float(atoms.get_potential_energy()) / len(atoms), _max_force(atoms))
            ]

            def _observe() -> None:
                history.append(
                    (float(atoms.get_potential_energy()) / len(atoms), _max_force(atoms))
                )

            optimizer = FIRE(target, maxstep=0.2, logfile=None)
            optimizer.attach(_observe)
            optimizer.run(fmax=fmax, steps=steps)
            initial_e, initial_f = history[0]
            final_e, final_f = history[-1]
            result = {
                **_meta(record),
                "status": "ok",
                "initial_e_per_atom": initial_e,
                "relaxed_e_per_atom": final_e,
                "relax_drop_per_atom": initial_e - final_e,
                "initial_max_force": initial_f,
                "final_max_force": final_f,
                "ionic_steps": len(history) - 1,
                "converged": bool(final_f <= fmax),
                "fmax_target": fmax,
                "relax_cell": relax_cell,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": "",
            }
        except Exception as exc:
            result = {
                **_meta(record),
                "status": "error",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        store.append(key, result)
        output.append(result)
        print(
            f"  relax {model_name}: {index}/{len(records)} "
            f"{record['material_id']} {result['status']}",
            flush=True,
        )
    return output
