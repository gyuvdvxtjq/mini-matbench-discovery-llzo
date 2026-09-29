"""MD protocol: Li-ion diffusivity from NVT-equilibrated NVE production runs.

Equilibration uses a Langevin thermostat; the production leg runs NVE so the
diffusion constant is not damped by thermostat friction. One record per
temperature, checkpointed; trajectories go to .traj files.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import CheckpointStore, make_key
from .config import MDSpec
from .models import to_atoms

KB_EV_PER_K = 8.617333262e-5


def _run_one_temperature(
    atoms0, calculator_factory, temperature: int, spec: MDSpec, traj_path: Path
) -> dict[str, Any]:
    from ase import units
    from ase.io import Trajectory
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    atoms = atoms0.copy()
    atoms.calc = calculator_factory()
    dt = spec.timestep_fs * units.fs
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    equil_steps = int(spec.equil_ps * 1000 / spec.timestep_fs)
    prod_steps = int(spec.prod_ps * 1000 / spec.timestep_fs)
    dyn = Langevin(atoms, dt, temperature_K=temperature, friction=0.01 / units.fs)
    dyn.run(equil_steps)

    traj_path.parent.mkdir(parents=True, exist_ok=True)
    traj = Trajectory(str(traj_path), "w", atoms)
    prod = VelocityVerlet(atoms, dt)
    prod.attach(traj.write, interval=spec.sample_every)
    prod.run(prod_steps)
    traj.close()
    return {
        "temperature": temperature,
        "equil_steps": equil_steps,
        "prod_steps": prod_steps,
        "trajectory": str(traj_path),
        "n_frames": prod_steps // spec.sample_every + 1,
    }


def run_md(
    record: dict[str, Any],
    calculator_factory,
    store: CheckpointStore,
    model_name: str,
    spec: MDSpec,
    out_dir: Path,
) -> list[dict[str, Any]]:
    params = {
        "kind": "md",
        "temps": spec.temperatures,
        "equil_ps": spec.equil_ps,
        "prod_ps": spec.prod_ps,
        "timestep_fs": spec.timestep_fs,
        "supercell": spec.supercell,
    }
    atoms0 = to_atoms(record["structure"]) * spec.supercell
    output = []
    for temperature in spec.temperatures:
        key = make_key(model_name, "md", params, f"{record['material_id']}@{temperature}K")
        cached = store.get_ok(key)
        if cached:
            output.append(cached)
            continue
        started = time.perf_counter()
        traj_path = out_dir / "trajectories" / f"{model_name}_{temperature}K.traj"
        try:
            detail = _run_one_temperature(
                atoms0, calculator_factory, temperature, spec, traj_path
            )
            result = {
                "material_id": record["material_id"],
                "formula": record["formula"],
                "n_atoms": len(atoms0),
                "status": "ok",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": "",
                **detail,
            }
        except Exception as exc:
            result = {
                "material_id": record["material_id"],
                "temperature": temperature,
                "status": "error",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "error": f"{type(exc).__name__}: {exc}",
            }
        store.append(key, result)
        output.append(result)
        print(f"  md {model_name} @{temperature}K: {result['status']}", flush=True)
    return output


def msd_from_trajectory(traj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Time-lag axis (frames) and mean-squared displacement (Å²)."""
    from ase.io import read

    frames = read(str(traj_path), index=":")
    positions = np.array([atoms.get_positions() for atoms in frames])
    n_frames = len(positions)
    msd = np.array(
        [
            np.mean(np.sum((positions[lag:] - positions[:-lag]) ** 2, axis=2))
            for lag in range(1, n_frames)
        ]
    )
    return np.arange(1, n_frames), msd


def analyze_md(
    records: list[dict[str, Any]], spec: MDSpec, out_dir: Path
) -> dict[str, Any]:
    """Diffusivity per temperature from the MSD slope, then Arrhenius fit."""
    ok = [r for r in records if r.get("status") == "ok"]
    dt_sample_ps = spec.sample_every * spec.timestep_fs / 1000.0
    points = []
    for record in ok:
        lags, msd = msd_from_trajectory(Path(record["trajectory"]))
        times_ps = lags * dt_sample_ps
        lo, hi = int(0.2 * len(times_ps)), int(0.8 * len(times_ps))
        slope, intercept = np.polyfit(times_ps[lo:hi], msd[lo:hi], 1)  # Å²/ps
        fitted = slope * times_ps[lo:hi] + intercept
        ss_res = float(np.sum((msd[lo:hi] - fitted) ** 2))
        ss_tot = float(np.sum((msd[lo:hi] - msd[lo:hi].mean()) ** 2))
        points.append(
            {
                "temperature": record["temperature"],
                "diffusivity_cm2_per_s": float(slope / 6 * 1e-4),  # Å²/ps → cm²/s
                "msd_final_A2": float(msd[-1]),
                "msd_fit_r2": 1 - ss_res / ss_tot if ss_tot > 0 else None,
            }
        )
    result: dict[str, Any] = {"points": points}
    if len(points) >= 3:
        temps = np.array([p["temperature"] for p in points], dtype=float)
        diff = np.array([p["diffusivity_cm2_per_s"] for p in points])
        slope, intercept = np.polyfit(1.0 / temps, np.log(diff), 1)
        result["arrhenius"] = {
            "activation_energy_eV": float(-slope * KB_EV_PER_K),
            "prefactor_cm2_per_s": float(np.exp(intercept)),
            "fit_temperatures": temps.tolist(),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / "md_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
