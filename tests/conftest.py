"""Shared test scaffolding: a stub MLIP and a synthetic MP snapshot.

The pipeline tests exercise the real config -> protocols -> checkpoint ->
report path without downloading a model or calling the Materials Project API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.calculators.calculator import Calculator, all_changes

from mlip_audit.config import AuditConfig, MDSpec, ModelSpec, RelaxSpec


class StubCalc(Calculator):
    """Deterministic stand-in for a real MLIP.

    Energy is a fixed term plus a small position-dependent term, so structures
    have a (crude but real) potential energy surface: relaxation actually moves
    atoms. Each model name maps to a different offset, so two models produce
    different -- and reproducible -- hull errors without any model weights.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    #: Per-model energy offset (eV/atom), so models are distinguishable.
    OFFSETS = {"m-a": 0.0, "m-b": 0.25}

    def __init__(self, model: str = "m", **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.calls = 0

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        energy = -1.0 * len(atoms) + 0.01 * float(np.sum(atoms.get_positions()))
        energy += self.OFFSETS.get(self.model, 0.1) * len(atoms)
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": np.zeros((len(atoms), 3)),
            "stress": np.zeros((3, 3)),
        }


def write_synthetic_snapshot(path: Path) -> Path:
    """A tiny Li-O snapshot: both elements plus mixed compositions.

    A phase diagram needs every element present as an endpoint, hence the
    three Li and two O entries.
    """
    from pymatgen.core import Lattice, Structure

    formulas = ["Li", "Li", "Li", "O", "O", "Li2O", "LiO2", "LiO"]
    records = []
    for index, formula in enumerate(formulas):
        species = ["Li"] * formula.count("Li") + ["O"] * formula.count("O")
        structure = Structure(
            Lattice.cubic(3.0 + 0.1 * index), species, [[0.0, 0.0, 0.0]] * len(species)
        )
        records.append(
            {
                "material_id": f"mp-t{index:03d}",
                "formula": formula,
                "chemsys": "-".join(sorted({"Li", "O"} & set(species))),
                "nsites": len(species),
                "mp_energy_per_atom": -3.0 - 0.1 * index,
                "mp_formation_energy_per_atom": -1.0 - 0.05 * index,
                "mp_energy_above_hull": 0.02 * index,
                "mp_is_stable": index == 0,
                "structure": structure.as_dict(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def checkpoint_counts(run_dir: str | Path) -> dict[str, list[dict]]:
    """Checkpoint records grouped by model, newest write per key wins."""
    grouped: dict[str, dict[str, dict]] = {}
    for line in (Path(run_dir) / "checkpoints.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        if line.strip():
            record = json.loads(line)
            grouped.setdefault(record["_key"].split("|", 1)[0], {})[record["_key"]] = record
    return {
        model: sorted(records.values(), key=lambda r: r["_key"])
        for model, records in grouped.items()
    }


def make_config(
    cache: Path, out_dir: Path, models: list[str], *, relax: bool = False
) -> AuditConfig:
    return AuditConfig(
        name="test",
        elements=["Li", "O"],
        models=[ModelSpec(name=name, type="chgnet") for name in models],
        relax=RelaxSpec(enabled=relax, steps=2),
        md=MDSpec(enabled=False),
        single_point=True,
        bootstrap_resamples=20,
        mp_cache=cache,
        out_dir=out_dir,
    )


@pytest.fixture
def synthetic_snapshot(tmp_path: Path) -> Path:
    return write_synthetic_snapshot(tmp_path / "mp.json")


@pytest.fixture
def stub_backend(monkeypatch):
    """Patch build_calculator in the runner; returns the list of built models."""
    built: list[str] = []

    def _build(spec: ModelSpec):
        built.append(spec.name)
        return StubCalc(spec.name)

    monkeypatch.setattr("mlip_audit.runner.build_calculator", _build)
    return built


@pytest.fixture
def stub_config(synthetic_snapshot: Path):
    """Factory for an AuditConfig pointing at the synthetic snapshot."""

    def _make(out_dir: Path, models: list[str], *, relax: bool = False) -> AuditConfig:
        return make_config(synthetic_snapshot, out_dir, models, relax=relax)

    return _make
