"""Declarative audit configuration (YAML) schema and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repository root, derived from this file so that configs can spell out
# repo-relative paths ("data/runs/llzo") without depending on the caller's
# working directory. Tests pass absolute paths and are unaffected.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Placeholder device meaning "cuda if a GPU is usable, otherwise cpu".
AUTO_DEVICE = "auto"


def repo_path(value: str | Path) -> Path:
    """Resolve a possibly-relative path against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass
class ModelSpec:
    name: str
    type: str  # chgnet | mace | deepmd
    enabled: bool = True
    device: str = AUTO_DEVICE
    variant: str = ""  # e.g. mace model size
    checkpoint: str = ""  # deepmd model file, local path or hf:// repo
    dtype: str = "float64"


@dataclass
class RelaxSpec:
    enabled: bool = True
    fmax: float = 0.1
    steps: int = 100
    relax_cell: bool = True
    optimizer: str = "FIRE"


@dataclass
class MDSpec:
    enabled: bool = False
    material_id: str = ""
    supercell: tuple[int, int, int] = (1, 1, 1)
    temperatures: list[int] = field(default_factory=lambda: [1000])
    equil_ps: float = 5.0
    prod_ps: float = 20.0
    timestep_fs: float = 1.0
    sample_every: int = 20


@dataclass
class AuditConfig:
    name: str
    elements: list[str]
    models: list[ModelSpec]
    relax: RelaxSpec
    md: MDSpec
    single_point: bool = True
    candidate_window_ev: float = 0.05
    bootstrap_resamples: int = 2000
    mp_cache: Path = Path("data/raw/mp_phase_space.json")
    out_dir: Path = Path("data/runs/audit")


def _model(raw: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        name=raw["name"],
        type=raw["type"],
        enabled=bool(raw.get("enabled", True)),
        device=str(raw.get("device", AUTO_DEVICE)),
        variant=str(raw.get("variant", "")),
        checkpoint=str(raw.get("checkpoint", "")),
        dtype=str(raw.get("dtype", "float64")),
    )


def load_config(path: str | Path) -> AuditConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    relax_raw = raw.get("relaxation", {}) or {}
    md_raw = raw.get("md", {}) or {}
    return AuditConfig(
        name=raw["name"],
        elements=list(raw["elements"]),
        models=[_model(m) for m in raw.get("models", [])],
        relax=RelaxSpec(
            enabled=bool(relax_raw.get("enabled", True)),
            fmax=float(relax_raw.get("fmax", 0.1)),
            steps=int(relax_raw.get("steps", 100)),
            relax_cell=bool(relax_raw.get("relax_cell", True)),
            optimizer=str(relax_raw.get("optimizer", "FIRE")),
        ),
        md=MDSpec(
            enabled=bool(md_raw.get("enabled", False)),
            material_id=str(md_raw.get("material_id", "")),
            supercell=tuple(md_raw.get("supercell", (1, 1, 1))),
            temperatures=[int(t) for t in md_raw.get("temperatures", [1000])],
            equil_ps=float(md_raw.get("equil_ps", 5.0)),
            prod_ps=float(md_raw.get("prod_ps", 20.0)),
            timestep_fs=float(md_raw.get("timestep_fs", 1.0)),
            sample_every=int(md_raw.get("sample_every", 20)),
        ),
        single_point=bool(raw.get("single_point", True)),
        candidate_window_ev=float(raw.get("candidate_window_ev", 0.05)),
        bootstrap_resamples=int(raw.get("bootstrap_resamples", 2000)),
        mp_cache=repo_path(raw.get("mp_cache", "data/raw/mp_phase_space.json")),
        out_dir=repo_path(raw.get("out_dir", "data/runs/audit")),
    )
