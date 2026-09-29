"""Model layer: every MLIP is reduced to an ASE calculator.

Protocols never see a model object — they only see `ase.Atoms` with a
calculator attached. Each adapter imports its backend lazily so that a
missing optional dependency (mace, deepmd) disables one model instead of
crashing the audit.
"""

from __future__ import annotations

from typing import Any

from ase import Atoms

from .config import ModelSpec


class ModelUnavailable(RuntimeError):
    """Raised when a model's optional dependency or checkpoint is missing."""


def to_atoms(structure_dict: dict[str, Any]) -> Atoms:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = Structure.from_dict(structure_dict)
    return AseAtomsAdaptor.get_atoms(structure)


def build_calculator(spec: ModelSpec):
    if spec.type == "chgnet":
        try:
            from chgnet.model import CHGNet, CHGNetCalculator
        except ImportError as exc:
            raise ModelUnavailable(f"chgnet not installed: {exc}") from exc
        model = CHGNet.load(use_device=spec.device)
        return CHGNetCalculator(model=model, use_device=spec.device)
    if spec.type == "mace":
        try:
            from mace.calculators import mace_mp
        except ImportError as exc:
            raise ModelUnavailable(f"mace-torch not installed: {exc}") from exc
        return mace_mp(
            model=spec.variant or "medium",
            device=spec.device,
            default_dtype=spec.dtype,
        )
    if spec.type == "deepmd":
        try:
            from deepmd.calculator import DP
        except ImportError as exc:
            raise ModelUnavailable(f"deepmd-kit not installed: {exc}") from exc
        if not spec.checkpoint:
            raise ModelUnavailable("deepmd model requires a checkpoint path")
        return DP(spec.checkpoint)
    raise ModelUnavailable(f"unknown model type: {spec.type}")
