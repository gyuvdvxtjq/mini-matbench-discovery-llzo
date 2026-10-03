"""Model layer: every MLIP is reduced to an ASE calculator.

Protocols never see a model object — they only see `ase.Atoms` with a
calculator attached. Each adapter imports its backend lazily so that a
missing optional dependency (mace, deepmd) disables one model instead of
crashing the audit.
"""

from __future__ import annotations

import os
from typing import Any

from ase import Atoms

from .config import ModelSpec

# Environment variable holding the DPA-4 (deepmd) checkpoint file. Weights are
# not committed (see .gitignore), so the path is supplied by the environment.
DPA4_CHECKPOINT_ENV = "DPA4_CHECKPOINT"

# model type -> (pip package providing it, setup script that installs it).
# Used to make a skip self-explanatory: without this a missing backend only
# says "No module named 'mace'", which reads like a bug in the audit.
BACKEND_SETUP: dict[str, tuple[str, str]] = {
    "chgnet": ("chgnet", "cloud/setup_base.sh"),
    "mace": ("mace-torch", "cloud/setup_mace.sh"),
    "deepmd": ("deepmd-kit", "cloud/setup_deepmd.sh"),
}


class ModelUnavailable(RuntimeError):
    """Raised when a model's optional dependency or checkpoint is missing."""

    def __init__(self, message: str, *, package: str = "", setup_script: str = ""):
        super().__init__(message)
        self.package = package
        self.setup_script = setup_script

    def report(self) -> str:
        """One-line reason for the log and for report.json's skipped_models."""
        parts = [str(self)]
        if self.package:
            parts.append(f"missing package: {self.package}")
        if self.setup_script:
            parts.append(f"install with: bash {self.setup_script}")
        return " | ".join(parts)


def detect_device() -> str:
    """Resolve ``auto`` to ``cuda`` when a GPU is usable, else ``cpu``.

    Configs ship without a ``device`` key so the same YAML runs on a laptop and
    on a Bohrium worker; an explicit ``--device`` on the CLI still wins.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # a broken CUDA driver must not abort the audit
        return "cpu"
    return "cpu"


def resolve_device(spec_device: str | None, override: str | None = None) -> str:
    """Final device for a model: CLI override > config value > auto-detection."""
    if override:
        return override
    device = (spec_device or "auto").strip().lower()
    if device in ("", "auto"):
        return detect_device()
    return device


def to_atoms(structure_dict: dict[str, Any]) -> Atoms:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = Structure.from_dict(structure_dict)
    return AseAtomsAdaptor.get_atoms(structure)


def scalar(value: Any) -> float:
    """Coerce a backend output (possibly a torch tensor) to a Python float."""
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(np.asarray(value).reshape(-1)[0])


def _unavailable(model_type: str, reason: str) -> ModelUnavailable:
    package, script = BACKEND_SETUP.get(model_type, ("", ""))
    return ModelUnavailable(reason, package=package, setup_script=script)


def _deepmd_checkpoint() -> str:
    path = os.getenv(DPA4_CHECKPOINT_ENV, "").strip()
    if not path:
        # deepmd-kit itself is installed here, so name the setup script but not
        # a missing package: the weights are what is absent.
        raise ModelUnavailable(
            f"{DPA4_CHECKPOINT_ENV} is not set: point it at the DPA-4 "
            f"checkpoint file (e.g. "
            f"{DPA4_CHECKPOINT_ENV}=models/dpa4/DPA4-Mini-OMat24-v20260805.pt). "
            "The .pt is not committed (CC-BY-NC-4.0, non-commercial); "
            "cloud/setup_deepmd.sh downloads it, see cloud/README.md.",
            setup_script="cloud/setup_deepmd.sh",
        )
    if not os.path.isfile(path):
        raise ModelUnavailable(
            f"{DPA4_CHECKPOINT_ENV} points at a missing file: {path}. "
            "Re-run cloud/setup_deepmd.sh or set the variable to a local copy.",
            setup_script="cloud/setup_deepmd.sh",
        )
    return path


def build_calculator(spec: ModelSpec):
    if spec.type == "chgnet":
        try:
            from chgnet.model import CHGNet, CHGNetCalculator
        except ImportError as exc:
            raise _unavailable("chgnet", f"chgnet not installed: {exc}") from exc
        model = CHGNet.load(use_device=spec.device)
        return CHGNetCalculator(model=model, use_device=spec.device)
    if spec.type == "mace":
        try:
            from mace.calculators import mace_mp
        except ImportError as exc:
            raise _unavailable("mace", f"mace-torch not installed: {exc}") from exc
        return mace_mp(
            model=spec.variant or "medium",
            device=spec.device,
            default_dtype=spec.dtype,
        )
    if spec.type == "deepmd":
        try:
            from deepmd.calculator import DP
        except ImportError as exc:
            raise _unavailable("deepmd", f"deepmd-kit not installed: {exc}") from exc
        return DP(_deepmd_checkpoint())
    raise _unavailable("", f"unknown model type: {spec.type}")
