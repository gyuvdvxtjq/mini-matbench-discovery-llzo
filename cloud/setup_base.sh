#!/bin/bash
# Common worker-side environment, shared by every model environment.
# Run from the repo root (cloud/run_split.sh does this for you).
#
# ---------------------------------------------------------------------------
# WHY THE SETUP IS SPLIT ACROSS THREE SCRIPTS
# ---------------------------------------------------------------------------
# mace-torch and deepmd-kit cannot be resolved by pip in the same environment
# (ResolutionImpossible: their torch / CUDA / numpy pins are mutually
# exclusive). That is the recorded root cause of DPA-4 being reported under
# `skipped_models` in the 2026-09-29 run (see data/runs/llzo/NOTES.md).
#
# The fix is NOT to make the conflict resolvable -- it is to stop trying:
# run the models in two separate environments, each with its own setup script,
# and let both write to the same checkpoint file. The audit is already
# checkpointed by (model, protocol, params-hash, material_id), so a record
# computed in the mace environment is simply skipped when the deepmd
# environment replays the same config. See cloud/run_split.sh.
#
# This script installs only what both environments agree on. It is idempotent:
# re-running skips anything already importable, so it is safe as the first
# step of either cloud/setup_mace.sh or cloud/setup_deepmd.sh.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------- core stack
# The two backends that ran for LLZO in the reference run, plus the
# scientific/data stack that mlip_audit imports unconditionally.
# Single source of truth: pyproject.toml ([project].dependencies). The pins
# here mirror cloud/setup.sh of the reference run and may be relaxed to
# "pip install -e ." once packaging exists.
$PYTHON -m pip install --quiet --upgrade pip

if $PYTHON -c "import chgnet, mp_api, dotenv, yaml, ase, pymatgen.core" >/dev/null 2>&1; then
    echo "core stack already present, skipping install"
else
    $PYTHON -m pip install --quiet \
        "chgnet==0.4.2" \
        "mp-api==0.46.5" \
        "python-dotenv==1.1.1" \
        "pyyaml>=6.0" \
        "ase>=3.23" \
        "pymatgen>=2024.1.1" \
        "torch==2.5.1" \
        "numpy" \
        "pandas" \
        "scipy" \
        "matplotlib"
fi

# ------------------------------------------------------------------ sanity
# Every backend is optional: a model that cannot be loaded is reported under
# `skipped_models` in report.json and never aborts the run.
$PYTHON - <<'PY'
import importlib.util
for name, module in {"chgnet": "chgnet", "mace": "mace", "deepmd": "deepmd"}.items():
    found = importlib.util.find_spec(module) is not None
    print(f"backend {name}: {'ok' if found else 'MISSING (model will be skipped)'}")
PY
