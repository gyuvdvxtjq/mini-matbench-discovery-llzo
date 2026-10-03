#!/bin/bash
# Common worker-side environment: the two backends plus the scientific stack.
# Run from the repo root (cloud/run_job.sh does this for you).
#
# This script installs everything the audit needs. It is idempotent: re-running
# skips anything already importable, so it is safe as the first step of
# cloud/setup_mace.sh.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------- core stack
# The two backends that ran for LLZO in the reference run, plus the
# scientific/data stack that mlip_audit imports unconditionally.
# Single source of truth: pyproject.toml ([project].dependencies).
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
# MACE is optional: if it cannot be loaded it is reported under
# `skipped_models` in report.json and never aborts the run.
$PYTHON - <<'PY'
import importlib.util
for name, module in {"chgnet": "chgnet", "mace": "mace"}.items():
    found = importlib.util.find_spec(module) is not None
    print(f"backend {name}: {'ok' if found else 'MISSING (model will be skipped)'}")
PY
