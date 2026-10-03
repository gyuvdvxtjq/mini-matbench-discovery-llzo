#!/bin/bash
# Environment for CHGNet + MACE-MP-0. Idempotent; run from the repo root.
#
# Must NOT share an environment with cloud/setup_deepmd.sh: pip cannot
# resolve mace-torch and deepmd-kit together (their torch/numpy/CUDA pins are
# mutually exclusive, ResolutionImpossible). That conflict is why DPA-4 was
# skipped in the reference run. Install one, run it, then use the other
# environment for DPA-4 -- the shared checkpoint file makes the two halves
# of the audit add up.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

bash "$(dirname "$0")/setup_base.sh"

if $PYTHON -c "import mace" >/dev/null 2>&1; then
    echo "mace-torch already present, skipping install"
else
    $PYTHON -m pip install --quiet "mace-torch>=0.3.0"
fi

$PYTHON -c "import mace; print('mace-torch ok')"
