#!/bin/bash
# Environment for CHGNet + MACE-MP-0. Idempotent; run from the repo root.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

bash "$(dirname "$0")/setup_base.sh"

if $PYTHON -c "import mace" >/dev/null 2>&1; then
    echo "mace-torch already present, skipping install"
else
    $PYTHON -m pip install --quiet "mace-torch>=0.3.0"
fi

$PYTHON -c "import mace; print('mace-torch ok')"
