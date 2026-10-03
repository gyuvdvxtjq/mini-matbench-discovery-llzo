#!/bin/bash
# Environment for DPA-4 (deepmd-kit). Idempotent; run from the repo root.
#
# Must NOT share an environment with cloud/setup_mace.sh: pip cannot resolve
# deepmd-kit and mace-torch together (ResolutionImpossible). Use this
# environment for the deepmd models only, and cloud/run_split.sh wires the
# two halves together through the shared checkpoint file.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

# ------------------------------------------------- DPA-4 frozen checkpoint
# The repo ships only the training-input .json under models/dpa4/; the frozen
# .pt is fetched here at job start. The weights are licensed CC-BY-NC-4.0
# (NON-COMMERCIAL) and are therefore not committed -- see .gitignore.
#
# The path is passed to the audit through the DPA4_CHECKPOINT environment
# variable, so the value must stay in sync with cloud/run_split.sh, which
# exports the same default.
DPA4_CHECKPOINT="${DPA4_CHECKPOINT:-models/dpa4/DPA4-Mini-OMat24-v20260805.pt}"
DPA4_HF_URL="${DPA4_HF_URL:-https://huggingface.co/deepmodeling/DPA4-Mini-OMat24/resolve/main/v20260805/DPA4-Mini-OMat24-v20260805.pt}"

bash "$(dirname "$0")/setup_base.sh"

if $PYTHON -c "import deepmd" >/dev/null 2>&1; then
    echo "deepmd-kit already present, skipping install"
else
    $PYTHON -m pip install --quiet "deepmd-kit"
fi

if [ -s "$DPA4_CHECKPOINT" ]; then
    echo "dpa4 checkpoint already present, skipping download"
elif ! curl -fsSL -o "$DPA4_CHECKPOINT" "$DPA4_HF_URL"; then
    echo "error: could not fetch the DPA-4 checkpoint from:" >&2
    echo "         $DPA4_HF_URL" >&2
    echo "       Set DPA4_HF_URL to your mirror, or point DPA4_CHECKPOINT at a" >&2
    echo "       local .pt. Without it the audit refuses to start: dpa4-mini-omat24" >&2
    echo "       is an explicit requirement of this environment, not an optional" >&2
    echo "       model, so silently skipping it would hide a failed job." >&2
    rm -f "$DPA4_CHECKPOINT"
    exit 1
fi

echo "DPA4_CHECKPOINT=$DPA4_CHECKPOINT"
$PYTHON -c "import deepmd; print('deepmd-kit ok')"
