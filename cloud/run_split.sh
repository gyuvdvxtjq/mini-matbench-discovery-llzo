#!/bin/bash
# Run one half of the LLZO audit in its own environment, then merge results.
#
#   bash cloud/run_split.sh --env mace     # chgnet-0.3.0 + mace-mp-0-medium
#   bash cloud/run_split.sh --env deepmd   # dpa4-mini-omat24
#
# WHY TWO ENVIRONMENTS: mace-torch and deepmd-kit are mutually unresolvable by
# pip (ResolutionImpossible), which is why the 2026-09-29 reference run
# reported DPA-4 under `skipped_models` (data/runs/llzo/NOTES.md). Rather than
# fight the resolver, each half gets its own environment.
#
# WHY THE TWO HALVES STILL ADD UP: every record is checkpointed in
# data/runs/llzo/checkpoints.jsonl under the key
# `model|protocol|params-hash|material_id`, which does NOT depend on which
# other models were in the same process. The second invocation therefore skips
# every record the first one already computed and only fills the missing
# model. Both invocations write the same checkpoint file; merge the job
# results by copying it back (see cloud/README.md).
#
# Usage:
#   --env mace|deepmd   which environment to build and run (required)
#   --config PATH       audit config (default: configs/llzo.yaml)
#   --limit N           debug: first N structures only
#   --device DEV        force cpu|cuda|mps (default: auto, see ModelSpec)
#   --setup-only        install the environment, do not run the audit
set -euo pipefail

CONFIG="configs/llzo.yaml"
ENV_NAME=""
LIMIT=()
DEVICE=()
SETUP_ONLY=0

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --env) ENV_NAME="${2:-}"; shift 2 ;;
        --config) CONFIG="${2:-}"; shift 2 ;;
        --limit) LIMIT=(--limit "${2:-}"); shift 2 ;;
        --device) DEVICE=(--device "${2:-}"); shift 2 ;;
        --setup-only) SETUP_ONLY=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

if [ -z "$ENV_NAME" ]; then
    echo "error: --env mace|deepmd is required" >&2
    usage 1
fi

cd "$(dirname "$0")/.."

# DPA-4 weights are read from the environment by mlip_audit/models.py; keep the
# default in sync with cloud/setup_deepmd.sh.
export DPA4_CHECKPOINT="${DPA4_CHECKPOINT:-models/dpa4/DPA4-Mini-OMat24-v20260805.pt}"

case "$ENV_NAME" in
    mace)
        # CHGNet lives in the base stack; MACE needs its own install.
        MODELS=(chgnet-0.3.0 mace-mp-0-medium)
        bash cloud/setup_mace.sh
        ;;
    deepmd)
        MODELS=(dpa4-mini-omat24)
        bash cloud/setup_deepmd.sh
        ;;
    *)
        echo "error: unknown --env '$ENV_NAME' (expected mace or deepmd)" >&2
        exit 1
        ;;
esac

if [ "$SETUP_ONLY" -eq 1 ]; then
    echo "setup complete for env=$ENV_NAME; not running the audit (--setup-only)"
    exit 0
fi

echo "== running env=$ENV_NAME models: ${MODELS[*]} =="
python3 run_audit.py --config "$CONFIG" --model "${MODELS[@]}" "${LIMIT[@]+"${LIMIT[@]}"}" "${DEVICE[@]+"${DEVICE[@]}"}"

echo
echo "== done. Merge the two halves with: =="
echo "   python3 scripts/merge_and_plot.py --run-dir data/runs/llzo"
