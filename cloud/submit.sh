#!/bin/bash
# Submit the LLZO audit as a Bohrium batch job.
#
# The job runs offline: .env (MP_API_KEY) is never uploaded, so the worker
# reads the cached Materials Project snapshot data/raw/mp_phase_space.json.
# Billing is per actual runtime, so the worker stops when the job exits.
#
# Usage:
#   DRY_RUN=1 bash cloud/submit.sh   # validate the input tree only (free)
#   bash cloud/submit.sh             # real submission
#   SKU_ID=<id> bash cloud/submit.sh # override the machine SKU
#
# Defaults come from cloud/README.md: SKU c16_m64_1xNVIDIA 4090 (yuan 6/h),
# image ubuntu:22.04-py3.10-cuda12.1. Confirm the exact submit flags against
# `bohr batchjob submit --help` on your Bohrium CLI version.
set -euo pipefail

# ------------------------------------------------------------------ config
SKU_ID="${SKU_ID:-9985}"
IMAGE="${IMAGE:-ubuntu:22.04-py3.10-cuda12.1}"
JOB_NAME="${JOB_NAME:-llzo-mlip-audit}"
INPUT_DIR="${INPUT_DIR:-.}"
CMD="bash cloud/run_job.sh"

# --------------------------------------------------------------- validate
required=(
    "run_audit.py"
    "cloud/run_job.sh"
    "cloud/setup.sh"
    "configs/llzo.yaml"
    "data/raw/mp_phase_space.json"
)
echo "== validating input tree =="
missing=0
for path in "${required[@]}"; do
    if [ -s "$INPUT_DIR/$path" ]; then
        echo "  ok   $path"
    else
        echo "  MISS $path"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "Refusing to submit: the worker runs offline and would fail" >&2
    echo "without the cached MP snapshot. Fetch it locally first" >&2
    echo "(python phase_space_benchmark.py --fetch-only) then re-submit." >&2
    exit 1
fi
if [ -f "$INPUT_DIR/.env" ]; then
    echo "  warn .env is present: it must be excluded from the upload" >&2
    echo "       (the job reads the cached snapshot, not your API key)" >&2
fi

# ------------------------------------------------------------------ submit
if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "== dry run (no job submitted, no balance charged) =="
    echo "  sku-id : $SKU_ID"
    echo "  image  : $IMAGE"
    echo "  name   : $JOB_NAME"
    echo "  cmd    : $CMD"
    echo "  input  : $INPUT_DIR"
    echo "Next: bash cloud/submit.sh"
    exit 0
fi

echo "== submitting =="
bohr auth whoami >/dev/null
bohr batchjob submit \
    --sku-id "$SKU_ID" \
    --image "$IMAGE" \
    --job-name "$JOB_NAME" \
    --cmd "$CMD" \
    "$INPUT_DIR"

echo "Submitted. Then:"
echo "  bohr batchjob list --status pending,running -o json"
echo "  bohr batchjob wait <job_id> --interval 30s --timeout 8h"
echo "  bohr batchjob download <job_id> --dest ./job_result"
echo "Copy job_result/data/runs/llzo/checkpoints.jsonl over the local one to"
echo "resume a follow-up job from these results."
