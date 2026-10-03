#!/bin/bash
# Job entrypoint executed on the Bohrium worker, from the repo root.
#
#   bash cloud/run_job.sh
set -euo pipefail

bash cloud/setup_mace.sh
python3 run_audit.py --config configs/llzo.yaml "$@"
