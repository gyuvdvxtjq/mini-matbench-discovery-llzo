#!/bin/bash
# Job entrypoint executed on the Bohrium worker, from the repo root.
#
# One Bohrium job builds one environment, because mace-torch and deepmd-kit
# cannot be installed together. Submit this script twice (ENV=mace, then
# ENV=deepmd) and merge the two results through the shared checkpoint file.
#
#   ENV=mace   bash cloud/run_job.sh    # chgnet-0.3.0 + mace-mp-0-medium
#   ENV=deepmd bash cloud/run_job.sh    # dpa4-mini-omat24
set -euo pipefail

bash cloud/run_split.sh --env "${ENV:-mace}" "${@}"
