#!/bin/bash
# Job entrypoint executed on the Bohrium worker, from the repo root.
set -e
bash cloud/setup.sh
python run_audit.py --config configs/llzo.yaml --device cuda
