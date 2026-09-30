#!/bin/bash
# Worker-side environment setup, run by cloud/run_job.sh from the repo root.
# Installs the MLIP backends the audit drives, then fetches the DPA-4
# checkpoint. Every model is optional: if a backend or its checkpoint is
# unavailable, the runner skips that model and still reports the rest
# (see report.json -> skipped_models).
set -uo pipefail

# ---------------------------------------------------------------- core stack
# The two models that ran for LLZO (see data/runs/llzo/NOTES.md).
python -m pip install --quiet --upgrade pip
python -m pip install --quiet \
    "chgnet==0.4.2" \
    "mace-torch>=0.3.0" \
    "mp-api==0.46.5" \
    "python-dotenv==1.1.1"

# --------------------------------------------------------------- deepmd-kit
# DPA-4 is reported under skipped_models rather than run: deepmd-kit cannot
# be resolved alongside mace-torch (pip ResolutionImpossible). We still try
# in isolation so that a future compatible release is picked up
# automatically; a failure here must not abort the job.
if ! python -m pip install --quiet "deepmd-kit"; then
    echo "warning: deepmd-kit unavailable (known conflict with mace-torch);" >&2
    echo "         dpa4-mini-omat24 will be skipped in this run." >&2
fi

# ---------------------------------------------------- DPA-4 frozen checkpoint
# configs/llzo.yaml points at models/dpa4/DPA4-Mini-OMat24-v20260805.pt.
# The repo ships only the training-input .json (no weights); the frozen .pt
# is fetched at job start from the HuggingFace mirror. It is licensed
# CC-BY-NC-4.0 (non-commercial). Override the URL if your mirror differs.
DPA4_PT="models/dpa4/DPA4-Mini-OMat24-v20260805.pt"
DPA4_HF_URL="${DPA4_HF_URL:-https://huggingface.co/deepmodeling/DPA4-Mini-OMat24/resolve/main/v20260805/DPA4-Mini-OMat24-v20260805.pt}"

if [ -s "$DPA4_PT" ]; then
    echo "dpa4 checkpoint already present, skipping download"
elif ! curl -fsSL -o "$DPA4_PT" "$DPA4_HF_URL"; then
    echo "warning: could not fetch the DPA-4 checkpoint from:" >&2
    echo "         $DPA4_HF_URL" >&2
    echo "         set DPA4_HF_URL to your mirror; dpa4 will be skipped." >&2
    rm -f "$DPA4_PT"
fi

# ------------------------------------------------------------------ sanity
python - <<'PY'
import importlib.util, pathlib, sys
backends = {"chgnet": "chgnet", "mace": "mace_torch", "deepmd": "deepmd"}
for name, module in backends.items():
    available = importlib.util.find_spec(module) is not None
    print(f"backend {name}: {'ok' if available else 'MISSING (model will be skipped)'}")
checkpoint = pathlib.Path("models/dpa4/DPA4-Mini-OMat24-v20260805.pt")
print(f"dpa4 checkpoint: {'ok' if checkpoint.is_file() and checkpoint.stat().st_size else 'MISSING'}")
PY
