# Running the audit on Bohrium

All heavy compute runs as Bohrium **batch jobs** — billing is per actual
runtime and the worker stops when the job exits, so cost is
`machine price × actual hours`, never idle time.

## Why the audit runs in two environments

`mace-torch` and `deepmd-kit` cannot be resolved by pip in the same
environment (`ResolutionImpossible` — their torch/numpy/CUDA pins are mutually
exclusive). That is the recorded root cause of DPA-4 appearing under
`skipped_models` in the 2026-09-29 run (`data/runs/llzo/NOTES.md`).

The fix is not to make the conflict resolvable; it is to stop trying to resolve
it in one environment:

| Script | Installs | Models it runs |
| --- | --- | --- |
| `cloud/setup_base.sh` | chgnet, mp-api, pyyaml, ase, pymatgen, torch, numpy/pandas/scipy/matplotlib | — |
| `cloud/setup_mace.sh` | base + `mace-torch` | `chgnet-0.3.0`, `mace-mp-0-medium` |
| `cloud/setup_deepmd.sh` | base + `deepmd-kit` + the DPA-4 `.pt` | `dpa4-mini-omat24` |

All three are idempotent: they skip anything already installed, so they are
safe to re-run and safe to chain.

`cloud/run_split.sh --env mace|deepmd` builds the environment and runs
`run_audit.py` for that environment's models. Both invocations append to the
**same** checkpoint file, so the halves add up.

## One-time setup

```bash
bohr auth whoami          # must be logged in
bohr billing balance      # needs a positive balance (jobs are billed)
```

## Submit

```bash
DRY_RUN=1 bash cloud/submit.sh    # validate the input tree, free, no balance needed
bash cloud/submit.sh              # real submission, ENV=mace
ENV=deepmd bash cloud/submit.sh   # the DPA-4 half
```

Defaults: SKU `c16_m64_1×NVIDIA 4090` (¥6/h, `--sku-id 9985`), image
`ubuntu:22.04-py3.10-cuda12.1` + pip-installed backends (see the setup scripts
above). Override with `SKU_ID=<id>` (see `bohr batchjob machine list
--choose-type gpu -o json`). Do not use a larger GPU: these models are small
and will not benefit.

Expected runtime for the full `configs/llzo.yaml` audit (single point +
relaxation over 150 structures, plus MD at 4 temperatures) is roughly 2–3 hours
on one 4090 per half, i.e. under ¥20 each.

## Monitor and collect

```bash
bohr batchjob list --status pending,running -o json
bohr batchjob describe <job_id> -o json
bohr batchjob wait <job_id> --interval 30s --timeout 8h
bohr batchjob download <job_id> --dest ./job_result   # dest must not exist
```

## Merge the two halves

```bash
# after downloading both jobs, copy each job_result checkpoint over the local one
cp job_result_mace/data/runs/llzo/checkpoints.jsonl   data/runs/llzo/checkpoints.jsonl
cp job_result_deepmd/data/runs/llzo/checkpoints.jsonl data/runs/llzo/checkpoints.jsonl

# then produce the cross-model figures and the README table
python3 scripts/merge_and_plot.py --run-dir data/runs/llzo --config configs/llzo.yaml
```

Copying one file over the other is safe: the jsonl is append-only and every
line carries its own `model|protocol|params-hash|material_id` key, so a later
copy simply re-reads the earlier keys. `report.json` is merged rather than
overwritten, so a half-run never erases the other half's metrics.

## Resume across jobs (断点重续)

Checkpoints live in `data/runs/llzo/checkpoints.jsonl`, keyed by
`model|protocol|params-hash|material_id`. To continue an interrupted run:

1. `bohr batchjob download <job_id> --dest ./job_result`
2. Copy `job_result/data/runs/llzo/checkpoints.jsonl` over the local one.
3. Re-submit. Completed keys are skipped; only missing work is computed.

Switching or adding a model in `configs/llzo.yaml` works the same way: the
new model's keys are absent, so only that model is computed; everything else
resumes. Changing a protocol parameter (e.g. `fmax`) changes the params hash,
which intentionally invalidates only the affected records.

The same property is what makes the two-environment split work, and it is
covered by `tests/test_pipeline.py::test_split_batches_match_a_single_run`:
running the models one at a time produces byte-identical records to running
them all at once.

## Notes

- `.env` (MP API key) is excluded from the upload; the job runs offline from
  the cached snapshot `data/raw/mp_phase_space.json`.
- The DPA-4 checkpoint (CC-BY-NC-4.0, **non-commercial**) is fetched by
  `cloud/setup_deepmd.sh` and passed to the audit through the
  `DPA4_CHECKPOINT` environment variable. It is never committed.
- Devices are resolved per model as `auto`: `cuda` when a GPU is usable, else
  `cpu`. No config hard-codes a device, so the same YAML runs on a laptop and
  on the worker. `--device` forces it.
- If a model backend fails to install or load, the runner skips that model,
  says which package and which setup script would fix it, and still delivers
  results for the rest — check `report.json`'s `skipped_models` before
  assuming a model ran.
