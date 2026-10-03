# Running the audit on Bohrium

All heavy compute runs as Bohrium **batch jobs** — billing is per actual
runtime and the worker stops when the job exits, so cost is
`machine price × actual hours`, never idle time.

## Environment

Both backends (CHGNet and MACE-MP-0) install into one environment:

| Script | Installs | Models it runs |
| --- | --- | --- |
| `cloud/setup_base.sh` | chgnet, mp-api, pyyaml, ase, pymatgen, torch, numpy/pandas/scipy/matplotlib | — |
| `cloud/setup_mace.sh` | base + `mace-torch` | `chgnet-0.3.0`, `mace-mp-0-medium` |

Both are idempotent: they skip anything already installed, so they are safe to
re-run. `cloud/run_job.sh` chains them and then runs the audit.

## One-time setup

```bash
bohr auth whoami          # must be logged in
bohr billing balance      # needs a positive balance (jobs are billed)
```

## Submit

```bash
DRY_RUN=1 bash cloud/submit.sh    # validate the input tree, free, no balance needed
bash cloud/submit.sh              # real submission
```

Defaults: SKU `c16_m64_1×NVIDIA 4090` (¥6/h, `--sku-id 9985`), image
`ubuntu:22.04-py3.10-cuda12.1` + pip-installed backends (see the setup scripts
above). Override with `SKU_ID=<id>` (see `bohr batchjob machine list
--choose-type gpu -o json`). Do not use a larger GPU: these models are small
and will not benefit.

Expected runtime for the full `configs/llzo.yaml` audit (single point +
relaxation over 150 structures, plus MD at 4 temperatures) is roughly 2–3 hours
on one 4090, i.e. under ¥20.

## Monitor and collect

```bash
bohr batchjob list --status pending,running -o json
bohr batchjob describe <job_id> -o json
bohr batchjob wait <job_id> --interval 30s --timeout 8h
bohr batchjob download <job_id> --dest ./job_result   # dest must not exist
```

Then produce the cross-model figures and the README table:

```bash
python3 scripts/merge_and_plot.py --run-dir data/runs/llzo --config configs/llzo.yaml
```

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

This is covered by
`tests/test_pipeline.py::test_split_batches_match_a_single_run`:
running the models one at a time produces byte-identical records to running
them all at once.

## Notes

- `.env` (MP API key) is excluded from the upload; the job runs offline from
  the cached snapshot `data/raw/mp_phase_space.json`.
- Devices are resolved per model as `auto`: `cuda` when a GPU is usable, else
  `cpu`. No config hard-codes a device, so the same YAML runs on a laptop and
  on the worker. `--device` forces it.
- If a model backend fails to install or load, the runner skips that model,
  says which package and which setup script would fix it, and still delivers
  results for the rest — check `report.json`'s `skipped_models` before
  assuming a model ran.
