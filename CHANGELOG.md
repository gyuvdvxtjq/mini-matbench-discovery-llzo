# Changelog

All notable changes to this benchmark are recorded here. The version refers to
`mlip_audit.__version__`.

## 0.3.0 — two-model scope, one dependency list, legacy archived

**Why 0.2 was refactored in the first place:** adding a second model meant
copying the entire pilot script. `battery_mlip_pilot.py` was
`phase_space_benchmark.py` with the hull swapped for a relaxation loop, and both
carried their own copies of `require_api_key`, the Materials Project field list,
the record assembly and `scalar()`. That is what produced 0.2, the
config-driven `mlip_audit` framework.

0.3 is not a science release: it changes no metric, no protocol and no recorded
number. It is the release that makes the benchmark honest about its own scope
and cleans up what the two generations left behind.

**Scope decision recorded here:** DPA-4 is removed from the benchmark. It was
never executed in the reference run — the weights and a CUDA-12-capable worker
were not available — and rather than ship a third model that only ever appeared
under `skipped_models`, it is cut entirely. The committed benchmark is a
two-model comparison (CHGNet 0.3.0 + MACE-MP-0 medium), and no claim in the
README or RESULTS.md depends on DPA-4. Its removal changes no recorded number.

### Added

- `cloud/setup_base.sh`, `cloud/setup_mace.sh`, replacing the single
  `cloud/setup.sh`. Both idempotent.
- `scripts/merge_and_plot.py` and `mlip_audit/merge.py`: turns the checkpoint
  into a per-chemsys, cross-model hull-MAE figure, a hull-parity figure and a
  Markdown summary table. Degrades to whatever models are present and says
  which configured models are absent.
- `device: auto` resolution (cuda when a GPU is usable, else cpu) with
  `--device` as the override.
- `mlip_audit/mp.py`: one home for `MP_FIELDS`, `require_api_key`, the record
  assembly, the snapshot load/fetch and the structure-stripping helper.
- `legacy/`: the v0.1 pilot, archived with a README mapping each script to the
  `RESULTS.md` section it produced.
- `pyproject.toml` as the single dependency manifest, with `[mace]` and `[dev]`
  extras, plus pytest and ruff configuration.
- `CHANGELOG.md`.
- 33 new tests (9 -> 42), covering device resolution, backend-skip reporting,
  split-run equivalence, report merging across passes, partial-model merging,
  and the shipped configs.
- Repository description changed from "Reproducible CHGNet relaxation-energy
  pilot on Materials Project LLZO structures" (which described only v0.1) to
  "Benchmarking universal MLIPs (CHGNet / MACE-MP-0) for stability prediction
  in the Li-La-Zr-O phase space". This needs a maintainer with write access:

  ```bash
  gh repo edit gyuvdvxtjq/mini-matbench-discovery-llzo \
    --description "Benchmarking universal MLIPs (CHGNet / MACE-MP-0) for stability prediction in the Li-La-Zr-O phase space"
  ```

### Changed

- **Run in passes instead of one process.** `run_audit` now takes several
  `--model` values, merges `report.json` across passes instead of overwriting
  it, and reads earlier passes' relaxation CSVs back so cross-model attribution
  spans the whole audit.
- **A missing backend explains itself.** Skips name the missing package and the
  setup script that installs it, in the log and in `skipped_models`.
- **Paths resolve from the repository root** (`mlip_audit.config.repo_path`)
  rather than the process working directory, so the audit and the legacy scripts
  run from anywhere and tests can pass temporary paths.
- `configs/*.yaml` no longer pin `device:`.
- `environment.yml` pins only Python 3.11 and installs `pip install -e .`.
- `mlip_audit/data.py` became `mlip_audit/mp.py`; `runner._slug` is one regex
  (with a test pinning the committed `data/runs/llzo/` directory names);
  `md.py`'s mid-function `import json` moved to the top.

### Removed

- **DPA-4 / deepmd support, in full.** The `deepmd` model type and its adapter,
  `DPA4_CHECKPOINT` handling, `cloud/setup_deepmd.sh`, `cloud/run_split.sh`,
  the `[deepmd]` extra, the `models/dpa4/` training-input directory, and the
  `skipped_models` entry in `report.json`. It never ran, and the split-
  environment machinery existed only to work around installing it alongside
  mace-torch — so both the model and the workaround go together.
- `requirements.txt` — it was missing `pyyaml` and `python-dotenv`, both
  imported on the first line of `mlip_audit`. `cloud/setup.sh` carried the only
  complete list, which is how the gap survived.
- `cloud/setup.sh`, superseded by the split scripts.
- `data.py`'s duplicate MP logic, replaced by `mp.py`.

### Unchanged, deliberately

- `mlip_audit/checkpoint.py`'s key composition and jsonl append format, so the
  existing `data/runs/llzo/` results stay valid.
- `RESULTS.md`, including its two now-stale file paths — it is a historical
  record; `legacy/README.md` carries the current path mapping.
- The convex-hull rebuild and scoring in `analysis.py`, and the Langevin
  equilibration + NVE production protocol in `md.py`.
- Every number in `RESULTS.md`.

## 0.2.0 — multi-model audit framework

Config-driven framework (`mlip_audit/`) driving several universal MLIPs through
one identical protocol: single point, cell relaxation, and NVT→NVE molecular
dynamics, every record checkpointed so runs resume and models can be added
without recomputation. CHGNet 0.3.0 and MACE-MP-0 (medium) were run over 150
Li–La–Zr–O structures. A third model (DPA-4) was planned but never executed;
it was removed in 0.3.0. Introduced `run_audit.py`, `configs/`, `tests/` and
the `cloud/` batch-job scripts. **Refactored from v0.1 because adding a second
model would otherwise have meant copying the whole pilot script.**

## 0.1.0 — CHGNet relaxation-energy pilot

Two standalone scripts: `phase_space_benchmark.py` (150 structures across every
Li/La/Zr/O subsystem, single-point energies, convex hull rebuilt from CHGNet
energies) and `battery_mlip_pilot.py` (full relaxation of the 3 exact
`Li-La-Zr-O` structures). Hard-coded CHGNet, no configuration, no multi-model
support. Produced the numbers in `RESULTS.md` sections 1–3 and the error
attribution in section 4. Archived under `legacy/` in 0.3.0.