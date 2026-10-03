# Changelog

All notable changes to this benchmark are recorded here. The version refers to
`mlip_audit.__version__`.

## 0.3.0 — make the third model runnable, and stop maintaining three copies of everything

**Why 0.2 was refactored in the first place:** adding a second model meant
copying the entire pilot script. `battery_mlip_pilot.py` was
`phase_space_benchmark.py` with the hull swapped for a relaxation loop, and both
carried their own copies of `require_api_key`, the Materials Project field list,
the record assembly and `scalar()`. That is what produced 0.2, the
config-driven `mlip_audit` framework.

0.3 is not a science release: it changes no metric, no protocol and no recorded
number. It is the release that makes the third model actually runnable and
cleans up what the two generations left behind.

**Scope decision recorded here:** DPA-4 remains unexecuted in the reference run.
The committed benchmark is two-wide (CHGNet 0.3.0 + MACE-MP-0 medium). Running
the third model needs a CUDA-12-capable worker plus the CC-BY-NC-4.0 weights,
which were not available during this work. Rather than block the release on
hardware, the split-environment path was built and tested with a stub backend,
and the run itself is left as one command
(`bash cloud/run_split.sh --env deepmd`). No claim in the README or RESULTS.md
depends on DPA-4 having run.

### Added

- `cloud/setup_base.sh`, `cloud/setup_mace.sh`, `cloud/setup_deepmd.sh`, replacing
  the single `cloud/setup.sh`. All idempotent, each documenting why the three
  installs must not be merged.
- `cloud/run_split.sh --env mace|deepmd`: builds one environment and runs that
  environment's models against the shared checkpoint file.
- `scripts/merge_and_plot.py` and `mlip_audit/merge.py`: turns a merged
  checkpoint into a per-chemsys, cross-model hull-MAE figure, a hull-parity
  figure and a Markdown summary table. Degrades to whatever models are present
  and says which configured models are absent.
- `DPA4_CHECKPOINT` environment variable as the only source of the DPA-4
  weights path. A missing variable is an error naming the variable and the
  setup script, not a silent skip.
- `device: auto` resolution (cuda when a GPU is usable, else cpu) with
  `--device` as the override.
- `mlip_audit/mp.py`: one home for `MP_FIELDS`, `require_api_key`, the record
  assembly, the snapshot load/fetch and the structure-stripping helper.
- `legacy/`: the v0.1 pilot, archived with a README mapping each script to the
  `RESULTS.md` section it produced.
- `pyproject.toml` as the single dependency manifest, with `[mace]`, `[deepmd]`
  and `[dev]` extras, plus pytest and ruff configuration.
- `CHANGELOG.md`.
- 37 new tests (9 -> 46), covering device resolution, backend-skip reporting,
  `DPA4_CHECKPOINT` handling, split-run equivalence, report merging across
  passes, partial-model merging, and the shipped configs.
- Repository description changed from "Reproducible CHGNet relaxation-energy
  pilot on Materials Project LLZO structures" (which described only v0.1) to
  "Benchmarking universal MLIPs (CHGNet / MACE-MP-0 / DPA) for stability
  prediction in the Li-La-Zr-O phase space". This needs a maintainer with write
  access:

  ```bash
  gh repo edit gyuvdvxtjq/mini-matbench-discovery-llzo \
    --description "Benchmarking universal MLIPs (CHGNet / MACE-MP-0 / DPA) for stability prediction in the Li-La-Zr-O phase space"
  ```

### Changed

- **Run in passes instead of one process.** `run_audit` now takes several
  `--model` values, merges `report.json` across passes instead of overwriting
  it, and reads earlier passes' relaxation CSVs back so cross-model attribution
  spans the whole audit. Without this the split environment silently destroyed
  the previous batch's metrics.
- **A missing backend explains itself.** Skips name the missing package and the
  setup script that installs it, in the log and in `skipped_models`.
- **Paths resolve from the repository root** (`mlip_audit.config.repo_path`)
  rather than the process working directory, so the audit and the legacy scripts
  run from anywhere and tests can pass temporary paths.
- `configs/*.yaml` no longer pin `device:`; the DPA-4 checkpoint path is no
  longer in the config.
- `environment.yml` pins only Python 3.11 and installs `pip install -e .`.
- `mlip_audit/data.py` became `mlip_audit/mp.py`; `runner._slug` is one regex
  (with a test pinning the committed `data/runs/llzo/` directory names);
  `md.py`'s mid-function `import json` moved to the top.

### Removed

- `requirements.txt` — it was missing `pyyaml` and `python-dotenv`, both
  imported on the first line of `mlip_audit`. `cloud/setup.sh` carried the only
  complete list, which is how the gap survived.
- `cloud/setup.sh`, superseded by the three split scripts.
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
Li–La–Zr–O structures; DPA-4 was reported under `skipped_models` because
deepmd-kit is not pip-resolvable alongside mace-torch. Introduced
`run_audit.py`, `configs/`, `tests/` and the `cloud/` batch-job scripts.
**Refactored from v0.1 because adding a second model would otherwise have meant
copying the whole pilot script.**

## 0.1.0 — CHGNet relaxation-energy pilot

Two standalone scripts: `phase_space_benchmark.py` (150 structures across every
Li/La/Zr/O subsystem, single-point energies, convex hull rebuilt from CHGNet
energies) and `battery_mlip_pilot.py` (full relaxation of the 3 exact
`Li-La-Zr-O` structures). Hard-coded CHGNet, no configuration, no multi-model
support. Produced the numbers in `RESULTS.md` sections 1–3 and the error
attribution in section 4. Archived under `legacy/` in 0.3.0.