# `legacy/` — the v0.1 single-model pilot (archived)

These two scripts are the **v0.1 CHGNet pilot**. Every conclusion they support
has since been superseded by the multi-model audit in `mlip_audit/`, which runs
CHGNet, MACE-MP-0 and DPA-4 through one identical protocol. They are kept, not
deleted, because they are the code that produced the numbers in
[`../RESULTS.md`](../RESULTS.md) sections 2–3:

| Script | Produced | `RESULTS.md` section |
| --- | --- | --- |
| `phase_space_benchmark.py` | `../data/phase_space/` (150 structures, single-point energies, convex hull, plots) | §1 (and the plots used in §4) |
| `battery_mlip_pilot.py` | `data/chgnet_vs_mp_llzo.csv`, `data/metrics.json` — full relaxation of the 3 exact `Li-La-Zr-O` structures | §2–3 |

`mlip_audit` reproduces the headline number from §1: its CHGNet single-point
hull MAE is 0.0376 eV/atom, the same value these scripts report (verified in
`data/runs/llzo/NOTES.md` and reproduced by
`python3 scripts/merge_and_plot.py`).

## Why archived rather than deleted

- They are the *provenance* of the published numbers; deleting them would make
  `RESULTS.md` unreproducible.
- They hard-code CHGNet, its own record schema, and its own output layout, and
  they carry no checkpointing beyond a material-id jsonl — all of which
  `mlip_audit` replaces. Maintaining both in the main tree invited the two
  generations to drift.

## Running them

Both scripts are offline-friendly: pass `--dry-run` to confirm the data source
resolves without loading a model.

```bash
# Section 1: 150-structure phase space. Needs data/raw/mp_phase_space.json
# (git-ignored) or a Materials Project API key in .env.
python3 legacy/phase_space_benchmark.py --dry-run
python3 legacy/phase_space_benchmark.py --use-cache      # from the cached snapshot

# Sections 2-3: the 3 exact Li-La-Zr-O structures. Always hits the API.
python3 legacy/battery_mlip_pilot.py --dry-run
python3 legacy/battery_mlip_pilot.py --fetch-only        # verify API access
python3 legacy/battery_mlip_pilot.py --fresh --no-relax-cell
```

Paths are resolved from the repository root via `__file__`, so the commands
work from any working directory. `battery_mlip_pilot.py --out-dir` overrides
where its outputs land (default `legacy/data/`).

## What changed when they were archived (v0.3.0)

Only mechanical, no behaviour change:

- Output paths are now derived from `__file__` instead of the process CWD.
- `require_api_key`, the MP field list, the record assembly and the
  `scalar()` tensor coercion are imported from `mlip_audit/mp.py` and
  `mlip_audit/models.py` rather than duplicated (they were triplicated).
- `battery_mlip_pilot.py` writes to `legacy/data/`; the two files that used to
  sit at the top of `data/` are archived here. `phase_space_benchmark.py`
  still writes to `data/phase_space/`, which is where `RESULTS.md` links its
  figures from.
- A `--dry-run` flag was added to both.

> **Note on stale paths in `RESULTS.md`.** `RESULTS.md` still points at
> `data/chgnet_vs_mp_llzo.csv` and `data/metrics.json`, which moved to
> `legacy/data/`. `RESULTS.md` is a historical record and its numbers are left
> untouched on purpose — the table above is the current path mapping.
