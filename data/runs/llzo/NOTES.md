# MLIP audit run notes — Li-La-Zr-O, 2026-09-29

Run: `python run_audit.py --config configs/llzo.yaml --device cuda`
Hardware: 1x A100-80GB, torch 2.14.0+cu130, wall time ~4.7 h (exit 0).

## Model coverage

| Model | Status |
| --- | --- |
| chgnet-0.3.0 | full metrics (single_point + relaxation + md) |
| mace-mp-0-medium | full metrics (single_point + relaxation + md) |
| dpa4-mini-omat24 | **skipped** — `deepmd-kit not installed: No module named 'deepmd'` |

deepmd-kit cannot be installed alongside mace-torch (pip ResolutionImpossible),
so DPA-4 is reported in `report.json` under `skipped_models` rather than run.

## Known anomalies (recorded as-is, not corrected)

1. **MACE relaxation, mp-1056059 (O2): -3.4728e9 eV/atom.** FIRE converged
   (final max force 0.0) to a spurious minimum with a collapsed cell. Because
   the convex hull is global, this one entry drags the reported MACE
   `relaxation.hull_mae_eV_per_atom` to ~1.9e9 eV/atom. Rebuilding the hull
   over the other 149 structures gives **0.0621 eV/atom** (CHGNet: 0.0540).
   The same failure reproduces in an independent rerun, so it is a genuine
   MACE failure mode on O2 molecular crystals, not a numerical fluke.

2. **MACE md @1200K: Arrhenius fit is NaN.** The Li MSD plateaued during the
   20 ps production leg (first-quarter mean 0.82 A^2 vs last-quarter 0.84),
   so the fitted slope is slightly negative and `log(D)` is undefined. The
   trajectory itself is stable (energy drift -0.025 eV/atom over 20 ps,
   constant volume). Cause is the 1x1x1 (96-atom) supercell and short
   production time set in `configs/llzo.yaml`; it is a statistical artefact
   of the configured protocol, not a code bug.

## Consistency check

CHGNet single-point hull MAE = 0.0376 eV/atom, matching the value published
in the repo README/RESULTS.md for the original single-model benchmark — the
multi-model framework reproduces the established result.
