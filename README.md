# mini-matbench-discovery-llzo

**A reproducible CHGNet benchmark for thermodynamic-stability prediction in the LLZO solid-electrolyte system**

[English](README.md) | [简体中文](README.zh-CN.md)

---

## Overview

This repository provides a focused, reproducible benchmark of how reliably **CHGNet**, a pretrained universal machine-learning interatomic potential (MLIP), predicts energies and thermodynamic stability across the Li–La–Zr–O chemical space — the phase space hosting the garnet solid-state electrolyte LLZO (Li₇La₃Zr₂O₁₂). Judging whether a crystal structure is stable traditionally requires expensive quantum-mechanical calculations (DFT, hours per structure); MLIPs like CHGNet reduce this to seconds and are widely used for materials screening. The question this benchmark answers: **in this system, is that speedup trustworthy — and where does it fail?**

The benchmark has two levels:

- **Phase-space layer**: 150 structures spanning every elemental, binary, ternary, and quaternary subsystem of Li/La/Zr/O (14 chemsyses). CHGNet single-point energies rebuild the convex hull; formation energies, hull distances, and stable-phase classification are compared against DFT references.
- **LLZO layer**: full structural relaxation (up to 100 ionic steps, force criterion 0.1 eV/Å) of the 3 exact quaternary LLZO structures on Materials Project, with relaxed energies compared against DFT within the same elemental space.

Failed runs are never deleted — they are written to checkpoints and final CSVs as benchmark evidence.

> Materials Project currently returns only 3 exact LLZO quaternary structures. Stability cannot be judged from 3 structures alone, so the phase-space evaluation adds the remaining 147 elemental and competing phases instead of padding with unrelated materials.

Raw DFT and CHGNet total energies carry different element-dependent reference offsets, so all cross-chemsys conclusions use formation energies and hull distances — never raw total-energy MAE.

---

## Key Results

| Metric | Value |
| --- | ---: |
| Hull-distance (E-hull) MAE | **0.0376 eV/atom** |
| Hull-distance Spearman ρ | **0.8645** |
| Candidate classification F1 at 50 meV/atom (57 positives) | **0.813** |
| Strict stable-phase classification F1 (15 positives) | 0.444 |
| Formation-energy MAE | 0.0418 eV/atom |
| LLZO relaxation: converged / energy MAE | 3 of 3 · 0.0450 eV/atom |

![Hull parity](data/phase_space/plots/hull_parity.png)

![Hull MAE by chemical system](data/phase_space/plots/hull_mae_by_chemsys.png)

Full metrics, methodology notes, and the error-attribution analysis — why two La-rich structures are systematically underestimated by ~60 meV/atom, ranking flips among elemental polymorphs, and a bimodal failure cluster on O₂ molecular crystals — are documented in [`RESULTS.md`](RESULTS.md) (Chinese).

---

## Repository Layout

```
.
├── phase_space_benchmark.py   # Phase-space benchmark (150 structures, single-point energies)
├── battery_mlip_pilot.py      # LLZO relaxation benchmark (3 quaternary structures)
├── environment.yml            # Conda environment specification
├── requirements.txt           # pip dependencies
├── RESULTS.md                 # Full results and error analysis (Chinese)
├── .env.example               # Template for MP_API_KEY (not committed)
└── data/                      # Outputs after running
    ├── chgnet_vs_mp_llzo.csv  #   relaxation-vs-DFT table
    ├── metrics.json           #   headline metrics
    ├── phase_space/           #   per-structure CSV + hull metrics + plots
    ├── raw/                   #   MP API snapshot (not committed)
    └── checkpoints/           #   resume checkpoints (not committed)
```

---

## Usage

### Environment setup

Requires [conda](https://docs.conda.io/) and a free [Materials Project](https://next-gen.materialsproject.org/) API key.

```bash
git clone https://github.com/gyuvdvxtjq/mini-matbench-discovery-llzo.git
cd mini-matbench-discovery-llzo
conda env create -f environment.yml --override-channels
conda activate mini-matbench-llzo
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
# open .env and fill in MP_API_KEY, then continue
python phase_space_benchmark.py
python battery_mlip_pilot.py
```

Both scripts resume from checkpoints: re-running after an interruption skips structures that already succeeded.

### Useful flags

```bash
# verify API access and cached data only
python battery_mlip_pilot.py --fetch-only

# rerun relaxations from scratch with fixed lattices
python battery_mlip_pilot.py --fresh --no-relax-cell
```

### Configuration

- **Model**: pretrained CHGNet `0.3.0` (package `chgnet==0.4.2`)
- **Relaxation**: max 100 ionic steps, convergence at max-force ≤ 0.1 eV/Å
- **Hardware**: runs on CPU by default; CUDA/MPS devices are supported via `--device`

Outputs are plain CSV tables, JSON metrics, and PNG figures, so everything can be inspected without rerunning.

---

## Scope & Limitations

CHGNet's training data comes from Materials Project, and so do all 150 structures evaluated here. This is therefore an **in-distribution** benchmark: it measures workflow correctness and predictive reliability within the chemical space the model was trained on. It does not measure extrapolation to unseen candidate structures — that requires an independent discovery set such as WBM under the full Matbench-Discovery protocol, and these results should not be read as an estimate of that score.

---

## Acknowledgments

Thanks to the CHGNet development team and the Materials Project consortium for open models and reference data that made this benchmark possible.

## Contact

Questions and suggestions are welcome via [GitHub Issues](https://github.com/gyuvdvxtjq/mini-matbench-discovery-llzo/issues).
