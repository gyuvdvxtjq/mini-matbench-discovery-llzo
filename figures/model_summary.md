### Cross-model summary (single-point, hull distance vs Materials Project DFT)

Covered: chgnet-0.3.0, mace-mp-0-medium.

| Model | n | Hull MAE (eV/atom) | Hull Spearman | Formation-energy MAE | Stable F1 | Candidate F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chgnet-0.3.0 | 150 | 0.0376 [0.0314, 0.0442] | 0.864 [0.787, 0.921] | 0.0418 [0.0351, 0.0491] | 0.444 [0.182, 0.667] | 0.813 [0.732, 0.885] |
| mace-mp-0-medium | 150 | 0.0454 [0.0349, 0.0559] | 0.868 [0.796, 0.921] | 0.2376 [0.2108, 0.2646] | 0.444 [0.189, 0.667] | 0.765 [0.673, 0.842] |

Values are point estimates with 95% percentile-bootstrap intervals over structures. Hull MAE is |model E-hull - MP E-hull| in eV/atom; the hull is rebuilt per model from the single-point energies in `checkpoints.jsonl`.
