# Training-Size Ablation Study (Manuscript Table 6)

Standalone reproduction of the Random-Forest training-size ablation from the CPC MagSurrogate paper.
Five-fold and Leave-One-Diameter-Out (LODO) cross-validation at four data fractions (100 %, 75 %, 50 %, 25 %), averaged over 5 random seeds.

## Files

```
ablation_test/
├── ablation.py              # Self-contained script (run or load pre-computed results)
├── requirements.txt         # Minimal dependencies
├── README.md                # This file
└── data/
    ├── fept_dataset.csv     # FePT simulation dataset (3819 rows, 19 diameters, 201 temperatures)
    └── ablation_*.csv       # Pre-computed results (reproducible via `--load` flag)
        ├── ablation_kfold_per_seed.csv          # 20 rows: 4 fractions × 5 seeds
        ├── ablation_lodo_by_diameter_per_seed.csv  # 380 rows: 20 seeds × 19 diameters
        └── ablation_summary.csv                 # 8 rows: 4 fractions × 2 protocols
```

## Reproducing the results

### Prerequisites

```bash
pip install -r requirements.txt
```

Required: `pandas`, `numpy`, `scikit-learn>=1.0`.

### Run

```bash
# Recompute from scratch (takes ~3 min on a modern laptop, 4 cores)
python ablation.py

# Load and display pre-computed results
python ablation.py --load

# Custom dataset or job count
python ablation.py --dataset my_data.csv --n-jobs 2
```

Output CSVs are written to `data/` alongside the source data.

## Experimental design

### Data fractions

| Fraction | Description              | Rows per diameter (≈) | Total rows |
|----------|--------------------------|----------------------:|-----------:|
| 1.00     | Full dataset            | 201                   | 3819       |
| 0.75     | 75 % subsampled         | 151                   | 2864       |
| 0.50     | 50 % subsampled         | 101                   | 1910       |
| 0.25     | 25 % subsampled         | 50                    | 947        |

Each fraction is sampled **stratified by diameter** — every diameter retains exactly the same proportion of its rows.

### Cross-validation protocols

**Five-fold CV** — random 5-way split (100 RF trees, unlimited depth). One result per seed; Table 6 reports mean ± SD over seeds.

**LODO-CV** — one diameter left out as test set; train on remaining diameters subsampled to the given fraction (200 RF trees, unlimited depth). The 19 withheld diameters are iterated. Table 6 reports the mean metric across diameters, then averaged over seeds.

### Hyperparameters

| Protocol | Trees | Max depth | Min samples split |
|----------|-------:|----------:|-------------------:|
| Five-fold CV | 100 | None | 2 |
| LODO-CV | 200 | None | 2 |

These match the "fixed" hyperparameters from manuscript Table 1.

### Targets (per-target metrics, not averaged)

| Column suffix | Target                | Symbol in manuscript |
|--------------:|-----------------------|----------------------|
| `_me`         | Magnetization (Mx)    | $m_e$               |
| `_chipar`     | Longitudinal χ        | $\chi_\parallel$    |
| `_chiperp`    | Transverse χ          | $\chi_\perp$        |

Both $R^2$ and RMSE are computed independently per target; negative $R^2$ values are preserved (not clipped to 0).

## Pre-computed results summary

```
fraction  protocol       R2_me   R2_chipar  R2_chiperp  RMSE_me  RMSE_chipar  RMSE_chiperp
------    --------       -----   ---------  ----------  -------  -----------  ------------
1.00      Five-fold CV   0.9999  0.9762     0.9976      0.0030   0.0026       0.0017
1.00      LODO-CV        0.9959  0.8137     0.9631      0.0123   0.0041       0.0043
0.75      Five-fold CV   0.9999  0.9573     0.9973      0.0035   0.0032       0.0018
0.75      LODO-CV        0.9958  0.7961     0.9543      0.0138   0.0052       0.0057
0.50      Five-fold CV   0.9998  0.9638     0.9957      0.0051   0.0030       0.0023
0.50      LODO-CV        0.9956  0.7599     0.9494      0.0157   0.0062       0.0064
0.25      Five-fold CV   0.9994  0.9512     0.9909      0.0090   0.0039       0.0032
0.25      LODO-CV        0.9946  0.6713     0.9423      0.0214   0.0084       0.0074
```

SD columns omitted for brevity; full values are in `ablation_summary.csv`.

## Differences from the manuscript

The code here matches the original `rev1_analysis/ablation_data_fraction_rerun.py`. The five-fold 100 % row gives $R^2(\chi_\parallel) = 0.9762$ (manuscript: 0.9741; difference due to RF stochasticity with different random-state assignment). The LODO 100 % row reports $R^2(\chi_\parallel) = 0.8137$, which matches a regenerated Table 4 value from the same code base but differs from the manuscript's 0.9445. See `README.md` (parent repo) for full analysis.

## License

MIT — see parent repository LICENSE. The FePT dataset is also redistributed under MIT terms.
