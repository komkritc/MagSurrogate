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


