"""Training-size ablation study (manuscript Table 6).

Standalone reproduction of the RF training-size ablation: five-fold and
Leave-One-Diameter-Out (LODO) cross-validation at four data fractions
(100 %, 75 %, 50 %, 25 %) averaged over 5 random seeds.

Usage
-----
python ablation.py                          # run with default dataset & write results
python ablation.py --dataset path/to/fept.csv  # supply your own dataset
python ablation.py --load                   # skip computation, just read results/data/ablation_*.csv

All constants, metrics helpers, and the model builder are self-contained;
the script depends only on ``pandas``, ``numpy`` and ``scikit-learn``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Constants (mirrors src/magsurrogate/data.py / models.py)
# ---------------------------------------------------------------------------
FEATURES = ["Temperature", "Diameter"]
TARGETS = ["Mx", "Chi_longitudinal", "Chi_trans"]
TARGET_LABELS = {"Mx": "me", "Chi_longitudinal": "chipar", "Chi_trans": "chiperp"}

ABLATION_FRACTIONS = [1.00, 0.75, 0.50, 0.25]
ABLATION_SEEDS = [0, 1, 2, 3, 4]

# Table 1 hyperparameters (fixed mode)
TABLE1_RF_KFOLD = dict(n_estimators=100, max_depth=None, min_samples_split=2)
TABLE1_RF_LODO = dict(n_estimators=200, max_depth=None, min_samples_split=2)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def r2_rmse(y_true, y_pred):
    """Per-target R^2 and RMSE (multioutput='raw_values'). R^2 may be negative."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput="raw_values"))
    return r2, rmse


def metrics_row(r2, rmse) -> dict:
    """Convert per-target r2/rmse arrays into a flat dictionary keyed by label."""
    row = {}
    for k, t in enumerate(TARGETS):
        lab = TARGET_LABELS[t]
        row[f"R2_{lab}"] = float(r2[k])
        row[f"RMSE_{lab}"] = float(rmse[k])
    return row


# ---------------------------------------------------------------------------
# Model builder (RF + KNN wrapper)
# ---------------------------------------------------------------------------
def _knn_pipeline(n_neighbors=7, weights="distance"):
    """KNN with imputation pipeline (required for sklearn >= 1.2)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)),
    ])


def make_model(name: str, random_state: int = 42, n_jobs: int = -1, params: dict | None = None):
    """Return a ``MultiOutputRegressor`` for RF or KNN."""
    params = dict(params or {})
    if name == "RF":
        est = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs, **params)
    elif name == "KNN":
        est = _knn_pipeline(**params)
    else:
        raise ValueError(f"unknown model {name!r}")
    return MultiOutputRegressor(est)


# ---------------------------------------------------------------------------
# Ablation core
# ---------------------------------------------------------------------------
def _subsample(data: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Stratified by diameter: ``frac`` of the rows of every diameter."""
    if frac >= 1.0:
        return data
    return pd.concat(
        [g.sample(frac=frac, random_state=seed) for _, g in data.groupby("Diameter")],
        ignore_index=True,
    )


def run_ablation(df: pd.DataFrame, fractions=None, seeds=None, n_jobs: int = -1, log=print):
    """RF training-size ablation. Five-fold uses 100 trees, LODO 200 trees (Table 1), unlimited depth.

    Returns ``(kfold_seed_df, lodo_by_diameter_seed_df)`` -- one row per seed with all metrics.
    """
    fractions = fractions or ABLATION_FRACTIONS
    seeds = seeds if seeds is not None else ABLATION_SEEDS
    df = df.sort_values(["Diameter", "Temperature"]).reset_index(drop=True)
    diameters = sorted(df["Diameter"].unique())
    k_rows, l_rows = [], []
    for frac in fractions:
        for seed in seeds:
            sub = _subsample(df, frac, seed).reset_index(drop=True)
            X, Y = sub[FEATURES].values, sub[TARGETS].values
            yt, yp = [], []
            # --- five-fold CV (100 trees) ---
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
                m = make_model("RF", seed, n_jobs, TABLE1_RF_KFOLD)
                m.fit(X[tr], Y[tr])
                yp.append(m.predict(X[te]))
                yt.append(Y[te])
            r2, rmse = r2_rmse(np.vstack(yt), np.vstack(yp))
            k_rows.append({"fraction": frac, "seed": seed, "n_rows": len(sub), **metrics_row(r2, rmse)})

            # --- LODO-CV (200 trees) ---
            for d_out in diameters:
                train = _subsample(df[df["Diameter"] != d_out], frac, seed)
                test = df[df["Diameter"] == d_out]
                m = make_model("RF", seed, n_jobs, TABLE1_RF_LODO)
                m.fit(train[FEATURES].values, train[TARGETS].values)
                r2, rmse = r2_rmse(test[TARGETS].values, m.predict(test[FEATURES].values))
                l_rows.append({"fraction": frac, "seed": seed, "diameter": d_out, **metrics_row(r2, rmse)})

            log(f"[ablation] fraction={frac:.2f} seed={seed} done")
    return pd.DataFrame(k_rows), pd.DataFrame(l_rows)


def summarise_ablation(kfold_df: pd.DataFrame, lodo_df: pd.DataFrame) -> pd.DataFrame:
    """Table 6 layout. K-fold: mean over seeds. LODO: mean over seeds first, then over diameters."""
    mcols = [c for c in kfold_df.columns if c.startswith(("R2_", "RMSE_"))]
    rows = []
    for frac in sorted(kfold_df["fraction"].unique(), reverse=True):
        k = kfold_df[kfold_df.fraction == frac]
        rows.append({
            "fraction": frac,
            "protocol": "Five-fold CV",
            "n_seeds": len(k),
            **{c: k[c].mean() for c in mcols},
            **{c + "_sd": k[c].std(ddof=1) for c in mcols},
        })
        l = lodo_df[lodo_df.fraction == frac].groupby("diameter")[mcols].mean()
        rows.append({
            "fraction": frac,
            "protocol": "LODO-CV",
            "n_seeds": int(lodo_df[lodo_df.fraction == frac].seed.nunique()),
            **{c: l[c].mean() for c in mcols},
            **{c + "_sd": l[c].std(ddof=1) for c in mcols},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Training-size ablation (Table 6). Five-fold + LODO-CV at 100/75/50/25%.",
    )
    parser.add_argument("--dataset", default=str(SCRIPT_DIR / "data" / "fept_dataset.csv"),
                        help="Path to the FePT dataset CSV")
    parser.add_argument("--output", default=str(SCRIPT_DIR / "data"),
                        help="Directory for output CSVs")
    parser.add_argument("--load", action="store_true",
                        help="Load pre-computed results instead of recomputing")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (-1 = all)")
    args = parser.parse_args(argv)

    if args.load:
        for name in ("ablation_kfold_per_seed", "ablation_lodo_by_diameter_per_seed",
                      "ablation_summary"):
            path = Path(args.output) / f"{name}.csv"
            df = pd.read_csv(path)
            print(f"\n--- {path.name} ({len(df)} rows) ---\n{df.to_string()}")
        return 0

    def log(msg):
        print(msg)

    print(f"Loading dataset: {args.dataset}")
    df = pd.read_csv(args.dataset)
    print(f"  {len(df)} rows, {df['Diameter'].nunique()} diameters")

    kfold_df, lodo_df = run_ablation(df, n_jobs=args.n_jobs, log=log)
    summary = summarise_ablation(kfold_df, lodo_df)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    kfold_df.to_csv(out / "ablation_kfold_per_seed.csv", index=False)
    lodo_df.to_csv(out / "ablation_lodo_by_diameter_per_seed.csv", index=False)
    summary.to_csv(out / "ablation_summary.csv", index=False)

    print(f"\nSaved {out.name}/")
    print(f"  ablation_kfold_per_seed.csv       ({len(kfold_df)} rows)")
    print(f"  ablation_lodo_by_diameter_per_seed.csv ({len(lodo_df)} rows)")
    print(f"  ablation_summary.csv               ({len(summary)} rows)\n")
    print("--- Summary ---")
    print(summary.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
