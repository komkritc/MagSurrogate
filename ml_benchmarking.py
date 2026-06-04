# =========================================================
# PHYSICS-AWARE ML BENCHMARK PIPELINE
# =========================================================
#
# Author:
#   Komkrit Chooruang
#
# =========================================================
# DESCRIPTION
# =========================================================
#
# This framework performs:
#
#   • Physics-aware ML benchmarking
#   • Leave-One-Diameter-Out Cross Validation (LODO-CV)
#   • Unseen geometry validation
#   • Classical interpolation benchmarking
#   • ML surrogate benchmarking
#   • Publication-quality plotting
#   • CSV export
#
# =========================================================
# INCLUDED MODELS
# =========================================================
#
# Classical Methods:
#
#   • Polynomial Order-9
#   • Cubic Spline
#
# Machine Learning Methods:
#
#   • Random Forest (RF)
#   • Gradient Boosting (GB)
#   • XGBoost (XGB)
#   • K-Nearest Neighbor (KNN)
#
# =========================================================
# GENERATED OUTPUTS
# =========================================================
#
# CSV:
#
#   • benchmark_LODO_summary.csv
#   • LODO_fold_results.csv
#   • LODO_predictions.csv
#   • Best_Hyperparameters.csv
#
# FIGURES:
#
#   • Fig_Benchmark_Comparison_LODO.png
#   • Fig_LODO_Generalization_5x3.png
#
# =========================================================
# IMPORTS
# =========================================================
import matplotlib
matplotlib.use('Agg')

import time
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

from scipy.interpolate import UnivariateSpline

# =========================================================
# SCIKIT-LEARN
# =========================================================
from sklearn.model_selection import (

    LeaveOneGroupOut,
    GridSearchCV
)

from sklearn.metrics import (

    r2_score,
    mean_squared_error
)

from sklearn.multioutput import (
    MultiOutputRegressor
)

from sklearn.pipeline import (
    Pipeline
)

from sklearn.preprocessing import (
    StandardScaler
)

# =========================================================
# ML MODELS
# =========================================================
from sklearn.ensemble import (

    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.neighbors import (
    KNeighborsRegressor
)

# =========================================================
# XGBOOST
# =========================================================
from xgboost import (
    XGBRegressor
)

# =========================================================
# VERSION
# =========================================================
VERSION = "1.0"

# =========================================================
# RANDOM SEED
# =========================================================
RANDOM_STATE = 42

# =========================================================
# LOGGER
# =========================================================
def log(msg):

    print(
        f"[{time.strftime('%H:%M:%S')}] {msg}",
        flush=True
    )

# =========================================================
# PUBLICATION STYLE
# =========================================================
plt.style.use('default')

plt.rcParams.update({

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 13,

    "axes.labelsize": 18,
    "axes.titlesize": 16,

    "axes.linewidth": 1.4,

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.top": True,
    "ytick.right": True,

    "legend.frameon": False,

    "grid.alpha": 0.25,
    "grid.linestyle": "--",

    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# =========================================================
# CONFIGURATION
# =========================================================
DATA_FILE = "fept_dataset.csv"

FEATURES = [

    "Temperature",
    "Diameter"
]

TARGETS = [

    "Mx",
    "Chi_longitudinal",
    "Chi_trans"
]

# =========================================================
# LOAD DATASET
# =========================================================
log("Loading dataset...")

df = pd.read_csv(DATA_FILE)

X = df[FEATURES]

y = df[TARGETS]

groups = df["Diameter"]

unique_diameters = np.sort(

    groups.unique()
)

log(f"Dataset shape = {df.shape}")

log(f"Diameters = {unique_diameters}")

# =========================================================
# LODO SETUP
# =========================================================
logo = LeaveOneGroupOut()

# =========================================================
# POLYNOMIAL REGRESSOR
# =========================================================
class PolynomialRegressor:

    """
    Classical polynomial regression baseline.
    """

    def __init__(self, degree=9):

        self.degree = degree

        self.models = {}

    def fit(self, X, y):

        T = X["Temperature"].values

        for col in y.columns:

            coeffs = np.polyfit(

                T,
                y[col].values,

                self.degree
            )

            self.models[col] = coeffs

    def predict(self, X):

        T = X["Temperature"].values

        preds = []

        for col in TARGETS:

            yp = np.polyval(

                self.models[col],
                T
            )

            preds.append(yp)

        return np.vstack(preds).T

# =========================================================
# SPLINE REGRESSOR
# =========================================================
class SplineRegressor:

    """
    Cubic spline interpolation.
    """

    def __init__(self, s=0):

        self.s = s

        self.models = {}

    def fit(self, X, y):

        self.models = {}

        diameters = np.sort(

            X["Diameter"].unique()
        )

        for d in diameters:

            mask = X["Diameter"] == d

            Xd = X[mask]

            yd = y[mask]

            order = np.argsort(

                Xd["Temperature"].values
            )

            T = Xd["Temperature"].values[order]

            self.models[d] = {}

            for col in y.columns:

                values = yd[col].values[order]

                spline = UnivariateSpline(

                    T,
                    values,
                    s=self.s
                )

                self.models[d][col] = spline

    def predict(self, X):

        preds_all = []

        for idx in range(len(X)):

            T = X.iloc[idx]["Temperature"]

            d = X.iloc[idx]["Diameter"]

            row_preds = []

            # =============================================
            # HANDLE UNSEEN DIAMETER
            # =============================================
            if d not in self.models:

                nearest_d = min(

                    self.models.keys(),

                    key=lambda x: abs(x - d)
                )

                d_use = nearest_d

            else:

                d_use = d

            for col in TARGETS:

                yp = self.models[d_use][col](T)

                row_preds.append(yp)

            preds_all.append(row_preds)

        return np.array(preds_all)

# =========================================================
# MODEL DEFINITIONS
# =========================================================
models = {

    # =====================================================
    # CLASSICAL BASELINES
    # =====================================================
    "POLY9": PolynomialRegressor(
        degree=9
    ),

    "SPLINE": SplineRegressor(
        s=0
    ),

    # =====================================================
    # RANDOM FOREST
    # =====================================================
    "RF": MultiOutputRegressor(

        RandomForestRegressor(

            random_state=RANDOM_STATE,

            n_jobs=-1
        )
    ),

    # =====================================================
    # GRADIENT BOOSTING
    # =====================================================
    "GB": MultiOutputRegressor(

        GradientBoostingRegressor(

            random_state=RANDOM_STATE
        )
    ),

    # =====================================================
    # XGBOOST
    # =====================================================
    "XGB": MultiOutputRegressor(

        XGBRegressor(

            random_state=RANDOM_STATE,

            objective='reg:squarederror',

            tree_method='hist',

            n_jobs=-1
        )
    ),

    # =====================================================
    # KNN
    # =====================================================
    "KNN": MultiOutputRegressor(

        Pipeline([

            (
                "scaler",
                StandardScaler()
            ),

            (
                "knn",
                KNeighborsRegressor()
            )
        ])
    )
}

# =========================================================
# HYPERPARAMETER SEARCH SPACE
# =========================================================
param_grids = {

    "RF": {

        "estimator__n_estimators": [100, 200],

        "estimator__max_depth": [10, 18, None]
    },

    "GB": {

        "estimator__n_estimators": [100, 200],

        "estimator__learning_rate": [0.03, 0.05, 0.1]
    },

    "XGB": {

        "estimator__n_estimators": [100, 200],

        "estimator__max_depth": [3, 5, 7],

        "estimator__learning_rate": [0.03, 0.05, 0.1]
    },

    "KNN": {

        "estimator__knn__n_neighbors": [3, 5, 7]
    }
}

# =========================================================
# METRIC FUNCTION
# =========================================================
def evaluate(y_true, y_pred):

    r2 = r2_score(

        y_true,
        y_pred,

        multioutput='raw_values'
    )

    rmse = np.sqrt(

        mean_squared_error(

            y_true,
            y_pred,

            multioutput='raw_values'
        )
    )

    return r2, rmse

# =========================================================
# STORAGE
# =========================================================
results = []

fold_results = []

prediction_storage = []

best_hyperparameters = {}

# =========================================================
# MAIN BENCHMARK LOOP
# =========================================================
for name, model in models.items():

    log("\n=================================================")

    log(f"MODEL = {name}")

    log("=================================================")

    # =====================================================
    # CLASSICAL METHODS
    # =====================================================
    if name in ["POLY9", "SPLINE"]:

        best_model = model

        if name == "POLY9":

            best_hyperparameters[name] = {

                "degree": 9
            }

        else:

            best_hyperparameters[name] = {

                "smoothing_factor": 0
            }

    # =====================================================
    # ML MODELS
    # =====================================================
    else:

        log("Running GridSearchCV with LODO-CV...")

        grid = GridSearchCV(

            estimator=model,

            param_grid=param_grids[name],

            cv=logo.split(X, y, groups),

            scoring='neg_mean_squared_error',

            n_jobs=-1,

            verbose=1
        )

        with warnings.catch_warnings():

            warnings.simplefilter("ignore")

            grid.fit(X, y)

        best_model = grid.best_estimator_

        best_hyperparameters[name] = grid.best_params_

        log(f"Best Parameters = {grid.best_params_}")

    # =====================================================
    # STORAGE
    # =====================================================
    r2_all = []

    rmse_all = []

    train_times = []

    infer_times = []

    # =====================================================
    # LODO LOOP
    # =====================================================
    for fold, (train_idx, test_idx) in enumerate(

        logo.split(X, y, groups),
        1
    ):

        Xtr = X.iloc[train_idx]
        Xte = X.iloc[test_idx]

        ytr = y.iloc[train_idx]
        yte = y.iloc[test_idx]

        test_diameter = Xte["Diameter"].iloc[0]

        log(f"\nFold {fold}")

        log(f"Test Diameter = {test_diameter} nm")

        # =================================================
        # TRAIN
        # =================================================
        t0 = time.time()

        best_model.fit(Xtr, ytr)

        train_time = time.time() - t0

        # =================================================
        # PREDICT
        # =================================================
        t0 = time.time()

        yp = best_model.predict(Xte)

        infer_time = (

            time.time() - t0

        ) / len(Xte)

        # =================================================
        # METRICS
        # =================================================
        r2, rmse = evaluate(yte, yp)

        log(f"R² = {r2}")

        log(f"RMSE = {rmse}")

        r2_all.append(r2)

        rmse_all.append(rmse)

        train_times.append(train_time)

        infer_times.append(infer_time)

        # =================================================
        # STORE FOLD RESULTS
        # =================================================
        fold_results.append({

            "Model": name,
            "Fold": fold,
            "Test_Diameter": test_diameter,

            "R2_Mx": r2[0],
            "R2_ChiPara": r2[1],
            "R2_ChiPerp": r2[2],

            "RMSE_Mx": rmse[0],
            "RMSE_ChiPara": rmse[1],
            "RMSE_ChiPerp": rmse[2],

            "Train_Time_sec": train_time,
            "Infer_Time_sec_per_sample": infer_time
        })

        # =================================================
        # STORE PREDICTIONS
        # =================================================
        for idx in range(len(Xte)):

            prediction_storage.append({

                "Model": name,

                "Fold": fold,

                "Diameter":
                    Xte.iloc[idx]["Diameter"],

                "Temperature":
                    Xte.iloc[idx]["Temperature"],

                "Mx_true":
                    yte.iloc[idx]["Mx"],

                "ChiPara_true":
                    yte.iloc[idx]["Chi_longitudinal"],

                "ChiPerp_true":
                    yte.iloc[idx]["Chi_trans"],

                "Mx_pred":
                    yp[idx,0],

                "ChiPara_pred":
                    yp[idx,1],

                "ChiPerp_pred":
                    yp[idx,2]
            })

    # =====================================================
    # STORE SUMMARY
    # =====================================================
    results.append([

        name,

        *np.mean(r2_all, axis=0),

        *np.mean(rmse_all, axis=0),

        np.mean(train_times),

        np.mean(infer_times)
    ])

# =========================================================
# SUMMARY DATAFRAME
# =========================================================
df_results = pd.DataFrame(results, columns=[

    "Model",

    "R2_Mx",
    "R2_ChiPara",
    "R2_ChiPerp",

    "RMSE_Mx",
    "RMSE_ChiPara",
    "RMSE_ChiPerp",

    "Train_time_sec",

    "Infer_time_sec_per_sample"
])

# =========================================================
# MEAN RMSE
# =========================================================
df_results["RMSE_mean"] = df_results[

    [
        "RMSE_Mx",
        "RMSE_ChiPara",
        "RMSE_ChiPerp"
    ]

].mean(axis=1)

# =========================================================
# SORT RESULTS
# =========================================================
df_results = df_results.sort_values(

    "RMSE_mean"
)

# =========================================================
# SAVE CSV FILES
# =========================================================
df_results.to_csv(

    "benchmark_LODO_summary.csv",
    index=False
)

pd.DataFrame(fold_results).to_csv(

    "LODO_fold_results.csv",
    index=False
)

pd.DataFrame(prediction_storage).to_csv(

    "LODO_predictions.csv",
    index=False
)

# =========================================================
# SAVE HYPERPARAMETERS
# =========================================================
hyper_rows = []

for model_name, params in best_hyperparameters.items():

    row = {

        "Model": model_name
    }

    for k, v in params.items():

        row[k] = v

    hyper_rows.append(row)

hyper_df = pd.DataFrame(hyper_rows)

hyper_df.to_csv(

    "Best_Hyperparameters.csv",
    index=False
)

log("Saved CSV files successfully")

# =========================================================
# CREATE CLEAN TERMINAL TABLE
# =========================================================
table_df = df_results.copy()

# =========================================================
# ROUND VALUES
# =========================================================
metric_cols = [

    "R2_Mx",
    "R2_ChiPara",
    "R2_ChiPerp",

    "RMSE_Mx",
    "RMSE_ChiPara",
    "RMSE_ChiPerp",

    "RMSE_mean"
]

for col in metric_cols:

    table_df[col] = table_df[col].round(4)

# =========================================================
# SCIENTIFIC NOTATION
# =========================================================
table_df["Train_time_sec"] = table_df[
    "Train_time_sec"
].map(lambda x: f"{x:.2e}")

table_df["Infer_time_sec_per_sample"] = table_df[
    "Infer_time_sec_per_sample"
].map(lambda x: f"{x:.1e}")

# =========================================================
# RENAME COLUMNS
# =========================================================
table_df = table_df.rename(columns={

    "R2_Mx": "R2_Mx",

    "R2_ChiPara": "R2_Chi||",

    "R2_ChiPerp": "R2_Chi⊥",

    "RMSE_Mx": "RMSE_Mx",

    "RMSE_ChiPara": "RMSE_Chi||",

    "RMSE_ChiPerp": "RMSE_Chi⊥",

    "Train_time_sec": "Train (s)",

    "Infer_time_sec_per_sample": "Infer (s)"
})

# =========================================================
# SELECT TABLE COLUMNS
# =========================================================
table_df = table_df[[

    "Model",

    "R2_Mx",
    "R2_Chi||",
    "R2_Chi⊥",

    "RMSE_Mx",
    "RMSE_Chi||",
    "RMSE_Chi⊥",

    "Train (s)",
    "Infer (s)"
]]

# =========================================================
# PRINT FINAL BENCHMARK TABLE
# =========================================================
print("\n")
print("=" * 100)
print("FINAL BENCHMARK SUMMARY")
print("=" * 100)

print(

    table_df.to_string(
        index=False
    )
)

# =========================================================
# PRINT BEST HYPERPARAMETERS
# =========================================================
print("\n")
print("=" * 100)
print("BEST HYPERPARAMETERS")
print("=" * 100)

for model_name, params in best_hyperparameters.items():

    print(f"\n{model_name}")

    for k, v in params.items():

        print(f"  {k} = {v}")

print("\n")

log("Benchmark table displayed successfully")

# =========================================================
# LOAD PREDICTIONS
# =========================================================
pred_df = pd.read_csv(

    "LODO_predictions.csv"
)

# =========================================================
# BENCHMARK COMPARISON FIGURE
# =========================================================
log("\nGenerating benchmark comparison figure...")

models_plot = df_results["Model"].values

r2_mx = df_results["R2_Mx"].values
r2_chiP = df_results["R2_ChiPara"].values
r2_chiT = df_results["R2_ChiPerp"].values

rmse_mx = df_results["RMSE_Mx"].values
rmse_chiP = df_results["RMSE_ChiPara"].values
rmse_chiT = df_results["RMSE_ChiPerp"].values

train_time = df_results["Train_time_sec"].values

fig = plt.figure(figsize=(9,7))

gs = GridSpec(

    2,
    2,

    figure=fig,

    hspace=0.35,
    wspace=0.30
)

x = np.arange(len(models_plot))

width = 0.25

colors = [

    '#1f77b4',
    '#ff7f0e',
    '#2ca02c'
]

# =========================================================
# (a) R² COMPARISON
# =========================================================
ax1 = fig.add_subplot(gs[0,0])

for i, (vals, label) in enumerate([

    (r2_mx, r"$M_x$"),

    (r2_chiP, r"$\chi_{\parallel}$"),

    (r2_chiT, r"$\chi_{\perp}$")
]):

    ax1.bar(

        x + (i-1)*width,

        vals,

        width,

        color=colors[i],

        edgecolor='black',

        linewidth=0.6,

        label=label
    )

ax1.set_ylabel(r"$R^2$")

ax1.set_title("(a)", loc='left')

ax1.set_xticks(x)

ax1.set_xticklabels(models_plot)

ax1.set_ylim(0,1.05)

ax1.grid()

# =========================================================
# (b) RMSE COMPARISON
# =========================================================
ax2 = fig.add_subplot(gs[0,1])

for i, (vals, label) in enumerate([

    (rmse_mx, r"$M_x$"),

    (rmse_chiP, r"$\chi_{\parallel}$"),

    (rmse_chiT, r"$\chi_{\perp}$")
]):

    ax2.bar(

        x + (i-1)*width,

        vals,

        width,

        color=colors[i],

        edgecolor='black',

        linewidth=0.6,

        label=label
    )

ax2.set_ylabel("RMSE")

ax2.set_yscale("log")

ax2.set_title("(b)", loc='left')

ax2.set_xticks(x)

ax2.set_xticklabels(models_plot)

ax2.legend()

ax2.grid(which='both')

# =========================================================
# (c) TRAINING TIME
# =========================================================
ax3 = fig.add_subplot(gs[1,0])

ax3.bar(

    models_plot,

    train_time,

    color='gray',

    edgecolor='black',

    linewidth=0.6
)

ax3.set_ylabel("Training Time (s)")

ax3.set_yscale("log")

ax3.set_title("(c)", loc='left')

ax3.grid()

# =========================================================
# (d) PERFORMANCE VS COST
# =========================================================
ax4 = fig.add_subplot(gs[1,1])

x_pos = np.log10(train_time)

y_pos = (

    np.array(r2_mx)

    + np.array(r2_chiP)

    + np.array(r2_chiT)

) / 3

for i, model in enumerate(models_plot):

    ax4.plot(

        x_pos[i],

        y_pos[i],

        'o',

        markersize=8
    )

    ax4.text(

        x_pos[i],

        y_pos[i],

        model,

        fontsize=10
    )

ax4.set_xlabel(

    r"log$_{10}$(Training Time)"
)

ax4.set_ylabel(

    r"Average $R^2$"
)

ax4.set_title("(d)", loc='left')

ax4.set_ylim(0.5,1.05)

ax4.grid()

plt.tight_layout()

plt.savefig(

    "Fig_Benchmark_Comparison_LODO.png",

    dpi=600
)

plt.close()

log("Saved Fig_Benchmark_Comparison_LODO.png")

# =========================================================
# 5×3 PUBLICATION FIGURE
# =========================================================
log("\nGenerating publication figure...")

selected_diameters = [

    2.0,
    3.0,
    5.0,
    7.0,
    10.0
]

fig, axes = plt.subplots(

    nrows=len(selected_diameters),

    ncols=3,

    figsize=(10,12),

    sharex=True
)

target_map = {

    "Mx": (

        "Mx_true",
        "Mx_pred",

        r"$M_x$"
    ),

    "Chi_longitudinal": (

        "ChiPara_true",
        "ChiPara_pred",

        r"$\chi_{\parallel}$"
    ),

    "Chi_trans": (

        "ChiPerp_true",
        "ChiPerp_pred",

        r"$\chi_{\perp}$"
    )
}

model_styles = {

    "POLY9": "--",

    "SPLINE": ":",

    "RF": "-",

    "GB": "-",

    "XGB": "-",

    "KNN": "-"
}

for row, d in enumerate(selected_diameters):

    for col, target in enumerate(TARGETS):

        ax = axes[row, col]

        true_col, pred_col, title = target_map[target]

        # =================================================
        # GROUND TRUTH
        # =================================================
        truth_subset = pred_df[

            (pred_df["Model"] == "RF") &
            (pred_df["Diameter"] == d)

        ].sort_values("Temperature")

        ax.plot(

            truth_subset["Temperature"],

            truth_subset[true_col],

            color='black',

            linewidth=2.8,

            label="Truth"
            if (row == 0 and col == 0)
            else None
        )

        # =================================================
        # MODEL PREDICTIONS
        # =================================================
        for name in models.keys():

            subset = pred_df[

                (pred_df["Model"] == name) &
                (pred_df["Diameter"] == d)

            ].sort_values("Temperature")

            ax.plot(

                subset["Temperature"],

                subset[pred_col],

                linewidth=1.6,

                linestyle=model_styles[name],

                label=name
                if (row == 0 and col == 0)
                else None
            )

        ax.grid(alpha=0.25)

        ax.set_xlim(0,1000)

        if row == 0:

            ax.set_title(title)

        if col == 0:

            ax.text(

                -0.32,

                0.5,

                rf"$D={d}\ \mathrm{{nm}}$",

                rotation=90,

                transform=ax.transAxes,

                ha='center',

                va='center',

                fontsize=13
            )

# =========================================================
# X LABELS
# =========================================================
for col in range(3):

    axes[-1, col].set_xlabel(

        "Temperature (K)"
    )

# =========================================================
# GLOBAL LEGEND
# =========================================================
handles, labels = axes[0,0].get_legend_handles_labels()

fig.legend(

    handles,
    labels,

    loc='upper center',

    ncol=7,

    frameon=False,

    fontsize=11
)

# =========================================================
# LAYOUT
# =========================================================
plt.tight_layout(

    rect=[0,0,1,0.965]
)

# =========================================================
# SAVE FIGURE
# =========================================================
plt.savefig(

    "Fig_LODO_Generalization_5x3.png",

    dpi=600
)

plt.close()

log("Saved Fig_LODO_Generalization_5x3.png")

# =========================================================
# FINISHED
# =========================================================
log("\nALL TASKS COMPLETED SUCCESSFULLY")

print("\nDONE.")
