# =========================================================
# MACHINE LEARNING + PHYSICALLY CONSISTENT
# UNIVERSAL FINITE-SIZE SCALING (FSS)
#
# VERSION 1.0
#
# =========================================================
#
#
# EXACT IMPLEMENTATION OF:
#
#   Eq.(1)
#
#       (Tc_inf - Tc(D))/Tc_inf = (d0/D)^k
#
#   Eq.(3)
#
#       m_e(epsilon)
#       ~ D^(-beta/nu)
#         * M_tilde(D^(1/nu) * epsilon)
#
# =========================================================
#
# IMPORTANT PHYSICS DESIGN
# =========================================================
#
# ML TRAINING:
#
#   • Uses FULL dataset (1–10 nm)
#
# Eq.(1):
#
#   • Uses FULL diameter range
#
# Eq.(3):
#
#   • Uses 2–9 nm
#
# because:
#
#   • Eq.(1) benefits from more points
#   • Eq.(3) requires asymptotic universality
#   • Ultra-small particles distort scaling collapse
#
# =========================================================
#
# IMPORTANT SIGN CONVENTION
# =========================================================
#
# The published paper figure orientation corresponds to:
#
#   epsilon = (T - Tc_inf)/Tc_inf
#
# NOT:
#
#   epsilon = (Tc_inf - T)/Tc_inf
#
# Therefore THIS CODE uses:
#
#   epsilon = (T - Tc_inf)/Tc_inf
#
# to reproduce the published collapse orientation.
#
# =========================================================

import matplotlib
matplotlib.use('Agg')

# =========================================================
# IMPORTS
# =========================================================

import os
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from time import perf_counter

from scipy.optimize import (
    curve_fit,
    minimize
)

from scipy.interpolate import interp1d

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error
)

# =========================================================
# APS / PRB STYLE
# =========================================================

plt.style.use('default')

plt.rcParams.update({

    "font.family": "serif",
    "mathtext.fontset": "cm",

    "font.size": 13,

    "axes.labelsize": 20,
    "axes.titlesize": 18,

    "xtick.labelsize": 14,
    "ytick.labelsize": 14,

    "legend.fontsize": 10,

    "axes.linewidth": 1.4,
    "axes.edgecolor": "black",

    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,

    "xtick.major.size": 6,
    "ytick.major.size": 6,

    "grid.alpha": 0.25,
    "grid.linestyle": "--",

    "savefig.dpi": 300,
    "savefig.bbox": "tight"

})

# =========================================================
# LOGGER
# =========================================================

logging.basicConfig(

    level=logging.INFO,

    format='[%(asctime)s] [%(levelname)s] %(message)s',

    datefmt='%H:%M:%S'

)

def log(msg):

    logging.info(msg)

# =========================================================
# TIMER
# =========================================================

TIMERS = {}

def tic(section):

    TIMERS[section] = {

        "start": perf_counter()

    }

    log(f"[START] {section}")

def toc(section):

    elapsed = perf_counter() - TIMERS[section]["start"]

    log(f"[DONE] {section} ({elapsed:.3f} s)")

# =========================================================
# USER CONFIGURATION
# =========================================================

DATA_FILE = "fept_dataset.csv"

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# RANDOM FOREST
# ---------------------------------------------------------

N_ESTIMATORS = 300

MAX_DEPTH = 18

RANDOM_STATE = 42

# ---------------------------------------------------------
# FIGURE SETTINGS
# ---------------------------------------------------------

FIGSIZE = (7.5, 5.5)

LINEWIDTH = 2.5

MARKERSIZE = 7

MARKEVERY = 6

# =========================================================
# Eq.(1) DIAMETERS
#
# FULL RANGE
# =========================================================

USE_FULL_EQ1_RANGE = True

CUSTOM_EQ1_DIAMETERS = [

    1,1.5,2,3,4,5,6,7,8,9,10

]

# =========================================================
# Eq.(3) DIAMETERS
#
# PAPER-MATCHING RANGE
# =========================================================

USE_FULL_EQ3_RANGE = False

CUSTOM_EQ3_DIAMETERS = [

    2,3,4,5,6,7,8,9

]

# =========================================================
# LOAD DATASET
# =========================================================

tic("Load Dataset")

df_full = pd.read_csv(DATA_FILE)

df_full = df_full.sort_values(

    by=["Diameter", "Temperature"]

).reset_index(drop=True)

log(f"Dataset shape = {df_full.shape}")

toc("Load Dataset")

# =========================================================
# VERIFY COLUMNS
# =========================================================

required_columns = [

    "Temperature",
    "Diameter",

    "Mx",

    "Chi_longitudinal",

    "Chi_trans"

]

for col in required_columns:

    if col not in df_full.columns:

        raise ValueError(
            f"Missing column: {col}"
        )

# =========================================================
# ALL DIAMETERS
# =========================================================

all_diameters = sorted(

    df_full["Diameter"].unique()

)

# =========================================================
# Eq.(1) DIAMETERS
# =========================================================

if USE_FULL_EQ1_RANGE:

    EQ1_DIAMETERS = all_diameters.copy()

else:

    EQ1_DIAMETERS = CUSTOM_EQ1_DIAMETERS

# =========================================================
# Eq.(3) DIAMETERS
# =========================================================

if USE_FULL_EQ3_RANGE:

    EQ3_DIAMETERS = all_diameters.copy()

else:

    EQ3_DIAMETERS = CUSTOM_EQ3_DIAMETERS

log(f"All diameters  = {all_diameters}")

log(f"Eq.(1) diameters = {EQ1_DIAMETERS}")

log(f"Eq.(3) diameters = {EQ3_DIAMETERS}")

# =========================================================
# ML FEATURES / TARGETS
#
# IMPORTANT:
#
# ML ALWAYS USES FULL DATASET
# =========================================================

X = df_full[[

    "Temperature",
    "Diameter"

]]

y = df_full[[

    "Mx",

    "Chi_longitudinal",

    "Chi_trans"

]]

# =========================================================
# RANDOM FOREST
# =========================================================

tic("Train Random Forest")

model = MultiOutputRegressor(

    RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        random_state=RANDOM_STATE,

        n_jobs=-1

    )

)

model.fit(X, y)

toc("Train Random Forest")

# =========================================================
# PREDICTIONS
# =========================================================

tic("Generate Predictions")

y_pred = model.predict(X)

toc("Generate Predictions")

# =========================================================
# BUILD PREDICTION DATAFRAME
# =========================================================

df_pred = pd.DataFrame({

    "Temperature": df_full["Temperature"],

    "Diameter": df_full["Diameter"],

    "m_e_RF": y_pred[:,0],

    "Chi_longitudinal_RF": y_pred[:,1],

    "Chi_trans_RF": y_pred[:,2]

})

# =========================================================
# EXPORT DATASET
# =========================================================

dataset_csv = os.path.join(

    OUTPUT_DIR,

    "ML_FSS_Dataset.csv"

)

df_pred.to_csv(dataset_csv, index=False)

log(f"Saved dataset: {dataset_csv}")

# =========================================================
# REFINED Tc EXTRACTION
#
# LOCAL QUADRATIC INTERPOLATION
# =========================================================

def extract_Tc(T, chi):

    idx = np.argmax(chi)

    # -----------------------------------------------------
    # SAFETY CHECK
    # -----------------------------------------------------

    if idx == 0 or idx == len(T)-1:

        return T[idx]

    # -----------------------------------------------------
    # LOCAL WINDOW
    # -----------------------------------------------------

    T_local = T[idx-1:idx+2]

    chi_local = chi[idx-1:idx+2]

    # -----------------------------------------------------
    # QUADRATIC FIT
    #
    # y = aT^2 + bT + c
    # -----------------------------------------------------

    coeff = np.polyfit(

        T_local,

        chi_local,

        2

    )

    a, b, c = coeff

    # -----------------------------------------------------
    # PARABOLA VERTEX
    # -----------------------------------------------------

    if abs(a) < 1e-12:

        return T[idx]

    Tc_refined = -b / (2*a)

    return Tc_refined

# =========================================================
# BUILD Tc TABLE
# =========================================================

tic("Extract Tc(D)")

tc_data = []

for d in EQ1_DIAMETERS:

    subset = df_pred[
        np.isclose(df_pred["Diameter"], d)
    ].sort_values("Temperature")

    T = subset["Temperature"].values

    chi = subset["Chi_trans_RF"].values

    Tc = extract_Tc(T, chi)

    tc_data.append({

        "Diameter": d,

        "Tc_RF": Tc

    })

    log(f"D = {d:.1f} nm --> Tc = {Tc:.4f} K")

toc("Extract Tc(D)")

df_tc = pd.DataFrame(tc_data)

# =========================================================
# Eq.(1)
# =========================================================

def scaling_law(D, Tc_inf, d0, k):

    return Tc_inf * (

        1.0 - (d0 / D)**k

    )

# =========================================================
# FIT Eq.(1)
# =========================================================

tic("Finite-Size Scaling Fit")

D_vals = df_tc["Diameter"].values

Tc_vals = df_tc["Tc_RF"].values

popt, pcov = curve_fit(

    scaling_law,

    D_vals,

    Tc_vals,

    p0=[675,0.7,1.0],

    bounds=(

        [500,0.1,0.5],

        [800,5.0,5.0]

    ),

    maxfev=10000

)

Tc_inf = popt[0]

d0_fit = popt[1]

k_fit = popt[2]

# ---------------------------------------------------------
# PARAMETER UNCERTAINTY
# ---------------------------------------------------------

perr = np.sqrt(np.diag(pcov))

Tc_inf_err = perr[0]

d0_err = perr[1]

k_err = perr[2]

# ---------------------------------------------------------
# FIT QUALITY
# ---------------------------------------------------------

Tc_fit = scaling_law(

    D_vals,

    Tc_inf,

    d0_fit,

    k_fit

)

r2 = r2_score(

    Tc_vals,

    Tc_fit

)

rmse = np.sqrt(

    mean_squared_error(

        Tc_vals,

        Tc_fit

    )

)

toc("Finite-Size Scaling Fit")

# =========================================================
# PRINT Eq.(1)
# =========================================================

print("\n" + "="*70)

print("Eq.(1) FINITE-SIZE Tc FIT")

print("="*70)

print(f"Tc_inf = {Tc_inf:.6f} ± {Tc_inf_err:.6f} K")

print(f"d0     = {d0_fit:.6f} ± {d0_err:.6f} nm")

print(f"k      = {k_fit:.6f} ± {k_err:.6f}")

print(f"R^2    = {r2:.6f}")

print(f"RMSE   = {rmse:.6f}")

# =========================================================
# FSS BUNDLE
# =========================================================

class FSSBundle:

    def __init__(self, diameter, T, M):

        self.size = diameter

        self.T = T

        self.M = M

# =========================================================
# BUILD Eq.(3) BUNDLES
# =========================================================

bundles = []

for d in EQ3_DIAMETERS:

    subset = df_pred[
        np.isclose(df_pred["Diameter"], d)
    ].sort_values("Temperature")

    bundles.append(

        FSSBundle(

            d,

            subset["Temperature"].values,

            subset["m_e_RF"].values

        )

    )

# =========================================================
# Eq.(3)
# =========================================================

class FitParameters:

    def __init__(self, par_array):

        self.update(par_array)

    def update(self, par_array):

        self.Tc = par_array[0]

        self.nu = par_array[1]

        self.beta = par_array[2]

    def scaling_transform(self, bundle):

        eps = (

            bundle.T - self.Tc

        ) / self.Tc

        x = (

            bundle.size ** (1.0 / self.nu)

        ) * eps

        y = (

            bundle.size ** (

                self.beta / self.nu

            )

        ) * bundle.M

        idx = np.argsort(x)

        x = x[idx]

        y = y[idx]

        return x, y

# =========================================================
# GENERATE INTERPOLATORS
# =========================================================

def generate_interpolators(

    fitpars,

    bundles

):

    interpolators = []

    for b in bundles:

        x, y = fitpars.scaling_transform(b)

        x_unique, idx_unique = np.unique(

            x,

            return_index=True

        )

        y_unique = y[idx_unique]

        interp = interp1d(

            x_unique,

            y_unique,

            kind='linear',

            bounds_error=False,

            fill_value=np.nan

        )

        interpolators.append(interp)

    return interpolators

# =========================================================
# COLLAPSE ERROR Pb
# =========================================================

def collapse_error(

    interpolators,

    fitpars,

    bundles

):

    pb = 0.0

    counter = 0

    for i in range(len(bundles)):

        for j in range(len(bundles)):

            if i == j:
                continue

            x1, y1 = fitpars.scaling_transform(

                bundles[i]

            )

            x2, y2 = fitpars.scaling_transform(

                bundles[j]

            )

            xmin = max(

                np.min(x1),

                np.min(x2)

            )

            xmax = min(

                np.max(x1),

                np.max(x2)

            )

            if xmax <= xmin:
                continue

            x_common = np.linspace(

                xmin,

                xmax,

                300

            )

            y1_interp = np.interp(

                x_common,

                x1,

                y1

            )

            y2_interp = np.interp(

                x_common,

                x2,

                y2

            )

            valid = (

                np.isfinite(y1_interp)

                &

                np.isfinite(y2_interp)

            )

            if np.sum(valid) < 10:
                continue

            diff = (

                y1_interp[valid]

                -

                y2_interp[valid]

            )

            pb += np.mean(diff**2)

            counter += 1

    if counter == 0:

        return 1e30

    return pb / counter

# =========================================================
# OBJECTIVE FUNCTION
# =========================================================

def objective_function(

    par_array,

    fitpars,

    bundles

):

    Tc, nu, beta = par_array

    if Tc <= 0:
        return 1e30

    if nu <= 0:
        return 1e30

    if beta <= 0:
        return 1e30

    fitpars.update(par_array)

    interpolators = generate_interpolators(

        fitpars,

        bundles

    )

    pb = collapse_error(

        interpolators,

        fitpars,

        bundles

    )

    return pb

# =========================================================
# INITIAL GUESS
# =========================================================

p0 = [

    Tc_inf,

    0.85,

    0.33

]

# =========================================================
# BOUNDS
# =========================================================

bounds = [

    (500,800),

    (0.4,1.5),

    (0.05,0.8)

]

# =========================================================
# OPTIMIZE Eq.(3)
# =========================================================

tic("Universal FSS Optimization")

fitpars = FitParameters(p0)

result = minimize(

    objective_function,

    p0,

    args=(fitpars, bundles),

    method='L-BFGS-B',

    bounds=bounds,

    options={

        'maxiter': 10000,

        'maxfun': 10000

    }

)

Tc_opt, nu_opt, beta_opt = result.x

fitpars.update(result.x)

interpolators_final = generate_interpolators(

    fitpars,

    bundles

)

collapse_error_final = collapse_error(

    interpolators_final,

    fitpars,

    bundles

)

toc("Universal FSS Optimization")

# =========================================================
# PRINT Eq.(3)
# =========================================================

print("\n" + "="*70)

print("Eq.(3) UNIVERSAL FSS RESULTS")

print("="*70)

print(f"Tc_inf = {Tc_opt:.6f} K")

print(f"nu     = {nu_opt:.6f}")

print(f"beta   = {beta_opt:.6f}")

print(f"Pb     = {collapse_error_final:.8e}")

# =========================================================
# STYLE MAPS
# =========================================================

colors = [

    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",

    "#ff7f0e",
    "#17becf",
    "#8c564b",
    "#e377c2",

    "#7f7f7f",
    "#bcbd22"

]

markers = [

    'o','s','^','D',
    'v','P','X','<',
    '>','h'

]

# =========================================================
# FIGURE 1
# =========================================================

tic("Figure 1")

plt.figure(figsize=FIGSIZE)

plt.scatter(

    D_vals,

    Tc_vals,

    marker='^',

    s=85,

    facecolors='white',

    edgecolors='black',

    linewidth=2,

    zorder=5,

    label='RF prediction'

)

D_plot = np.linspace(

    min(D_vals),

    max(D_vals),

    400

)

Tc_plot = scaling_law(

    D_plot,

    Tc_inf,

    d0_fit,

    k_fit

)

plt.plot(

    D_plot,

    Tc_plot,

    '--',

    linewidth=2.8,

    color='red',

    label='Finite-size scaling fit'

)

plt.axhline(

    y=Tc_inf,

    linestyle=':',

    linewidth=2,

    color='black'

)

plt.xlabel(

    r"Diameter $D$ (nm)"

)

plt.ylabel(

    r"$T_c(D)$ (K)"

)

plt.title(

    "Finite-Size Scaling of Curie Temperature"

)

plt.ylim(400,800)

plt.grid(alpha=0.3)

plt.legend()

plt.text(

    0.05,

    0.74,

    rf"$T_c^\infty = {Tc_inf:.2f} \ \mathrm{{K}}$",

    transform=plt.gca().transAxes,

    fontsize=12,

    verticalalignment='top'

)

plt.tight_layout()

fig1 = os.path.join(

    OUTPUT_DIR,

    "Fig1_Tc_vs_D.png"

)

plt.savefig(fig1)

plt.close()

toc("Figure 1")

# =========================================================
# FIGURE 2
# =========================================================

tic("Figure 2")

plt.figure(figsize=FIGSIZE)

for i, d in enumerate(all_diameters):

    subset = df_pred[
        np.isclose(df_pred["Diameter"], d)
    ].sort_values("Temperature")

    T = subset["Temperature"].values

    M = subset["m_e_RF"].values

    plt.plot(

        T,

        M,

        color=colors[i % len(colors)],

        marker=markers[i % len(markers)],

        linewidth=LINEWIDTH,

        markersize=MARKERSIZE,

        markerfacecolor='white',

        markevery=MARKEVERY,

        label=f"{d:.1f} nm"

    )

plt.xlabel(

    r"Temperature $T$ (K)"

)

plt.ylabel(

    r"$m_e(T,D)$"

)

plt.title(

    "Predicted Equilibrium Magnetization Curves"

)

plt.grid(alpha=0.3)

plt.legend(

    fontsize=9,

    ncol=2

)

plt.tight_layout()

fig2 = os.path.join(

    OUTPUT_DIR,

    "Fig2_me_vs_T.png"

)

plt.savefig(fig2)

plt.close()

toc("Figure 2")

# =========================================================
# FIGURE 3
# =========================================================

tic("Figure 3")

plt.figure(figsize=FIGSIZE)

for i, d in enumerate(EQ3_DIAMETERS):

    subset = df_pred[
        np.isclose(df_pred["Diameter"], d)
    ].sort_values("Temperature")

    T = subset["Temperature"].values

    M = subset["m_e_RF"].values

    eps = (

        T - Tc_opt

    ) / Tc_opt

    x_scaled = (

        d ** (1.0 / nu_opt)

    ) * eps

    y_scaled = (

        d ** (

            beta_opt / nu_opt

        )

    ) * M

    plt.plot(

        x_scaled,

        y_scaled,

        color=colors[i % len(colors)],

        marker=markers[i % len(markers)],

        linewidth=LINEWIDTH,

        markersize=MARKERSIZE,

        markerfacecolor='white',

        markevery=MARKEVERY,

        label=f"{d:.1f} nm"

    )

plt.xlabel(

    r"$D^{1/\nu}(T-{T_c^{\infty}})/{T_c^{\infty}}$"

)

plt.ylabel(

    r"$D^{\beta/\nu}m_e(T,D)$"

)

plt.title(

    "Universal Finite-Size Scaling Collapse"

)

plt.grid(alpha=0.3)

plt.minorticks_on()

plt.xlim(-6,6)

plt.ylim(-0.05,2.4)

plt.legend(

    fontsize=9,

    ncol=2,

    loc='upper right'

)

plt.text(

    0.73,

    0.45,

    rf"${{T_c^{{\infty}}}}$ = {Tc_opt:.2f} K" + "\n"
    rf"$\nu$ = {nu_opt:.4f}" + "\n"
    rf"$\beta$ = {beta_opt:.4f}" + "\n"
    rf"$P_b$ = {collapse_error_final:.6e}",

    transform=plt.gca().transAxes,

    fontsize=11,

    verticalalignment='top',

    bbox=dict(

        facecolor='white',

        edgecolor='gray',

        boxstyle='round,pad=0.35',

        alpha=0.96

    )

)

plt.tight_layout()

fig3 = os.path.join(

    OUTPUT_DIR,

    "Fig3_Universal_FSS.png"

)

plt.savefig(fig3)

plt.close()

toc("Figure 3")

# =========================================================
# EXPORT Tc TABLE
# =========================================================

tc_csv = os.path.join(

    OUTPUT_DIR,

    "Tc_vs_D.csv"

)

df_tc.to_csv(tc_csv, index=False)

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n" + "="*70)

print("ML + UNIVERSAL FSS COMPLETED")

print("="*70)

print("\nEq.(1) DIAMETERS")

print("-"*40)

print(EQ1_DIAMETERS)

print("\nEq.(3) DIAMETERS")

print("-"*40)

print(EQ3_DIAMETERS)

print("\nEq.(1) PARAMETERS")

print("-"*40)

print(f"Tc_inf = {Tc_inf:.6f} ± {Tc_inf_err:.6f} K")

print(f"d0     = {d0_fit:.6f} ± {d0_err:.6f} nm")

print(f"k      = {k_fit:.6f} ± {k_err:.6f}")

print(f"R^2    = {r2:.6f}")

print(f"RMSE   = {rmse:.6f}")

print("\nEq.(3) PARAMETERS")

print("-"*40)

print(f"Tc_inf = {Tc_opt:.6f} K")

print(f"nu     = {nu_opt:.6f}")

print(f"beta   = {beta_opt:.6f}")

print(f"\nCollapse error Pb = {collapse_error_final:.8e}")

print("\nOUTPUT DIRECTORY")

print("-"*40)

print(OUTPUT_DIR)

print("\nGENERATED FILES")

print("-"*40)

print("1. ML_FSS_Dataset.csv")
print("2. Tc_vs_D.csv")
print("3. Fig1_Tc_vs_D.png")
print("4. Fig2_me_vs_T.png")
print("5. Fig3_Universal_FSS.png")

print("\n" + "="*70)

print("DONE")

print("="*70)