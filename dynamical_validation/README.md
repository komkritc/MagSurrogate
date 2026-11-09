# dynamical_validation/

Comparison of relaxation dynamics of one FePt grain (manuscript Sec. 5.4, Figs 9-10): atomistic spin dynamics
(**VAMPIRE**, LLG, constant damping α₀ = 0.1) against a single-spin Landau-Lifshitz-Bloch model (**MARS-RF**) whose
m_e(T), χ∥(T), χ⊥(T) come from the RF surrogate look-up table. The test checks the RF-derived constitutive relations
in a dynamical setting. 

## A. Simulation generation (needs external executables; not run here)

Neither VAMPIRE nor MARS source code or binaries are distributed here. VAMPIRE is open source
(<https://vampire.york.ac.uk>); the MARS executable is not included and its licence status is not documented in the available materials.

| item | location |
|---|---|
| VAMPIRE input template, material file, one 0 K example | `input/vampire/` |
| MARS-RF main configuration and constant-damping material (T_c = 639 K) | `input/marsrf/` |
| RF look-up table used by every MARS-RF run (`m_s`, `chi_L`, `chi_T`, 2-10 nm, 0-1000 K, 1 K) | `data/lookup_table/marsrf_lut_fept_2d_2-10nm.csv` |
| VAMPIRE run script (field 1 T, 320 000 steps, seeds 1001+n) | `scripts/run_vampire.sh` |
| MARS-RF run script (`FieldDC` = 10 000, read as 1 T, seeds 1001+n) | `scripts/run_marsrf.sh` |

```bash
dynamical_validation/scripts/run_vampire.sh /path/to/vampire-serial 50 100 200 300     # runs/vampire/T<K>_s<n>.dat
dynamical_validation/scripts/run_marsrf.sh  /path/to/MARS_exe       50 100 200 300     # runs/marsrf/H10000_M<K>_s<n>.dat
```

Setup (from the input files): cylinder 4.7 nm diameter x 10 nm height (unit cell 2.72 x 2.72 x 3.85 Å, `effective-damping`
program, LLG Heun integrator, time step 1e-16 s). The system is equilibrated for 200 000 steps (20 ps) at the target
temperature, all spins are rotated 30° from z, and the relaxation is recorded. The MARS-RF runs use `Struct:Avg_Grain_width = 4.7` with
the two-dimensional (T, D) look-up table; how MARS interpolates in D is not documented in the available materials. The analytical
Eq. (19) curve uses m(T) of a 4.7 nm grain taken as 0.6 m(4.5 nm) + 0.4 m(5.0 nm) from the same table (`magsurrogate.pipeline`).

Notes on provenance:

* `input/vampire/vampire_input_template.input` carries `applied-field-strength=10 !T`; the run script overrides it with 1 T
  (the value used for the stored traces). The stored 300 K example input matches the original run input (checked).
* `FieldDC = 1e4` in MARS is read as 1 T. This is an inference (`documentation/01_source_integration_report.md`); no MARS
  documentation was available to confirm the unit. The MARS material file sets `Mat:Ms = 1051` (µs/(a²c) with µs = 3.23 µB and
  the VAMPIRE unit cell gives 1051.65 emu/cm³), against 1.043e6 A/m in the manuscript's Table 2.
* The original run scripts with machine-specific paths are not included; `scripts/` holds the portable versions.

## B. Post-processing (runs from the stored traces)

```bash
python scripts/run_dynamics_postprocessing.py     # fits all 502 traces -> results/dynamics/*.csv
python scripts/reproduce_figures.py                # Figs 9-10 (results/figures/fig09_*, fig10_*)
python dynamical_validation/analysis/analyse_dynamics_sample_size.py   # 50- versus 100-seed comparison at 400 K
```

| quantity | method (`src/magsurrogate/dynamics.py`) |
|---|---|
| magnetisation trajectories | VAMPIRE columns 7-9 after the first 20.01 ps; MARS-RF columns 0-2, renormalised to \|m\| = 1; window 0-10 ps |
| α_eff, f_res | m∥ = tanh(k₁t + d), m⊥ = sech(k₁t + d) sin(k₂t + φ); α_eff = k₁/k₂, f_res = k₂/2π |
| normalised damping | α_eff(T)/α_eff(0 K) of the same code |
| statistics | mean, SD and standard error of the mean over seeds, per temperature (`results/dynamics/summary_per_temperature.csv`) |
| analytical curve | Eq. (19) with α₀ = 0.1, T_c = 639 K and m(T) from the look-up table (`analytical_eq19.csv`) |

The macrospin damping model is exact only for a noiseless macrospin of constant |m|, so at T > 0 it is an approximation.

### Results as regenerated from the stored traces

| T (K) | seeds | α_eff VAMPIRE (mean ± SEM) | α_eff MARS-RF (mean ± SEM) | f_res VAMPIRE / MARS-RF (THz) |
|---|---|---|---|---|
| 0 | 1 | 0.0975 | 0.0975 | 0.4976 / 0.5030 |
| 100 | 50 | 0.0957 ± 0.0013 | 0.0976 ± 0.0001 | 0.4432 / 0.4462 |
| 200 | 50 | 0.0983 ± 0.0023 | 0.0987 ± 0.0002 | 0.3852 / 0.3875 |
| 300 | 50 | 0.1031 ± 0.0032 | 0.1016 ± 0.0003 | 0.3250 / 0.3255 |
| 400 | 100 | 0.1069 ± 0.0034 | 0.1066 ± 0.0003 | 0.2617 / 0.2575 |

α_eff is **fitted**, not an input (α₀ = 0.1 in both codes). Its rise with temperature is a simulation result; the
analytical curve is an independent comparison, not a fit.
