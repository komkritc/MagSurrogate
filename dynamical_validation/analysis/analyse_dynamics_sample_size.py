#!/usr/bin/env python3
"""Seed-count sensitivity of the 400 K dynamics comparison: the manuscript states 50 seeds; the stored data hold
100. Writes results/dynamics/seed_count_sensitivity.csv (needs --stage dynamics first)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
per = pd.read_csv(ROOT / "results" / "dynamics" / "per_seed_results.csv")
rows = []
for code in ("AM", "MM"):
    g = per[(per.code == code) & (per["T_K"] == 400)].sort_values("seed")
    for n in (50, 100):
        s = g[g.seed <= n]
        rows.append(dict(code=code, n_seeds=len(s), alpha_mean=s.alpha.mean(), alpha_sem=s.alpha.std(ddof=1) / np.sqrt(len(s)),
                         f_THz_mean=s.f_THz.mean()))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "dynamics" / "seed_count_sensitivity.csv", index=False)
print(out.round(5).to_string(index=False))
