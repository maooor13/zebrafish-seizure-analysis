"""
Step 6b — Assumption checks: Shapiro-Wilk (per group) + Levene, raw and sqrt.

Justifies the sqrt + HC3 robust-SE path used in step6a_primary.py.
Self-contained (scipy only) — no external skill/repo dependency.

Run: python step6b_assumptions.py   (from code/)
"""
import numpy as np
import pandas as pd
from scipy import stats
from dataset_config import CSV, LABEL

df = pd.read_csv(CSV)
df['Total_distance_sqrt'] = np.sqrt(df['Total_distance_cm'])
print(f"[dataset: {LABEL}]  input: {CSV}")

for name, col in [('raw', 'Total_distance_cm'), ('sqrt', 'Total_distance_sqrt')]:
    groups = {g: s[col].dropna().to_numpy() for g, s in df.groupby('Fish Type')}
    print(f"\n--- {name} ---")
    for g, x in groups.items():
        w, p = stats.shapiro(x)
        print(f"  Shapiro  {g:12s}  W={w:.4f}  p={p:.4g}  {'FAIL' if p < 0.05 else 'ok'}")
    f, p = stats.levene(*groups.values())
    print(f"  Levene   F={f:.3f}  p={p:.4g}  {'FAIL' if p < 0.05 else 'ok'}")
