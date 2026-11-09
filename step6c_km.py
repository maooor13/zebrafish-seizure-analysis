"""
Step 6c — Kaplan-Meier on latency to first burst
Answers: does the mutant seize EARLIER than WT?

Uses Latency_first_event_ThrA from PerFish_Primary.csv:
  - Responders: latency = seconds to first event
  - Non-responders: latency = null -> censored at 3600s (recording end)

Stratified by Trial (device-L artifact absorbed — stratification made the
headline MORE significant: p=0.025 -> p=0.007, so it's the correct primary).

Run: python step6c_km.py  (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from statsmodels.duration.survfunc import survdiff
from dataset_config import CSV, LABEL, T_CUTOFF, FIGDIR

# Publication font settings (scientific-visualization skill)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})


def format_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'

df = pd.read_csv(CSV)
print(f"[dataset: {LABEL}]  input: {CSV}")
df['Trial'] = df['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()

if T_CUTOFF is None or 'Latency_first_event_ThrA' not in df.columns:
    print("No burst/latency metrics for this dataset (no per-second data) — skipping step6c_km.")
    import sys; sys.exit(0)

# Censored fish (null latency) get duration=recording end, event=0
REC_END = T_CUTOFF  # 3600 for 60-min, 1200 for 20-min
df['duration'] = df['Latency_first_event_ThrA'].fillna(REC_END)
df['event'] = df['Latency_first_event_ThrA'].notna().astype(int)

print("=" * 70)
print("STEP 6c — KAPLAN-MEIER: Latency to first burst (stratified by Trial)")
print("=" * 70)

# Responder counts per group
print("\n--- Responder counts per group ---")
resp = df.groupby('Fish Type').agg(
    n=('duration', 'count'),
    responders=('event', 'sum'),
    censored=('event', lambda x: (x == 0).sum()),
    median_latency=('duration', lambda x: x[x < REC_END].median() if (x < REC_END).any() else np.nan)
).round(1)
resp['pct_responders'] = (resp['responders'] / resp['n'] * 100).round(1)
print(resp)

# 4-group stratified log-rank
print("\n--- 4-group stratified log-rank (by Trial) ---")
chisq_4g, p_4g = survdiff(
    df['duration'].values, df['event'].values,
    df['Fish Type'].values, strata=df['Trial'].values
)
df_4g = df['Fish Type'].nunique() - 1
print(f"  chi2 = {chisq_4g:.3f}, df = {df_4g}, p = {p_4g:.6f}")

# Headline pairwise: wt_ptz10 vs mut_ptz10 (stratified)
print("\n--- Headline pairwise: wt_ptz10 vs mut_ptz10 (stratified by Trial) ---")
ptz = df[df['Fish Type'].isin(['wt_ptz10', 'mut_ptz10'])]
chisq_pw, p_pw = survdiff(
    ptz['duration'].values, ptz['event'].values,
    ptz['Fish Type'].values, strata=ptz['Trial'].values
)
print(f"  chi2 = {chisq_pw:.3f}, p = {p_pw:.6f} {'*' if p_pw < 0.05 else 'ns'}")

# Baseline sanity: wt_control vs mut_control (stratified)
print("\n--- Baseline: wt_control vs mut_control (stratified by Trial) ---")
ctrl = df[df['Fish Type'].isin(['wt_control', 'mut_control'])]
chisq_ctrl, p_ctrl = survdiff(
    ctrl['duration'].values, ctrl['event'].values,
    ctrl['Fish Type'].values, strata=ctrl['Trial'].values
)
print(f"  chi2 = {chisq_ctrl:.3f}, p = {p_ctrl:.6f} {'*' if p_ctrl < 0.05 else 'ns'}")

# KM plot
okabe_ito = {
    'wt_control': '#56B4E9',
    'wt_ptz10': '#E69F00',
    'mut_control': '#009E73',
    'mut_ptz10': '#D55E00',
}

fig, ax = plt.subplots(figsize=(8, 6))
for group in ['wt_control', 'mut_control', 'wt_ptz10', 'mut_ptz10']:
    sub = df[df['Fish Type'] == group]
    kmf = KaplanMeierFitter(label=group)
    kmf.fit(sub['duration'], sub['event'])
    kmf.plot_survival_function(
        ax=ax, color=okabe_ito[group], ci_show=True,
        show_censors=True, censor_styles={'marker': '|', 'ms': 8}
    )

ax.set_xlabel('Time to first burst event (seconds)', fontsize=12)
ax.set_ylabel('Proportion burst-free', fontsize=12)
ax.set_title(f'Latency to first burst event (2.8 cm/s threshold) — {LABEL}', fontsize=13)
ax.legend(title='Group', fontsize=10, loc='upper right')
ax.set_xlim(0, REC_END + 100)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

textstr = (f'4-group log-rank (strat): {format_p(p_4g)}\n'
           f'wt_ptz10 vs mut_ptz10 (strat): {format_p(p_pw)}')
ax.text(0.55, 0.05, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

caveat = 'Note: PTZ added at t=0 without acclimation;\nearly latencies may reflect handling stress'
ax.text(0.02, 0.02, caveat, transform=ax.transAxes, fontsize=7, style='italic',
        color='gray', verticalalignment='bottom')

import os
os.makedirs(FIGDIR, exist_ok=True)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/km_latency.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGDIR}/km_latency.pdf', bbox_inches='tight')
print(f"\nPlot saved to {FIGDIR}/km_latency.png + .pdf")
