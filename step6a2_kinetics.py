"""
Step 6a" — Time×group kinetics (collapsed Group × Time_bin mixed ANOVA)
Answers: WHEN does the mutant's excess emerge across the 60-min arc?

5-min × 12 bins, Greenhouse-Geisser corrected (sphericity violated —
autocorrelated time bins). Standard approach (Turrini 2022, Afrikanova 2013).

Run: python step6a2_kinetics.py  (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pingouin as pg
from dataset_config import CSV, LABEL, T_CUTOFF, FIGDIR

if T_CUTOFF is None:
    print("No per-second data for this dataset — skipping step6a2_kinetics.")
    import sys; sys.exit(0)

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

df = pd.read_csv('../AllData.csv')
print(f"[dataset: {LABEL}]  per-second cutoff: {T_CUTOFF}s")

# Strip trial names first so exclusion list matches (AllData has "Trial    28" with spaces)
df['Etho_Trial'] = df['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Manual exclusions (phantom, contamination — things the PerFish filter wouldn't catch)
fish_to_exclude = [
    ('Trial 28', 'C8'),  # phantom tracking — no fish in video
]
for trial, fish in fish_to_exclude:
    n_before = len(df)
    df = df[~((df['Etho_Trial'] == trial) & (df['Fish'] == fish))]
    print(f"Excluded {trial}/{fish}: {n_before - len(df)} rows removed")

# Filter to fish in the dataset's combined CSV (phantom C8 already excluded there too)
szep_fish = pd.read_csv(CSV)[['Etho_Trial', 'Fish']].drop_duplicates()
szep_fish['Etho_Trial'] = szep_fish['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()
df = df.merge(szep_fish, on=['Etho_Trial', 'Fish'], how='inner')

# Filter to the dataset's time window and bin
df = df[df['Time_seconds'] < T_CUTOFF]
BIN_SEC = 300  # 5-min bins
df['Time_bin'] = df['Time_seconds'] // BIN_SEC
n_bins = df['Time_bin'].nunique()
df['Fish_id'] = df['Etho_Trial'] + '_' + df['Fish']  # Etho_Trial already stripped above
df['Fish Type'] = df['Fish Type'].str.strip()

# Per-fish per-bin mean velocity (182 fish × 12 bins)
perfish_bin = df.groupby(['Fish_id', 'Fish Type', 'Etho_Trial', 'Time_bin'], as_index=False)['Velocity'].mean()
perfish_bin['Velocity'] = perfish_bin['Velocity'].astype(float)

print(f"\nData: {perfish_bin['Fish_id'].nunique()} fish × {perfish_bin['Time_bin'].nunique()} bins = {len(perfish_bin)} rows")

# ----------------------------------------------------------------------
# Mixed ANOVA: Group (4) × Time_bin (12), Greenhouse-Geisser
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("MIXED ANOVA: Group × Time_bin (Greenhouse-Geisser corrected)")
print("=" * 70)
aov = pg.mixed_anova(
    dv='Velocity',
    between='Fish Type',
    within='Time_bin',
    subject='Fish_id',
    data=perfish_bin,
    correction='auto'  # Greenhouse-Geisser if sphericity violated
)
print(aov.to_string(index=False))

# Sphericity check
print("\n--- Sphericity (Mauchly) ---")
sph = pg.sphericity(data=perfish_bin, dv='Velocity', within='Time_bin', subject='Fish_id')
print(f"  W = {sph.W:.4f}, p = {sph.pval:.6f} {'-> GG corrected' if sph.pval < 0.05 else '-> sphericity OK'}")

# ----------------------------------------------------------------------
# Post-hoc: when does mut_ptz10 diverge from wt_ptz10?
# Pairwise at each time bin, Bonferroni-corrected across 12 bins
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"POST-HOC: mut_ptz10 vs wt_ptz10 at each time bin (Bonferroni over {n_bins})")
print("=" * 70)
ptz = perfish_bin[perfish_bin['Fish Type'].isin(['wt_ptz10', 'mut_ptz10'])]

pw_rows = []
for tb in sorted(ptz['Time_bin'].unique()):
    wt = ptz[(ptz['Fish Type'] == 'wt_ptz10') & (ptz['Time_bin'] == tb)]['Velocity']
    mut = ptz[(ptz['Fish Type'] == 'mut_ptz10') & (ptz['Time_bin'] == tb)]['Velocity']
    r = pg.mwu(wt, mut, alternative='two-sided')
    p_raw = r['p_val'].values[0]
    p_bonf = min(p_raw * n_bins, 1.0)
    pw_rows.append({
        'time_bin': tb,
        'min_range': f"{tb*5}-{(tb+1)*5} min",
        'wt_mean': wt.mean().round(3),
        'mut_mean': mut.mean().round(3),
        'gap': round(mut.mean() - wt.mean(), 3),
        'p_raw': round(p_raw, 6),
        'p_bonf': round(p_bonf, 6),
        'sig': '***' if p_bonf < 0.001 else ('**' if p_bonf < 0.01 else ('*' if p_bonf < 0.05 else 'ns'))
    })
pw_df = pd.DataFrame(pw_rows)
print(pw_df.to_string(index=False))

# ----------------------------------------------------------------------
# Plot: time-course of mean velocity per group
# ----------------------------------------------------------------------
okabe_ito = {
    'wt_control': '#56B4E9',
    'wt_ptz10': '#E69F00',
    'mut_control': '#009E73',
    'mut_ptz10': '#D55E00',
}

fig, ax = plt.subplots(figsize=(10, 6))

x_max_min = n_bins * 5
time_minutes = np.arange(n_bins) * 5 + 2.5  # bin centers

for group in ['wt_control', 'mut_control', 'wt_ptz10', 'mut_ptz10']:
    sub = perfish_bin[perfish_bin['Fish Type'] == group]
    means = sub.groupby('Time_bin')['Velocity'].mean()
    sems = sub.groupby('Time_bin')['Velocity'].sem()
    ax.plot(time_minutes, means.values, color=okabe_ito[group], label=group, linewidth=2)
    ax.fill_between(time_minutes, means.values - sems.values, means.values + sems.values,
                    color=okabe_ito[group], alpha=0.15)

# Epoch annotations (60-min only; skip for shorter windows)
if x_max_min >= 60:
    epoch_bands = [(0, 15, 'Onset'), (15, 30, 'Peak'), (30, 45, 'Plateau'), (45, 60, 'Exhaustion')]
    for start, end, label in epoch_bands:
        ax.axvspan(start, end, alpha=0.03, color='gray')
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.97, label, ha='center', fontsize=8,
                color='gray', style='italic')

ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_ylabel('Mean Velocity (cm/s)', fontsize=12)
ax.set_title(f'Time-course of velocity by group (5-min bins) — {LABEL}', fontsize=13)
ax.legend(title='Group', fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 0.92))
ax.set_xlim(0, x_max_min)

# Interaction p in text box
interaction_row = aov[aov['Source'] == 'Interaction']
if not interaction_row.empty:
    # pingouin column names vary across versions — find the p column
    p_col = None
    for col in ['p-GG-corr', 'p-unc', 'p_unc', 'pval', 'p-val']:
        if col in interaction_row.columns:
            val = interaction_row[col].values[0]
            if not np.isnan(val):
                p_col = col
                break
    if p_col:
        p_inter = interaction_row[p_col].values[0]
        textstr = f'Group × Time interaction: {format_p(p_inter)}'
        ax.text(0.02, 0.88, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

import os
os.makedirs(FIGDIR, exist_ok=True)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/timecourse_velocity.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGDIR}/timecourse_velocity.pdf', bbox_inches='tight')
print(f"\nPlot saved to {FIGDIR}/timecourse_velocity.png + .pdf")
