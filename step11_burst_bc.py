"""
Step 11 — Burst difference plot (10mM only)
  (mut_ptz10 − wt_ptz10) event count per 5-min bin, with 95% CI band
  (±1.96×SE of the difference, Welch), zero line, per-bin MW stars on event
  count (Bonferroni over window bins).

Event count = number of distinct contiguous burst episodes (velocity > 2.8 cm/s)
per fish per bin, via detect_events (same logic as step11_raster.py).

Per-window Bonferroni divisor (12 for 60-min, 4 for 20-min) — see notes.md
"Per-bin Bonferroni divisor decision": the family is the bins of the protocol.

Self-skips for 5mM (no per-second data).

Run: DS=10mM_full (or 10mM_20min) python step11_burst_bc.py
"""
import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pingouin as pg
from dataset_config import CSV, LABEL, FIGDIR, T_CUTOFF

if T_CUTOFF is None:
    print("No per-second data for this dataset — skipping step11_burst_bc.")
    import sys; sys.exit(0)

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
})
THRESH = 2.8


def sig_stars(p):
    if p < 0.0001: return '****'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'


def count_events(velocity_series, threshold=THRESH):
    """Number of distinct contiguous runs above threshold (per fish, per bin)."""
    v = (velocity_series.values > threshold).astype(int)
    if v.sum() == 0:
        return 0
    diff = np.diff(np.concatenate([[0], v, [0]]))
    return int((diff == 1).sum())


# --- data prep: per-fish per-bin event count (from AllData per-second) ---
alldata = pd.read_csv('../AllData.csv')
alldata['Etho_Trial'] = alldata['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()
alldata['Fish'] = alldata['Fish'].astype(str).str.strip()
alldata['Fish Type'] = alldata['Fish Type'].astype(str).str.strip()
alldata = alldata[~((alldata['Etho_Trial'] == 'Trial 28') & (alldata['Fish'] == 'C8'))]  # phantom
fish_df = pd.read_csv(CSV)[['Etho_Trial', 'Fish']].drop_duplicates()
fish_df['Etho_Trial'] = fish_df['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()
ad = alldata.merge(fish_df, on=['Etho_Trial', 'Fish'], how='inner')
ad = ad[ad['Time_seconds'] < T_CUTOFF].copy()
ad['Time_bin'] = ad['Time_seconds'] // 300
ad['Fish_id'] = ad['Etho_Trial'] + '_' + ad['Fish']

n_bins = int(T_CUTOFF // 300)
x_max_min = n_bins * 5
time_minutes = np.arange(n_bins) * 5 + 2.5  # bin centers

# per-fish per-bin event count
perfish_ev = (ad.groupby(['Fish_id', 'Fish Type', 'Time_bin'])['Velocity']
              .apply(lambda s: count_events(s)).reset_index(name='events'))

wt_by_bin = {tb: perfish_ev[(perfish_ev['Fish Type'] == 'wt_ptz10') & (perfish_ev['Time_bin'] == tb)]['events'] for tb in range(n_bins)}
mut_by_bin = {tb: perfish_ev[(perfish_ev['Fish Type'] == 'mut_ptz10') & (perfish_ev['Time_bin'] == tb)]['events'] for tb in range(n_bins)}

# per-bin MW (per-window Bonferroni)
pw = []
for tb in range(n_bins):
    wt, mut = wt_by_bin[tb], mut_by_bin[tb]
    p_raw = pg.mwu(wt, mut, alternative='two-sided')['p_val'].values[0]
    p_bonf = min(p_raw * n_bins, 1.0)
    pw.append({'tb': tb, 'p_raw': p_raw, 'p_bonf': p_bonf, 'sig': sig_stars(p_bonf)})
pw_df = pd.DataFrame(pw)

print(f"[dataset: {LABEL}]  per-bin MW on event count (Bonferroni over {n_bins}):")
print(pw_df.round(4).to_string(index=False))

# 95% CI for the difference (mut - wt) per bin: ±1.96×SE of the difference (Welch)
def diff_ci(wt, mut):
    d = mut.mean() - wt.mean()
    se = np.sqrt(wt.var(ddof=1)/len(wt) + mut.var(ddof=1)/len(mut))
    return d, d - 1.96*se, d + 1.96*se

diff_rows = [diff_ci(wt_by_bin[tb], mut_by_bin[tb]) for tb in range(n_bins)]
diff_means = [r[0] for r in diff_rows]
diff_lo = [r[1] for r in diff_rows]
diff_hi = [r[2] for r in diff_rows]

os.makedirs(FIGDIR, exist_ok=True)

# --- difference plot (mut − wt) with 95% CI band + zero line + stars ---
fig, ax = plt.subplots(figsize=(max(8, n_bins * 0.8), 5))
ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=1)
ax.fill_between(time_minutes, diff_lo, diff_hi, color='#D55E00', alpha=0.2, label='95% CI (Welch)')
ax.plot(time_minutes, diff_means, 'o-', color='#D55E00', linewidth=2, markersize=6, label='mut − wt')

y_span = max(abs(np.array(diff_hi).max()), abs(np.array(diff_lo).min())) or 1
for _, row in pw_df.iterrows():
    if row['sig'] != 'ns':
        ax.text(row['tb'] * 5 + 2.5, y_span * 1.12, row['sig'], ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Time (minutes)', fontsize=10)
ax.set_ylabel('Difference (mut − wt), burst events per fish', fontsize=10)
ax.set_title(f'Burst event difference (mut − wt) per bin — {LABEL}\n(95% CI of the difference, Welch SE; stars = per-bin MW on event count, Bonferroni over {n_bins})',
             fontsize=11, fontweight='bold')
ax.legend(title='', fontsize=9, loc='upper left')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_xlim(0, x_max_min)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/burst_difference.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGDIR}/burst_difference.pdf', bbox_inches='tight')
plt.close()
print(f'-> {FIGDIR}/burst_difference.png + .pdf')
