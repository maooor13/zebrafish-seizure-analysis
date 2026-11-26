"""
Step 10 — Figures for the seizure state machine (steps 7-9).

Seizure-based, NOT burst-based: reads the outputs of step 7/7b/8, so numbers
here differ from the existing burst figures (two adjacent bursts = one seizure
here). Existing figures in figures/ are left untouched; these use new names.

Conventions match step11_figures.py / step6c_km.py:
  - Okabe-Ito palette, font sizes 9/7/7
  - bars = mean ± SD + individual jittered dots (black, alpha 0.3)
  - PNG 300 DPI + PDF vector, bbox_inches='tight', format_p p-values

Outputs (into FIGDIR):
  seizure_count_by_group.png   — seizures per fish, bar + dots
  seizure_latency_km.png       — KM latency to first seizure, 4 groups
  seizure_count_quarters.png   — seizures per fish per 15-min quarter
  seizure_burden_by_group.png  — % of tracked time seizing, bar + dots

Run: python step10_figures.py   (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_config import LABEL, FIGDIR, T_CUTOFF

ROLLUP = '../seizure_fish_rollup.csv'
EVENTS = '../seizure_events_iqr_filtered.csv'
GROUP_COL = 'Fish Type'
GROUPS = ['wt_control', 'wt_ptz10', 'mut_control', 'mut_ptz10']
GROUP_LABELS = ['WT\ncontrol', 'WT\nPTZ 10', 'mut\ncontrol', 'mut\nPTZ 10']
OKABE = {'wt_control': '#56B4E9', 'wt_ptz10': '#E69F00',
         'mut_control': '#009E73', 'mut_ptz10': '#D55E00'}
REC_END = T_CUTOFF or 3600

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 9,
    'axes.labelsize': 9, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
})
np.random.seed(42)


def format_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def save_fig(fig, name):
    """Save figure as PNG (300 DPI) + PDF (vector) per scientific-visualization skill."""
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(f'{FIGDIR}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{FIGDIR}/{name}.pdf', bbox_inches='tight')


def sig_stars(p):
    if p < 0.0001: return '****'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'


def headline_p(df, col):
    a = df.loc[df[GROUP_COL] == 'wt_ptz10', col].dropna()
    b = df.loc[df[GROUP_COL] == 'mut_ptz10', col].dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic').pvalue


def mw_p(df, col, g1, g2):
    a = df.loc[df[GROUP_COL] == g1, col].dropna()
    b = df.loc[df[GROUP_COL] == g2, col].dropna()
    return stats.mannwhitneyu(a, b, alternative='two-sided', method='asymptotic').pvalue


def bar_with_dots(ax, df, col, ylabel, scale=1.0, show_bracket=True):
    """Bar chart with jittered dots + significance brackets. Error bars = SD."""
    means = [df.loc[df[GROUP_COL] == g, col].dropna().mean() * scale for g in GROUPS]
    sds = [df.loc[df[GROUP_COL] == g, col].dropna().std() * scale for g in GROUPS]
    x = np.arange(len(GROUPS))
    colors = [OKABE[g] for g in GROUPS]

    ax.bar(x, means, yerr=sds, color=colors, capsize=5,
           edgecolor='black', linewidth=0.5, alpha=0.7)
    for i, g in enumerate(GROUPS):
        vals = df.loc[df[GROUP_COL] == g, col].dropna().to_numpy() * scale
        if len(vals) == 0:
            continue
        jitter = np.random.normal(i, 0.06, size=len(vals))
        ax.scatter(jitter, vals, color='black', alpha=0.3, s=15, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_LABELS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_bracket:
        # 4 brackets at increasing heights: WT PTZ, Mut PTZ, baseline genotype, genotype under PTZ
        y_base = max(m + s for m, s in zip(means, sds))
        brackets = [
            (0, 1, GROUPS[0], GROUPS[1], y_base * 1.10),   # WT PTZ effect
            (2, 3, GROUPS[2], GROUPS[3], y_base * 1.10),   # Mut PTZ effect
            (0, 2, GROUPS[0], GROUPS[2], y_base * 1.22),  # Baseline genotype
            (1, 3, GROUPS[1], GROUPS[3], y_base * 1.34),  # Genotype under PTZ
        ]
        for i1, i2, g1, g2, h in brackets:
            p = mw_p(df, col, g1, g2)
            ax.plot([i1, i1, i2, i2], [h, h * 1.03, h * 1.03, h], 'k-', lw=0.8)
            ax.text((i1 + i2) / 2, h * 1.05, sig_stars(p), ha='center',
                    va='bottom', fontsize=9)


def km_curve(times, events):
    """Product-limit estimator + Greenwood 95% CI -> (times, S, lo, hi)."""
    order = np.argsort(times, kind='mergesort')
    t, e = times[order], events[order]
    uniq = np.unique(t)
    s, out, cum = 1.0, [], []
    acc = 0.0
    for ut in uniq:
        n = np.sum(t >= ut)
        d = np.sum((t == ut) & (e == 1))
        if n > 0 and d > 0:
            s *= 1.0 - d / n
            if n > d:
                acc += d / (n * (n - d))
        out.append(s)
        cum.append(acc)
    surv = np.array(out)
    se = surv * np.sqrt(np.array(cum))
    return uniq, surv, np.clip(surv - 1.96 * se, 0, 1), np.clip(surv + 1.96 * se, 0, 1)


def logrank(times, events, groups):
    gs = np.unique(groups)
    k = len(gs)
    O, E, V = np.zeros(k), np.zeros(k), np.zeros((k, k))
    for ut in np.unique(times[events == 1]):
        at_risk = times >= ut
        died = (times == ut) & (events == 1)
        n_g = np.array([np.sum(at_risk & (groups == g)) for g in gs], float)
        d_g = np.array([np.sum(died & (groups == g)) for g in gs], float)
        n, d = n_g.sum(), d_g.sum()
        if n <= 1 or d == 0:
            continue
        O += d_g
        E += d * n_g / n
        for i in range(k):
            for j in range(k):
                V[i, j] += d * (n_g[i] / n) * ((1 if i == j else 0) - n_g[j] / n) * (n - d) / (n - 1)
    diff = (O - E)[:-1]
    chi2 = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    return chi2, float(stats.chi2.sf(chi2, k - 1))


for path in (ROLLUP, EVENTS):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run step7/7b/8 first.")
        sys.exit(1)

roll = pd.read_csv(ROLLUP)
ev = pd.read_csv(EVENTS)
print(f"[dataset: {LABEL}]  {len(roll)} fish, {len(ev)} seizures  ->  figures: {FIGDIR}")

# ---------------------------------------------------------- 1. count by group
fig, ax = plt.subplots(figsize=(6, 5))
bar_with_dots(ax, roll, 'seizure_count', 'Seizures per fish')
p = headline_p(roll, 'seizure_count')
ax.set_title(f'Seizures per fish ({format_p(p)}) — {LABEL}', fontsize=11, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'seizure_count_by_group')
plt.close()

# ------------------------------------------------------------- 2. KM latency
fig, ax = plt.subplots(figsize=(8, 6))
t = roll['latency_first_s'].fillna(REC_END).to_numpy(float)
e = roll['latency_first_s'].notna().astype(int).to_numpy()
g = roll[GROUP_COL].to_numpy()
for group in ['wt_control', 'mut_control', 'wt_ptz10', 'mut_ptz10']:
    m = g == group
    uniq, surv, lo, hi = km_curve(t[m], e[m])
    ax.fill_between(np.r_[0, uniq], np.r_[1.0, lo], np.r_[1.0, hi],
                    step='post', color=OKABE[group], alpha=0.2, linewidth=0)
    ax.step(np.r_[0, uniq], np.r_[1.0, surv], where='post',
            color=OKABE[group], label=group, linewidth=1.6)
    ct = t[m & (e == 0)]
    if len(ct):
        cs = np.array([surv[max(0, np.searchsorted(uniq, x, side='right') - 1)] for x in ct])
        ax.plot(ct, cs, '|', color=OKABE[group], ms=8, mew=1.2)

chi2, p = logrank(t, e, g)
ax.set_xlim(0, REC_END + 100)
ax.set_ylim(0, 1.05)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Time to first seizure (seconds)', fontsize=12)
ax.set_ylabel('Proportion seizure-free', fontsize=12)
ax.set_title(f'Latency to first seizure (2.8 cm/s threshold) — {LABEL}', fontsize=13)
ax.legend(title='Group', fontsize=10, loc='upper right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
textstr = f'4-group log-rank: {format_p(p)}'
ax.text(0.55, 0.05, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
caveat = 'Note: PTZ added at t=0 without acclimation;\nearly latencies may reflect handling stress'
ax.text(0.02, 0.02, caveat, transform=ax.transAxes, fontsize=7, style='italic',
        color='gray', verticalalignment='bottom')
plt.tight_layout()
save_fig(fig, 'seizure_latency_km')
plt.close()

# --------------------------------------------------- 3. count across quarters
QUARTER = REC_END / 4
ev['quarter'] = np.minimum((ev['burst_start_s'] // QUARTER).astype(int), 3)
counts = (ev.groupby([GROUP_COL, 'Etho_Trial', 'Fish', 'quarter']).size()
          .unstack(fill_value=0).reindex(columns=[0, 1, 2, 3], fill_value=0).fillna(0))
roster = roll[[GROUP_COL, 'Etho_Trial', 'Fish']].set_index(['Etho_Trial', 'Fish'])
per_fish = roster.join(counts.droplevel(0), how='left').fillna(0)

fig, ax = plt.subplots(figsize=(10, 6))
centers = np.arange(4) * (QUARTER / 60) + (QUARTER / 120)
for group in ['wt_control', 'mut_control', 'wt_ptz10', 'mut_ptz10']:
    sub = per_fish[per_fish[GROUP_COL] == group]
    means = sub[[0, 1, 2, 3]].mean().values
    sems = sub[[0, 1, 2, 3]].sem().values
    ax.plot(centers, means, color=OKABE[group], label=group, linewidth=2)
    ax.fill_between(centers, means - sems, means + sems, color=OKABE[group], alpha=0.15)

for start, end, label in [(0, 15, 'Onset'), (15, 30, 'Peak'),
                          (30, 45, 'Plateau'), (45, 60, 'Exhaustion')]:
    ax.axvspan(start, end, alpha=0.03, color='gray')
    ax.text((start + end) / 2, ax.get_ylim()[1] * 0.97, label, ha='center',
            fontsize=8, color='gray', style='italic')

ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_ylabel('Seizures per fish per quarter', fontsize=12)
ax.set_title(f'Seizure frequency across the hour (15-min quarters) — {LABEL}', fontsize=13)
ax.legend(title='Group', fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 0.92))
ax.set_xticks(centers)
ax.set_xticklabels([f'{int(c)}-{int(c + QUARTER / 60)}' for c in centers - QUARTER / 120])
ax.set_xlim(0, QUARTER / 60 * 4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'seizure_count_quarters')
plt.close()

# -------------------------------------------------- 5-min bin timecourse + per-bin MWU
BIN_SEC = 300
N_BINS = REC_END // BIN_SEC  # 12

ev['bin'] = np.minimum((ev['burst_start_s'] // BIN_SEC).astype(int), N_BINS - 1)
bin_counts = (ev.groupby([GROUP_COL, 'Etho_Trial', 'Fish', 'bin']).size()
              .unstack(fill_value=0).reindex(columns=list(range(N_BINS)), fill_value=0).fillna(0))
roster = roll[[GROUP_COL, 'Etho_Trial', 'Fish']].set_index(['Etho_Trial', 'Fish'])
per_fish_bin = roster.join(bin_counts.droplevel(0), how='left').fillna(0)

# Per-bin Mann-Whitney: mut_ptz10 vs wt_ptz10, Bonferroni over N_BINS bins
print(f"\n--- Per-bin MWU: mut_ptz10 vs wt_ptz10 (Bonferroni over {N_BINS} bins) ---")
bin_results = []
for tb in range(N_BINS):
    wt = per_fish_bin.loc[per_fish_bin[GROUP_COL] == 'wt_ptz10', tb].to_numpy()
    mut = per_fish_bin.loc[per_fish_bin[GROUP_COL] == 'mut_ptz10', tb].to_numpy()
    p_raw = stats.mannwhitneyu(wt, mut, alternative='two-sided', method='asymptotic').pvalue
    p_bonf = min(p_raw * N_BINS, 1.0)
    bin_results.append({
        'bin': tb,
        'min_range': f'{tb*5}-{(tb+1)*5} min',
        'wt_mean': round(wt.mean(), 3),
        'mut_mean': round(mut.mean(), 3),
        'gap': round(mut.mean() - wt.mean(), 3),
        'p_raw': round(p_raw, 6),
        'p_bonf': round(p_bonf, 6),
        'sig': sig_stars(p_bonf)
    })
bin_df = pd.DataFrame(bin_results)
print(bin_df.to_string(index=False))

# Figure: line + SEM ribbon per group, epoch bands, Bonferroni stars at top
fig, ax = plt.subplots(figsize=(10, 6))
centers = np.arange(N_BINS) * 5 + 2.5
y_max_data = 0
series = {}
for group in ['wt_control', 'mut_control', 'wt_ptz10', 'mut_ptz10']:
    sub = per_fish_bin[per_fish_bin[GROUP_COL] == group]
    means = sub[list(range(N_BINS))].mean().values
    sems = sub[list(range(N_BINS))].sem().values
    series[group] = (means, sems)
    ax.plot(centers, means, color=OKABE[group], label=group, linewidth=2)
    ax.fill_between(centers, means - sems, means + sems, color=OKABE[group], alpha=0.15)
    y_max_data = max(y_max_data, (means + sems).max())

for start, end, label in [(0, 15, 'Onset'), (15, 30, 'Peak'),
                          (30, 45, 'Plateau'), (45, 60, 'Exhaustion')]:
    ax.axvspan(start, end, alpha=0.03, color='gray')
    ax.text((start + end) / 2, y_max_data * 1.14, label, ha='center',
            fontsize=8, color='gray', style='italic')

# Bonferroni stars above the PTZ curves at each significant bin
wt_m, wt_s = series['wt_ptz10']
mut_m, mut_s = series['mut_ptz10']
for r in bin_results:
    if r['sig'] != 'ns':
        tb = r['bin']
        y_star = max(wt_m[tb] + wt_s[tb], mut_m[tb] + mut_s[tb]) + y_max_data * 0.05
        ax.text(centers[tb], y_star, r['sig'], ha='center', va='bottom',
                fontsize=11, fontweight='bold')

ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_ylabel('Seizures per fish per 5-min bin', fontsize=12)
ax.set_title(f'Seizure frequency across the hour (5-min bins) — {LABEL}', fontsize=13)
ax.legend(title='Group', fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 0.92))
ax.set_xticks(centers)
ax.set_xticklabels([f'{int(c*5)}-{int(c*5+5)}' for c in range(N_BINS)], fontsize=8)
ax.set_xlim(0, N_BINS * 5)
ax.set_ylim(0, y_max_data * 1.24)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig, 'seizure_timecourse')
plt.close()

# ------------------------------------------------------------- 4. burden
fig, ax = plt.subplots(figsize=(6, 5))
bar_with_dots(ax, roll, 'burden_frac', '% of tracked time seizing', scale=100)
p = headline_p(roll, 'burden_frac')
ax.set_title(f'Seizure burden ({format_p(p)}) — {LABEL}', fontsize=11, fontweight='bold')
plt.tight_layout()
save_fig(fig, 'seizure_burden_by_group')
plt.close()

print("Saved 5 figures (png + pdf): seizure_count_by_group, seizure_latency_km,")
print("                             seizure_count_quarters, seizure_timecourse,")
print("                             seizure_burden_by_group")
