"""
Step 11 — Poster figures
Generates 6 new figures + 1 CSV summary table (F3-F8).
F1 (km_latency.png) and F2 (timecourse_velocity.png) already exist.

Reuses Okabe-Ito palette, MW p-values, and significance conventions
from step6a scripts. All figures: bars=mean±SEM, dots=individual fish.

Run: python step11_figures.py  (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
from string import ascii_uppercase
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pingouin as pg
from dataset_config import CSV, LABEL, FIGDIR, T_CUTOFF

# Publication font settings (scientific-visualization skill)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

np.random.seed(42)


def format_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def save_fig(fig, name):
    """Save figure as PNG (300 DPI) + PDF (vector) per scientific-visualization skill."""
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(f'{FIGDIR}/{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{FIGDIR}/{name}.pdf', bbox_inches='tight')


def add_panel_labels(axes):
    """Add bold A, B, C... labels to subplot axes."""
    for i, ax in enumerate(axes):
        ax.text(-0.12, 1.05, ascii_uppercase[i], transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='top')

df = pd.read_csv(CSV)
print(f"[dataset: {LABEL}]  input: {CSV}  ->  figures: {FIGDIR}")

OKABE = {'wt_control': '#56B4E9', 'wt_ptz10': '#E69F00', 'mut_control': '#009E73', 'mut_ptz10': '#D55E00',
         'wt_ptz5': '#E69F00', 'mut_ptz5': '#D55E00'}
# Group set depends on dataset (ptz10 for 10mM, ptz5 for 5mM)
_ptz = 'mut_ptz5' if LABEL.startswith('5mM') else 'mut_ptz10'
_wtptz = 'wt_ptz5' if LABEL.startswith('5mM') else 'wt_ptz10'
GROUPS = ['wt_control', _wtptz, 'mut_control', _ptz]
GROUP_LABELS = ['WT\ncontrol', f'WT\n{("PTZ 5" if LABEL.startswith("5mM") else "PTZ 10")}',
                'mut\ncontrol', f'mut\n{("PTZ 5" if LABEL.startswith("5mM") else "PTZ 10")}']
PTZ_PAIR = [_wtptz, _ptz]
HAS_BURST = T_CUTOFF is not None and 'Event_count_ThrA' in df.columns
SEC = T_CUTOFF if T_CUTOFF else 3600  # seconds in window (for per-sec/per-min distance figures)
# Genotype / Treatment factors (for F8 η²_p)
df['Genotype'] = df['Fish Type'].str.split('_').str[0]
df['Treatment'] = df['Fish Type'].str.split('_').str[1]


def sig_stars(p):
    if p < 0.0001: return '****'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return 'ns'


def mw_p(data, col, g1=None, g2=None):
    if g1 is None: g1 = PTZ_PAIR[0]
    if g2 is None: g2 = PTZ_PAIR[1]
    a = data[data['Fish Type'] == g1][col]
    b = data[data['Fish Type'] == g2][col]
    return pg.mwu(a, b, alternative='two-sided')['p_val'].values[0]


def bar_with_dots(ax, data, col, groups=GROUPS, labels=GROUP_LABELS,
                  ylabel='', title='', show_bracket=True):
    """Bar chart with jittered dots + significance bracket. Error bars = SD."""
    means = [data[data['Fish Type'] == g][col].mean() for g in groups]
    sds = [data[data['Fish Type'] == g][col].std() for g in groups]
    x = np.arange(len(groups))
    colors = [OKABE[g] for g in groups]

    ax.bar(x, means, yerr=sds, color=colors, capsize=5, edgecolor='black', linewidth=0.5, alpha=0.7)
    for i, g in enumerate(groups):
        vals = data[data['Fish Type'] == g][col].dropna()
        jitter = np.random.normal(i, 0.06, size=len(vals))
        ax.scatter(jitter, vals, color='black', alpha=0.3, s=15, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
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
            p = mw_p(data, col, g1, g2)
            ax.plot([i1, i1, i2, i2], [h, h * 1.03, h * 1.03, h], 'k-', lw=0.8)
            ax.text((i1 + i2) / 2, h * 1.05, sig_stars(p), ha='center', va='bottom', fontsize=9)


def pairwise_bar(ax, data, col, ylabel='', title=''):
    """2-bar pairwise: wt_ptz10 vs mut_ptz10. Error bars = SD."""
    means = [data[data['Fish Type'] == g][col].mean() for g in PTZ_PAIR]
    sds = [data[data['Fish Type'] == g][col].std() for g in PTZ_PAIR]
    x = np.arange(2)
    colors = [OKABE[g] for g in PTZ_PAIR]

    ax.bar(x, means, yerr=sds, color=colors, capsize=5, edgecolor='black', linewidth=0.5, alpha=0.7)
    for i, g in enumerate(PTZ_PAIR):
        vals = data[data['Fish Type'] == g][col].dropna()
        jitter = np.random.normal(i, 0.06, size=len(vals))
        ax.scatter(jitter, vals, color='black', alpha=0.3, s=15, zorder=3)

    p = mw_p(data, col)
    y_max = max(m + s for m, s in zip(means, sds)) * 1.15
    ax.plot([0, 0, 1, 1], [y_max, y_max * 1.05, y_max * 1.05, y_max], 'k-', lw=0.8)
    ax.text(0.5, y_max * 1.08, sig_stars(p), ha='center', va='bottom', fontsize=10)

    ax.set_xticks(x)
    _plab = 'PTZ 5' if LABEL.startswith('5mM') else 'PTZ 10'
    ax.set_xticklabels([f'WT {_plab}', f'mut {_plab}'], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ======================================================================
# F3 — Grouped bar: Total_distance
# ======================================================================
fig, ax = plt.subplots(figsize=(6, 5))
bar_with_dots(ax, df, 'Total_distance_cm', ylabel='Total distance (cm)',
              title=f'Total distance — genotype × treatment interaction\n({format_p(mw_p(df, "Total_distance_cm", _wtptz, _ptz))}) — {LABEL}')
plt.tight_layout()
save_fig(fig, 'fig3_bar_total_distance')
plt.close()
print('F3: fig3_bar_total_distance.png + .pdf')

# ======================================================================
# F4 — Grouped bar: Event_count + Time_above (2 subplots)
# ======================================================================
if HAS_BURST:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    bar_with_dots(axes[0], df, 'Event_count_ThrA', ylabel='Burst count',
                  title=f'Burst count ({format_p(mw_p(df, "Event_count_ThrA", _wtptz, _ptz))})')
    bar_with_dots(axes[1], df, 'Time_above_ThrA', ylabel='Time above threshold (s)',
                  title=f'Time above threshold ({format_p(mw_p(df, "Time_above_ThrA", _wtptz, _ptz))})')
    add_panel_labels(axes)
    plt.tight_layout()
    save_fig(fig, 'fig4_bar_burst_metrics')
    plt.close()
    print('F4: fig4_bar_burst_metrics.png + .pdf')
else:
    print('F4: skipped (no burst metrics)')

# ======================================================================
# F5 — Grouped bar: Max_velocity
# ======================================================================
if HAS_BURST:
    fig, ax = plt.subplots(figsize=(6, 5))
    bar_with_dots(ax, df, 'Max_velocity', ylabel='Max velocity (cm/s)',
                  title=f'Max velocity — same peak intensity ({format_p(mw_p(df, "Max_velocity", _wtptz, _ptz))}) — {LABEL}')
    plt.tight_layout()
    save_fig(fig, 'fig5_bar_max_velocity')
    plt.close()
    print('F5: fig5_bar_max_velocity.png + .pdf')
else:
    print('F5: skipped (no burst metrics)')

# ======================================================================
# F6 — Pairwise: mut vs wt PTZ (4 subplots if burst, else just distance)
# ======================================================================
if HAS_BURST:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    pairwise_bar(axes[0], df, 'Total_distance_cm', ylabel='Total distance (cm)', title='Total distance')
    pairwise_bar(axes[1], df, 'Event_count_ThrA', ylabel='Burst count', title='Burst count')
    pairwise_bar(axes[2], df, 'Time_above_ThrA', ylabel='Time above threshold (s)', title='Time above threshold')
    pairwise_bar(axes[3], df, 'Max_velocity', ylabel='Max velocity (cm/s)', title='Max velocity')
    add_panel_labels(axes)
    fig.suptitle(f'Pairwise: mut vs WT ({("PTZ 5" if LABEL.startswith("5mM") else "PTZ 10")})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig6_pairwise_mut_vs_wt')
    plt.close()
    print('F6: fig6_pairwise_mut_vs_wt.png + .pdf')
else:
    fig, axes = plt.subplots(1, 1, figsize=(5, 4.5))
    pairwise_bar(axes, df, 'Total_distance_cm', ylabel='Total distance (cm)', title='Total distance')
    fig.suptitle(f'Pairwise: mut vs WT ({("PTZ 5" if LABEL.startswith("5mM") else "PTZ 10")})', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig6_pairwise_mut_vs_wt')
    plt.close()
    print('F6: fig6_pairwise_mut_vs_wt.png + .pdf (distance only — no burst)')

# ======================================================================
# F7 — Box plots: all key metrics (4 subplots if burst, else just distance)
# ======================================================================
if HAS_BURST:
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    box_metrics = [
        ('Total_distance_cm', 'Total distance (cm)'),
        ('Event_count_ThrA', 'Burst count'),
        ('Time_above_ThrA', 'Time above threshold (s)'),
        ('Max_velocity', 'Max velocity (cm/s)'),
    ]
else:
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes = [axes]
    box_metrics = [('Total_distance_cm', 'Total distance (cm)')]
for ax, (col, ylabel) in zip(axes, box_metrics):
    data = [df[df['Fish Type'] == g][col].dropna() for g in GROUPS]
    bp = ax.boxplot(data, positions=np.arange(4), widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', linewidth=1.5))
    for patch, g in zip(bp['boxes'], GROUPS):
        patch.set_facecolor(OKABE[g])
        patch.set_alpha(0.5)
    for i, g in enumerate(GROUPS):
        vals = df[df['Fish Type'] == g][col].dropna()
        jitter = np.random.normal(i, 0.05, size=len(vals))
        ax.scatter(jitter, vals, color='black', alpha=0.25, s=12, zorder=3)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(GROUP_LABELS, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
add_panel_labels(axes)
fig.suptitle('Distribution by group (box + individual data)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, 'fig7_boxplots_all_metrics')
plt.close()
print('F7: fig7_boxplots_all_metrics.png + .pdf')

# ======================================================================
# F8 — Summary table (image + CSV)
# ======================================================================
metrics = [('Total distance (cm)', 'Total_distance_cm', 'mean')]
if HAS_BURST:
    metrics += [
        ('Burst count', 'Event_count_ThrA', 'mean'),
        ('Time above threshold (s)', 'Time_above_ThrA', 'mean'),
        ('Max velocity (cm/s)', 'Max_velocity', 'mean'),
        ('Latency to first burst (s)', 'Latency_first_event_ThrA', 'median'),
        ('Group × Time interaction', None, 'special'),
    ]

tests = {
    'Total distance (cm)': 'HC3 OLS + MW',
    'Burst count': 'Mann-Whitney (primary)',
    'Time above threshold (s)': 'Mann-Whitney (primary)',
    'Max velocity (cm/s)': 'HC3 OLS + MW',
    'Latency to first burst (s)': 'Log-rank (strat)',
    'Group × Time interaction': 'Mixed ANOVA (GG)',
}

rows = []
for label, col, agg in metrics:
    row = {'Metric': label, 'Test': tests[label]}
    if col is None:
        row.update({g: '—' for g in GROUPS})
        row['p-value'] = 'see step6a2'  # filled by kinetics script
        row['Effect size'] = 'η²p see step6a2'
        row['Direction'] = 'see step6a2'
    elif agg == 'median':
        for g in GROUPS:
            vals = df[df['Fish Type'] == g][col].dropna()
            row[g] = f'{vals.median():.1f}' if len(vals) > 0 else '—'
        row['p-value'] = 'see step6c'  # filled by KM script (log-rank)
        row['Effect size'] = 'log-rank χ²'
        row['Direction'] = 'mut shorter (caveat: onset window)'
    else:
        for g in GROUPS:
            vals = df[df['Fish Type'] == g][col]
            row[g] = f'{vals.mean():.1f} ± {vals.std():.1f}'
        row['p-value'] = format_p(mw_p(df, col, _wtptz, _ptz))
        # η²_p (interaction) from unblocked 2-way ANOVA on sqrt-transformed metric
        import pingouin as pg
        df['_sq'] = np.sqrt(df[col].clip(lower=0))
        aov = pg.anova(dv='_sq', between=['Genotype', 'Treatment'], data=df, detailed=True)
        inter = aov[aov['Source'] == 'Genotype * Treatment']
        eta = inter['np2'].values[0] if not inter.empty else np.nan
        row['Effect size'] = f'η²p={eta:.3f}'
        mut_v = df[df['Fish Type'] == _ptz][col].mean()
        wt_v = df[df['Fish Type'] == _wtptz][col].mean()
        pct = (mut_v - wt_v) / wt_v * 100 if wt_v else 0
        row['Direction'] = f'mut {"+" if pct >= 0 else ""}{pct:.0f}% vs WT'
    rows.append(row)

table_df = pd.DataFrame(rows)
os.makedirs(FIGDIR, exist_ok=True)
table_df.to_csv(f'{FIGDIR}/summary_table.csv', index=False)
print('F8 CSV: summary_table.csv')

# Table as image — wider figure + auto column widths to fit text
fig, ax = plt.subplots(figsize=(20, 4))
ax.axis('off')
col_labels = ['Metric', 'WT ctrl', 'mut ctrl', 'WT PTZ', 'mut PTZ', 'Test', 'p', 'Effect', 'Direction']
cell_text = []
for _, r in table_df.iterrows():
    cell_text.append([r['Metric'], r[GROUPS[0]], r[GROUPS[2]], r[GROUPS[1]], r[GROUPS[3]],
                      r['Test'], r['p-value'], r['Effect size'], r['Direction']])
table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.8)
table.auto_set_column_width(list(range(len(col_labels))))
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#D55E00')
    table[0, j].set_text_props(color='white', fontweight='bold')
plt.savefig(f'{FIGDIR}/fig8_summary_table.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGDIR}/fig8_summary_table.pdf', bbox_inches='tight')
plt.close()
print('F8: fig8_summary_table.png + .pdf')

# ======================================================================
# F9 — Mean distance per minute (Total_distance / minutes in window)
# ======================================================================
df['Avg_distance_per_min'] = df['Total_distance_cm'] / (SEC / 60)
fig, ax = plt.subplots(figsize=(6, 5))
bar_with_dots(ax, df, 'Avg_distance_per_min', ylabel='Mean distance (cm/min)',
              title=f'Mean distance per minute ({format_p(mw_p(df, "Total_distance_cm", _wtptz, _ptz))}) — {LABEL}')
save_fig(fig, 'fig9_bar_distance_per_min')
plt.close()
print('F9: fig9_bar_distance_per_min.png + .pdf')

# ======================================================================
# F10 — Mean distance per second (Total_distance / seconds in window)
# ======================================================================
df['Avg_distance_per_sec'] = df['Total_distance_cm'] / SEC
fig, ax = plt.subplots(figsize=(6, 5))
bar_with_dots(ax, df, 'Avg_distance_per_sec', ylabel='Mean distance (cm/s)',
              title=f'Mean distance per second ({format_p(mw_p(df, "Total_distance_cm", _wtptz, _ptz))}) — {LABEL}')
save_fig(fig, 'fig10_bar_distance_per_sec')
plt.close()
print('F10: fig10_bar_distance_per_sec.png + .pdf')

print(f'\nAll figures saved to {FIGDIR}/ (PNG 300 DPI + PDF vector)')
print('F1 (existing): km_latency.png')
print('F2 (existing): timecourse_velocity.png')
for i in range(3, 11):
    print(f'F{i}: fig{i}_*.png + .pdf')
