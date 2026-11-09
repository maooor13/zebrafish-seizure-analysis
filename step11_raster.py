"""
Step 11 — Event raster: burst timelines for top-burst fish
Shows WHEN each fish seized (velocity > 2.8 cm/s) so you can jump to that
timestamp in the video.

Top-N fish per PTZ group by event count, horizontal bars = event duration.

Run: python step11_raster.py  (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pingouin as pg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dataset_config import CSV, LABEL, FIGDIR, T_CUTOFF

if T_CUTOFF is None:
    print("No per-second data for this dataset — skipping step11_raster.")
    import sys; sys.exit(0)

# Publication font settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
})

TOP_N = 5  # per group — adjustable

OKABE = {'wt_ptz10': '#E69F00', 'mut_ptz10': '#D55E00'}

# --- Select top-N burst fish from PerFish_Primary ---
perfish = pd.read_csv(CSV)
print(f"[dataset: {LABEL}]  input: {CSV}  ->  figures: {FIGDIR}")
perfish['Trial'] = perfish['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()

ptz = perfish[perfish['Fish Type'].isin(['wt_ptz10', 'mut_ptz10'])].copy()
top_fish = (
    ptz.sort_values('Event_count_ThrA', ascending=False)
    .groupby('Fish Type')
    .head(TOP_N)
    .sort_values(['Fish Type', 'Event_count_ThrA'], ascending=[True, False])
)

fish_ids = top_fish[['Trial', 'Fish', 'Fish Type', 'Event_count_ThrA']].values.tolist()
n_fish = len(fish_ids)
print(f'Selected {n_fish} fish ({TOP_N} per group)')

# --- Load AllData, filter to selected fish only (and to dataset time window) ---
alldata = pd.read_csv('../AllData.csv')
alldata['Trial'] = alldata['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()
alldata['Fish'] = alldata['Fish'].str.strip()
alldata = alldata[alldata['Time_seconds'] < T_CUTOFF]

selected = alldata.merge(
    top_fish[['Trial', 'Fish']],
    on=['Trial', 'Fish'],
    how='inner'
)

print(f'Loaded {len(selected)} rows for {selected[["Trial","Fish"]].drop_duplicates().shape[0]} fish')

# --- Detect contiguous runs above threshold (vectorized) ---
def detect_events(velocity_series, threshold=2.8):
    """Return list of (start_second, duration_seconds) for contiguous runs above threshold."""
    v = (velocity_series.values > threshold).astype(int)
    if v.sum() == 0:
        return []
    diff = np.diff(np.concatenate([[0], v, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(s, e - s) for s, e in zip(starts, ends)]

# --- Plot ---
fig, ax = plt.subplots(figsize=(14, max(4, n_fish * 0.4 + 1)))

bar_height = 0.6
y_positions = list(range(n_fish))
current_group = None
group_boundaries = []

for i, (trial, fish, ftype, event_count) in enumerate(fish_ids):
    sub = selected[(selected['Trial'] == trial) & (selected['Fish'] == fish)].sort_values('Time_seconds')
    if sub.empty:
        continue

    events = detect_events(sub['Velocity'])
    y = n_fish - 1 - i  # top-to-bottom order

    # Track group boundaries for separator
    if current_group is not None and ftype != current_group:
        group_boundaries.append(y + 0.5)
    current_group = ftype

    # Plot event bars (convert seconds to minutes for x-axis)
    if events:
        xranges = [(s/60, d/60) for s, d in events]
        ax.broken_barh(xranges, (y - bar_height/2, bar_height),
                       facecolors=OKABE[ftype], edgecolors='black', linewidth=0.3)

    # Fish label on y-axis
    label = f'{trial.replace("Trial ", "T")} {fish} ({int(event_count)})'
    ax.text(-0.5, y, label, ha='right', va='center', fontsize=7)

# Group separator lines
for yb in group_boundaries:
    ax.axhline(y=yb, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

x_max_min = T_CUTOFF / 60

# Group labels on right margin
mid_wt = n_fish - 1 - (n_fish // 2 if n_fish > 2 else 0)
if any(f[2] == 'mut_ptz10' for f in fish_ids):
    mut_indices = [i for i, f in enumerate(fish_ids) if f[2] == 'mut_ptz10']
    wt_indices = [i for i, f in enumerate(fish_ids) if f[2] == 'wt_ptz10']
    if mut_indices:
        y_mut = np.mean([n_fish - 1 - i for i in mut_indices])
        ax.text(x_max_min + 2, y_mut, 'mut PTZ 10', fontsize=9, fontweight='bold',
                color=OKABE['mut_ptz10'], va='center')
    if wt_indices:
        y_wt = np.mean([n_fish - 1 - i for i in wt_indices])
        ax.text(x_max_min + 2, y_wt, 'WT PTZ 10', fontsize=9, fontweight='bold',
                color=OKABE['wt_ptz10'], va='center')

ax.set_xlim(-0.5, x_max_min)
ax.set_ylim(-0.5, n_fish - 0.5)
ax.set_xlabel('Time (minutes)', fontsize=10)
ax.set_title(f'Burst event timeline — top {TOP_N} per group by event count — {LABEL}\n'
             f'(bars = velocity > 2.8 cm/s; label = Trial Well (event count))',
             fontsize=11, fontweight='bold')
ax.set_yticks([])
ax.set_xticks(range(0, int(x_max_min) + 1, 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

# Epoch shading (60-min only)
if x_max_min >= 60:
    for start, end, color in [(0, 15, '#f0f0f0'), (15, 30, '#ffffff'), (30, 45, '#f5f5f5'), (45, 60, '#ffffff')]:
        ax.axvspan(start, end, color=color, alpha=0.3, zorder=0)

os.makedirs(FIGDIR, exist_ok=True)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/burst_raster.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGDIR}/burst_raster.pdf', bbox_inches='tight')
plt.close()
print(f'Plot saved to {FIGDIR}/burst_raster.png + .pdf')

# ======================================================================
# Population figure: binary presence + count heatmap (2 panels)
# Uses ALL PTZ fish (not just top-N), filtered to PerFish_Primary (phantom C8 excluded)
# ======================================================================
szep_fish = pd.read_csv(CSV)[['Etho_Trial', 'Fish']].copy()
szep_fish.rename(columns={'Etho_Trial': 'Trial'}, inplace=True)
szep_fish['Trial'] = szep_fish['Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()
szep_fish['Fish'] = szep_fish['Fish'].str.strip()

ptz_all = alldata[alldata['Fish Type'].str.strip().isin(['wt_ptz10', 'mut_ptz10'])].copy()
ptz_all['Trial'] = ptz_all['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()
ptz_all['Fish'] = ptz_all['Fish'].str.strip()
# Filter to dataset fish (excludes phantom C8) and dataset time window
ptz_all = ptz_all.merge(szep_fish, on=['Trial', 'Fish'], how='inner')
ptz_all = ptz_all[ptz_all['Time_seconds'] < T_CUTOFF]
# Fish_id = Trial + Fish (well codes repeat across trials)
ptz_all['Fish_id'] = ptz_all['Trial'] + '_' + ptz_all['Fish']

pop_groups = ['mut_ptz10', 'wt_ptz10']
n_seconds = T_CUTOFF
count_matrix = np.zeros((2, n_seconds))

for i, g in enumerate(pop_groups):
    sub = ptz_all[ptz_all['Fish Type'].str.strip() == g]
    above = (sub.pivot_table(index='Time_seconds', columns='Fish_id', values='Velocity') > 2.8)
    counts = above.sum(axis=1).reindex(range(n_seconds), fill_value=0)
    count_matrix[i, :] = counts.values
    print(f'  {g}: {sub["Fish_id"].nunique()} fish, max simultaneous above threshold: {int(counts.max())}')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1, 0.6]})

# Panel A: Binary presence (broken_barh)
for i, g in enumerate(pop_groups):
    presence = count_matrix[i] > 0
    diff = np.diff(np.concatenate([[0], presence.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    xranges = [(s / 60, (e - s) / 60) for s, e in zip(starts, ends)]
    ax1.broken_barh(xranges, (i - 0.3, 0.6), facecolors=OKABE[g], edgecolors='black', linewidth=0.3)

ax1.set_yticks([0, 1])
ax1.set_yticklabels(pop_groups, fontsize=9)
ax1.set_ylim(-0.5, 1.5)
ax1.set_xlim(0, x_max_min)
ax1.set_ylabel('Group', fontsize=10)
ax1.set_title('Burst presence (any fish above threshold)', fontsize=11, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Panel B: log1p(count) heatmap (variance-stabilizes per-second count; expands low end)
im = ax2.imshow(np.log1p(count_matrix), aspect='auto', cmap='cividis',
                extent=[0, x_max_min, -0.5, 1.5], origin='lower', interpolation='nearest')
ax2.set_yticks([0, 1])
ax2.set_yticklabels(pop_groups, fontsize=9)
ax2.set_ylim(-0.5, 1.5)
ax2.set_xlim(0, x_max_min)
ax2.set_ylabel('Group', fontsize=10)
ax2.set_title('Burst burden (log1p fish count above threshold)', fontsize=11, fontweight='bold')

divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='2%', pad=0.05)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('log1p(fish count above threshold)', fontsize=9)

# Panel C: (mut − wt) difference in raw count per second, diverging colormap centered 0
diff_row = count_matrix[0] - count_matrix[1]  # mut_ptz10 − wt_ptz10
vmax = int(np.abs(diff_row).max()) or 1
im3 = ax3.imshow(diff_row.reshape(1, -1), aspect='auto', cmap='RdBu_r',
                 extent=[0, x_max_min, -0.5, 0.5], origin='lower',
                 interpolation='nearest', vmin=-vmax, vmax=vmax)
ax3.set_yticks([0])
ax3.set_yticklabels(['mut − wt'], fontsize=9)
ax3.set_ylim(-0.5, 0.5)
ax3.set_xlim(0, x_max_min)
ax3.set_xlabel('Time (minutes)', fontsize=10)
ax3.set_title('Burst burden difference (mut − wt, fish count)', fontsize=11, fontweight='bold')
divider3 = make_axes_locatable(ax3)
cax3 = divider3.append_axes('right', size='2%', pad=0.05)
cbar3 = fig.colorbar(im3, cax=cax3)
cbar3.set_label('mut − wt (fish count)', fontsize=9)

# --- per-bin MW stars on event count (top of figure) ---
def count_events(velocity_series, threshold=2.8):
    v = (velocity_series.values > threshold).astype(int)
    if v.sum() == 0: return 0
    diff = np.diff(np.concatenate([[0], v, [0]]))
    return int((diff == 1).sum())

ptz_all['Time_bin_evc'] = ptz_all['Time_seconds'] // 300
perfish_ev = (ptz_all.groupby(['Fish_id', 'Fish Type', 'Time_bin_evc'])['Velocity']
             .apply(lambda s: count_events(s)).reset_index(name='events'))
n_bins_pop = int(T_CUTOFF // 300)
sig_stars = lambda p: '****' if p < 0.0001 else '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
print(f'  per-bin MW on event count (Bonferroni over {n_bins_pop}):')
for tb in range(n_bins_pop):
    wt = perfish_ev[(perfish_ev['Fish Type'] == 'wt_ptz10') & (perfish_ev['Time_bin_evc'] == tb)]['events']
    mut = perfish_ev[(perfish_ev['Fish Type'] == 'mut_ptz10') & (perfish_ev['Time_bin_evc'] == tb)]['events']
    p_raw = pg.mwu(wt, mut, alternative='two-sided')['p_val'].values[0]
    p_bonf = min(p_raw * n_bins_pop, 1.0)
    s = sig_stars(p_bonf)
    if s != 'ns':
        # stars at top of figure, above ax1, at bin center (in minutes)
        ax1.text(tb * 5 + 2.5, 1.6, s, ha='center', va='bottom', fontsize=10, fontweight='bold',
                 transform=ax1.transData, clip_on=False)
        print(f'    bin {tb} ({tb*5}-{(tb+1)*5} min): p_raw={p_raw:.4f} p_bonf={p_bonf:.4f} {s}')

# Panel labels
ax1.text(-0.06, 1.05, 'A', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top')
ax2.text(-0.06, 1.05, 'B', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top')
ax3.text(-0.06, 1.05, 'C', transform=ax3.transAxes, fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/burst_population.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGDIR}/burst_population.pdf', bbox_inches='tight')
plt.close()
print(f'Plot saved to {FIGDIR}/burst_population.png + .pdf')

# ======================================================================
# Burst time-course: seconds above threshold per bin + per-bin stats
# ======================================================================
ptz_all['Time_bin'] = ptz_all['Time_seconds'] // 300
ptz_all['above'] = (ptz_all['Velocity'] > 2.8).astype(int)

perfish_burst = ptz_all.groupby(['Fish_id', 'Fish Type', 'Time_bin'], as_index=False)['above'].sum()

n_bins = int(T_CUTOFF // 300)
burst_groups = ['wt_ptz10', 'mut_ptz10']
OKABE_BURST = {'wt_ptz10': '#E69F00', 'mut_ptz10': '#D55E00'}

# Per-bin MW tests
print('\n' + '=' * 70)
print(f'PER-BIN MW: mut_ptz10 vs wt_ptz10 seconds above threshold (Bonferroni over {n_bins})')
print('=' * 70)

pw_rows = []
for tb in range(n_bins):
    wt = perfish_burst[(perfish_burst['Fish Type'] == 'wt_ptz10') & (perfish_burst['Time_bin'] == tb)]['above']
    mut = perfish_burst[(perfish_burst['Fish Type'] == 'mut_ptz10') & (perfish_burst['Time_bin'] == tb)]['above']
    r = pg.mwu(wt, mut, alternative='two-sided')
    p_raw = r['p_val'].values[0]
    p_bonf = min(p_raw * n_bins, 1.0)
    pw_rows.append({
        'time_bin': tb,
        'min_range': f'{tb*5}-{(tb+1)*5} min',
        'wt_mean': round(wt.mean(), 2),
        'mut_mean': round(mut.mean(), 2),
        'gap': round(mut.mean() - wt.mean(), 2),
        'p_raw': round(p_raw, 6),
        'p_bonf': round(p_bonf, 6),
        'sig': '****' if p_bonf < 0.0001 else ('***' if p_bonf < 0.001 else ('**' if p_bonf < 0.01 else ('*' if p_bonf < 0.05 else 'ns')))
    })
pw_df = pd.DataFrame(pw_rows)
print(pw_df.to_string(index=False))

# Plot: 2 lines + SEM ribbons + significance stars
fig, ax = plt.subplots(figsize=(10, 6))

time_minutes = np.arange(n_bins) * 5 + 2.5  # bin centers

for group in burst_groups:
    sub = perfish_burst[perfish_burst['Fish Type'] == group]
    means = sub.groupby('Time_bin')['above'].mean()
    sems = sub.groupby('Time_bin')['above'].sem()
    ax.plot(time_minutes, means.values, color=OKABE_BURST[group], label=group, linewidth=2)
    ax.fill_between(time_minutes, means.values - sems.values, means.values + sems.values,
                    color=OKABE_BURST[group], alpha=0.15)

# Epoch annotations (60-min only)
if x_max_min >= 60:
    for start, end, label in [(0, 15, 'Onset'), (15, 30, 'Peak'), (30, 45, 'Plateau'), (45, 60, 'Exhaustion')]:
        ax.axvspan(start, end, alpha=0.03, color='gray')
        ax.text((start + end) / 2, ax.get_ylim()[1] * 0.97, label, ha='center', fontsize=8,
                color='gray', style='italic')

# Significance stars above significant bins
y_max = perfish_burst.groupby('Time_bin')['above'].mean().max()
for _, row in pw_df.iterrows():
    if row['sig'] != 'ns':
        x = row['time_bin'] * 5 + 2.5
        ax.text(x, y_max * 1.08, row['sig'], ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Time (minutes)', fontsize=12)
ax.set_ylabel('Seconds above threshold per fish', fontsize=12)
ax.set_title(f'Burst activity over time (seconds > 2.8 cm/s per 5-min bin) — {LABEL}', fontsize=13, fontweight='bold')
ax.legend(title='Group', fontsize=10, loc='upper right', bbox_to_anchor=(1.0, 0.97))
ax.set_xlim(0, x_max_min)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{FIGDIR}/burst_timecourse.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{FIGDIR}/burst_timecourse.pdf', bbox_inches='tight')
plt.close()
print(f'\nPlot saved to {FIGDIR}/burst_timecourse.png + .pdf')
