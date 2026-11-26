"""
Step 8 — Per-fish seizure rollup.

One row per fish, combining:
  - event metrics from seizure_events_iqr_filtered.csv
  - per-phase second counts from seizure_seconds.csv
  - group label from seizure_seconds.csv

Fish with zero events are kept (counts 0, latency/mean/max NaN).
Percentages and burden use tracked (non-null) seconds as denominator.

Output: ../seizure_fish_rollup.csv
Run: python step8_fish_rollup.py   (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seizure_config import TRIAL_COL, FISH_COL, PHASE_COL
from dataset_config import LABEL

SECONDS = '../seizure_seconds.csv'
EVENTS = '../seizure_events_iqr_filtered.csv'
OUT = '../seizure_fish_rollup.csv'
GROUP_COL = 'Fish Type'
KEY = [TRIAL_COL, FISH_COL]
PHASES = ['rest', 'hyper', 'burst', 'catatonic', 'null']

for path in (SECONDS, EVENTS):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run step7/step7b first.")
        sys.exit(1)

sec = pd.read_csv(SECONDS, usecols=KEY + [GROUP_COL, PHASE_COL])
ev = pd.read_csv(EVENTS)
print(f"[dataset: {LABEL}]  {len(sec):,} fish-seconds, {len(ev)} events")

# --- Roster: one row per fish ---
rollup = sec.groupby(KEY, as_index=False)[GROUP_COL].first()

# --- Per-phase second counts ---
phase_s = (
    sec.groupby(KEY + [PHASE_COL]).size().unstack(fill_value=0)
    .reindex(columns=PHASES, fill_value=0)
    .rename(columns={p: f'seconds_{p}' for p in PHASES})
)
rollup = rollup.merge(phase_s, left_on=KEY, right_index=True, how='left')

# Denominator: tracked seconds (null = gap, excluded)
tracked = rollup[['seconds_rest', 'seconds_hyper', 'seconds_burst', 'seconds_catatonic']].sum(axis=1)
for p in ['rest', 'hyper', 'burst', 'catatonic']:
    rollup[f'pct_{p}'] = (rollup[f'seconds_{p}'] / tracked).round(4)

# --- Event metrics ---
ev_agg = ev.groupby(KEY).agg(
    seizure_count=('event_id', 'size'),
    total_seizing_s=('burst_duration_s', 'sum'),
    mean_duration_s=('burst_duration_s', 'mean'),
    max_duration_s=('burst_duration_s', 'max'),
    latency_first_s=('burst_start_s', 'min'),
    catatonic_tail_count=('has_catatonic_tail', 'sum'),
    catatonic_tail_s=('catatonic_duration_s', 'sum'),
).round(3)

rollup = rollup.merge(ev_agg, on=KEY, how='left')

count_cols = ['seizure_count', 'catatonic_tail_count']
rollup[count_cols] = rollup[count_cols].fillna(0).astype(int)
rollup[['total_seizing_s', 'catatonic_tail_s']] = rollup[['total_seizing_s', 'catatonic_tail_s']].fillna(0)
rollup['burden_frac'] = (rollup['total_seizing_s'] / tracked).round(4)

rollup = rollup.sort_values(KEY).reset_index(drop=True)

# --- Self-check: events can never exceed burst-labelled seconds ---
bad = rollup[rollup['total_seizing_s'] > rollup['seconds_burst']]
assert bad.empty, f"seizing time exceeds burst seconds:\n{bad[KEY + ['total_seizing_s', 'seconds_burst']]}"

rollup.to_csv(OUT, index=False)

n_with = int((rollup['seizure_count'] > 0).sum())
print(f"Fish: {len(rollup)}  |  with >=1 seizure: {n_with}  |  none: {len(rollup) - n_with}")
print("\nPer-group summary:")
print(rollup.groupby(GROUP_COL).agg(
    fish=('seizure_count', 'size'),
    seizures=('seizure_count', 'sum'),
    mean_per_fish=('seizure_count', 'mean'),
    median_per_fish=('seizure_count', 'median'),
).round(2).to_string())
print(f"\nSaved rollup -> {OUT}")
