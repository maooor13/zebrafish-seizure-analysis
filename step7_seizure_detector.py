"""
Step 7 — Per-second seizure state machine (Phase 1, vectorized).

Phases per fish-second:
  rest       : velocity < REST_FLOOR
  hyper      : REST_FLOOR <= velocity < BURST_GATE
  burst      : velocity > BURST_GATE
  catatonic  : velocity < FREEZE_FLOOR for 1-3 consecutive s right after a burst
  null       : missing velocity (gap — breaks burst contiguity)

Burst runs with >= MIN_BURST_DURATION seconds become seizure events; each may
carry a post-burst catatonic tail (1-3 consecutive freeze seconds).

Outputs:
  ../seizure_seconds.csv  — per-second phase labels
  ../seizure_events.csv   — one row per detected seizure event

Run: python step7_seizure_detector.py   (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seizure_config import (
    REST_FLOOR, BURST_GATE, FREEZE_FLOOR, MIN_BURST_DURATION,
    FREEZE_WINDOW, FREEZE_DURATION_MIN, FREEZE_DURATION_MAX,
    VEL_COL, TIME_COL, TRIAL_COL, FISH_COL, PHASE_COL, EVENT_ID_COL,
)
from dataset_config import T_CUTOFF, LABEL

ALLDATA = '../AllData.csv'
df = pd.read_csv(ALLDATA)
print(f"[dataset: {LABEL}]  per-second input: {ALLDATA}  ({len(df):,} rows)")

df[TRIAL_COL] = df[TRIAL_COL].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
df[FISH_COL] = df[FISH_COL].astype(str).str.strip()

# Manual exclusions — phantom tracking (no fish in video; EthoVision fabricated velocity)
fish_to_exclude = [
    ('Trial 28', 'C8'),
    ('Trial 29', 'A5'),
]
for trial, fish in fish_to_exclude:
    n_before = len(df)
    df = df[~((df[TRIAL_COL] == trial) & (df[FISH_COL] == fish))]
    print(f"  Excluded {trial}/{fish}: {n_before - len(df)} rows removed")

if T_CUTOFF is not None:
    n0 = len(df)
    df = df[df[TIME_COL] < T_CUTOFF]
    print(f"  time cutoff < {T_CUTOFF}s applied: {n0:,} -> {len(df):,} rows")

df = df.sort_values([TRIAL_COL, FISH_COL, TIME_COL], kind='mergesort').reset_index(drop=True)
df['pos'] = df.index

# --- Vectorized phase classification ---
v = df[VEL_COL]
is_null = v.isna()
is_burst = v > BURST_GATE
is_hyper = (v >= REST_FLOOR) & (v <= BURST_GATE)
df[PHASE_COL] = np.select(
    [is_null, is_burst, is_hyper],
    ['null', 'burst', 'hyper'],
    default='rest'
)

# --- Run-length encode contiguous phase runs per fish ---
fish_id = df[TRIAL_COL] + '_' + df[FISH_COL]
run_change = (df[PHASE_COL] != df[PHASE_COL].shift()).astype(int)
fish_change = (fish_id != fish_id.shift()).astype(int)
df['run_id'] = (run_change | fish_change).cumsum()

# --- Summarize runs ---
run_summary = df.groupby('run_id').agg(
    trial=(TRIAL_COL, 'first'),
    fish=(FISH_COL, 'first'),
    phase=(PHASE_COL, 'first'),
    pos_start=('pos', 'min'),
    pos_end=('pos', 'max'),
    duration=('pos', 'size'),
    peak_vel=(VEL_COL, 'max'),
    mean_vel=(VEL_COL, 'mean'),
).reset_index()

# Keep only burst runs with enough duration
burst_runs = run_summary[
    (run_summary['phase'] == 'burst') & (run_summary['duration'] >= MIN_BURST_DURATION)
].copy().reset_index(drop=True)

# Per-fish sequential event IDs
burst_runs['ev_num'] = burst_runs.groupby(['trial', 'fish']).cumcount() + 1
burst_runs['event_id'] = (
    burst_runs['trial'] + '_' + burst_runs['fish'] + '_ev'
    + burst_runs['ev_num'].astype(str)
)

# --- Catatonic tail scan (loop over events only — small count) ---
trials_arr = df[TRIAL_COL].to_numpy()
fishes_arr = df[FISH_COL].to_numpy()
freeze_eligible = (
    (df[PHASE_COL] == 'rest') & (df[VEL_COL] < FREEZE_FLOOR) & df[VEL_COL].notna()
).to_numpy()

cat_starts = [None] * len(burst_runs)
cat_ends = [None] * len(burst_runs)
cat_durs = [0] * len(burst_runs)

for idx, br in burst_runs.iterrows():
    pos_end = int(br['pos_end'])
    trial = br['trial']
    fish = br['fish']
    end_limit = min(len(df), pos_end + 1 + FREEZE_WINDOW)
    j = pos_end + 1
    while j < end_limit:
        if freeze_eligible[j] and trials_arr[j] == trial and fishes_arr[j] == fish:
            k = j
            while (k < end_limit and freeze_eligible[k]
                   and trials_arr[k] == trial and fishes_arr[k] == fish):
                k += 1
            run_dur = k - j
            if FREEZE_DURATION_MIN <= run_dur <= FREEZE_DURATION_MAX:
                cat_starts[idx] = j
                cat_ends[idx] = k - 1
                cat_durs[idx] = run_dur
                break
        j += 1

burst_runs['cat_start_pos'] = cat_starts
burst_runs['cat_end_pos'] = cat_ends
burst_runs['cat_dur'] = cat_durs
burst_runs['has_catatonic_tail'] = burst_runs['cat_start_pos'].notna()

# --- Tag catatonic seconds in phase column ---
cat_positions = []
for cs, ce in zip(cat_starts, cat_ends):
    if cs is not None:
        cat_positions.extend(range(cs, ce + 1))
if cat_positions:
    df.loc[cat_positions, PHASE_COL] = 'catatonic'

# --- Build event list ---
events_df = pd.DataFrame({
    TRIAL_COL: burst_runs['trial'].values,
    FISH_COL: burst_runs['fish'].values,
    EVENT_ID_COL: burst_runs['event_id'].values,
    'burst_start_s': df.loc[burst_runs['pos_start'].values, TIME_COL].values,
    'burst_end_s': df.loc[burst_runs['pos_end'].values, TIME_COL].values,
    'burst_duration_s': burst_runs['duration'].values,
    'burst_peak_velocity': burst_runs['peak_vel'].round(3).values,
    'burst_mean_velocity': burst_runs['mean_vel'].round(3).values,
    'has_catatonic_tail': burst_runs['has_catatonic_tail'].values,
    'catatonic_start_s': [df.loc[cs, TIME_COL] if cs is not None else None for cs in cat_starts],
    'catatonic_end_s': [df.loc[ce, TIME_COL] if ce is not None else None for ce in cat_ends],
    'catatonic_duration_s': cat_durs,
})

# --- Clean up helper columns ---
df = df.drop(columns=['pos', 'run_id', 'fish_id', '_freeze_eligible'], errors='ignore')

# --- Save outputs ---
df.to_csv('../seizure_seconds.csv', index=False)
events_df.to_csv('../seizure_events.csv', index=False)

n_fish_with_events = events_df.groupby([TRIAL_COL, FISH_COL]).size().shape[0] if len(events_df) else 0
print(f"\nDetected {len(events_df)} seizure events.")
print(f"Fish with >=1 detected event: {n_fish_with_events}")
if len(events_df):
    print(f"Catatonic tails tagged: {int(events_df['has_catatonic_tail'].sum())} / {len(events_df)}")
    print(f"Burst duration stats (s): {events_df['burst_duration_s'].describe().round(2).to_dict()}")
print(f"Saved per-second labels -> ../seizure_seconds.csv")
print(f"Saved event list        -> ../seizure_events.csv")
