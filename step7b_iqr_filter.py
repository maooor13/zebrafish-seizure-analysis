"""
Step 7b — Hard cap on seizure burst duration.

Reads seizure_events.csv (Phase 1 output) and drops bursts longer than
MAX_BURST_DURATION seconds (phantom-tracking / artifact guard). A flat cap
is used instead of per-group IQR so every group gets the same standard.

seizure_seconds.csv (per-second labels) is left untouched — only the event
list is filtered.

Outputs:
  ../seizure_events_iqr_filtered.csv  — events after the cap
  ../seizure_events_iqr_outliers.csv  — dropped bursts (audit trail)

Run: python step7b_iqr_filter.py   (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seizure_config import TRIAL_COL, FISH_COL
from dataset_config import LABEL

EVENTS = '../seizure_events.csv'
ALLDATA = '../AllData.csv'
GROUP_COL = 'Fish Type'
MAX_BURST_DURATION = 10.0  # seconds

if not os.path.exists(EVENTS):
    print(f"ERROR: {EVENTS} not found. Run step7_seizure_detector.py first.")
    sys.exit(1)

events = pd.read_csv(EVENTS)
print(f"[dataset: {LABEL}]  loaded {EVENTS}  ({len(events)} events)")

# Group mapping (for the audit table only)
all_data = pd.read_csv(ALLDATA, usecols=[TRIAL_COL, FISH_COL, GROUP_COL])
all_data[TRIAL_COL] = all_data[TRIAL_COL].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
all_data[FISH_COL] = all_data[FISH_COL].astype(str).str.strip()
group_map = all_data.drop_duplicates(subset=[TRIAL_COL, FISH_COL])
events = events.merge(group_map, on=[TRIAL_COL, FISH_COL], how='left')
events[GROUP_COL] = events[GROUP_COL].fillna('unknown')

# --- Hard cap on burst duration ---
kept = events[events['burst_duration_s'] <= MAX_BURST_DURATION].copy()
dropped = events[events['burst_duration_s'] > MAX_BURST_DURATION].copy()

print(f"  Max burst duration cap: {MAX_BURST_DURATION}s")
print(f"  Dropped {len(dropped)} events longer than {MAX_BURST_DURATION}s (of {len(events)} total)")

filtered = kept.sort_values([TRIAL_COL, FISH_COL, 'burst_start_s']).reset_index(drop=True)
dropped = dropped.sort_values([TRIAL_COL, FISH_COL, 'burst_start_s']).reset_index(drop=True)

filtered.to_csv('../seizure_events_iqr_filtered.csv', index=False)
dropped.to_csv('../seizure_events_iqr_outliers.csv', index=False)

print(f"\nKept {len(filtered)} events -> ../seizure_events_iqr_filtered.csv")
print(f"Dropped {len(dropped)} outlier events -> ../seizure_events_iqr_outliers.csv")

if len(dropped):
    print("\nDropped bursts (audit):")
    print(dropped[[TRIAL_COL, FISH_COL, GROUP_COL, 'burst_start_s',
                   'burst_duration_s']].to_string(index=False))
