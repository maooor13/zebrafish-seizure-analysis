"""
Phase 0 — Merge instructor per-fish CSVs with our burst metrics.

Produces 3 combined CSVs in /code/ with old-script-compatible headers:
  1. PerFish_10mM_combined.csv       (191 fish, 60-min, regular + burst)
  2. PerFish_10mM_20min_combined.csv (191 fish, first 20-min, regular + burst)
  3. PerFish_5mM_combined.csv       (191 fish, 60-min, regular only — no per-second)

Output schema (matches existing Step 6 scripts that read PerFish_Primary.csv):
  Etho_Trial, Fish, Fish Type, Fish_ID, Total_distance_cm, Mean_distance,
  Mean_velocity, Mean_movement, Mean_min_accel, Mean_max_accel, Mean_mobility,
  Event_count_ThrA, Time_above_ThrA, Latency_first_event_ThrA, Max_velocity (10mM only),
  Decision, Outlier_Flag (carried, unfiltered; 10mM 60-min only)

Distance: Total_distance_cm = Mean_distance * (60 * window_minutes). Mean_distance
is the instructor's per-fish mean distance rate (cm/s); multiplying by recording
seconds gives total cm. 60-min identity validated vs our PerFish_Primary.csv
Total_distance_cm (corr 1.0, ratio 1.0). 20-min uses *1200 (self-consistent with
instructor's 20-min export; noted caveat in methods: corr 0.80 vs per-second sum
due to windowing differences). 5mM uses *3600 (rate x time; no per-second to check).

Burst (10mM only): 60-min reused from PerFish_Primary.csv (already computed/validated);
20-min recomputed from AllData.csv per-second, Time_seconds < 1200. 5mM has no
per-second data -> no burst metrics.

Etho_Trial carries EthoVision labels (Trial 26/27/28/29) for 10mM via reverse
crosswalk, so step6a2_kinetics.py / step11_raster.py joins with AllData.csv still
work. 5mM has no AllData join -> keeps instructor label (8.6L/31.5L).

Crosswalk (instructor plate <-> EthoVision trial), validated 191/191 merge:
  24.5R <-> Trial 26, 25.5R <-> Trial 27, 24.5L <-> Trial 28, 25.5L <-> Trial 29

Run: PYTHONPATH=/code/code/venv/lib/python3.12/site-packages /usr/bin/python3.12 code/merge_perfish.py
"""
import os
import numpy as np
import pandas as pd

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
os.chdir(ROOT)

# instructor plate -> EthoVision trial label
INS_TO_ETHO = {'24.5R': 'Trial 26', '25.5R': 'Trial 27', '24.5L': 'Trial 28', '25.5L': 'Trial 29'}
BURST_COLS = ['Event_count_ThrA', 'Time_above_ThrA', 'Latency_first_event_ThrA', 'Max_velocity']


def clean_etho(s):
    return s.astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()


def fish_burst(g):
    """Per-fish burst metrics from per-second Velocity (>2.8 cm/s)."""
    v = g.sort_values('Time_seconds')
    above = (v['Velocity'] > 2.8).astype(int).values
    starts = 0
    prev = 0
    for a in above:
        if a == 1 and prev == 0:
            starts += 1
        prev = a
    lat = v[v['Velocity'] > 2.8]
    return pd.Series({
        'Event_count_ThrA': starts,
        'Time_above_ThrA': int(above.sum()),
        'Latency_first_event_ThrA': float(lat['Time_seconds'].iloc[0]) if len(lat) else np.nan,
        'Max_velocity': float(v['Velocity'].max()),
    })


def compute_burst_from_alldata(seconds_cutoff=None):
    """Recompute burst metrics from AllData.csv per-second, optionally time-filtered."""
    ad = pd.read_csv('AllData.csv')
    ad['Trial'] = clean_etho(ad['Etho_Trial'])
    ad['Fish'] = ad['Fish'].astype(str).str.strip()
    if seconds_cutoff is not None:
        ad = ad[ad['Time_seconds'] < seconds_cutoff].copy()
    burst = ad.groupby(['Trial', 'Fish'], as_index=False).apply(fish_burst, include_groups=False).reset_index(drop=True)
    burst = burst.loc[:, ~burst.columns.duplicated()]
    return burst.rename(columns={'Trial': 'Etho_Trial', 'Fish': 'Fish'})


# ---------------------------------------------------------------------------
# 1. 10mM full (60-min)
# ---------------------------------------------------------------------------
def merge_10mM_full():
    ins = pd.read_csv('PerFish_10mM_full_minutes.csv')
    prim = pd.read_csv('PerFish_Primary.csv')

    ins['Fish'] = ins['Fish_ID'].str.split('_').str[1]
    ins['Etho_Trial'] = ins['Trial'].map(INS_TO_ETHO)
    prim['Etho_Trial'] = clean_etho(prim['Etho_Trial'])
    prim['Fish'] = prim['Fish'].astype(str).str.strip()

    m = ins.merge(
        prim[['Etho_Trial', 'Fish'] + BURST_COLS],
        on=['Etho_Trial', 'Fish'], how='inner'
    )
    m['Total_distance_cm'] = m['Mean_distance'] * 3600.0

    keep = ['Etho_Trial', 'Fish', 'Fish_Type', 'Fish_ID',
            'Total_distance_cm', 'Mean_distance', 'Mean_velocity', 'Mean_movement',
            'Mean_min_accel', 'Mean_max_accel', 'Mean_mobility',
            'Decision', 'Outlier_Flag'] + BURST_COLS
    out = m[keep].rename(columns={'Fish_Type': 'Fish Type'}).copy()

    print(f"[10mM full] merged {len(out)}/191 | groups: {dict(out['Fish Type'].value_counts().sort_index())}")
    print(f"[10mM full] unmatched: {sorted(set(ins['Fish_ID']) - set(out['Fish_ID']))}")
    out.to_csv('PerFish_10mM_combined.csv', index=False)
    print("[10mM full] -> PerFish_10mM_combined.csv")
    return out


# ---------------------------------------------------------------------------
# 2. 10mM first-20-min (regular from instructor export, burst from AllData <1200s)
# ---------------------------------------------------------------------------
def merge_10mM_20min():
    ins = pd.read_csv('PerFish_10mM_20min.csv')
    burst = compute_burst_from_alldata(seconds_cutoff=1200)

    ins['Fish'] = ins['Fish_ID'].str.split('_').str[1]
    ins['Etho_Trial'] = ins['Trial'].map(INS_TO_ETHO)
    m = ins.merge(burst, on=['Etho_Trial', 'Fish'], how='inner')
    m['Total_distance_cm'] = m['Mean_distance'] * 1200.0

    keep = ['Etho_Trial', 'Fish', 'Fish_Type', 'Fish_ID',
            'Total_distance_cm', 'Mean_distance', 'Mean_velocity', 'Mean_movement',
            'Mean_mobility'] + BURST_COLS
    out = m[keep].rename(columns={'Fish_Type': 'Fish Type'}).copy()

    print(f"[10mM 20min] merged {len(out)}/191 | groups: {dict(out['Fish Type'].value_counts().sort_index())}")
    print(f"[10mM 20min] burst non-null Event_count: {out['Event_count_ThrA'].notna().sum()}")
    out.to_csv('PerFish_10mM_20min_combined.csv', index=False)
    print("[10mM 20min] -> PerFish_10mM_20min_combined.csv")
    return out


# ---------------------------------------------------------------------------
# 3. 5mM (regular only, no burst)
# ---------------------------------------------------------------------------
def merge_5mM():
    df = pd.read_csv('PerFish_5mM_full_minutes.csv')
    before = len(df)
    df = df[df['Decision'] != 'exclude'].copy()
    print(f"[5mM] dropped {before - len(df)} exclude -> {len(df)}")

    df['Fish'] = df['Fish_ID'].str.split('_').str[1]
    df['Etho_Trial'] = df['Trial']  # no AllData join for 5mM; keep instructor label
    df['Total_distance_cm'] = df['Mean_distance'] * 3600.0

    keep = ['Etho_Trial', 'Fish', 'Fish_Type', 'Fish_ID',
            'Total_distance_cm', 'Mean_distance', 'Mean_velocity', 'Mean_movement',
            'Mean_min_accel', 'Mean_max_accel', 'Mean_mobility', 'Decision']
    out = df[keep].rename(columns={'Fish_Type': 'Fish Type'}).copy()

    print(f"[5mM] groups: {dict(out['Fish Type'].value_counts().sort_index())}")
    out.to_csv('PerFish_5mM_combined.csv', index=False)
    print("[5mM] -> PerFish_5mM_combined.csv")
    return out


if __name__ == '__main__':
    print("=" * 70)
    merge_10mM_full()
    print("=" * 70)
    merge_10mM_20min()
    print("=" * 70)
    merge_5mM()
    print("=" * 70)
    print("Phase 0 complete.")
