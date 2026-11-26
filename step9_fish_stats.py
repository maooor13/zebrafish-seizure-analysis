"""
Step 9 — Statistical analysis on per-fish seizure metrics.

Reads ../seizure_fish_rollup.csv (fish-level) and ../seizure_events_iqr_filtered.csv
(burst-level, temporal split only).

Sections:
  1. Between-group (6 pairwise Mann-Whitney U + Cliff's delta, Holm-corrected)
     on seizure_count, total_seizing_s, mean_duration_s, max_duration_s, burden_frac.
  2. Latency to first seizure: Kaplan-Meier + log-rank (zero-event fish censored
     at T_CUTOFF). KM/log-rank hand-rolled in numpy — no lifelines dependency.
  3. Temporal: first half (0-1800s) vs second half (1800-3600s) per fish
     (Wilcoxon signed-rank on counts and median burst duration), plus Friedman
     across 4 quarters.
  4. Per-burst duration distribution (K-S, descriptive only — bursts within a
     fish are autocorrelated).

Output: ../seizure_fish_stats.csv (flat table of every test run).
Run: python step9_fish_stats.py   (from /code/code/)
"""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_config import T_CUTOFF, LABEL

ROLLUP = '../seizure_fish_rollup.csv'
EVENTS = '../seizure_events_iqr_filtered.csv'
OUT = '../seizure_fish_stats.csv'
GROUP_COL = 'Fish Type'
GROUPS = ['mut_control', 'mut_ptz10', 'wt_control', 'wt_ptz10']
METRICS = ['seizure_count', 'total_seizing_s', 'mean_duration_s',
           'max_duration_s', 'burden_frac']
HALF = (T_CUTOFF or 3600) / 2
QUARTER = (T_CUTOFF or 3600) / 4

for path in (ROLLUP, EVENTS):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found.")
        sys.exit(1)

roll = pd.read_csv(ROLLUP)
ev = pd.read_csv(EVENTS)
print(f"[dataset: {LABEL}]  {len(roll)} fish, {len(ev)} bursts")

rows = []


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


# ---------------------------------------------------------------- 1. groups
print("\n" + "=" * 78)
print("1. BETWEEN-GROUP (Mann-Whitney U, Holm-corrected, Cliff's delta)")
print("=" * 78)
print("n per group: " + ", ".join(f"{g}={int((roll[GROUP_COL] == g).sum())}" for g in GROUPS))

for metric in METRICS:
    print(f"\n{metric}")
    pairs = list(combinations(GROUPS, 2))
    stats_, pvals, ns = [], [], []
    for a, b in pairs:
        xa = roll.loc[roll[GROUP_COL] == a, metric].dropna().to_numpy()
        xb = roll.loc[roll[GROUP_COL] == b, metric].dropna().to_numpy()
        ns.append((len(xa), len(xb)))
        if len(xa) < 3 or len(xb) < 3:
            stats_.append(np.nan); pvals.append(np.nan)
            continue
        try:
            res = stats.mannwhitneyu(xa, xb, alternative='two-sided', method='asymptotic')
            u, p = res.statistic, res.pvalue
        except ValueError:
            u, p = np.nan, np.nan
        stats_.append(u); pvals.append(p)

    valid = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    adj = [np.nan] * len(pvals)
    if valid:
        for i, a in zip(valid, holm([pvals[i] for i in valid])):
            adj[i] = a

    for (a, b), (na, nb), u, p, pa in zip(pairs, ns, stats_, pvals, adj):
        delta = 2 * u / (na * nb) - 1 if not np.isnan(u) and na and nb else np.nan
        sig = '***' if pa < 0.001 else '**' if pa < 0.01 else '*' if pa < 0.05 else 'ns'
        print(f"  {a:12s} vs {b:12s}  U={u:8.1f}  p={p:.4g}  p_holm={pa:.4g}  "
              f"delta={delta:+.2f}  {sig}")
        rows.append(dict(section='between_group', metric=metric, group_a=a, group_b=b,
                         n_a=na, n_b=nb, stat=u, p=p, p_holm=pa, effect=delta))

# ------------------------------------------------------------- 2. latency
print("\n" + "=" * 78)
print("2. LATENCY TO FIRST SEIZURE (Kaplan-Meier, log-rank)")
print("=" * 78)


def km(times, events):
    """Product-limit estimator. Returns (unique event times, survival prob)."""
    uniq = np.unique(times)
    s, out = 1.0, []
    for ut in uniq:
        at_risk = np.sum(times >= ut)
        d = np.sum((times == ut) & (events == 1))
        if at_risk > 0 and d > 0:
            s *= 1.0 - d / at_risk
        out.append(s)
    return uniq, np.array(out)


def logrank(times, events, groups):
    """k-sample log-rank chi-square (df = k-1)."""
    gs = np.unique(groups)
    k = len(gs)
    O = np.zeros(k); E = np.zeros(k); V = np.zeros((k, k))
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
                base = d * (n_g[i] / n) * ((1 if i == j else 0) - n_g[j] / n)
                V[i, j] += base * (n - d) / (n - 1)
    diff = (O - E)[:-1]
    chi2 = float(diff @ np.linalg.pinv(V[:-1, :-1]) @ diff)
    return chi2, float(stats.chi2.sf(chi2, k - 1))


lat_t = roll['latency_first_s'].fillna(T_CUTOFF or 3600).to_numpy(float)
lat_e = roll['latency_first_s'].notna().astype(int).to_numpy()
lat_g = roll[GROUP_COL].to_numpy()

for g in GROUPS:
    m = lat_g == g
    t, s = km(lat_t[m], lat_e[m])
    below = np.where(s <= 0.5)[0]
    med = t[below[0]] if len(below) else np.nan
    n_ev = int(lat_e[m].sum())
    med_str = f"{med:.0f}s" if not np.isnan(med) else 'not reached'
    print(f"  {g:12s}  n={m.sum():3d}  seizures={n_ev:3d}  "
          f"censored={m.sum() - n_ev:3d}  median latency={med_str}")
    rows.append(dict(section='latency', metric='latency_first_s', group_a=g, group_b='',
                     n_a=int(m.sum()), n_b=n_ev, stat=med, p=np.nan, p_holm=np.nan,
                     effect=np.nan))

chi2, p = logrank(lat_t, lat_e, lat_g)
print(f"\n  Log-rank across 4 groups: chi2={chi2:.3f}, df=3, p={p:.4g}")
rows.append(dict(section='latency_logrank', metric='latency_first_s', group_a='all',
                 group_b='', n_a=len(roll), n_b=int(lat_e.sum()), stat=chi2, p=p,
                 p_holm=np.nan, effect=np.nan))

# ------------------------------------------------------------ 3. temporal
print("\n" + "=" * 78)
print("3. TEMPORAL - first half vs second half (Wilcoxon signed-rank)")
print("=" * 78)

ev['half'] = np.where(ev['burst_start_s'] < HALF, 'first', 'second')
ev['quarter'] = np.minimum((ev['burst_start_s'] // QUARTER).astype(int), 3)

for g in GROUPS:
    sub = ev[ev[GROUP_COL] == g]
    if sub.empty:
        continue
    fish_ids = sub[['Etho_Trial', 'Fish']].drop_duplicates()

    counts = (sub.groupby(['Etho_Trial', 'Fish', 'half']).size()
              .unstack(fill_value=0).reindex(fish_ids.set_index(['Etho_Trial', 'Fish']).index,
                                             fill_value=0).fillna(0))
    for h in ('first', 'second'):
        if h not in counts:
            counts[h] = 0
    d_counts = counts['second'] - counts['first']

    med = (sub.groupby(['Etho_Trial', 'Fish', 'half'])['burst_duration_s'].median()
           .unstack().reindex(columns=['first', 'second']))
    both = med.dropna()
    d_med = (both['second'] - both['first']) if len(both) >= 3 else np.array([])

    def wilcox(d):
        d = np.asarray(d, float)
        if len(d) < 3 or np.all(d == 0):
            return np.nan, np.nan
        try:
            r = stats.wilcoxon(d)
            return r.statistic, r.pvalue
        except ValueError:
            return np.nan, np.nan

    w1, p1 = wilcox(d_counts)
    w2, p2 = wilcox(d_med)

    q = (sub.groupby(['Etho_Trial', 'Fish', 'quarter']).size()
         .unstack(fill_value=0)
         .reindex(columns=[0, 1, 2, 3], fill_value=0).fillna(0))
    if len(q) >= 3 and q.to_numpy().std() > 0:
        try:
            f_stat, f_p = stats.friedmanchisquare(*[q[c].to_numpy() for c in q.columns])
        except ValueError:
            f_stat, f_p = np.nan, np.nan
    else:
        f_stat, f_p = np.nan, np.nan

    print(f"\n  {g}  (fish with events: {len(fish_ids)})")
    print(f"    count      first={counts['first'].median():.1f}  second={counts['second'].median():.1f}  "
          f"median delta={d_counts.median():+.1f}  W={w1:.1f}  p={p1:.4g}")
    print(f"    duration   first={both['first'].median() if len(both) else np.nan:.2f}s  "
          f"second={both['second'].median() if len(both) else np.nan:.2f}s  "
          f"n_paired={len(both)}  W={w2:.1f}  p={p2:.4g}")
    print(f"    quarters   medians={[round(float(q[c].median()), 1) for c in q.columns]}  "
          f"Friedman chi2={f_stat:.3f}  p={f_p:.4g}")

    rows.append(dict(section='temporal_count', metric='seizure_count', group_a=g, group_b='',
                     n_a=len(fish_ids), n_b=len(d_counts), stat=w1, p=p1, p_holm=np.nan,
                     effect=d_counts.median()))
    rows.append(dict(section='temporal_duration', metric='median_duration_s', group_a=g,
                     group_b='', n_a=len(fish_ids), n_b=len(both), stat=w2, p=p2,
                     p_holm=np.nan, effect=(both['second'] - both['first']).median()
                     if len(both) else np.nan))
    rows.append(dict(section='temporal_quarter_trend', metric='seizure_count', group_a=g,
                     group_b='', n_a=len(q), n_b=4, stat=f_stat, p=f_p, p_holm=np.nan,
                     effect=np.nan))

# ------------------------------------------------- 4. burst distribution
print("\n" + "=" * 78)
print("4. PER-BURST DURATION DISTRIBUTION (descriptive; autocorrelation caveat)")
print("=" * 78)
for g in GROUPS:
    d = ev.loc[ev[GROUP_COL] == g, 'burst_duration_s'].dropna()
    if len(d):
        print(f"  {g:12s}  n={len(d):5d}  median={d.median():.1f}s  "
              f"IQR={d.quantile(.25):.1f}-{d.quantile(.75):.1f}s  max={d.max():.0f}s")

print("\n  K-S tests (per-burst, secondary):")
for a, b in combinations(GROUPS, 2):
    da = ev.loc[ev[GROUP_COL] == a, 'burst_duration_s'].dropna()
    db = ev.loc[ev[GROUP_COL] == b, 'burst_duration_s'].dropna()
    if len(da) < 3 or len(db) < 3:
        continue
    ks = stats.ks_2samp(da, db)
    print(f"    {a:12s} vs {b:12s}  D={ks.statistic:.3f}  p={ks.pvalue:.4g}")
    rows.append(dict(section='burst_ks', metric='burst_duration_s', group_a=a, group_b=b,
                     n_a=len(da), n_b=len(db), stat=ks.statistic, p=ks.pvalue,
                     p_holm=np.nan, effect=np.nan))

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nSaved {len(rows)} test rows -> {OUT}")
