"""
Step 6a' — Burst-specific metrics
Tests Event_count_ThrA, Time_above_ThrA, Max_velocity
Model: sqrt + HC3 robust OLS + Mann-Whitney supplementary.

PRIMARY TEST BY METRIC TYPE (chosen by data structure, not applied uniformly):
  - Continuous, non-zero (Max_velocity): HC3 robust OLS is primary; MW supplementary.
  - Zero-inflated counts (Event_count, Time_above): Mann-Whitney is primary;
    OLS is supplementary because sqrt(0)=0 cannot stabilize zero-inflated variance.

sqrt on counts: variance-stabilizing — Poisson counts have Var=Mean, so groups
  with more events have more variance (the heteroscedasticity Levene caught).
  sqrt makes variance ~constant regardless of mean. Fails when zeros dominate.
HC3 robust SE: handles residual heteroscedasticity after transform; best
  small-sample (n<250) HC estimator, downweights high-leverage points (zero-inflated controls).
Mann-Whitney: rank-based, distribution-free — handles zero-inflation naturally
  (zeros share lowest ranks). Primary for count metrics where OLS breaks down.

Pairwise set (4 comparisons, Bonferroni over 4):
  - wt_ptz10 vs mut_ptz10  (headline: mutant vs WT seizure response)
  - wt_control vs mut_control  (baseline sanity, expected null)
  - wt_ptz10 vs wt_control  (PTZ effect in WT, expected sig)
  - mut_ptz10 vs mut_control  (PTZ effect in mutant, expected sig)

Run: python step6a_burst_metrics.py  (from /code/code/, your venv env)
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pingouin as pg
from dataset_config import CSV, LABEL, T_CUTOFF

df = pd.read_csv(CSV)
print(f"[dataset: {LABEL}]  input: {CSV}")
df['Genotype'] = df['Fish Type'].str.split('_').str[0]
df['Treatment'] = df['Fish Type'].str.split('_').str[1]
df['Trial'] = df['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()

if T_CUTOFF is None or 'Event_count_ThrA' not in df.columns:
    print("No burst metrics for this dataset (no per-second data) — skipping step6a_burst_metrics.")
    import sys; sys.exit(0)

# Metrics to test
dvs = ['Event_count_ThrA', 'Time_above_ThrA', 'Max_velocity']

# Pairwise comparisons (4, Bonferroni over 4)
pairs = [
    ('wt_ptz10', 'mut_ptz10'),       # headline: mutant vs WT seizure response
    ('wt_control', 'mut_control'),   # baseline sanity (expected null)
    ('wt_ptz10', 'wt_control'),       # PTZ effect in WT (expected sig)
    ('mut_ptz10', 'mut_control'),    # PTZ effect in mutant (expected sig)
]

summary_rows = []

for dv in dvs:
    # Handle nulls (Latency had them, but counts shouldn't — be safe)
    sub = df.dropna(subset=[dv]).copy()
    sub[f'{dv}_sqrt'] = np.sqrt(sub[dv])

    print("\n" + "#" * 70)
    print(f"# {dv}")
    print("#" * 70)

    # --- Descriptive stats (raw) ---
    print(f"\n--- Descriptive Statistics (raw {dv}) ---")
    desc = sub.groupby('Fish Type')[dv].agg(
        ['count', 'mean', 'std', 'sem']
    ).round(2)
    desc['95% CI lower'] = (desc['mean'] - 1.96 * desc['sem']).round(2)
    desc['95% CI upper'] = (desc['mean'] + 1.96 * desc['sem']).round(2)
    print(desc)

    # --- HC3 robust OLS ---
    model = smf.ols(
        f'{dv}_sqrt ~ C(Genotype) * C(Treatment) + C(Trial)',
        data=sub
    ).fit(cov_type='HC3')

    # Interaction term (headline)
    interaction_term = None
    for term in model.params.index:
        if 'Genotype' in term and 'Treatment' in term:
            interaction_term = term
            break

    ols_interaction_p = None
    ols_interaction_coef = None
    if interaction_term:
        wald = model.wald_test(interaction_term, use_f=True, scalar=True)
        coef = model.params[interaction_term]
        ols_interaction_p = wald.pvalue
        ols_interaction_coef = coef
        print(f"\nHEADLINE — Genotype x Treatment interaction:")
        print(f"  Coefficient: {coef:+.4f}")
        print(f"  F({wald.df_num:.0f},{wald.df_denom:.0f}) = {wald.fvalue:.3f}, p = {wald.pvalue:.6f}")
        print(f"  Robust SE (HC3): {model.bse[interaction_term]:.4f}")
    else:
        print("\nWARNING: interaction term not found")

    # Main effects
    g_term = 'C(Genotype)[T.wt]'
    if g_term in model.params.index:
        wald_g = model.wald_test(g_term, use_f=True, scalar=True)
        print(f"\nMain effect Genotype: F({wald_g.df_num:.0f},{wald_g.df_denom:.0f}) = {wald_g.fvalue:.3f}, p = {wald_g.pvalue:.6f}")

    t_term = 'C(Treatment)[T.ptz10]'
    if t_term in model.params.index:
        wald_t = model.wald_test(t_term, use_f=True, scalar=True)
        print(f"Main effect Treatment: F({wald_t.df_num:.0f},{wald_t.df_denom:.0f}) = {wald_t.fvalue:.3f}, p = {wald_t.pvalue:.6f}")

    # --- Effect size (eta²_p) ---
    aov = pg.anova(
        dv=f'{dv}_sqrt',
        between=['Genotype', 'Treatment'],
        data=sub,
        detailed=True
    )
    interaction_row = aov[aov['Source'] == 'Genotype * Treatment']
    if not interaction_row.empty:
        np2 = interaction_row['np2'].values[0]
        print(f"\nEffect size (eta²_p, interaction): {np2:.4f}")

    # --- Mann-Whitney pairwise (4 comparisons, Bonferroni over 4) ---
    print(f"\n--- Mann-Whitney U Pairwise (Bonferroni over {len(pairs)}) ---")
    mw_results = []
    for g1, g2 in pairs:
        a = sub[sub['Fish Type'] == g1][dv]
        b = sub[sub['Fish Type'] == g2][dv]
        r = pg.mwu(a, b, alternative='two-sided')
        r.insert(0, 'comparison', f'{g1} vs {g2}')
        r.insert(1, 'n1', len(a))
        r.insert(2, 'n2', len(b))
        mw_results.append(r)

    mw_out = pd.concat(mw_results, ignore_index=True)
    n_tests = len(pairs)
    mw_out['p_bonferroni'] = (mw_out['p_val'] * n_tests).clip(upper=1.0)
    mw_out['sig_bonf'] = mw_out['p_bonferroni'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
    )
    print(mw_out.to_string(index=False))

    # --- Agreement check ---
    headline_mw = mw_out[mw_out['comparison'] == 'wt_ptz10 vs mut_ptz10']
    if not headline_mw.empty and ols_interaction_p is not None:
        mw_p = headline_mw['p_val'].values[0]
        mw_bonf = headline_mw['p_bonferroni'].values[0]
        ols_sig = ols_interaction_p < 0.05
        mw_sig = mw_bonf < 0.05
        agree = ols_sig == mw_sig
        print(f"\nAgreement check (OLS interaction vs MW headline pairwise):")
        print(f"  OLS interaction p = {ols_interaction_p:.6f} ({'sig' if ols_sig else 'ns'})")
        print(f"  MW headline p (raw) = {mw_p:.6f}, (Bonf) = {mw_bonf:.6f} ({'sig' if mw_sig else 'ns'})")
        if agree:
            print(f"  → AGREE — robust finding")
        else:
            print(f"  → DISAGREE — zero-inflation likely affecting OLS;")
            print(f"    report MW as primary for this metric, OLS as supplementary")

    # Collect summary
    summary_rows.append({
        'metric': dv,
        'ols_interaction_p': ols_interaction_p,
        'ols_coef': ols_interaction_coef,
        'eta2_p': np2 if not interaction_row.empty else None,
        'mw_headline_p_raw': mw_p if not headline_mw.empty else None,
        'mw_headline_p_bonf': mw_bonf if not headline_mw.empty else None,
        'agree': agree if (not headline_mw.empty and ols_interaction_p is not None) else None,
    })

# --- Final summary table ---
print("\n\n" + "=" * 70)
print("SUMMARY ACROSS ALL BURST METRICS")
print("=" * 70)
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
print("\n(Compare to Total_distance interaction — see step6a_primary.py)")
