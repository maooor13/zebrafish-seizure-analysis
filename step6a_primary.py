"""
Step 6a' — Primary inferential analysis
Headline: genotype x treatment interaction on Total_distance (sqrt-transformed)
- HC3 robust SE OLS (heteroscedasticity-robust)
- Effect size (eta²_p) from Type III ANOVA on sqrt-transformed data
- Mann-Whitney pairwise supplementary (Bonferroni-corrected)

Run: python step6a_primary.py  (from /code/code/, your venv env)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pingouin as pg
from dataset_config import CSV, LABEL

df = pd.read_csv(CSV)

print(f"[dataset: {LABEL}]  input: {CSV}")

# Split Fish Type into Genotype and Treatment
df['Genotype'] = df['Fish Type'].str.split('_').str[0]
df['Treatment'] = df['Fish Type'].str.split('_').str[1]

# PTZ level name (ptz10 for 10mM, ptz5 for 5mM) — derived from data, not hardcoded
_treatments = df['Treatment'].unique().tolist()
_ptz = [t for t in _treatments if t != 'control'][0]
_wtptz, _mutptz = f'wt_{_ptz}', f'mut_{_ptz}'

# sqrt transform (Bandara 2020; passes normality for PTZ groups per 6b recheck)
df['Total_distance_sqrt'] = np.sqrt(df['Total_distance_cm'])

# Clean trial name (has extra spaces: "Trial    28" -> "Trial 28")
df['Trial'] = df['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()

print("=" * 70)
print("STEP 6a' — PRIMARY ANALYSIS")
print("Total_distance (sqrt-transformed), HC3 robust SE")
print("=" * 70)

# ----------------------------------------------------------------------
# PART 1: HC3 robust OLS — primary inference
# ----------------------------------------------------------------------
# C(Trial) as blocking covariate (device L/R confound, per plan :251-260)
# Interaction term is the headline (plan :14)
model = smf.ols(
    'Total_distance_sqrt ~ C(Genotype) * C(Treatment) + C(Trial)',
    data=df
).fit(cov_type='HC3')

print("\n--- HC3 Robust OLS Summary ---")
print(model.summary().tables[1])

# Wald test on the interaction term
# Reference category is alphabetical: Genotype mut, Treatment control
# Interaction coef = C(Genotype)[T.wt]:C(Treatment)[T.ptz10]
interaction_term = None
for term in model.params.index:
    if 'Genotype' in term and 'Treatment' in term:
        interaction_term = term
        break

if interaction_term:
    wald = model.wald_test(interaction_term, use_f=True, scalar=True)
    coef = model.params[interaction_term]
    print(f"\nHEADLINE — Genotype x Treatment interaction:")
    print(f"  Coefficient (wt:ptz effect relative to mut:ctrl): {coef:+.4f}")
    print(f"  F({wald.df_num:.0f},{wald.df_denom:.0f}) = {wald.fvalue:.3f}, p = {wald.pvalue:.6f}")
    print(f"  Robust SE (HC3): {model.bse[interaction_term]:.4f}")
else:
    print("\nWARNING: interaction term not found in model params")
    print("Available terms:", list(model.params.index))

# Main effects
print(f"\nMain effect of Genotype:")
g_term = 'C(Genotype)[T.wt]'
if g_term in model.params.index:
    wald_g = model.wald_test(g_term, use_f=True, scalar=True)
    print(f"  F({wald_g.df_num:.0f},{wald_g.df_denom:.0f}) = {wald_g.fvalue:.3f}, p = {wald_g.pvalue:.6f}")

print(f"\nMain effect of Treatment:")
t_term = f'C(Treatment)[T.{_ptz}]'
if t_term in model.params.index:
    wald_t = model.wald_test(t_term, use_f=True, scalar=True)
    print(f"  F({wald_t.df_num:.0f},{wald_t.df_denom:.0f}) = {wald_t.fvalue:.3f}, p = {wald_t.pvalue:.6f}")

# ----------------------------------------------------------------------
# PART 2: Effect size (eta²_p) from Type III ANOVA
# ----------------------------------------------------------------------
# Standard (non-robust) ANOVA on sqrt data for SS decomposition -> eta²_p
# Effect size for magnitude; robust p above for inference (standard practice)
print("\n--- Effect Size (partial eta-squared) ---")
aov = pg.anova(
    dv='Total_distance_sqrt',
    between=['Genotype', 'Treatment'],
    data=df,
    detailed=True
)
print("\nPingouin 2-way ANOVA (Genotype x Treatment, sqrt-transformed):")
print(aov.to_string(index=False))
# ponytail: eta²_p is unblocked 2x2 (magnitude); inferential p comes from HC3 model with Trial block

# ----------------------------------------------------------------------
# PART 3: Descriptive stats per group (for reporting)
# ----------------------------------------------------------------------
print("\n--- Descriptive Statistics (raw Total_distance_cm) ---")
desc = df.groupby('Fish Type')['Total_distance_cm'].agg(
    ['count', 'mean', 'std', 'sem']
).round(2)
desc['95% CI lower'] = (desc['mean'] - 1.96 * desc['sem']).round(2)
desc['95% CI upper'] = (desc['mean'] + 1.96 * desc['sem']).round(2)
print(desc)

# ----------------------------------------------------------------------
# PART 4: Mann-Whitney pairwise (supplementary)
# ----------------------------------------------------------------------
print("\n--- Mann-Whitney U Pairwise (supplementary, Bonferroni-corrected) ---")
pairs = [
    ('wt_control', _wtptz),       # WT PTZ response
    ('mut_control', _mutptz),     # Mut PTZ response
    (_wtptz, _mutptz),            # Headline pairwise
    ('wt_control', 'mut_control'),# Genotype baseline
    ('wt_control', _mutptz),
    ('mut_control', _wtptz),
]

mw_results = []
for g1, g2 in pairs:
    a = df[df['Fish Type'] == g1]['Total_distance_cm']
    b = df[df['Fish Type'] == g2]['Total_distance_cm']
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

print("\n" + "=" * 70)
print("DONE — interpret interaction p-value (Part 1) as the headline")
print("=" * 70)
