"""
Step 6a' — Per-trial consistency gate
Plan :262: interaction direction must be consistent across all 4 trials.
If one flips -> device-L artifact flag.

Run: python step6a_trial_gate.py  (from /code/code/, your venv env)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')
from dataset_config import CSV, LABEL

df = pd.read_csv(CSV)
print(f"[dataset: {LABEL}]  input: {CSV}")
df['Genotype'] = df['Fish Type'].str.split('_').str[0]
df['Treatment'] = df['Fish Type'].str.split('_').str[1]
df['Total_distance_sqrt'] = np.sqrt(df['Total_distance_cm'])
df['Trial'] = df['Etho_Trial'].str.replace(r'\s+', ' ', regex=True).str.strip()

print("=" * 70)
print("PER-TRIAL CONSISTENCY GATE (plan :262)")
print("Interaction direction must be consistent across all 4 trials")
print("=" * 70)
print(f"{'Trial':<12} {'n':>4} {'coef':>10} {'SE':>8} {'p':>8} {'dir':>6} {'sig':>5}")
print("-" * 60)

coefs = []
for trial in sorted(df['Trial'].unique()):
    sub = df[df['Trial'] == trial].copy()
    n = len(sub)
    
    # Check we have all 4 groups in this trial
    groups_present = sub['Fish Type'].nunique()
    if groups_present < 4:
        print(f"{trial:<12} {n:>4}  -- only {groups_present} groups, skipping")
        continue
    
    try:
        m = smf.ols(
            'Total_distance_sqrt ~ C(Genotype) * C(Treatment)',
            data=sub
        ).fit(cov_type='HC3')
        
        # Find interaction term
        interaction_term = None
        for term in m.params.index:
            if 'Genotype' in term and 'Treatment' in term:
                interaction_term = term
                break
        
        if interaction_term:
            coef = m.params[interaction_term]
            se = m.bse[interaction_term]
            p = m.pvalues[interaction_term]
            direction = '+' if coef > 0 else '-'
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"{trial:<12} {n:>4} {coef:>+10.4f} {se:>8.4f} {p:>8.4f} {direction:>6} {sig:>5}")
            coefs.append((trial, coef, p))
        else:
            print(f"{trial:<12} {n:>4}  interaction term not estimable")
    except Exception as e:
        print(f"{trial:<12} {n:>4}  error: {str(e)[:40]}")

# Summary
print("-" * 60)
if coefs:
    directions = ['+' if c > 0 else '-' for _, c, _ in coefs]
    consistent = len(set(directions)) == 1
    print(f"\nDirection consistency: {'PASS — all same direction' if consistent else 'CHECK — direction flips!'}")
    print(f"  Directions: {dict(zip([t for t, _, _ in coefs], directions))}")
    
    n_sig = sum(1 for _, _, p in coefs if p < 0.05)
    print(f"  Significant (p<0.05): {n_sig}/{len(coefs)} trials")
    
    if not consistent:
        print(f"\n  ⚠ One or more trials flip direction.")
        print(f"    This may indicate a device-L artifact or contamination.")
        print(f"    Report this honestly — do not hide (plan :262).")
