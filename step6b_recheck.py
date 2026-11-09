import matplotlib
matplotlib.use('Agg')
import sys
sys.path.insert(0, '../scientific-agent-skills/skills/statistical-analysis/scripts')
import pandas as pd
import numpy as np
from scipy import stats
from assumption_checks import comprehensive_assumption_check
from dataset_config import CSV, LABEL

df = pd.read_csv(CSV)
print(f"[dataset: {LABEL}]  input: {CSV}")
df['Total_distance_sqrt'] = np.sqrt(df['Total_distance_cm'])

print("=" * 60)
print("RECHECK: sqrt(Total_distance_cm)")
print("=" * 60)

results = comprehensive_assumption_check(
    data=df,
    value_col='Total_distance_sqrt',
    group_col='Fish Type',
    alpha=0.05
)
