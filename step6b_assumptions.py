import matplotlib
matplotlib.use('Agg')  # headless: plt.show() becomes no-op, numeric output still prints
import sys
sys.path.insert(0, '../scientific-agent-skills/skills/statistical-analysis/scripts/')
import pandas as pd
from assumption_checks import comprehensive_assumption_check
from dataset_config import CSV, LABEL

df = pd.read_csv(CSV)
print(f"[dataset: {LABEL}]  input: {CSV}")
results = comprehensive_assumption_check(
    data=df, value_col='Total_distance_cm', group_col='Fish Type', alpha=0.05)
