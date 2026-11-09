"""
Master analysis runner — runs all Step 6 / Step 11 scripts in order.
Each script selects its dataset via the DS env var (see dataset_config.py):
  DS=10mM_full    -> PerFish_10mM_combined.csv       (figures/)
  DS=10mM_20min   -> PerFish_10mM_20min_combined.csv  (figures_20min/)
  DS=5mM          -> PerFish_5mM_combined.csv         (figures_5mM/)
Default DS=10mM_full. Burst/kinetics/raster/KM scripts self-skip for 5mM (no per-second data).

Run (set DS before the command; the env is inherited by subprocesses):
  DS=10mM_full  python run_all.py
  DS=10mM_20min python run_all.py
  DS=5mM        python run_all.py

Each script runs in isolation (subprocess) — no namespace collisions.
Comment out a line in STEPS to skip that step.
"""
import subprocess
import sys
import time

STEPS = [
    ('6b: Assumption checks (raw)', 'step6b_assumptions.py'),
    ('6b: Assumption checks (sqrt recheck)', 'step6b_recheck.py'),
    ("6a': Primary analysis (Total_distance)", 'step6a_primary.py'),
    ("6a': Burst-specific metrics", 'step6a_burst_metrics.py'),
    ("6a': Per-trial consistency gate", 'step6a_trial_gate.py'),
    ('6c: Kaplan-Meier latency', 'step6c_km.py'),
    ('6a": Time-course kinetics', 'step6a2_kinetics.py'),
    ('11: Poster figures', 'step11_figures.py'),
    ('11: Burst raster (top burst fish)', 'step11_raster.py'),
    ('11: Burst difference (event count)', 'step11_burst_bc.py'),
]

py = sys.executable
t0 = time.time()

for label, script in STEPS:
    print("\n" + "#" * 70)
    print(f"# {label}")
    print(f"# > python {script}")
    print("#" * 70)
    t1 = time.time()
    result = subprocess.run([py, script], capture_output=False)
    dt = time.time() - t1
    status = 'OK' if result.returncode == 0 else f'FAILED (exit {result.returncode})'
    print(f"\n[{status}] {label} ({dt:.1f}s)")
    if result.returncode != 0:
        print("Stopping — fix the error above before continuing.")
        sys.exit(1)

print("\n" + "=" * 70)
print(f"ALL STEPS COMPLETE ({time.time() - t0:.1f}s total)")
print("=" * 70)
