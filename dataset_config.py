"""
Shared dataset config — imported by all Step 6 / Step 11 scripts.
Selects input CSV, figure dir, and per-second time cutoff via the DS env var.

  DS=10mM_full   -> PerFish_10mM_combined.csv       (figures/)
  DS=10mM_20min  -> PerFish_10mM_20min_combined.csv  (figures_20min/)
  DS=5mM         -> PerFish_5mM_combined.csv         (figures_5mM/)

Default: 10mM_full (backward-compatible with the original PerFish_Primary.csv runs).
For 5mM there is no per-second data, so kinetics/raster skip themselves.
"""
import os

DS = os.environ.get('DS', '10mM_full')

CONFIG = {
    '10mM_full':  {'csv': '../PerFish_10mM_combined.csv',       'figdir': '../figures',        't_cutoff': 3600, 'label': '10mM (60 min)'},
    '10mM_20min': {'csv': '../PerFish_10mM_20min_combined.csv', 'figdir': '../figures_20min', 't_cutoff': 1200, 'label': '10mM (first 20 min)'},
    '5mM':        {'csv': '../PerFish_5mM_combined.csv',        'figdir': '../figures_5mM',   't_cutoff': None, 'label': '5mM (60 min)'},
}[DS]

CSV = CONFIG['csv']
FIGDIR = CONFIG['figdir']
T_CUTOFF = CONFIG['t_cutoff']  # seconds; None means no per-second data (5mM)
LABEL = CONFIG['label']
