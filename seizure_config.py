"""
Seizure detection parameters — Phase 1 (per-second state machine).
Single source of truth; edit here, detector reads from here.
Mirrors dataset_config.py pattern (env-var driven datasets live there).
"""
# Velocity thresholds (cm/s)
REST_FLOOR = 0.6        # v < this   -> rest
BURST_GATE = 2.8        # v > this   -> burst (seizure-swim)
FREEZE_FLOOR = 0.4      # v < this   -> catatonic / post-ictal freeze

# Temporal rules (seconds)
MIN_BURST_DURATION = 1     # min contiguous burst seconds for 1 event
FREEZE_WINDOW = 10          # s after burst end to scan for catatonic tail
FREEZE_DURATION_MIN = 1    # min consecutive freeze s to tag tail
FREEZE_DURATION_MAX = 3    # max consecutive freeze s to tag tail

# Missing data
NULL_HANDLING = 'ignore'   # null velocity -> gap (breaks burst contiguity)

# Column names
VEL_COL = 'Velocity'
TIME_COL = 'Time_seconds'
TRIAL_COL = 'Etho_Trial'
FISH_COL = 'Fish'
PHASE_COL = 'phase'
EVENT_ID_COL = 'event_id'
