# IQSEC2 × PTZ zebrafish seizure analysis

Analysis code for a 2×2 larval-zebrafish study of pentylenetetrazole (PTZ)-induced
seizures in an IQSEC2 mutant line versus wild type. Fish were tracked per second with
EthoVision XT for 60 min; PTZ was added at t = 0 (no acclimation).

The code has two pipelines that share the same raw data:

- **Burst pipeline** — an "event" is any contiguous run of seconds above 2.8 cm/s.
  Produces the statistics and figures for the poster/paper.
- **Seizure state machine** — labels every fish-second as `rest` / `hyper` / `burst` /
  `catatonic` / `null`, gates bursts at ≥ 1 s, caps them at 10 s, and tags a 1–3 s
  post-ictal freeze tail. Answers "how many discrete convulsive episodes, and does each
  have a post-ictal freeze?" — numbers are **not** interchangeable with the burst pipeline.

---

## Design

| | |
|---|---|
| Genotypes | wild type (WT), IQSEC2 mutant |
| Treatments | control, PTZ (10 mM primary; 5 mM sub-maximal) |
| Groups | `wt_control`, `wt_ptz10`, `mut_control`, `mut_ptz10` (and `wt_ptz5` / `mut_ptz5` for 5 mM) |
| N | 191 fish (burst pipeline); 190 in the seizure pipeline, where A5 is also excluded |
| Age | 7 dpf |
| Recording | 60 min (20-min window for the feasibility dataset); PTZ at t = 0, no acclimation |
| Trials | 4 (`Trial 26`, `27`, `28`, `29`), 48 wells each, two EthoVision rigs (R: 26–27, L: 28–29) |
| Tracking | EthoVision XT: per-second velocity (cm/s), distance, movement, min/max acceleration, mobility |
| Excluded | `Trial 28/C8` (phantom — no fish in the well), `Trial 29/A5` (phantom, seizure pipeline only) |

Group sizes, 10 mM: 44 `wt_control`, 40 `wt_ptz10`, 51 `mut_control`, 56 `mut_ptz10`.
5 mM: 48 `wt_control`, 48 `wt_ptz5`, 48 `mut_control`, 47 `mut_ptz5`.

Three datasets are selectable at run time:

| `DS` | Window | Used for |
|---|---|---|
| `10mM_full` (default) | 60 min | primary analysis |
| `10mM_20min` | first 20 min | temporal dissection / feasibility |
| `5mM` | 60 min | sub-maximal dose control |

---

## Repository layout

Only this `code/` directory is in the repository. The data and all outputs live **one
level up**, because every script uses `../` paths (`../AllData.csv`, `../figures`).
Clone/copy the repo as `code/` inside a project directory:

```
<project>/
├── AllData.csv                        input   per-second tracking, 10 mM
├── PerFish_10mM_full_minutes.csv      input   instructor per-fish 60-min aggregates
├── PerFish_10mM_20min.csv             input   instructor per-fish 20-min aggregates
├── PerFish_5mM_full_minutes.csv       input   instructor per-fish 5 mM aggregates
├── PerFish_Primary.csv                input   burst metrics at 2.8 cm/s
├── PerFish_10mM_combined.csv          output  ┐
├── PerFish_10mM_20min_combined.csv    output  ├ merge_perfish.py
├── PerFish_5mM_combined.csv           output  ┘
├── seizure_*.csv                      output  steps 7–9
├── figures/  figures_20min/  figures_5mM/   output  step 6/10/11
└── code/                              this repo
```

### Scripts

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `dataset_config.py` | maps `DS` → input CSV, figure dir, time cutoff | env `DS` | — |
| `seizure_config.py` | all detector thresholds in one place | — | — |
| `merge_perfish.py` | Phase 0: joins instructor per-fish exports with burst metrics | 4 instructor CSVs, `AllData.csv`, `PerFish_Primary.csv` | `PerFish_*_combined.csv` |
| `run_all.py` | burst pipeline runner (9 scripts, subprocess) | — | — |
| `step6b_assumptions.py` | Shapiro-Wilk per group + Levene, raw and √ | `CSV` | stdout |
| `step6a_primary.py` | headline: genotype × treatment on total distance (HC3 OLS) + η²p + pairwise MW | `CSV` | stdout |
| `step6a_burst_metrics.py` | same model for event count, time above threshold, max velocity | `CSV` | stdout |
| `step6a_trial_gate.py` | interaction direction per trial (consistency gate) | `CSV` | stdout |
| `step6c_km.py` | Kaplan-Meier + log-rank, stratified by trial | `CSV` | `km_latency.png/.pdf` |
| `step6a2_kinetics.py` | group × 5-min-bin mixed ANOVA (Greenhouse-Geisser) | `AllData.csv`, `CSV` | `timecourse_velocity.png/.pdf` |
| `step11_figures.py` | poster figures F3–F10 + summary table | `CSV` | `fig3…fig10`, `summary_table.csv` |
| `step11_raster.py` | burst timelines for the top-5 fish per PTZ group | `AllData.csv`, `CSV` | `burst_raster`, `burst_population`, `burst_timecourse` |
| `step11_burst_bc.py` | (mut − WT) event count per 5-min bin | `AllData.csv`, `CSV` | `burst_difference.png/.pdf` |
| `step7_seizure_detector.py` | per-second phase state machine | `AllData.csv` | `seizure_seconds.csv`, `seizure_events.csv` |
| `step7b_iqr_filter.py` | flat 10 s burst-duration cap | `seizure_events.csv`, `AllData.csv` | `seizure_events_iqr_filtered.csv`, `…_outliers.csv` |
| `step8_fish_rollup.py` | one row per fish, phase + event metrics | step 7/7b outputs | `seizure_fish_rollup.csv` |
| `step9_fish_stats.py` | MWU + Holm + Cliff's δ, KM/log-rank, Wilcoxon/Friedman, K-S | rollup, filtered events | `seizure_fish_stats.csv` |
| `step10_figures.py` | 5 seizure figures | rollup, filtered events | `seizure_*.png/.pdf` |

---

## Installation

Python 3.12+ (developed on 3.14). All inputs are CSV — no Excel reader is needed.

```bash
cd code
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

---

## Running

All commands are run **from `code/`**. `DS` is an environment variable, so its syntax
differs by shell.

### Phase 0 — rebuild the combined per-fish CSVs (only if the instructor exports change)

```bash
python merge_perfish.py
```

This produces `PerFish_10mM_combined.csv`, `PerFish_10mM_20min_combined.csv`, and
`PerFish_5mM_combined.csv` in the project root. It does not use `DS` and always does all three.

### Pipeline A — burst analysis and figures

PowerShell:

```powershell
$env:DS="10mM_full";  python run_all.py
$env:DS="10mM_20min"; python run_all.py
$env:DS="5mM";        python run_all.py
```

bash:

```bash
DS=10mM_full  python run_all.py
DS=10mM_20min python run_all.py
DS=5mM        python run_all.py
```

`run_all.py` stops on the first failure. Comment out a line in `STEPS` to skip a step.
It prints all statistics to stdout and writes figures to the dataset's figure directory.

### Pipeline B — seizure state machine (manual, in this order)

```bash
python step7_seizure_detector.py
python step7b_iqr_filter.py
python step8_fish_rollup.py
python step9_fish_stats.py
python step10_figures.py
```

`DS` applies here too, but this pipeline only makes sense for `10mM_full` or
`10mM_20min`. **Do not run it with `DS=5mM`**: there is no 5 mM per-second data, and
the scripts would silently reuse the 10 mM `AllData.csv`.

### What runs per dataset

| Step | 10 mM 60-min | 10 mM 20-min | 5 mM |
|---|---|---|---|
| Assumption checks, primary, burst metrics, trial gate, figures | ✓ | ✓ | ✓ (burst metrics skipped) |
| KM latency, kinetics, raster, burst difference | ✓ | ✓ | skipped — no per-second data |
| Seizure pipeline (steps 7–10) | ✓ | ✓ (20-min window) | not applicable |

Scripts that need per-second data check `T_CUTOFF is None` and exit cleanly with a
message instead of failing.

---

## Configuration

`dataset_config.py`:

| `DS` | Input CSV | Figure dir | `T_CUTOFF` |
|---|---|---|---|
| `10mM_full` | `PerFish_10mM_combined.csv` | `figures/` | 3600 s |
| `10mM_20min` | `PerFish_10mM_20min_combined.csv` | `figures_20min/` | 1200 s |
| `5mM` | `PerFish_5mM_combined.csv` | `figures_5mM/` | `None` |

`seizure_config.py`:

| Constant | Value | Role |
|---|---|---|
| `REST_FLOOR` | 0.6 cm/s | rest / hyper boundary |
| `BURST_GATE` | 2.8 cm/s | seizure-swim gate |
| `FREEZE_FLOOR` | 0.4 cm/s | catatonic / post-ictal freeze |
| `MIN_BURST_DURATION` | 1 s | minimum contiguous burst to count as a seizure |
| `FREEZE_WINDOW` | 10 s | how far after burst end to scan for a tail |
| `FREEZE_DURATION_MIN/MAX` | 1 / 3 s | valid catatonic tail length |

`MAX_BURST_DURATION = 10.0 s` (the phantom guard) is defined in `step7b_iqr_filter.py`,
not in the config file.

---

## Inputs

| File | Contents |
|---|---|
| `AllData.csv` | Per-second tracking for the 10 mM experiment: 691,296 rows × 192 fish × 4 trials. Columns: `Etho_Trial`, `Fish` (well), `Fish Type` (group), `Time_seconds`, `Distance`, `Velocity`, `Movement`, `Min_Accel`, `Max_Accel`, `Mobility`, `Untracked`. |
| `PerFish_10mM_full_minutes.csv` | Instructor's per-fish 60-min aggregates (mean distance/velocity/movement/acceleration/mobility, z-scores, outlier flags). |
| `PerFish_10mM_20min.csv` | Same, first 20 min. |
| `PerFish_5mM_full_minutes.csv` | Same, 5 mM, 60 min. |
| `PerFish_Primary.csv` | Per-fish burst metrics already computed at 2.8 cm/s (`Event_count_ThrA`, `Time_above_ThrA`, `Latency_first_event_ThrA`, `Max_velocity`). |

The original per-second workbook (`analysis_with_seconds.xlsx`, 4 sheets) is the source
of `AllData.csv` but is not read by any script.

## Outputs

| File | Contents |
|---|---|
| `PerFish_*_combined.csv` | One row per fish: instructor aggregates + burst metrics + group, in the schema the step-6 scripts expect. |
| `seizure_seconds.csv` | One row per fish-second with the `phase` label. Large (~80 MB). |
| `seizure_events.csv` | One row per detected seizure: start/end/duration, peak and mean velocity, catatonic-tail fields. |
| `seizure_events_iqr_filtered.csv` | Analysis-ready event list after the 10 s cap. |
| `seizure_events_iqr_outliers.csv` | Dropped bursts — audit trail. |
| `seizure_fish_rollup.csv` | One row per fish: phase seconds/percentages, seizure count, total seizing time, latency, burden. |
| `seizure_fish_stats.csv` | Every test run by `step9_fish_stats.py` (section, metric, groups, n, statistic, p, adjusted p, effect). |
| `figures*/` | PNG (300 dpi) + PDF (vector) pairs, plus `summary_table.csv` in each. |

Figures written to `figures/` (10 mM, 60 min):

| Figure | Content |
|---|---|
| `km_latency` | KM curves, latency to first burst, 4 groups |
| `timecourse_velocity` | Mean velocity per 5-min bin, 4 groups |
| `fig3_bar_total_distance` … `fig10_bar_distance_per_sec` | Bar/box/paired figures, mean ± SD with individual fish |
| `fig8_summary_table` + `summary_table.csv` | Per-metric summary: group means, p, η²p, direction |
| `burst_raster`, `burst_population`, `burst_timecourse`, `burst_difference` | Burst timelines, population view, per-bin difference |
| `seizure_count_by_group`, `seizure_latency_km`, `seizure_count_quarters`, `seizure_timecourse`, `seizure_burden_by_group` | State-machine outputs |

`figures_20min/` holds the same burst figures for the 20-min window. `figures_5mM/`
holds only the figures that do not need per-second data (`fig3`, `fig6`–`fig10`).

---

## Methods summary

- **Burst gate 2.8 cm/s** — literature-cited seizure-swim threshold (Whyte-Fagundes 2025 /
  Baraban-lab convention). A 2.0 cm/s sensitivity analysis (Lee 2022) is *not* implemented.
  A data-driven cross-check (P99.9 of control velocity = 2.40 cm/s) is computed but used
  only as a sanity check, never for inference.
- **Assumption checks** — Shapiro-Wilk per group + Levene, raw and √-transformed.
  Normality fails on raw data; √ (Bandara 2020) improves it but only `wt_ptz10` passes
  (p = 0.097), and Levene still fails at 10 mM (PTZ groups have several-fold the variance
  of controls). Hence the robust-SE path.
- **Primary inference** — HC3 heteroscedasticity-robust OLS on √(total distance) with
  genotype × treatment plus `C(Trial)` as a blocking covariate. The **interaction term
  is the headline**; its Wald test is the reported p. η²p comes from an unblocked Type-III
  two-way ANOVA on the same transformed data (magnitude vs inference split).
- **Pairwise** — Mann-Whitney U, Bonferroni-corrected. For the zero-inflated count
  metrics (`Event_count_ThrA`, `Time_above_ThrA`) MW is the *primary* test and OLS is
  supplementary: √(0) = 0 cannot stabilise variance when controls are mostly zeros.
- **Latency** — Kaplan-Meier; fish with no event are right-censored at the recording end.
  Log-rank is stratified by trial to absorb the device-L/R baseline shift.
- **Kinetics** — group × 5-min-bin mixed ANOVA on per-fish per-bin mean velocity,
  Greenhouse-Geisser corrected (time bins are autocorrelated). Per-bin MW post-hoc,
  Bonferroni over the bins of the window.
- **Per-trial gate** — the interaction is re-estimated inside each trial; direction must
  agree. 2/4 trials are significant, both in the expected direction; the two
  non-significant trials sit near zero (underpowered, not reversed).
- **Error bars** — SD on bar figures (individual fish overlaid), SEM on time-course ribbons.
- **Seizure pipeline** — MWU with Holm correction and Cliff's δ; hand-rolled KM/log-rank
  (no `lifelines` dependency); Wilcoxon signed-rank for first vs second half; Friedman
  across quarters; K-S on per-burst durations, descriptive only (bursts within a fish are
  autocorrelated).

---

## Key findings

10 mM PTZ, 60 min, 191 fish:

| Comparison | Test | Result |
|---|---|---|
| Genotype × treatment, total distance | HC3 OLS, trial blocked | **p = 0.007**, η²p = 0.035; mutant +32% vs WT under PTZ (89.8 vs 67.9 cm) |
| Event count, `mut_ptz10` vs `wt_ptz10` | Mann-Whitney, Bonferroni | **p = 0.004**; +40% (31.8 vs 22.8) |
| Time above 2.8 cm/s | Mann-Whitney, Bonferroni | **p = 0.005**; +66% (57.1 vs 34.3 s) |
| Peak velocity | Mann-Whitney, Bonferroni | ns (p = 0.10); +10% — same peak intensity |
| Latency to first event | log-rank, stratified | 4 groups **p < 0.001**; `wt_ptz10` vs `mut_ptz10` p = 0.013 — mutant seizes earlier |
| Group × time (5-min bins) | mixed ANOVA, GG | **p = 3.9e-8** — divergence at 45–55 min |
| 5 mM genotype × treatment | HC3 OLS | ns (p = 0.586); +6% — no genotype effect at sub-maximal dose |

The mutation changes **how often and how long** the fish seize, not how fast the
individual bursts are. Baseline (control) genotypes do not differ. At 5 mM PTZ both
genotypes activate, but the genotype-specific effect disappears.

These are the numbers from the final 191-fish run; re-running `run_all.py` regenerates
them and writes the summary to `figures*/summary_table.csv`.

---

## Gotchas and limitations

1. **Run everything from `code/`.** All paths are relative (`../AllData.csv`, `../figures`).
2. **`code/` must sit one level below the data root.** Cloned standalone, nothing resolves.
3. **`run_all.py` covers the burst pipeline only.** Steps 7–10 must be run manually, in order.
4. **`DS` is an environment variable.** PowerShell uses `$env:DS="…"`, bash uses `DS=…`.
5. **Never run steps 7–10 with `DS=5mM`.** There is no 5 mM per-second data; the scripts
   would silently analyse the 10 mM file.
6. **Excel locks output files.** If `seizure_events.csv` is open in Excel, the detector
   fails with `PermissionError`. Close it first.
7. **Phantom fish are hardcoded by trial/well** in `step7_seizure_detector.py` and
   `step6a2_kinetics.py`. If the raw export changes, re-check those exclusions.
8. **20-min total distance is derived:** `Mean_distance × 1200`. It correlates only 0.80
   with a per-second recomputation (window-boundary differences), so absolute totals are
   not comparable to the 60-min numbers.
9. **Rotation is not used.** Stage II (whirlpool) behaviour is therefore not detected;
   adding it is on the backlog.
10. **Catatonic-tail rate is 99%** (2,848 / 2,867 events) — biologically implausible and
    still unresolved. Any catatonic-derived conclusion is provisional.
11. **Median burst duration = 1 s**, i.e. exactly `MIN_BURST_DURATION`. Many "seizures"
    are single-second threshold crossings.
12. **No gap merge and no refractory period.** Two bursts separated by one sub-threshold
    second count as two events. This is deliberate.
13. **No electrophysiology** — behaviour only.
14. **Data files are not in the repo.** `AllData.csv` is ~78 MB, `seizure_seconds.csv`
    ~80 MB, and the source workbook ~229 MB. Keep them out of version control.

---

## References

Thresholds, staging and statistical choices follow these papers:

- Baraban et al. (2005) — PTZ-induced changes in zebrafish behaviour, neural activity and c-fos. *Neuroscience*
- Baraban (2021) — A zebrafish-centric approach to antiepileptic drug development. *Dis Models Mech*
- Whyte-Fagundes et al. (2025) — Automated detection of complex zebrafish seizure behaviour at scale. *Commun Biol*
- Afrikanova et al. (2013) — Validation of the zebrafish PTZ seizure model: locomotor vs electrographic responses. *PLoS ONE*
- Mussulini et al. (2013) — Seizures induced by PTZ in adult zebrafish: behavioural characterization. *PLoS ONE*
- Bandara et al. (2020) — Susceptibility of larval zebrafish to GABA-A receptor antagonists. *Neurotoxicology*
- Lee et al. (2022) — Seizurogenic effect of PFOS in zebrafish larvae. *Neurotoxicology*
- Milder et al. (2022) — Neural activity correlates with behavioural effects of anti-seizure drugs. *Front Pharmacol*
- Shaw et al. (2022) — Optimization of a locomotion-based zebrafish seizure model. *J Neurosci Methods*
- Szep et al. (2023) — Optimizing PTZ and pilocarpine-induced epilepsy conditions in zebrafish larvae. *PLoS ONE*
- Turrini et al. (2022) — Multimodal characterization of seizures in zebrafish larvae. *Biomedicines*
- Hamanaka et al. (2020) — De novo truncating SEMA6B variants cause progressive myoclonic epilepsy. *AJHG*
- Baxendale et al. (2012) — Identification of compounds with anti-convulsant properties. *Dis Models Mech*
- Bartoszek et al. (2023) — Optimizing the PTZ seizure model for caffeine/topiramate interactions. *Int J Mol Sci*
- Cintra et al. (2024) — Baicalin pre-treatment and PTZ seizures in zebrafish. *Epilepsy Behav*
- Bauer, Mally & Liedtke (2021) — Zebrafish embryos and larvae as alternative toxicity models. *Int J Mol Sci*
