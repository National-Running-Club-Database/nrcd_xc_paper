# National Running Club Database (NRCD) — Cross Country Analysis

This repository contains the empirical analysis code for a study of collegiate
**club cross country** built on the National Running Club Database (NRCD). It
predicts within-season improvement, examines multi-season progression, and
relates team race frequency to national-championship placement.

The **dataset and the performance-standardization method** are published
separately via the public NRCD export and the `nrcd` Python package; this
repository consumes them rather than re-deriving them. A plain-language summary
of results is in [`findings.md`](findings.md).

## Analysis Sample

We analyze the **comprehensive (current) era** of the NRCD Cross Country subset
(meet `start_date` on or after 2023-08-01), covering the 2023–2025 seasons:

- **23,355** race results, **7,083** athletes, **280** meets
- **~97.7%** course-details coverage and **~97.7%** weather coverage
  (vs. only ~4% weather coverage in the historical 2004–Jul 2023 era)

The historical era is excluded because credible environmental standardization
requires the near-complete metadata that only the comprehensive era provides.
Primary analyses use **Standardized** times (distance + weather + elevation);
**Converted Only** (distance-only) times are reported as a sensitivity check.

## Dataset Setup

The public NRCD export is linked as a **git submodule** at `data_public/`:

```bash
git submodule update --init --recursive
```

`scripts/load_nrcd_data.py` resolves data in this order:

1. `$NRCD_DATA_DIR` — an explicit directory containing the CSV export.
2. `<repo>/data_public/` — the git submodule (public GitHub export).
3. Initializes the submodule if needed (`git submodule update --init`).
4. `<repo>/data/` — legacy fallback (e.g. a local symlink).

To use a pre-downloaded copy instead:

```bash
export NRCD_DATA_DIR=/path/to/nrcd/csv/export
```

The export provides `result.csv`, `meet.csv`, `sport.csv`, `athlete.csv`,
`running_event.csv`, and `course_details.csv`.

## Setup

### Virtual environment (recommended)

This project and the `nrcd` package require **Python ≥ 3.10** (declared in
`pyproject.toml` as `requires-python = ">=3.10"`; developed on 3.12):

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
```

### Dependencies

```bash
pip install -r requirements.txt
# or: pip install .
```

This installs pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib,
seaborn, plotly, and **`nrcd[data]>=0.1.5`** from PyPI. Standardization uses
`nrcd.standardize.standardize_dataframe` over parallel process chunks (many race
rows at once). Converted Only and Standardized tiers share one prepared frame
and are memoized under `output/.cache/` (set `NRCD_XC_DISABLE_CACHE=1` to skip).

## Performance Standardization

Raw finishing times are not comparable across courses, so every performance is
standardized with the `nrcd` package (implementing the NRCD framework):

- **Converted Only** — corrects measured course length to 8,000 m (men) /
  6,000 m (women) via Riegel scaling.
- **Standardized** — additionally adjusts for temperature–dew point and course
  grade (elevation).

Because autumn weather typically cools over a season, Converted Only can absorb
weather-driven gains into apparent fitness improvement; **Standardized is the
primary metric** throughout, with Converted Only used for sensitivity.
Both tiers are produced via nrcd's batch DataFrame API (`standardize_dataframe`)
after a single course-details join. Key performance models are also **trained
separately by gender**; pooled fits are diagnostics only.

## Statistical Methods

- **Temporal validation** (primary): train on 2023, test on 2024, preventing
  leakage across seasons; secondary splits train 2023 → test 2025 and train
  2023+2024 → test 2025.
- **Athlete-season units and endpoint exclusion**: each row is one athlete in
  one season; all time-derived features use pre-final races only, and
  `last_time` is never a model feature.
- **5-fold cross-validation** on the training years for model stability.
- **Bootstrap confidence intervals** (2,000 resamples) for held-out R².
- **Fixed reproducibility controls**: seed 42 for bootstrap resampling,
  Random Forest, Gradient Boosting, learning-curve subsampling, and team
  permutations; SVR is deterministic for fixed inputs. Runs log these settings
  and write machine-readable reproducibility manifests.
- **Multiple-comparison control** via Bonferroni correction within families of
  related tests (e.g. the six year × gender nationals tests use α = 0.05/6 =
  0.0083); a multiplicity-accounting table enumerates every test family.
- **Effect sizes** reported alongside tests (Cohen's d for contrasts; risk
  ratios and odds ratios for the team association).
- **Noise ceiling / reliability**: split-half (Spearman–Brown) reliability of
  the improvement outcome and race-time ICC, bounding the achievable R².
- **Permutation null for held-out R²** and a **top-quartile-improver
  classification** (AUC) check complement the regression null result.
- **Team association robustness**: threshold sensitivity (3+/4+/5+),
  DerSimonian–Laird random-effects pooling of risk ratios, and a cluster-robust
  (GEE) / random-intercept model on continuous team race count.
- **Robustness**: feature ablation, temporal learning curves, four alternative
  outcome-trimming rules, year-stratified team-correlation permutation tests,
  and mixed-effects (athlete random-intercept) explanatory models, including a
  gender × race-count interaction model.
- Associations are interpreted as **associational, not causal**.

## Running the Analysis

Run all scripts **from the repository root** (not from inside `scripts/`).

### Run everything

```bash
python scripts/run_all.py       # sample summary → RQ1 → RQ2 → RQ3
```

### Run by research question

```bash
python scripts/rq1.py           # within-season improvement + team/nationals analyses
python scripts/rq2.py           # multi-season progression (race-count consistency filter)
python scripts/rq3.py           # gender participation and racing-frequency differences
```

Outputs are written to `output/rq1/`, `output/rq2/`, and `output/rq3/`.

### Sample / coverage summary

```bash
python scripts/sample_summary.py   # -> output/sample_summary{,_by_year_gender}.csv, .md
```

### Selected individual scripts

```bash
# RQ1 building blocks
python scripts/first_to_last_improvement.py
python scripts/numberOfRacesQuestion.py
python scripts/numberOfRacesBrokenDown.py
python scripts/ml_improvement_prediction.py        # main ML model (temporal validation)
python scripts/nationals_overlap_analysis.py       # top-15 teams vs 4+ race athletes
python scripts/top25_team_analysis.py              # top-25 team rank correlations
python scripts/feature_ablation_robustness.py      # feature ablation
python scripts/sensitivity_analysis_sweep.py       # sensitivity sweep
python scripts/prediction_diagnostics.py           # learning curves + outlier sensitivity
python scripts/rq1_null_result_diagnostics.py      # noise ceiling, permutation-R2, classification AUC
python scripts/feature_exclusion_audit.py          # leakage vs over-exclusion; compact vs legacy
python scripts/compact_model_suite.py              # six-model + learning curves + outlier + perm on compact
# Feature policy (single source of truth): scripts/feature_policy.py
python scripts/cikm_r2_discrepancy_audit.py         # resource-paper vs leakage-controlled R2 audit
python scripts/mixed_effects_explanatory_model.py  # mixed-effects model
python scripts/mixed_effects_interaction.py        # gender x race-count interaction model
python scripts/rq2_temporal_splits.py              # multi-season R2 across all temporal splits
python scripts/team_association_robustness.py      # thresholds, cluster-robust, partial pooling
python scripts/underexplored_mechanisms.py         # roster depth, peer density, quantile CATE
python scripts/mathematical_contributions.py       # weather-path identity, SER, ERO index
python scripts/paper_enrichment_analyses.py        # dose-response, weather inflation, retention

# Overlay figures (parametrized)
python scripts/create_combined_overlay_2023_2024_2025.py   # men + women, 3x2 grid
python scripts/create_combined_2023_2024_2025_all_plots_grid.py
python scripts/create_rq1_overlay.py 2023 2024 2025        # per-year overlays

# RQ2 / RQ3
python scripts/multi_season_analysis_rq2.py
python scripts/gender_race_participation_test.py
```

> **Note on overlay scripts.** Overlays write under `output/rq1/overlay_plots/`.
> `rq1.py` calls the gender-parametrized combined overlay for both sexes and the
> all-plots grid. Per-year overlays remain available via `create_rq1_overlay.py`.

## Main ML Model

`ml_improvement_prediction.py` evaluates each athlete-season's **improvement
rate** (seconds/day; negative = faster). Time-derived predictors use only races
before the final outcome race; for a two-race season, `slope` is zero rather
than the first-to-final difference. Absolute performance summaries are retained
only after Standardization, so they represent baseline ability without
reintroducing course/weather bias. The script evaluates six models separately
by gender (OLS, Ridge, Lasso, Random Forest, Gradient Boosting, SVR):

- `raw_data_athlete_features.csv` — engineered athlete-season features
- `raw_data_model_performance.csv` — R², RMSE, MAE with bootstrap 95% CIs
- `raw_data_gender_model_performance.csv` — primary separately trained models
- `raw_data_feature_importance.csv` / `raw_data_gender_feature_importance_*.csv`
- `raw_data_subgroup_analysis.csv` — performance by year and gender
- `raw_data_important_statistics.csv` — summary statistics
- prediction/diagnostic figures (PDF)
- `reproducibility_manifest.json` — seeds, bootstrap count, and split policy

## Output Organization

- **`output/rq1/`** — within-season improvement: ML model results
  (`raw_data_*.csv/pdf`), first-to-last improvement, race-count analyses, team
  participation, `top25_teams/`, `nationals_overlap/`, `weekly_participation/`,
  `overlay_plots/`, `robustness_feature_ablation/`, `sensitivity_sweep/`,
  `prediction_diagnostics/`, `feature_exclusion_audit/`, `null_result_diagnostics/`,
  `mathematical_contributions/`, `mixed_effects/`, and the state results map.
- **`output/rq2/`** — multi-season progression (race-count consistency filter),
  including `temporal_splits/` (all three temporal splits with bootstrap CIs):
  pooled ML performance, feature importance, best-time trajectory summaries.
- **`output/rq3/`** — gender participation and race-count distributions, plus
  `team_association_robustness/` (threshold sensitivity, pooled and cluster-robust
  team race-frequency / nationals models).
- **`output/sample_summary*`** — sample size and metadata coverage tables.

## Key Findings

See [`findings.md`](findings.md) for the full write-up. Headline results
(Standardized times, temporal validation train 2023 → test 2024):

0. **Headline negative result.** Under strict leakage control, a comprehensive
   race-result feature set does **not** support out-of-year forecasting of
   individual improvement (held-out R² ≈ 0 or negative across six models and
   both sexes). The improvement outcome's own split-half reliability is only
   ~0.23–0.28, so this near-zero R² reflects an intrinsic noise ceiling, not
   just failed models; a permutation test still finds the observed R² above a
   shuffled-label null (p ≈ 0.001).

1. **RQ1 — weak individual prediction; weather inflation is the robust
   descriptive finding.** Separately trained models remain weak (men SVR
   R² = 0.043; women −0.029), near a split-half reliability ceiling of
   ~0.23–0.28. Converted Only overstates mean first→last gains by **21 s / 15 s**
   vs Standardized. Race-count cell means (e.g. men 14→47 s at 2→3 races) are
   exploratory/uncorrected — only the men's 4-vs-2 contrast survives
   multiple-comparison correction — and are not treated as a primary claim; the
   mixed-effects quadratic term is an associational shape reconciliation.
   Learning curves flatten near zero by 80–100% of training data.

2. **Team depth and peers, not just max races (cross-sectional).** Top-15
   association survives pooling/GEE (RR 2.09; OR 2.56/SD). Roster depth remains
   significant conditional on max races (OR 1.74/SD). **Within-team** year-to-year
   Δvolume/Δdepth do not predict Δtop15, and roster-size adjustment cuts the
   max-race OR to ~1.69 — treat the team finding as between-program selection,
   not a schedule experiment. Peer co-start density still predicts faster
   improvement/retention after controlling for own race count.

3. **RQ2 — retention rises with prior race load; multi-season prediction is not
   robust.** Next-season retention is 45%→72% (men) and 41%→61% (women) from 2
   to 4 prior races. The primary 2023→2024 split is non-positive for all models
   and both sexes; some men's linear models turn positive predicting 2025
   (Lasso R² up to 0.68) but this does not replicate for women or SVR and is
   confounded by heavy train/test athlete overlap (the consistency filter reuses
   the same athletes across years). 59.2% of stable participants improved
   2023→2025.

4. **RQ3 — imbalance is at entry, not intensity.** Men outnumber women each
   year (~60–63% of athletes; χ² p ≪ 0.001), but mean races per athlete are
   nearly identical (men 2.00–2.07, women 1.92–1.97). ~40–46% of athletes race
   once and only ~9–12.5% race four or more times, with modest gender gaps.

All results are associative: athletes who race more may differ systematically in
health, availability, and program support.

## Repository Layout

- **`scripts/`** — analysis and figure code (`run_all.py` → sample summary +
  `rq1.py`/`rq2.py`/`rq3.py`; `ml_improvement_prediction.py` is the main model;
  `feature_policy.py` is the compact/legacy feature source of truth;
  `load_nrcd_data.py` + `utils.py` handle data loading and standardization).
- **`output/`** — generated results by research question (see above).
- **`data_public/`** — git submodule with the public NRCD CSV export.
- **`pyproject.toml`** — declares `requires-python = ">=3.10"` and dependencies.

## Reproducibility

A claim → script → CSV checklist is in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
Plain-language results: [`findings.md`](findings.md).

- Initialize the dataset submodule (`git submodule update --init --recursive`)
  or set `$NRCD_DATA_DIR`.
- All models use temporal splits; features are computed without future-season
  information.
- Bonferroni corrections and bootstrap CIs are applied throughout.
- All results are saved to CSV/PDF with consistent naming.
- After regenerating figures: `bash papers/sync_figures.sh`.
