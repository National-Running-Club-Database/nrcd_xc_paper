# Findings — Collegiate Club Cross Country (NRCD)

A plain-language summary of the empirical results in this repository. All
numbers use the **comprehensive-era** NRCD Cross Country sample and, unless
noted, **Standardized** times (distance + weather + elevation adjusted) with
**temporal validation** (train on 2023, test on 2024). Findings are
**associational, not causal**.

Source artifacts are listed under each section so every number can be traced to
a generated file.

**Headline (a negative result).** Under strict leakage control and temporal
validation, a compact race-result feature set (17 within-sex predictors after
excluding endpoint leakage and pruning near-duplicates; see
`scripts/feature_policy.py`) does **not** support out-of-year forecasting of
individual improvement: held-out R² is at or below zero almost everywhere
(best SVR: men 0.043, women −0.029). Crucially, the
improvement outcome is itself only weakly reliable — its split-half reliability
is ~0.23 (men) / ~0.28 (women) — so this near-zero R² reflects an **intrinsic
noise ceiling**, not merely six failed models. Adding `last_time` back raises
men’s R² by +0.44 (leakage confirmed); dropping `slope` does not create the null.

**Building on the prior CIKM resource paper (not contradicting it).** The NRCD
resource paper comes first and reports illustrative RF/GB R² ≈ 0.5–0.74 under
Converted vs Standardized labels as a *paired* standardization check — not a
forecasting benchmark. Absolute R² there is high by construction (features
include `last_time` while the label is `(last−first)/duration`). This JQAS
paper asks the follow-on forecasting question with leakage control and gets
R² ≈ 0. An audit in `output/rq1/cikm_r2_discrepancy_audit/` reproduces that
drop for reproducibility; the two papers are complementary. Set against this null, the data yield a robust,
moderately large **positive** finding at the team level: team race frequency is
associated with top-15 nationals placement — dose-monotone in the threshold and
stable under cluster-robust and partial-pooling reanalysis (pooled RR = 2.09;
cluster-robust OR = 2.56 per SD). **Roster depth** (athletes with ≥3 starts)
explains placement better than a single workhorse's max race count (depth OR =
1.74 per SD conditional on max races). Athletes who race with more teammates
improve faster and retain more after controlling for their own race count.
Quantile / BH-stratum analyses suggest any individual volume association
concentrates among mid-to-slow starters. We also quantify weather inflation and
next-season retention, reported separately by gender. **Mathematical add-ons**
(see `output/rq1/mathematical_contributions/FORMULAS.md`; defined in Methods of `papers/arxiv.tex`): (i) weather inflation
obeys the identity Δ_conv−Δ_std = a₁−a_L; (ii) Signal Extraction Ratio
SER = R²/ρ reframes the individual null; (iii) Effective Racing Opportunity
ERO = eᴴ n̄ beats max-race-count alone for out-of-year top-15 AUC (0.84 vs 0.74).
Race-count dose–response
cell means (the *individual*-level analysis, distinct from the team dose curve
above) are exploratory and mostly fail multiplicity correction, so they are not
treated as a primary claim.

---

## 1. Sample and coverage

- **23,355** race results, **7,083** athletes, **280** meets (2023–2025).
- Course-details coverage **~97.7%**; weather coverage **~97.7%**.
- By comparison, the excluded historical era (2004–Jul 2023) has only **~4.4%**
  weather coverage — too sparse for credible environmental standardization.
- Analyses that define within-season improvement exclude national-championship
  meets and require ≥ 2 regular-season races per athlete-season (leaving 19,742
  results).

> Sources: `output/sample_summary.md`, `output/sample_summary.csv`,
> `output/sample_summary_by_year_gender.csv`.

| Year | Gender | Results | Athletes | Meets |
|------|--------|---------|----------|-------|
| 2023 | Women | 2,604 | 1,155 | 61 |
| 2023 | Men | 4,592 | 1,929 | 73 |
| 2024 | Women | 2,870 | 1,236 | 90 |
| 2024 | Men | 4,960 | 2,124 | 101 |
| 2025 | Women | 3,178 | 1,381 | 96 |
| 2025 | Men | 5,151 | 2,097 | 104 |

---

## 2. RQ1 — Within-season improvement

**Question:** How much do club runners improve within a season after adjusting
for course and weather, and which early-season features predict end-of-season
improvement rate?

**Estimand:** improvement rate = (last − first standardized time) / days between
them (seconds/day; negative = getting faster).

**One primary metric.** First-to-last improvement (above) is the single primary
estimand. It drives the ML models (per-day rate) and the dose–response table
(same quantity in total seconds; they differ only by the days factor). Two other
definitions appear **only as robustness checks** of it: (i) the per-race linear
trajectory `slope` (less endpoint-sensitive, ignores the final race; Appendix
subgroup tables) and (ii) first-to-fastest improvement (bounded ≥ 0, so
mechanically larger; combined figure). All three agree qualitatively — slower
starters and, up to a point, more frequent racers improve more — but differ in
magnitude and can differ in sign for near-flat athletes because of endpoint
choice, not a substantive contradiction.

### Leakage-controlled feature policy

- Each row is one **athlete-season**, so future seasons cannot enter a prior
  season's predictors or target.
- Every time-derived predictor uses **pre-final Standardized performances**.
  `last_time` is an endpoint constituent and is never a model feature.
- For two-race seasons, trajectory `slope` is zero; using race 2 would exactly
  reproduce the target numerator.
- Absolute Standardized performance summaries remain valid baseline-ability
  covariates because course and weather effects have already been adjusted.

### Primary gender-separated performance (train 2023 → test 2024)

Compact feature set (17 within-sex predictors after leakage exclusions and
near-duplicate pruning; see `scripts/feature_policy.py`):

| Gender | Best model | Train n | Test n | Test R² | 95% CI | MAE |
|--------|------------|---------|--------|---------|--------|-----|
| Men | SVR | 1,146 | 1,162 | **0.043** | [−0.024, 0.102] | 3.75 |
| Women | SVR | 631 | 666 | **−0.029** | [−0.105, 0.035] | 4.24 |

Neither confidence interval excludes zero. Thus these schedule and pre-final
race-result features do **not** reliably forecast improvement out of year.
Random Forest R² is negative for both genders, so its feature importance is
retained only as an exploratory artifact, not evidence of predictive drivers.
The intervals use 2,000 fixed-seed bootstrap replicates. Bootstrap resampling,
RF, and GB use seed 42; SVR is deterministic. These settings are logged in
`output/rq1/feature_exclusion_audit/compact_suite_reproducibility.json`.

> Sources: `output/rq1/feature_exclusion_audit/compact_six_models.csv`,
> `compact_primary_svr_bootstrap.csv`. Legacy (pre-compact) tables remain under
> `raw_data_gender_model_performance.csv` for provenance only.

### Descriptive pattern / race-count dose–response (medians)

**Not a primary claim.** Prefer **medians** (slow-starter tails inflate means —
e.g. men’s 3-race mean 48 s vs median 18 s). Exploratory; only men’s 4-vs-2
pairwise contrast survives multiplicity among six gender×bin tests.

**A. Overall median improve (s)**

| Gender | 2 | 3 | 4 | 5+ |
|--------|---|---|---|-----|
| Men | 12.2 (n=1644) [6.9, 18.0] | 18.3 (1167) [12.3, 23.0] | 19.7 (589) [12.2, 27.4] | 7.7 (136) [−12.3, 24.9] |
| Women | 13.8 (1010) [8.1, 20.2] | 22.0 (632) [15.5, 28.7] | 29.7 (317) [19.8, 39.6] | 20.2 (72) [−8.3, 27.5] |

Kruskal–Wallis: men H = 10.7 (p = 0.014), women H = 8.2 (p = 0.042).
Cohen’s d vs 2 (means): men 0.18 / 0.07 / −0.01; women 0.11 / 0.12 / −0.04.

**B–C. By starting-time quartile (median s; Q1 = fastest)**

| Gender | Quartile | 2 | 3 | 4 | 5+ |
|--------|----------|---|---|---|-----|
| Men | Q1 | −22.3 (n=331) | −16.4 (300) | −3.7 (191) | −23.0 (63) |
| Men | Q2 | 6.2 (398) | −0.5 (282) | 16.1 (171) | 25.5 (33) |
| Men | Q3 | 27.5 (450) | 42.1 (282) | 51.1 (127) | 65.0 (24) |
| Men | Q4 | 55.4 (465) | 91.4 (303) | 103.7 (100) | 73.6 (16) |
| Women | Q1 | −10.1 (203) | 3.9 (172) | 10.9 (101) | −12.4 (33) |
| Women | Q2 | 11.0 (250) | 10.3 (151) | 20.6 (87) | 26.5 (19) |
| Women | Q3 | 15.3 (270) | 19.4 (150) | 53.2 (72) | 23.7 (14) |
| Women | Q4 | 64.2 (287) | 73.5 (159) | 101.8 (57) | 101.5* (6*) |

\* $n<10$ (women’s Q4 5+ only). Volume associations concentrate in Q3–Q4; Q1 often
non-positive.

> Sources: `output/rq1/enrichment/race_count_dose_response.csv`,
> `starting_ability_by_race_count.csv`,
> `race_count_dose_by_starting_quartile.pdf`.

### Ability-matched volume (4+ vs 2 races)

Nearest-neighbor match on first Standardized time within gender×year (caliper
0.25 SD; mean |Δfirst| ≈ 1 s) and IPW ATE adjusting for first time + year:

| Method | Gender | n | Mean Δ improve (s) | Note |
|--------|--------|---|--------------------|------|
| 1NN pooled | Men | 515 | **+27.6** | Wilcoxon p ≈ 2e-5; d = 0.18 |
| 1NN pooled | Women | 304 | **+24.8** | Wilcoxon p ≈ 1e-4; d = 0.19 |
| IPW ATE | Men | 2,369 | +33.0 | propensity on first time + year |
| IPW ATE | Women | 1,399 | +28.1 | propensity on first time + year |

After matching on starting ability, 4+ race seasons still improve ~25–28 s more
than 2-race seasons — a small-to-moderate associational contrast that survives
ability balancing but is **not** a causal schedule effect (who gets to race 4×
remains selected on health/availability).

> Source: `output/rq1/robustness_checks/ability_matched_summary.csv`.

### Why we do not use historical-era “experience”

Pre-2023 race counts look like a natural experience covariate, but the
historical export is **fragmentary**: **43.9%** of historical athlete–years have
only one recorded race, weather coverage on historical course-detail rows is
~18%, and only ~19% of comprehensive-era athletes even appear in both eras. A
recorded count of “2” can easily mean half (or more) of meets were never
entered. We therefore leave experience / grade year as Limitations rather than
encode a systematically undercounted proxy.

> Source: `output/rq1/robustness_checks/WHY_NOT_HISTORICAL_EXPERIENCE.md`.

### Weather inflation (why Standardized is primary)

Paired athlete–seasons: Converted Only overstates mean first→last improvement
by **21.0 s** for men (95% CI [20.2, 21.9]) and **15.1 s** for women
([14.2, 15.9]); Wilcoxon p ≪ 0.001. This is the quantitative case for
Standardized key statistics.

Leave-one-year-out re-fit of the environment residual
\(a=t_{\mathrm{conv}}-t_{\mathrm{std}}\) from temperature, dew point, elevation,
pressure, and day-of-year: held-out corr(pred, a) ≈ **0.66–0.92** (MAE ≈ 4–10 s).
The nrcd weather/elevation structure is not contradicted on this sample (this is
not a full re-derivation of package coefficients).

> Sources: `output/rq1/enrichment/weather_inflation.csv`,
> `output/rq1/robustness_checks/weather_holdout.csv`.

### Relative finish & club field-relative course factors (LACCTiC-motivated)

Complement to metadata standardization: [LACCTiC](https://www.lacctic.com/)
infers course difficulty from overlapping athletes’ times (no weather inputs).
We derive within-meet place / finish percentile from clock times and fit a
transparent club analogue (damped median EM for meet factor α on raw times;
median α pinned to 1; no track-PR calibration). Field size ≥ 5 required.
Head-to-head vs NRCD Standardized: `output/rq1/relative_finish_course_factors/VERDICT.md`.

**Within-meet order:** Raw / Converted / Standardized preserve finish order
almost perfectly (women: 100% exact place agreement across 193 meets; men:
≈99.8% across 238) — as expected when env adjustments are mostly meet-level.

**Time vs place season change** (n = 1,971 women / 3,484 men athlete-seasons):
Standardized mean first→last time improvement ≈ **26 s / 28 s**, but mean
finish-percentile change is slightly **negative** (−0.013 / −0.010); only
~46% improve their relative place. Spearman(time Δ, place Δ) under
Standardized is moderate (ρ ≈ 0.46 / 0.49). Place trajectories are essentially
identical across Raw/Conv/Std (ρ ≈ 1).

**Field-adjusted (α·raw) vs Standardized:** season Δ agrees in rank
(Spearman ρ ≈ **0.70** women / **0.77** men) but field adjustment shrinks mean
improvement to ~2–4 s (vs ~26–28 s Standardized). Meet α is only weakly related
to the weather/elevation residual (men ρ ≈ −0.14, p ≈ 0.04; women n.s.).

**Are actual course factors better than LACCTiC-style α?** For *this paper’s*
improvement estimand: **yes — prefer Standardized.** Env residual tracks
temperature at Spearman ρ ≈ **0.86 / 0.89**; field hardness only ≈ 0.17 / 0.14.
Field α *does* win next-raw-time prediction (MAE ≈ 46–50 s vs Std ≈ 65–72 s)
and split-half ability reliability — expected for a batch-effect correction on
the clock you are predicting (LACCTiC’s ranking use case). It is a good
**fallback / sensitivity** when weather is missing, not a replacement while
comprehensive-era coverage stays ~98%.

> Sources: `output/rq1/relative_finish_course_factors/SUMMARY.md`,
> `VERDICT.md`, `next_race_prediction_summary.csv`,
> `difficulty_metadata_alignment.csv`, `method_comparison_verdict.csv`,
> `tier_improvement_summary.csv`, `field_adjusted_vs_standardized.csv`,
> `course_factors.csv`, `within_meet_order_agreement.csv`.

### Early-window prediction check (coach estimand)

**Performance-only** features available after meet 1 (or after meets 1–2 among
athletes with ≥3 races total); temporal split train 2023 → test 2024; no
future schedule length in the feature set.

| Gender | Window | Best model | Test R² | MAE skill vs mean | SER (clipped) |
|--------|--------|------------|---------|-------------------|---------------|
| Men | First race only | SVR | 0.025 | +8% | 0.11 |
| Men | First two races | Ridge | 0.190 | +10% | 0.83 |
| Women | First race only | SVR | −0.015 | +8% | 0.00 |
| Women | First two races | SVR | 0.081 | +14% | 0.29 |

**Takeaway:** after the opener, forecasts are essentially useless for
individualizing season plans (near-zero R² / SER). After two races the men’s
Ridge R² rises on the ≥3-race subsample, but MAE still improves only ~10% over
predicting the training mean, and women remain weak — not enough to support
athlete-level early-season triage. Calibration plots are in
`early_window_decision.pdf`.

> Sources: `output/rq1/enrichment/early_window_decision.csv`,
> `early_window_calibration.csv`, `early_window_prediction.csv`.

### Robustness

- `last_time` is excluded by construction rather than tested after the fact.
- **Learning curves (compact SVR):** from 20% to 100% of the 2023 training data,
  mean test R² moves only 0.026→0.043 (men) and −0.039→−0.029 (women), flattening
  between 80% and 100%. GB improves with sample size but remains negative; RF
  remains negative throughout. This does not prove an irreducible ceiling, but
  weakens “the same features just need modestly more data” as an explanation.
- **Outcome trimming (compact SVR):** remains near zero under ±20, ±50, ±100 s/day
  and training-only 1.5-IQR rules (men R² −0.031 to 0.043; women −0.029 to 0.014).
  RF and GB remain negative under the primary suite.
- Fixed Random Forest baselines are negative (men −0.422, women −0.289), so
  ablation rankings are instability diagnostics only.
- Dropping all absolute Standardized time markers changes RF R² by only +0.003
  (men) and +0.044 (women); neither model becomes predictive.
- Gender-separated mixed-effects models are descriptive and now include a
  **quadratic race-count term** to match the non-monotone dose–response. For men
  the linear term is −0.548 s/day per SD (p < 0.001, more races → faster) with a
  significant positive curvature +0.170 (p = 0.048, diminishing/reversing
  returns); women show the same-direction but non-significant curvature (+0.092,
  p = 0.45) and a marginal linear term (−0.375, p = 0.08). A single linear term
  would have masked this inverted-U.

> Sources: `output/rq1/feature_exclusion_audit/compact_learning_curves.csv`,
> `compact_outlier_sensitivity.csv`, `output/rq1/robustness_feature_ablation/`,
> `output/rq1/sensitivity_sweep/`, `output/rq1/mixed_effects/`.

### Why the null is a signal ceiling, not just failed models

- **Reliability / noise ceiling.** A single Standardized race time is reliable
  within a season (ICC 0.78 men, 0.82 women), but the season improvement *slope*
  has a split-half reliability of only **0.23** (men) / **0.28** (women). Because
  a construct can't be predicted better than it's measured, this is an
  approximate upper bound on any model's test R² — and it's near zero. (Caveat:
  the odd/even split needs ≥4 races, so this ceiling is estimated on a
  more-persistent subgroup than the full ≥2-race prediction sample; it's an
  approximation for the broader sample it bounds.) The Signal Extraction Ratio
  SER = R²/ρ is **≈0.19** for men (compact SVR) and clipped at 0 for women —
  almost none of the already-small reliable signal is captured.
- **Full-sample (≥2-race) reliability checks.** Across-season test–retest of
  the improvement *rate* is near zero (Pearson r ≈ 0.05 for both sexes) —
  confirming that year-to-year rates do not even predict themselves. A
  simulation noise floor for first→last Δ on the full ≥2-race sample gives
  φ ≈ 0.35 (men) / 0.68 (women). The conservative R² ceiling for the prediction
  sample remains the split-half ≥4-race value (~0.23 / 0.28); SER vs that
  ceiling is unchanged (~0.19 / non-positive).
- **Permutation null (compact SVR).** Shuffling the 2023 training outcome and
  refitting 1,000× (seed 42): observed R² sits above the shuffled null (men
  0.043 vs null mean −0.077; women −0.029 vs −0.115; both permutation p ≈ 0.001).
  A tiny, statistically detectable but practically useless amount of information
  exists.
- **Classification of the tail.** Predicting the top-quartile improver reaches
  AUC ≈ 0.79–0.81 with all features, but drops to **0.66–0.69** once
  baseline-coupled time features are removed — so most of the tail skill is the
  mechanical baseline→change (regression-to-the-mean) coupling, not new signal.
- **Gender × race-count interaction.** A single joint mixed model finds no
  significant sex difference in the linear (p = 0.49) or quadratic (p = 0.60)
  race-count term: the dose–response shape is shared across sexes.

> Sources: `output/rq1/null_result_diagnostics/`,
> `output/rq1/feature_exclusion_audit/compact_permutation_r2.csv`,
> `output/rq1/mathematical_contributions/signal_extraction_ratio.csv`,
> `output/rq1/robustness_checks/full_sample_reliability.csv`,
> `output/rq1/mixed_effects/interaction_fixed_effects.csv`.

### Standardized vs. Converted Only (sensitivity)

Under the same pooled SVR and split, R² is 0.041 for **Standardized**, 0.044 for
**Converted Only**, and 0.035 for unadjusted times. These negligible differences
do not change the weak-prediction conclusion. Standardized remains primary
because it supplies scientifically comparable performances, not because it
maximizes R².

---

## 3. Team race frequency and nationals placement

**Question:** Are teams with athletes who race frequently over-represented among
top national finishers?

We test whether teams with ≥ 1 athlete racing **4+** regular-season times are
over-represented in the **top 15** at nationals, per year × gender.

| Group | Teams | 4+ teams | Top-15 with 4+ | RR | p (χ²) | Bonferroni sig. |
|-------|-------|----------|----------------|----|--------|------------------|
| 2023 Men | 92 | 35 (38%) | 7/14 (50%) | 1.63 | 0.483 | No |
| 2023 Women | 83 | 25 (30%) | 7/15 (47%) | 2.03 | 0.218 | No |
| 2024 Men | 104 | 36 (35%) | 8/15 (53%) | 2.16 | 0.176 | No |
| 2024 Women | 91 | 28 (31%) | 9/15 (60%) | 3.38 | 0.017 | No |
| 2025 Men | 106 | 39 (37%) | 6/15 (40%) | 1.15 | 1.000 | No |
| 2025 Women | 97 | 33 (34%) | 9/15 (60%) | 2.91 | 0.044 | No |

Point estimates consistently favor over-representation (**RR ≈ 1.15–3.38**,
strongest for women), but **no** individual split survives Bonferroni correction
(α = 0.0083) — because each cell has only 15 top-15 slots, i.e. underpowered, not
because there is no association.

**Once power is recovered, the association is robust** (all point the same way):

- **Random-effects pooling** (DerSimonian–Laird) of the six log-RRs gives
  RR = **2.09** (95% CI [1.42, 3.06]) at 4+ races, with no between-cell
  heterogeneity (τ² = 0, Q p = 0.66). The noisy cells shrink to this value, so
  the extremes (2024 women 3.38; 2025 men 1.15) are not distinguishable from
  RR ≈ 2.1.
- **Dose-monotone in the threshold:** pooled RR rises 1.32 (≥3) → 2.09 (≥4) →
  3.04 (≥5), which is what a real signal (not a cut-point artifact) should do.
- **Cluster-robust continuous model** (drops the 4+ dichotomy): odds of top-15
  per SD of team max race count = **2.56** [1.78, 3.67], p < 1e-6 (GEE,
  cluster-robust by team); a team-random-intercept Bayesian mixed GLM agrees
  (OR 2.53 [1.88, 3.41]). n = 573 team-years, 126 teams.
- **Placement curve (why top-15?).** Top-15 is conventional, not privileged.
  On the ranked top-25 panel, pooled RR at ≥4 races declines smoothly with the
  cut: top-5 **5.71** → top-10 **5.44** → top-15 **4.32** → top-20 **3.68** →
  top-25 **3.28** (all CIs exclude 1). Continuous OR stays ~2.7–3.6 across the
  same cuts. Spearman(rank, max races) among top-25 = **−0.24** (p = 0.003).
  So the association is strongest among elite finishers and is not an artifact
  of choosing k=15 (`topk_placement_*.csv`).

Still **observational**: stronger, better-resourced programs plausibly both race
more and place higher, so this is not evidence that adding races *causes* better
placement.

**Selection-robust checks** (`output/rq3/team_association_robustness/SELECTION_ROBUSTNESS.md`):

- **Within-team first differences** (328 consecutive team–year pairs): Δ max
  race count, Δ depth, and Δ ERO do **not** significantly predict Δ top-15 under
  HC1 OLS (all p > 0.08); Spearman for Δ ERO is only ρ = 0.13 (p = 0.016).
  Among pairs with top-25 ranks both years (n = 78), Δ schedule metrics also fail
  to track Δ rank. Year-to-year schedule changes inside a program do not move
  placement the way the cross-sectional association suggests.
- **Roster-size adjustment:** GEE OR for z(max races) alone is 2.57; adding
  z(`n_athletes`) cuts it to **1.69** [1.05, 2.71]. In the joint
  max-race + depth + size model, depth OR collapses to ~1.05 (n.s.) while size
  remains large. Much of the cross-sectional “depth/volume” signal is entangled
  with program size / between-team selection.
- **Non-top-7 (bench) depth:** Athletes outside the season-best top 7 with ≥3
  starts — OR **2.36** alone [1.68, 3.33], but **1.11** (n.s.) after size
  control (corr with roster size 0.81–0.85). Top-15 mean beyond-7≥3: men 11.6
  vs 3.0; women 5.7 vs 1.3 — tracking club headcount (40 vs 18 men; 25 vs 12
  women). Same story as overall depth: **club size, not an independent schedule
  effect.**

> Sources: `within_team_first_diff_summary.csv`, `roster_size_adjusted_models.csv`,
> `beyond_top7_depth_models.csv`, `beyond_top7_depth_descriptives.csv`.

**Confounder proxies (size, travel, longevity):** Adding roster size + mean
meet-travel (km from season meet centroid) cuts the max-race GEE OR from 2.57
to **1.67** [1.03, 2.72]; travel itself is null. Within-team demeaned LPM:
Δmax-race and Δdepth ≈ 0; only demeaned roster size remains positive. Top-15
team-years are more often returning programs (70% prior-year presence vs 56%
among non–top-15) — descriptive selection, not a logistic that separates cleanly.

> Source: `output/rq1/robustness_checks/team_confounder_models.csv`.

### Roster depth, peer density, and who benefits

Three follow-ons bridge the individual null and the team positive finding
(`output/rq1/underexplored_mechanisms/`):

1. **Roster depth, not a workhorse — but entangled with size.** Top-15 teams
   field ~16 (men) / ~9.5 (women) athletes with ≥3 starts vs ~5 / ~3 among other
   qualifying teams (d ≈ 1.3–1.4). Conditional on max race count, depth OR =
   **1.74** [1.26, 2.42]; after roster size, depth ≈ **1.05** (n.s.). Non-top-7
   bench depth shows the same size channel (OR 2.36 → 1.11).
2. **Peer co-start density.** Athletes who race with more same-team teammates
   improve faster (−0.37 s/day per SD mean peers, p < 1e-4) controlling for own
   race count and starting ability, and retain more (OR = 1.14 per SD peers).
3. **Quantile / BH “who benefits.”** Race-volume slopes are largest in the lower
   improvement tail (men τ=0.25: +12 s/race; τ=0.75: +7 s). BH-corrected 4+ vs 2
   contrasts survive for mid-to-slow starters (5/8 strata), not for mid-pack Q2.

A separate 10,000-draw, year-stratified permutation test (seed 42) checks the
top-25 rank correlations. Empirical p-values closely match parametric results:
for max races, p = 0.110 (men, r = −0.190) and 0.048 (women, r = −0.232), so
neither survives the within-sex three-test Bonferroni threshold (0.0167). Only
men's season duration correlation meets that threshold (r = −0.283,
permutation p = 0.016).

> Sources: `output/rq1/nationals_overlap/nationals_overlap_summary.csv`,
> `output/rq1/top25_teams/correlations_comprehensive.csv`,
> `output/rq1/top25_teams/correlation_permutation_null.csv`,
> `output/rq3/team_association_robustness/`,
> `output/rq1/underexplored_mechanisms/`.

---

## 4. RQ2 — Multi-season progression

**Question:** Do athletes with stable multi-season participation keep improving?

Athletes are retained only if their race count differs by ≤ 1 between adjacent
seasons (isolating comparable workload).

- **Multi-season prediction is not robust.** On the primary split (train 2023 →
  test 2024) every model is non-positive for both sexes (best: men SVR −0.09,
  women SVR −0.08). On the 2025 splits, men's *linear* models turn positive
  (Lasso R² = 0.68, 95% CI [0.33, 0.82] for 2023+2024 → 2025), but the same
  setup is negative for women (Lasso −0.30) and for SVR everywhere. We report
  this openly rather than claiming uniform failure, but do not treat it as
  robust forecasting: the consistency filter forces the *same athletes* into
  train and test years (88–93% overlap), so the positive fits partly recover
  athlete-specific baselines (identity autocorrelation) rather than forecasting
  a new cohort, and the effect does not replicate across sex, model, or target
  year. Feature importances from these models are not interpreted substantively.
- Among retained athletes, **59.2%** of both men (n = 262) and women (n = 120)
  improved their best time from 2023 to 2025.
- Gains are front-loaded: mean best-time change is larger from 2023→2024
  (men −0.18, women −0.23 min) than 2024→2025 (men +0.07, women −0.06 min).
- **Retention** (race next season after a ≥2-race season): men 55%, women 52%
  overall; rises with prior race count (men 45% after 2 races → 72% after 4;
  women 41% → 61%).

Grade year and pre-2023 training history are unobserved, so these describe the
filtered three-season panel rather than a full collegiate career.

> Sources: `output/rq2/temporal_splits/temporal_splits_r2.csv`,
> `output/rq2/multi_season_model_performance.csv`,
> `output/rq2/athletes_all_seasons_summary.csv`,
> `output/rq2/gender_time_comparison.csv`,
> `output/rq1/enrichment/season_retention.csv`.
> `output/rq2/multi_season_gender_feature_importance_comparison.csv`.

---

## 5. RQ3 — Gender differences in participation

**Question:** How do racing frequency and improvement differ by gender?

- **Participation is imbalanced.** Men make up ~60–63% of athletes and races
  each year; χ² tests against equal shares give p ≪ 0.001 for both athlete and
  race counts.
- **Racing intensity is not.** Conditional on competing, mean races per athlete
  are nearly identical (men 2.00–2.07; women 1.92–1.97).
- **Distribution.** ~40–46% of athletes race once and only ~9–12.5% race four or
  more times, with modest gender gaps within bins.

The imbalance is concentrated at **entry into competition**, not in how often
active athletes race — suggesting recruitment and initial-participation barriers
rather than differences in commitment among those who compete.

> Sources: `output/rq3/gender_analysis/gender_analysis_summary.csv`,
> `output/rq3/gender_analysis/gender_race_participation.csv`,
> `output/rq3/gender_analysis/race_count_distribution.csv`.

---

## 6. Interpretation and limitations

- **Association, not causation.** Athletes who race 4+ times are a selected group
  (healthier, more available, more committed, or on programs offering more
  opportunities); we cannot identify a causal effect of "assigning more races."
- Finish times omit tactics, mid-race injury, and intentional conservation; we
  lack training logs, illness, and menstrual-cycle data.
- Only three seasons (2023–2025) are covered; grade year and prior training are
  unobserved.
- Standardization formulas (including the weather coefficient) come from the
  NRCD resource paper and are not independently re-validated here.
- Club NIRCA populations overlap with, but are not identical to, NCAA/NAIA/NJCAA
  athletes, so transfer to varsity contexts should be cautious.

---

## 7. Reproducing these numbers

```bash
python scripts/sample_summary.py   # sample / coverage tables (Section 1)
python scripts/rq1.py              # Sections 2-3
python scripts/rq2.py              # Section 4
python scripts/rq3.py              # Section 5
# or run everything:
python scripts/run_all.py
```

See [`README.md`](README.md) for setup, data resolution, and method details.
