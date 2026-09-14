# Robustness / sensitivity checks -- summary

Script: `scripts/robustness_checks.py`.

## Skipped: historical experience covariates

See `WHY_NOT_HISTORICAL_EXPERIENCE.md`. Historical seasons are fragmentary
(**44%** of historical athlete-years have only one recorded race;
weather coverage ~18% on course-detail rows).
Race counts would understate true volume -- often by a large fraction.

## 1. Ability-matched volume (4+ vs 2 races)

| Method | Gender | n | Mean d improve (s) | p / note |
|--------|--------|---|--------------------|----------|
| IPW_ATE | Women | 1399 | 28.1 | IPW ATE |
| IPW_ATE | Men | 2369 | 33.0 | IPW ATE |
| 1NN_caliper_pooled_years | Women | 304 | 24.8 | Wilcoxon p=0.000113 |
| 1NN_caliper_pooled_years | Men | 515 | 27.6 | Wilcoxon p=2.04e-05 |

## 2. Team confounder proxies

| Model | Term | OR or LPM | 95% CI | p |
|-------|------|-----------|--------|---|
| maxrace | z_maxrace | 2.574 | [1.790, 3.702] | 3.41e-07 |
| maxrace_size_travel | z_maxrace | 1.674 | [1.030, 2.722] | 0.0375 |
| maxrace_size_travel | z_size | 2.369 | [1.382, 4.058] | 0.00169 |
| maxrace_size_travel | z_travel | 1.153 | [0.762, 1.744] | 0.501 |
| full | z_maxrace | 1.616 | [1.036, 2.520] | 0.0342 |
| full | z_depth | 1.090 | [0.493, 2.410] | 0.831 |
| full | z_size | 2.228 | [0.923, 5.375] | 0.0747 |
| full | z_travel | 1.164 | [0.761, 1.780] | 0.484 |
| within_team_demean_LPM | z_maxrace_dm | -0.000 | [-0.046, 0.045] | 0.985 |
| within_team_demean_LPM | z_depth_dm | -0.043 | [-0.120, 0.035] | 0.28 |
| within_team_demean_LPM | z_size_dm | 0.081 | [0.004, 0.159] | 0.0392 |
| within_team_demean_LPM | z_travel_dm | -0.001 | [-0.031, 0.030] | 0.968 |
| descriptive_longevity | share_appeared_prior_top15 | 0.701 | [nan, nan] | nan |
| descriptive_longevity | share_appeared_prior_not_top15 | 0.555 | [nan, nan] | nan |

## 4. Full-sample reliability

| Gender | Test-retest r | phi (>=2) | Cons. ceiling | SER (cons.) |
|--------|---------------|-----------|---------------|-------------|
| Men | 0.047 | 0.35 | 0.228 | 0.188 |
| Women | 0.048 | 0.68 | 0.279 | -0.104 |

## 5. Weather residual holdout

| Holdout year | corr(pred,a) | MAE (s) | R2 |
|--------------|--------------|---------|----|
| 2023 | 0.659 | 4.30 | 0.045 |
| 2024 | 0.915 | 8.30 | 0.801 |
| 2025 | 0.922 | 9.97 | 0.663 |
