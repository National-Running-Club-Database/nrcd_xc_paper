# Robustness / sensitivity checks -- summary

Script: `scripts/robustness_checks.py`.

## Skipped: historical experience covariates

See `WHY_NOT_HISTORICAL_EXPERIENCE.md`. Historical seasons are fragmentary
(**48%** of historical athlete-years have only one recorded race;
weather coverage ~53% on course-detail rows).
Race counts would understate true volume -- often by a large fraction.

## 1. Ability-matched volume (4+ vs 2 races)

| Method | Gender | n | Mean d improve (s) | p / note |
|--------|--------|---|--------------------|----------|
| IPW_ATE | Women | 1403 | 27.1 | IPW ATE |
| IPW_ATE | Men | 2371 | 32.3 | IPW ATE |
| 1NN_caliper_pooled_years | Women | 303 | 25.2 | Wilcoxon p=0.000326 |
| 1NN_caliper_pooled_years | Men | 527 | 28.6 | Wilcoxon p=9.49e-06 |

## 2. Team confounder proxies

| Model | Term | OR or LPM | 95% CI | p |
|-------|------|-----------|--------|---|
| maxrace | z_maxrace | 2.574 | [1.790, 3.702] | 3.41e-07 |
| maxrace_size_travel | z_maxrace | 1.676 | [1.033, 2.719] | 0.0363 |
| maxrace_size_travel | z_size | 2.379 | [1.387, 4.084] | 0.00166 |
| maxrace_size_travel | z_travel | 1.147 | [0.745, 1.766] | 0.534 |
| full | z_maxrace | 1.623 | [1.042, 2.528] | 0.0321 |
| full | z_depth | 1.081 | [0.486, 2.403] | 0.849 |
| full | z_size | 2.251 | [0.928, 5.461] | 0.0726 |
| full | z_travel | 1.157 | [0.741, 1.807] | 0.52 |
| within_team_demean_LPM | z_maxrace_dm | -0.001 | [-0.047, 0.045] | 0.975 |
| within_team_demean_LPM | z_depth_dm | -0.043 | [-0.121, 0.035] | 0.281 |
| within_team_demean_LPM | z_size_dm | 0.083 | [0.005, 0.161] | 0.037 |
| within_team_demean_LPM | z_travel_dm | 0.000 | [-0.033, 0.034] | 0.989 |
| descriptive_longevity | share_appeared_prior_top15 | 0.701 | [nan, nan] | nan |
| descriptive_longevity | share_appeared_prior_not_top15 | 0.555 | [nan, nan] | nan |

## 4. Full-sample reliability

| Gender | Test-retest r | phi (>=2) | Cons. ceiling | SER (cons.) |
|--------|---------------|-----------|---------------|-------------|
| Men | 0.047 | 0.34 | 0.228 | 0.188 |
| Women | 0.050 | 0.69 | 0.279 | -0.104 |

## 5. Weather residual holdout

| Holdout year | corr(pred,a) | MAE (s) | R2 |
|--------------|--------------|---------|----|
| 2023 | 0.655 | 4.38 | 0.022 |
| 2024 | 0.919 | 8.24 | 0.805 |
| 2025 | 0.922 | 9.96 | 0.668 |
