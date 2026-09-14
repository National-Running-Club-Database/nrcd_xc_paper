# Feature exclusion policy and audit

Source of truth: `scripts/feature_policy.py`.
Primary model: gender-separated SVR, train 2023 → test 2024.

Compact primary features (17): `year`, `num_races`, `num_races_squared`, `season_duration`, `season_duration_squared`, `first_time`, `best_time`, `cv_time`, `race_frequency`, `starting_percentile`, `starting_percentile_squared`, `experience_level`, `slope`, `avg_days_between_races`, `race_to_race_improvement_std`, `best_race_timing`, `bad_race_count`.

Redundancy-pruned (kept out of primary): `worst_time`, `avg_time`, `time_std`, `time_range`, `best_to_avg_ratio`, `worst_to_avg_ratio`, `variability_score`, `consistency_score`, `best_race_timing_ratio`.

## Policy

| Feature | Status | Class | Reason |
|---|---|---|---|
| `last_time` | excluded | A_endpoint_leakage | Endpoint of y=(last-first)/duration. |
| `total_improvement` | excluded | A_endpoint_leakage | Numerator of y. |
| `late_season_performance` | excluded | A_endpoint_leakage | Alias of last_time (counted as A, not also C). |
| `progression_improvement` | excluded | A_endpoint_leakage | Alias of total_improvement (counted as A, not also C). |
| `improvement_per_race` | excluded | B_derived_leakage | total_improvement / num_races. |
| `improvement_to_variability_ratio` | excluded | B_derived_leakage | Uses total_improvement in the numerator. |
| `early_season_performance` | excluded | C_exact_duplicate | Alias of retained first_time. |
| `races_duration_ratio` | excluded | C_exact_duplicate | Identical to retained race_frequency. |
| `gender_encoded` | dropped_within_sex | E_within_sex_constant | Constant after sex-stratified fitting. |
| `gender_year` | dropped_within_sex | E_within_sex_constant | Constant after sex-stratified fitting. |

## Compact vs legacy full

Men compact R²=0.043 (n_feat=17); legacy R²=0.051 (n_feat=26); Δ=-0.007.

## Leakage / duplicate / prune add-back

Primary (compact) R² men=0.043.
`+last_time` ΔR²=+0.438 → leakage_confirmed.
`+total_improvement` ΔR²=+0.651 → leakage_confirmed.
`+races_duration_ratio` ΔR²=-0.004 → duplicate_ok.

## Over-exclusion

- `drop_slope`: R²=0.035 (Δ=-0.009) — null_robust_to_dropping_block
- `drop_absolute_times`: R²=0.043 (Δ=+0.000) — null_robust_to_dropping_block
- `drop_schedule`: R²=0.039 (Δ=-0.004) — null_robust_to_dropping_block
- `schedule_only`: R²=-0.056 (Δ=-0.099) — still_near_zero
- `early_external_proxy`: R²=0.029 (Δ=-0.014) — still_near_zero

Reproduce: `python scripts/feature_exclusion_audit.py`
