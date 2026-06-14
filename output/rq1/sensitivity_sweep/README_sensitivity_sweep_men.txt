Sensitivity sweep
=================

Input table: output/rq1/raw_data_athlete_features.csv
Gender subset: M (gender_encoded / gender_year excluded)
Model: RandomForestRegressor with varying n_estimators (see CSV)
Splits: temporal (2023→2024, 2023→2025, 2023+2024→2025)

Outputs:
- sensitivity_sweep_results_men.csv (all scenarios/splits + deltas vs baseline)
- sensitivity_sweep_delta_r2_train_2023_test_2024_men.pdf
- sensitivity_sweep_r2_across_splits_men.pdf
