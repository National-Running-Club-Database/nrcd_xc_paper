Feature ablation robustness checks
=================================

Input table: output/rq1/raw_data_athlete_features.csv
Gender subset: M (gender_encoded / gender_year excluded)
Model: RandomForestRegressor(n_estimators=100, random_state=42)
Primary split: train 2023, test 2024

Baseline (train 2023 → test 2024): R²=-0.4195, RMSE=7.5605, MAE=4.4567

Largest performance drops (single-feature removal; sorted by ΔR² ascending):
               removed  delta_r2        r2     rmse      mae
                 slope -0.023863 -0.443324 7.623830 4.510997
      experience_level -0.007602 -0.427063 7.580763 4.470838
        race_frequency -0.003062 -0.422523 7.568694 4.453020
     consistency_score -0.000797 -0.420259 7.562667 4.460372
     best_to_avg_ratio -0.000314 -0.419775 7.561380 4.460527
       season_duration -0.000269 -0.419730 7.561259 4.457837
            worst_time -0.000104 -0.419566 7.560822 4.465630
     num_races_squared  0.000580 -0.418881 7.558999 4.461219
avg_days_between_races  0.000727 -0.418734 7.558607 4.460492
             num_races  0.000867 -0.418594 7.558234 4.461041
