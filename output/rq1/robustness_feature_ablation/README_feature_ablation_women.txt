Feature ablation robustness checks
=================================

Input table: output/rq1/raw_data_athlete_features.csv
Gender subset: F (gender_encoded / gender_year excluded)
Model: RandomForestRegressor(n_estimators=100, random_state=42)
Primary split: train 2023, test 2024

Baseline (train 2023 → test 2024): R²=-0.3061, RMSE=7.5902, MAE=4.6680

Largest performance drops (single-feature removal; sorted by ΔR² ascending):
               removed  delta_r2        r2     rmse      mae
     best_to_avg_ratio -0.002380 -0.308503 7.597083 4.667385
               cv_time  0.003094 -0.303029 7.581175 4.664526
     variability_score  0.003656 -0.302467 7.579540 4.668574
   starting_percentile  0.004490 -0.301633 7.577114 4.656347
              time_std  0.004630 -0.301493 7.576705 4.665465
best_race_timing_ratio  0.004643 -0.301480 7.576667 4.651205
    worst_to_avg_ratio  0.004949 -0.301175 7.575778 4.662300
                 slope  0.005506 -0.300617 7.574156 4.671776
            time_range  0.005901 -0.300222 7.573005 4.654407
        bad_race_count  0.007162 -0.298961 7.569332 4.652464
