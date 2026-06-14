Feature ablation robustness checks
=================================

Input table: output/rq1/raw_data_athlete_features.csv
Gender subset: M (gender_encoded / gender_year excluded)
Model: RandomForestRegressor(n_estimators=100, random_state=42)
Primary split: train 2023, test 2024

Baseline (train 2023 → test 2024): R²=0.8193, RMSE=2.3557, MAE=0.5822

Largest performance drops (single-feature removal; sorted by ΔR² ascending):
               removed  delta_r2       r2     rmse      mae
        bad_race_count -0.025435 0.793852 2.516016 0.588181
      experience_level -0.006647 0.812640 2.398627 0.608012
        race_frequency -0.002587 0.816700 2.372495 0.580484
best_race_timing_ratio -0.002323 0.816964 2.370785 0.595308
                 slope -0.000664 0.818623 2.360017 0.601970
      best_race_timing  0.000322 0.819610 2.353589 0.587920
            time_range  0.000508 0.819795 2.352379 0.585072
    worst_to_avg_ratio  0.000905 0.820192 2.349783 0.584188
               cv_time  0.001021 0.820308 2.349026 0.579808
       season_duration  0.001185 0.820472 2.347953 0.589127
