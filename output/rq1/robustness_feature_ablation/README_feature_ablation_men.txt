Feature ablation robustness checks
=================================

Input table: output/rq1/raw_data_athlete_features.csv
Gender subset: M (gender_encoded / gender_year excluded)
Model: RandomForestRegressor(n_estimators=100, random_state=42)
Primary split: train 2023, test 2024

Baseline (train 2023 → test 2024): R²=-0.4033, RMSE=7.3850, MAE=4.3231

Largest performance drops (single-feature removal; sorted by ΔR² ascending):
                     removed  delta_r2        r2     rmse      mae
                       slope -0.054201 -0.457459 7.526276 4.393968
              bad_race_count -0.012240 -0.415498 7.417142 4.334454
                   best_time -0.010338 -0.413596 7.412157 4.341374
race_to_race_improvement_std -0.008967 -0.412225 7.408562 4.338775
                  first_time -0.008432 -0.411689 7.407157 4.328481
                   num_races -0.008244 -0.411501 7.406664 4.332570
           num_races_squared -0.008244 -0.411501 7.406664 4.332570
            best_race_timing -0.006723 -0.409981 7.402674 4.330115
      avg_days_between_races -0.006056 -0.409313 7.400921 4.342075
            experience_level -0.005106 -0.408364 7.398428 4.336764
