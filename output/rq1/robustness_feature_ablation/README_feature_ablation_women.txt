Feature ablation robustness checks
=================================

Input table: output/rq1/raw_data_athlete_features.csv
Gender subset: F (gender_encoded / gender_year excluded)
Model: RandomForestRegressor(n_estimators=100, random_state=42)
Primary split: train 2023, test 2024

Baseline (train 2023 → test 2024): R²=-0.2837, RMSE=7.4048, MAE=4.4814

Largest performance drops (single-feature removal; sorted by ΔR² ascending):
                     removed  delta_r2        r2     rmse      mae
                       slope -0.006796 -0.290472 7.424357 4.491704
                  first_time -0.003599 -0.287275 7.415155 4.488251
                   best_time  0.001197 -0.282479 7.401327 4.473510
            experience_level  0.002492 -0.281184 7.397591 4.480289
race_to_race_improvement_std  0.002595 -0.281081 7.397294 4.479298
         starting_percentile  0.004538 -0.279138 7.391682 4.477587
              bad_race_count  0.004705 -0.278971 7.391198 4.471140
     season_duration_squared  0.005137 -0.278539 7.389950 4.467800
             season_duration  0.005213 -0.278463 7.389731 4.468134
              race_frequency  0.005585 -0.278091 7.388656 4.476779
