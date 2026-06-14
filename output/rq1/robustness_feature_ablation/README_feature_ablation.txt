Feature ablation robustness checks
=================================

Input table: output/rq1/raw_data_athlete_features.csv
Model: RandomForestRegressor(n_estimators=100, random_state=42)
Primary split: train 2023, test 2024

Baseline (train 2023 → test 2024): R²=0.9038, RMSE=1.5749, MAE=0.4281

Largest performance drops (single-feature removal; sorted by ΔR² ascending):
                removed  delta_r2       r2     rmse      mae
         bad_race_count -0.019095 0.884704 1.724158 0.425085
                  slope -0.006487 0.897312 1.627158 0.423755
         race_frequency -0.004156 0.899642 1.608589 0.437656
        season_duration -0.001469 0.902330 1.586906 0.433348
              num_races -0.001325 0.902474 1.585737 0.432873
season_duration_squared -0.000682 0.903117 1.580496 0.432024
              last_time -0.000468 0.903331 1.578749 0.438597
                   year -0.000463 0.903335 1.578714 0.427570
      best_to_avg_ratio -0.000251 0.903547 1.576983 0.431883
                cv_time  0.000058 0.903856 1.574454 0.430132
