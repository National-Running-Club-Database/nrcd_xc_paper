Feature ablation robustness checks
=================================

Input table: output/rq1/raw_data_athlete_features.csv
Gender subset: F (gender_encoded / gender_year excluded)
Model: RandomForestRegressor(n_estimators=100, random_state=42)
Primary split: train 2023, test 2024

Baseline (train 2023 → test 2024): R²=0.8883, RMSE=1.3876, MAE=0.5475

Largest performance drops (single-feature removal; sorted by ΔR² ascending):
                    removed  delta_r2       r2     rmse      mae
             bad_race_count -0.007295 0.881053 1.432267 0.551517
           experience_level -0.006104 0.882245 1.425075 0.564326
             race_frequency -0.005194 0.883154 1.419560 0.554848
                  last_time -0.004656 0.883693 1.416285 0.560323
            season_duration -0.004606 0.883743 1.415983 0.557302
starting_percentile_squared -0.003785 0.884564 1.410973 0.552878
        starting_percentile -0.003757 0.884592 1.410801 0.552824
                       year -0.003473 0.884876 1.409065 0.549802
     best_race_timing_ratio -0.002943 0.885405 1.405822 0.555378
                   avg_time -0.002697 0.885652 1.404306 0.551450
