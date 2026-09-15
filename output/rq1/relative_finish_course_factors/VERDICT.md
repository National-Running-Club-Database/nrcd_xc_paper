# Method comparison verdict

Question: are NRCD **actual course factors** (distance + weather +
elevation → Standardized / Converted) a better indicator than a
club **LACCTiC-style field α**?

## Short answer

**Yes — for this paper’s estimand, prefer actual NRCD Standardized.**
Field α is better at predicting *raw clock times* across meets (that is
what LACCTiC is designed for), but it is **not** a better *course*
indicator than metadata standardization when weather coverage is high.

| Goal | Better method | Why |
|------|---------------|-----|
| Measure within-season improvement without weather bias | **Standardized** | Env residual tracks temperature at ρ≈0.86–0.89; weather inflation identity |
| Predict next raw finish time across courses | **Field α** | MAE ~46–50 s vs Std ~65–72 s |
| Course difficulty that means weather/terrain | **Standardized** | Field α barely tracks temperature (ρ≈0.14–0.17) |
| Robustness when weather is missing | **Field α** | Needs only overlapping athletes |

## Win counts (naive criterion × gender)

Naive wins overweight raw-time tasks (field α’s specialty):

- **field_alpha**: 7 wins
- **standardized**: 2 wins
- **standardized_full**: 1 wins

## Bottom line for the manuscript

Keep **Standardized as primary**. Report field α / relative-finish as a
**complementary sensitivity**: season Δ ranks agree (Spearman ρ≈0.70–0.77)
but field α shrinks mean improvement toward ~0–4 s because it absorbs
field-composition and unmeasured meet effects, not only course/weather.
Do **not** replace Standardized with LACCTiC-style α while comprehensive-era
weather coverage remains ~98%.

## Criterion detail

- Men / next_race_mae_last / converted: MAE=80.3s [75.7, 85.4]
- Men / next_race_mae_last / field_alpha: MAE=50.1s [47.7, 52.7] ✓
- Men / next_race_mae_last / raw: MAE=121.0s [115.5, 126.9]
- Men / next_race_mae_last / standardized: MAE=71.1s [67.0, 75.4]
- Women / next_race_mae_last / converted: MAE=74.5s [69.8, 79.4]
- Women / next_race_mae_last / field_alpha: MAE=45.7s [42.1, 49.7] ✓
- Women / next_race_mae_last / raw: MAE=92.6s [87.2, 98.5]
- Women / next_race_mae_last / standardized: MAE=65.4s [61.1, 70.0]
- Men / holdout_within_athlete_sd / field_alpha: SD=77.1s ✓
- Men / holdout_within_athlete_sd / standardized: SD=97.6s
- Men / holdout_within_athlete_sd / converted: SD=101.7s
- Men / holdout_within_athlete_sd / raw: SD=171.1s
- Women / holdout_within_athlete_sd / field_alpha: SD=67.8s ✓
- Women / holdout_within_athlete_sd / standardized: SD=78.7s
- Women / holdout_within_athlete_sd / converted: SD=82.3s
- Women / holdout_within_athlete_sd / raw: SD=120.1s
- Men / split_half_reliability / raw: ρ=0.625
- Men / split_half_reliability / converted: ρ=0.891
- Men / split_half_reliability / standardized: ρ=0.903
- Men / split_half_reliability / field_alpha: ρ=0.940 ✓
- Women / split_half_reliability / raw: ρ=0.758
- Women / split_half_reliability / converted: ρ=0.896
- Women / split_half_reliability / standardized: ρ=0.908
- Women / split_half_reliability / field_alpha: ρ=0.947 ✓
- Women / tracks_temperature / field_alpha: |ρ|=0.160 vs temperature
- Women / tracks_temperature / standardized: |ρ|=0.873 vs temperature ✓
- Women / field_size_confound / field_alpha: |ρ|=0.164 vs field_size (lower=less confound)
- Women / field_size_confound / standardized: |ρ|=0.303 vs field_size (lower=less confound)
- Women / field_size_confound / standardized_full: |ρ|=0.113 vs field_size (lower=less confound) ✓
- Men / tracks_temperature / field_alpha: |ρ|=0.149 vs temperature
- Men / tracks_temperature / standardized: |ρ|=0.896 vs temperature ✓
- Men / field_size_confound / field_alpha: |ρ|=0.032 vs field_size (lower=less confound) ✓
- Men / field_size_confound / standardized: |ρ|=0.271 vs field_size (lower=less confound)
- Men / field_size_confound / standardized_full: |ρ|=0.096 vs field_size (lower=less confound)

## Interpretation of the horse race

- **Field α wins raw MAE, holdout residual SD, and split-half reliability**
  because it is an empirical batch-effect correction on the *same clock*
  you are predicting. That is valuable for rankings/simulations (LACCTiC’s
  use case), not proof it isolates *course* difficulty.
- **Standardized wins environmental alignment**: mean env residual vs
  temperature Spearman ρ ≈ 0.86 (women) / 0.89 (men); field hardness
  only ρ ≈ 0.17 / 0.14. So “actual course factors” are the better
  indicator of weather-driven course difficulty.
- Field α ↔ env residual agreement is weak (ρ ≈ 0.08–0.14) — the two
  adjustments are largely different information.
- Neither proxy is strongly confounded with field size (|ρ| ≲ 0.16).

