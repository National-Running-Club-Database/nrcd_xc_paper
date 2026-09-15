# Relative finish & club course factors

Generated under `/Users/jonathankarr/Documents/nrcd_xc_paper/output/rq1/relative_finish_course_factors/`.

Motivation: [LACCTiC](https://www.lacctic.com/) adjusts XC times for course
difficulty from overlapping athletes' relative performances (not weather).
This club analysis derives within-meet place/percentile, compares Raw /
Converted / Standardized season change, and fits a transparent
LACCTiC-style course factor α (EM on raw times; median α pinned to 1;
no track-PR calibration).

**Head-to-head verdict** (metadata Standardized vs field α): see
[`VERDICT.md`](VERDICT.md).

Inclusion: field size ≥ 5 at first and last race for
athlete-season relative metrics; same floor for α estimation.

## Within-meet order agreement

- Women raw_vs_conv: exact place agree=1.000 [1.000, 1.000], perfect meets=100.0%, mean Spearman=1.0000, n_meets=193
- Women raw_vs_std: exact place agree=1.000 [1.000, 1.000], perfect meets=100.0%, mean Spearman=1.0000, n_meets=193
- Women conv_vs_std: exact place agree=1.000 [1.000, 1.000], perfect meets=100.0%, mean Spearman=1.0000, n_meets=193
- Men raw_vs_conv: exact place agree=0.998 [0.995, 1.000], perfect meets=99.6%, mean Spearman=0.9999, n_meets=238
- Men raw_vs_std: exact place agree=0.998 [0.995, 1.000], perfect meets=99.6%, mean Spearman=0.9999, n_meets=238
- Men conv_vs_std: exact place agree=1.000 [1.000, 1.000], perfect meets=100.0%, mean Spearman=1.0000, n_meets=238

## First→last by clock tier (time and finish percentile)

- Women raw: time Δ=-41.5s [-50.2, -33.2] (52.0% faster); finish-pct Δ=-0.013 [-0.020, -0.006] (46.5% better place), n=1971
- Women conv: time Δ=39.8s [34.6, 45.3] (67.4% faster); finish-pct Δ=-0.013 [-0.020, -0.006] (46.5% better place), n=1971
- Women std: time Δ=25.7s [20.6, 30.8] (60.3% faster); finish-pct Δ=-0.013 [-0.020, -0.005] (46.5% better place), n=1971
- Women Converted−Std time inflation: 14.2s [13.3, 15.0], Wilcoxon p=2.22e-162, n=1971
- Men raw: time Δ=-27.1s [-36.9, -17.7] (63.4% faster); finish-pct Δ=-0.010 [-0.015, -0.004] (45.6% better place), n=3484
- Men conv: time Δ=48.1s [42.2, 54.3] (65.5% faster); finish-pct Δ=-0.010 [-0.015, -0.005] (45.5% better place), n=3484
- Men std: time Δ=28.1s [22.2, 34.0] (58.0% faster); finish-pct Δ=-0.010 [-0.015, -0.005] (45.5% better place), n=3484
- Men Converted−Std time inflation: 20.0s [19.2, 20.9], Wilcoxon p=0.00e+00, n=3484

## Concordance (Spearman)

- Women raw_time_vs_raw_place: ρ=0.318 (p=1.10e-47), sign agree=0.626, n=1971
- Women conv_time_vs_conv_place: ρ=0.444 (p=4.83e-96), sign agree=0.635, n=1971
- Women std_time_vs_std_place: ρ=0.463 (p=2.21e-105), sign agree=0.665, n=1971
- Women raw_time_vs_std_time: ρ=0.648 (p=1.92e-235), sign agree=0.773, n=1971
- Women conv_time_vs_std_time: ρ=0.976 (p=0.00e+00), sign agree=0.909, n=1971
- Women raw_place_vs_std_place: ρ=1.000 (p=0.00e+00), sign agree=1.000, n=1971
- Women raw_place_vs_conv_place: ρ=1.000 (p=0.00e+00), sign agree=1.000, n=1971
- Men raw_time_vs_raw_place: ρ=0.408 (p=1.33e-139), sign agree=0.617, n=3484
- Men conv_time_vs_conv_place: ρ=0.479 (p=1.67e-199), sign agree=0.630, n=3484
- Men std_time_vs_std_place: ρ=0.494 (p=1.25e-213), sign agree=0.658, n=3484
- Men raw_time_vs_std_time: ρ=0.784 (p=0.00e+00), sign agree=0.862, n=3484
- Men conv_time_vs_std_time: ρ=0.964 (p=0.00e+00), sign agree=0.902, n=3484
- Men raw_place_vs_std_place: ρ=1.000 (p=0.00e+00), sign agree=0.999, n=3484
- Men raw_place_vs_conv_place: ρ=1.000 (p=0.00e+00), sign agree=0.999, n=3484

## Course factor α vs weather residual

- Women: Spearman(α, mean env residual)=-0.088 (p=2.25e-01), n_meets=191, sd(α)=0.0914
- Men: Spearman(α, mean env residual)=-0.137 (p=3.59e-02), n_meets=236, sd(α)=0.2094

## Field-adjusted (α·raw) vs Standardized improvement

- Women field_adj: 2.0s [-1.9, 6.0], %improved=52.1, n=1971
- Women standardized: 25.7s [20.6, 30.9], %improved=60.3, n=1971
- Women raw: -41.5s [-49.5, -33.3], %improved=52.0, n=1971
- Women converted: 39.8s [34.5, 45.2], %improved=67.4, n=1971
- Women field_adj−Std: -23.7s [-26.8, -20.7], Wilcoxon p=2.56e-45, Spearman ρ=0.703
- Men field_adj: 3.5s [0.2, 7.0], %improved=50.9, n=3484
- Men standardized: 28.1s [22.1, 34.3], %improved=58.0, n=3484
- Men raw: -27.1s [-37.5, -17.7], %improved=63.4, n=3484
- Men converted: 48.1s [42.3, 54.4], %improved=65.5, n=3484
- Men field_adj−Std: -24.5s [-29.3, -20.2], Wilcoxon p=1.09e-56, Spearman ρ=0.770

## Interpretation notes

- If weather/elevation adjustments are mostly meet-level, within-meet
  place order is nearly identical for Raw / Converted / Standardized;
  relative-finish Δ then mainly reflects field composition change, not
  environmental standardization.
- Field-inferred α is a *complement* to NRCD Standardized: agreement
  with Standardized Δ supports robustness; disagreement highlights meets
  where field strength and weather residuals diverge.

## Head-to-head (see VERDICT.md)

- field_alpha: 6 criterion×gender wins
- standardized: 3 criterion×gender wins
- standardized_full: 1 criterion×gender wins

Full write-up: `VERDICT.md`.

