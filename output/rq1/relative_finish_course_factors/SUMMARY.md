# Relative finish & club course factors

Generated under `output/rq1/relative_finish_course_factors/`.

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
- Men raw_vs_conv: exact place agree=0.994 [0.984, 1.000], perfect meets=99.2%, mean Spearman=0.9995, n_meets=238
- Men raw_vs_std: exact place agree=0.994 [0.984, 1.000], perfect meets=99.2%, mean Spearman=0.9995, n_meets=238
- Men conv_vs_std: exact place agree=1.000 [1.000, 1.000], perfect meets=100.0%, mean Spearman=1.0000, n_meets=238

## First→last by clock tier (time and finish percentile)

- Women raw: time Δ=-41.5s [-49.8, -32.8] (52.0% faster); finish-pct Δ=-0.013 [-0.020, -0.006] (46.5% better place), n=1975
- Women conv: time Δ=39.9s [34.8, 45.0] (67.3% faster); finish-pct Δ=-0.013 [-0.020, -0.006] (46.5% better place), n=1975
- Women std: time Δ=24.6s [19.8, 29.9] (59.8% faster); finish-pct Δ=-0.013 [-0.020, -0.006] (46.5% better place), n=1975
- Women Converted−Std time inflation: 15.3s [14.4, 16.1], Wilcoxon p=1.20e-175, n=1975
- Men raw: time Δ=-27.5s [-36.9, -18.1] (63.3% faster); finish-pct Δ=-0.010 [-0.015, -0.004] (45.5% better place), n=3489
- Men conv: time Δ=47.9s [42.0, 53.6] (65.5% faster); finish-pct Δ=-0.010 [-0.015, -0.004] (45.6% better place), n=3489
- Men std: time Δ=26.7s [21.0, 32.2] (57.4% faster); finish-pct Δ=-0.010 [-0.015, -0.004] (45.6% better place), n=3489
- Men Converted−Std time inflation: 21.2s [20.4, 22.0], Wilcoxon p=0.00e+00, n=3489

## Concordance (Spearman)

- Women raw_time_vs_raw_place: ρ=0.318 (p=1.08e-47), sign agree=0.626, n=1975
- Women conv_time_vs_conv_place: ρ=0.444 (p=4.21e-96), sign agree=0.635, n=1975
- Women std_time_vs_std_place: ρ=0.464 (p=6.77e-106), sign agree=0.665, n=1975
- Women raw_time_vs_std_time: ρ=0.649 (p=2.92e-236), sign agree=0.773, n=1975
- Women conv_time_vs_std_time: ρ=0.975 (p=0.00e+00), sign agree=0.907, n=1975
- Women raw_place_vs_std_place: ρ=1.000 (p=0.00e+00), sign agree=1.000, n=1975
- Women raw_place_vs_conv_place: ρ=1.000 (p=0.00e+00), sign agree=1.000, n=1975
- Men raw_time_vs_raw_place: ρ=0.407 (p=1.53e-139), sign agree=0.617, n=3489
- Men conv_time_vs_conv_place: ρ=0.480 (p=2.17e-200), sign agree=0.630, n=3489
- Men std_time_vs_std_place: ρ=0.497 (p=2.24e-217), sign agree=0.661, n=3489
- Men raw_time_vs_std_time: ρ=0.786 (p=0.00e+00), sign agree=0.861, n=3489
- Men conv_time_vs_std_time: ρ=0.963 (p=0.00e+00), sign agree=0.899, n=3489
- Men raw_place_vs_std_place: ρ=1.000 (p=0.00e+00), sign agree=0.999, n=3489
- Men raw_place_vs_conv_place: ρ=1.000 (p=0.00e+00), sign agree=0.999, n=3489

## Course factor α vs weather residual

- Women: Spearman(α, mean env residual)=-0.108 (p=1.35e-01), n_meets=191, sd(α)=0.0915
- Men: Spearman(α, mean env residual)=-0.184 (p=4.68e-03), n_meets=236, sd(α)=0.2094

## Field-adjusted (α·raw) vs Standardized improvement

- Women field_adj: 2.1s [-2.0, 6.2], %improved=52.2, n=1975
- Women standardized: 24.6s [19.6, 29.9], %improved=59.8, n=1975
- Women raw: -41.5s [-49.7, -32.9], %improved=52.0, n=1975
- Women converted: 39.9s [34.4, 45.2], %improved=67.3, n=1975
- Women field_adj−Std: -22.5s [-25.3, -19.8], Wilcoxon p=1.09e-42, Spearman ρ=0.709
- Men field_adj: 3.5s [-0.0, 7.0], %improved=50.8, n=3489
- Men standardized: 26.7s [20.8, 33.0], %improved=57.4, n=3489
- Men raw: -27.5s [-37.4, -18.3], %improved=63.3, n=3489
- Men converted: 47.9s [42.1, 54.1], %improved=65.5, n=3489
- Men field_adj−Std: -23.1s [-27.9, -18.7], Wilcoxon p=1.34e-51, Spearman ρ=0.772

## Interpretation notes

- If weather/elevation adjustments are mostly meet-level, within-meet
  place order is nearly identical for Raw / Converted / Standardized;
  relative-finish Δ then mainly reflects field composition change, not
  environmental standardization.
- Field-inferred α is a *complement* to NRCD Standardized: agreement
  with Standardized Δ supports robustness; disagreement highlights meets
  where field strength and weather residuals diverge.

## Head-to-head (see VERDICT.md)

- field_alpha: 7 criterion×gender wins
- standardized: 2 criterion×gender wins
- standardized_full: 1 criterion×gender wins

Full write-up: `VERDICT.md`.

