# Selection robustness for team race-frequency → top-15

Cross-sectional team associations (pooled RR, GEE OR, ERO AUC) can
reflect stable program quality rather than schedule choices. These
checks ask whether **within-team year-to-year changes** in racing
volume/depth still track placement, and whether controlling for
**roster size** absorbs the association.

Within-team consecutive pairs: **328**.

## Within-team first differences (Δtop15)

| Predictor | n | OLS coef | p | Spearman ρ |
|-----------|---|----------|---|------------|
| Δ max_race_count | 328 | -0.0087 | 0.613 | -0.036 |
| Δ n_athletes_ge3 | 328 | 0.0042 | 0.445 | 0.063 |
| Δ ERO | 328 | 0.0033 | 0.0903 | 0.132 |
| Δ n_athletes | 328 | 0.0069 | 0.0982 | 0.099 |

## Within-team Δrank (among top-25 both years; lower = better)

| Predictor | n | OLS coef | p | Spearman ρ |
|-----------|---|----------|---|------------|
| Δ max_race_count | 78 | 0.7206 | 0.395 | 0.080 |
| Δ n_athletes_ge3 | 78 | -0.0400 | 0.776 | -0.116 |
| Δ ERO | 78 | -0.0136 | 0.768 | -0.109 |

## Roster-size-adjusted GEE (cluster-robust by team)

| Model | Term | OR/SD | 95% CI | p |
|-------|------|-------|--------|---|
| maxrace_only | z_maxrace | 2.574 | [1.790, 3.702] | 3.41e-07 |
| depth_only | z_depth | 2.666 | [1.872, 3.797] | 5.41e-08 |
| maxrace_plus_size | z_maxrace | 1.689 | [1.052, 2.710] | 0.0299 |
| maxrace_plus_size | z_n_athletes | 2.419 | [1.404, 4.168] | 0.00146 |
| depth_plus_maxrace_plus_size | z_maxrace | 1.661 | [1.072, 2.572] | 0.0231 |
| depth_plus_maxrace_plus_size | z_depth | 1.043 | [0.468, 2.325] | 0.918 |
| depth_plus_maxrace_plus_size | z_n_athletes | 2.349 | [0.967, 5.707] | 0.0595 |
| ERO_plus_size | z_ERO | 3.155 | [0.866, 11.498] | 0.0817 |
| ERO_plus_size | z_n_athletes | 1.002 | [0.243, 4.135] | 0.998 |

## Non-top-7 (bench) depth vs roster size

Athletes outside the season-best top 7 with ≥3 starts.

| Gender | Top-15 mean beyond-7≥3 | Not | Top-15 size | Not | corr(beyond7, size) |
|--------|------------------------|-----|-------------|-----|---------------------|
| F | 5.7 | 1.3 | 25.0 | 12.0 | 0.813 |
| M | 11.6 | 3.0 | 40.2 | 17.7 | 0.854 |

| Model | Term | OR/SD | 95% CI | p |
|-------|------|-------|--------|---|
| beyond7_depth_only | z_n_ge3_beyond7 | 2.368 | [1.681, 3.334] | 7.98e-07 |
| beyond7_plus_size | z_n_ge3_beyond7 | 1.102 | [0.507, 2.398] | 0.806 |
| beyond7_plus_size | z_n_athletes | 2.703 | [0.979, 7.466] | 0.055 |
| beyond7_plus_maxrace | z_n_ge3_beyond7 | 1.857 | [1.350, 2.556] | 0.000144 |
| beyond7_plus_maxrace | z_max_race_count | 1.768 | [1.269, 2.463] | 0.00075 |
| full | z_n_ge3_beyond7 | 0.876 | [0.418, 1.834] | 0.725 |
| full | z_n_ge3_top7 | 1.539 | [0.872, 2.716] | 0.137 |
| full | z_n_athletes | 2.521 | [0.947, 6.714] | 0.0642 |
| full | z_max_race_count | 1.289 | [0.716, 2.321] | 0.397 |
| size_only | z_n_athletes | 2.956 | [1.704, 5.127] | 0.000115 |
| overall_depth_only | z_n_athletes_ge3 | 2.666 | [1.872, 3.797] | 5.41e-08 |
| overall_depth_plus_size | z_n_athletes_ge3 | 1.414 | [0.637, 3.136] | 0.395 |
| overall_depth_plus_size | z_n_athletes | 2.200 | [0.844, 5.731] | 0.107 |

## Interpretation (associational)

If within-team Δvolume/Δdepth coefficients are near zero while
cross-sectional ORs remain large, treat the primary team finding as
**between-program selection** rather than evidence that changing a
given team's schedule moves nationals placement. Roster-size
adjustment speaks to whether depth/volume proxies for program size.
Non-top-7 depth likewise collapses once size is controlled — it is
largely a club-size channel, not an independent schedule effect.

Artifacts: `within_team_first_diff.csv`,
`within_team_first_diff_summary.csv`,
`roster_size_adjusted_models.csv`,
`beyond_top7_depth_descriptives.csv`,
`beyond_top7_depth_models.csv`.
