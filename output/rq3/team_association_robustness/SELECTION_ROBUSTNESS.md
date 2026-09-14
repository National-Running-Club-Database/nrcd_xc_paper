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
| Δ n_athletes_ge3 | 328 | 0.0040 | 0.465 | 0.065 |
| Δ ERO | 328 | 0.0033 | 0.0884 | 0.133 |
| Δ n_athletes | 328 | 0.0068 | 0.101 | 0.095 |

## Within-team Δrank (among top-25 both years; lower = better)

| Predictor | n | OLS coef | p | Spearman ρ |
|-----------|---|----------|---|------------|
| Δ max_race_count | 78 | 0.7206 | 0.395 | 0.080 |
| Δ n_athletes_ge3 | 78 | -0.0330 | 0.814 | -0.114 |
| Δ ERO | 78 | -0.0135 | 0.769 | -0.107 |

## Roster-size-adjusted GEE (cluster-robust by team)

| Model | Term | OR/SD | 95% CI | p |
|-------|------|-------|--------|---|
| maxrace_only | z_maxrace | 2.574 | [1.790, 3.702] | 3.41e-07 |
| depth_only | z_depth | 2.665 | [1.870, 3.799] | 5.92e-08 |
| maxrace_plus_size | z_maxrace | 1.690 | [1.054, 2.711] | 0.0295 |
| maxrace_plus_size | z_n_athletes | 2.410 | [1.400, 4.146] | 0.00149 |
| depth_plus_maxrace_plus_size | z_maxrace | 1.656 | [1.070, 2.565] | 0.0236 |
| depth_plus_maxrace_plus_size | z_depth | 1.052 | [0.473, 2.340] | 0.901 |
| depth_plus_maxrace_plus_size | z_n_athletes | 2.325 | [0.960, 5.631] | 0.0616 |
| ERO_plus_size | z_ERO | 3.292 | [0.903, 11.999] | 0.071 |
| ERO_plus_size | z_n_athletes | 0.959 | [0.234, 3.938] | 0.954 |

## Non-top-7 (bench) depth vs roster size

Athletes outside the season-best top 7 with ≥3 starts.

| Gender | Top-15 mean beyond-7≥3 | Not | Top-15 size | Not | corr(beyond7, size) |
|--------|------------------------|-----|-------------|-----|---------------------|
| F | 5.7 | 1.3 | 25.0 | 12.0 | 0.812 |
| M | 11.6 | 3.0 | 40.3 | 17.8 | 0.854 |

| Model | Term | OR/SD | 95% CI | p |
|-------|------|-------|--------|---|
| beyond7_depth_only | z_n_ge3_beyond7 | 2.364 | [1.677, 3.333] | 9.11e-07 |
| beyond7_plus_size | z_n_ge3_beyond7 | 1.106 | [0.511, 2.393] | 0.799 |
| beyond7_plus_size | z_n_athletes | 2.687 | [0.980, 7.365] | 0.0547 |
| beyond7_plus_maxrace | z_n_ge3_beyond7 | 1.854 | [1.347, 2.551] | 0.00015 |
| beyond7_plus_maxrace | z_max_race_count | 1.772 | [1.274, 2.466] | 0.000678 |
| full | z_n_ge3_beyond7 | 0.881 | [0.424, 1.831] | 0.734 |
| full | z_n_ge3_top7 | 1.563 | [0.877, 2.787] | 0.13 |
| full | z_n_athletes | 2.492 | [0.942, 6.589] | 0.0657 |
| full | z_max_race_count | 1.271 | [0.699, 2.309] | 0.432 |
| size_only | z_n_athletes | 2.946 | [1.701, 5.102] | 0.000116 |
| overall_depth_only | z_n_athletes_ge3 | 2.665 | [1.870, 3.799] | 5.92e-08 |
| overall_depth_plus_size | z_n_athletes_ge3 | 1.423 | [0.642, 3.151] | 0.385 |
| overall_depth_plus_size | z_n_athletes | 2.180 | [0.840, 5.658] | 0.109 |

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
