# Reproducibility checklist

Map of manuscript headlines → scripts → artifacts. Canonical prose summary:
[`findings.md`](findings.md). Do **not** cite
[`output/FINDINGS_EXPLANATION.md`](output/FINDINGS_EXPLANATION.md) (legacy leakage-era).

## Setup

```bash
git submodule update --init --recursive
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py          # full stack (slow)
# or targeted:
python scripts/mathematical_contributions.py
python scripts/paper_enrichment_analyses.py
python scripts/relative_finish_course_factors.py
python scripts/team_association_robustness.py
python scripts/robustness_checks.py
bash papers/sync_figures.sh        # copy PDFs into papers/figures/
```

Standardized frames are memoized under `output/.cache/` (gitignored). Set
`NRCD_XC_DISABLE_CACHE=1` for a cold recompute.

## Claim → artifact

| Claim | Script | Artifact |
|-------|--------|----------|
| Compact null R² (SVR 0.044 / −0.018) | `compact_model_suite.py` via `rq1.py` | `output/rq1/feature_exclusion_audit/compact_six_models.csv` |
| SER ≈ 0.19 (men) | `mathematical_contributions.py` | `output/rq1/mathematical_contributions/signal_extraction_ratio.csv`, `FORMULAS.md` |
| Weather inflation identity | `mathematical_contributions.py` | `weather_path_identity*.csv`, `weather_path_identity.pdf` |
| ERO AUC 0.84 vs max-race 0.74 | `mathematical_contributions.py` | `ero_out_of_year_auc.csv`, `ero_vs_maxrace.pdf` |
| Team pooled RR ≈ 2.09; GEE OR ≈ 2.56 | `team_association_robustness.py` | `output/rq3/team_association_robustness/threshold_pooled_rr.csv`, `continuous_race_count_model.csv` |
| Roster depth OR ≈ 1.74 | `underexplored_mechanisms.py` | `output/rq1/underexplored_mechanisms/roster_depth_gee_models.csv` |
| Within-team Δschedule ↛ Δtop15; size attenuates OR | `team_association_robustness.py` | `within_team_first_diff_summary.csv`, `roster_size_adjusted_models.csv`, `SELECTION_ROBUSTNESS.md` |
| Ability-matched 4+ vs 2 volume | `robustness_checks.py` | `output/rq1/robustness_checks/ability_matched_summary.csv` |
| Team size/travel confounders | `robustness_checks.py` | `team_confounder_models.csv` |
| Full-sample reliability / SER bounds | `robustness_checks.py` | `full_sample_reliability.csv` |
| Weather residual year-holdout | `robustness_checks.py` | `weather_holdout.csv` |
| Why not historical experience | `robustness_checks.py` | `WHY_NOT_HISTORICAL_EXPERIENCE.md` |
| Early-window coach estimand | `paper_enrichment_analyses.py` | `early_window_decision.csv`, `early_window_decision.pdf` |
| Relative finish / club course α (LACCTiC-motivated) | `relative_finish_course_factors.py` | `output/rq1/relative_finish_course_factors/` (`SUMMARY.md`, `VERDICT.md`, `next_race_prediction_*.csv`, `field_adjusted_vs_standardized.*`) |
| Sample sizes | `sample_summary` via `run_all.py` | `output/sample_summary.md` |

## Figure sync

`papers/sync_figures.sh` copies regenerated PDFs into `papers/figures/` for
Overleaf / local LaTeX builds (including weather-path, ERO, and early-window
decision figures).

## Follow-ups (not yet automated)

- Unit tests for feature policy / standardization cache keys.
- CI that asserts compact R² and SER CSVs match manuscript constants.
