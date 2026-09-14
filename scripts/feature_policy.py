"""Single source of truth for RQ1/RQ2 feature inclusion / exclusion.

Policy classes (one per feature — no double-counting)
------------------------------------------------------
A  endpoint leakage   — algebraic constituent of y = (t_L - t_1) / Δ
B  derived leakage    — transform of that constituent
C  exact duplicate    — identical to a retained column (non-leaky)
D  retained           — pre-final / schedule; may be non-external
E  within-sex drop    — constant after gender-stratified fits

Primary modeling uses ``PRIMARY_FEATURES`` (compact): one representative per
near-duplicate cluster (|corr| ≥ 0.95 in the exclusion audit). The fuller
``LEGACY_FULL_FEATURES`` list is retained only for sensitivity / ablation
comparisons.

Reproduce tests: ``python scripts/feature_exclusion_audit.py``
"""

from __future__ import annotations

from typing import Dict, List, Sequence

# ---------------------------------------------------------------------------
# Exclusion registry (documented in paper § Feature exclusion policy)
# ---------------------------------------------------------------------------

EXCLUSION_POLICY: List[Dict[str, str]] = [
    {
        "feature": "last_time",
        "status": "excluded",
        "class": "A_endpoint_leakage",
        "reason": "Endpoint of y=(last-first)/duration.",
    },
    {
        "feature": "total_improvement",
        "status": "excluded",
        "class": "A_endpoint_leakage",
        "reason": "Numerator of y.",
    },
    {
        "feature": "late_season_performance",
        "status": "excluded",
        "class": "A_endpoint_leakage",
        "reason": "Alias of last_time (counted as A, not also C).",
    },
    {
        "feature": "progression_improvement",
        "status": "excluded",
        "class": "A_endpoint_leakage",
        "reason": "Alias of total_improvement (counted as A, not also C).",
    },
    {
        "feature": "improvement_per_race",
        "status": "excluded",
        "class": "B_derived_leakage",
        "reason": "total_improvement / num_races.",
    },
    {
        "feature": "improvement_to_variability_ratio",
        "status": "excluded",
        "class": "B_derived_leakage",
        "reason": "Uses total_improvement in the numerator.",
    },
    {
        "feature": "early_season_performance",
        "status": "excluded",
        "class": "C_exact_duplicate",
        "reason": "Alias of retained first_time.",
    },
    {
        "feature": "races_duration_ratio",
        "status": "excluded",
        "class": "C_exact_duplicate",
        "reason": "Identical to retained race_frequency.",
    },
    {
        "feature": "gender_encoded",
        "status": "dropped_within_sex",
        "class": "E_within_sex_constant",
        "reason": "Constant after sex-stratified fitting.",
    },
    {
        "feature": "gender_year",
        "status": "dropped_within_sex",
        "class": "E_within_sex_constant",
        "reason": "Constant after sex-stratified fitting.",
    },
]

# ---------------------------------------------------------------------------
# Compact primary set (post-audit)
# ---------------------------------------------------------------------------
# Dropped from the legacy full set as near-duplicates (|corr|≥0.95), keeping
# one representative per cluster:
#   worst_time, avg_time          → covered by first_time / best_time
#   time_std, time_range,
#   best_to_avg_ratio, worst_to_avg_ratio,
#   variability_score, consistency_score → covered by cv_time
#   best_race_timing_ratio        → covered by best_race_timing
#
# Squared terms kept (nonlinear schedule / starting-ability shapes).

PRIMARY_FEATURES: List[str] = [
    "gender_encoded",
    "year",
    "num_races",
    "num_races_squared",
    "season_duration",
    "season_duration_squared",
    "first_time",
    "best_time",
    "cv_time",
    "race_frequency",
    "starting_percentile",
    "starting_percentile_squared",
    "gender_year",
    "experience_level",
    "slope",
    "avg_days_between_races",
    "race_to_race_improvement_std",
    "best_race_timing",
    "bad_race_count",
]

# Pre-compact full list (sensitivity / historical ablations only)
LEGACY_FULL_FEATURES: List[str] = [
    "gender_encoded",
    "year",
    "num_races",
    "season_duration",
    "first_time",
    "best_time",
    "worst_time",
    "avg_time",
    "time_std",
    "time_range",
    "cv_time",
    "race_frequency",
    "starting_percentile",
    "gender_year",
    "starting_percentile_squared",
    "num_races_squared",
    "season_duration_squared",
    "best_to_avg_ratio",
    "worst_to_avg_ratio",
    "variability_score",
    "consistency_score",
    "experience_level",
    "slope",
    "avg_days_between_races",
    "race_to_race_improvement_std",
    "best_race_timing",
    "best_race_timing_ratio",
    "bad_race_count",
]

# Back-compat alias used by older scripts
DEFAULT_FEATURE_COLUMNS = PRIMARY_FEATURES

WITHIN_SEX_DROP = {"gender_encoded", "gender_year"}

REDUNDANCY_PRUNED = [
    "worst_time",
    "avg_time",
    "time_std",
    "time_range",
    "best_to_avg_ratio",
    "worst_to_avg_ratio",
    "variability_score",
    "consistency_score",
    "best_race_timing_ratio",
]


def features_for_gender_model(feature_columns: Sequence[str] | None = None) -> List[str]:
    """Drop within-sex constants from a feature list."""
    cols = list(feature_columns or PRIMARY_FEATURES)
    return [c for c in cols if c not in WITHIN_SEX_DROP]


def exclusion_policy_frame():
    """Return policy as a list of dicts (callers may wrap in DataFrame)."""
    return [dict(r) for r in EXCLUSION_POLICY]
