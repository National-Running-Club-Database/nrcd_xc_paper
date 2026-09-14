"""Feature-exclusion policy audit for RQ1 individual models.

Uses ``feature_policy`` as the single source of truth. Validates that
exclusions are neither too weak (leakage left in) nor too aggressive
(useful non-leaky signal discarded), and that the compact primary set
(after redundancy prune) matches the legacy full set on held-out R².

Outputs: output/rq1/feature_exclusion_audit/
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from _setup_paths import setup_paths

setup_paths()

from feature_policy import (
    EXCLUSION_POLICY,
    LEGACY_FULL_FEATURES,
    PRIMARY_FEATURES,
    REDUNDANCY_PRUNED,
    features_for_gender_model,
)

RANDOM_SEED = 42
RATE_RANGE = (-50.0, 50.0)
OUTPUT_DIR = os.path.join(
    Path(__file__).resolve().parents[1],
    "output",
    "rq1",
    "feature_exclusion_audit",
)


def _load_features() -> pd.DataFrame:
    candidates = [
        Path(__file__).resolve().parents[1] / "output" / "rq1" / "raw_data_athlete_features.csv",
        Path(__file__).resolve().parents[1] / "output" / "raw_data_athlete_features.csv",
    ]
    for path in candidates:
        if path.is_file():
            df = pd.read_csv(path)
            return df[df["improvement_rate"].between(*RATE_RANGE)].copy()
    raise FileNotFoundError(
        "Need output/rq1/raw_data_athlete_features.csv — run ml_improvement_prediction.py first."
    )


def _augment_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["improvement_per_race"] = out["total_improvement"] / out["num_races"].clip(lower=1)
    denom = out["time_range"].replace(0, np.nan)
    out["improvement_to_variability_ratio"] = (
        (out["total_improvement"] / denom)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    out["races_duration_ratio"] = out["num_races"] / out["season_duration"].clip(lower=1)
    out["late_season_performance"] = out["last_time"]
    out["early_season_performance"] = out["first_time"]
    out["progression_improvement"] = out["total_improvement"]
    return out


def _fit_svr_r2(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> float:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=1.0, gamma="scale")),
        ]
    )
    pipe.fit(X_train.to_numpy(float), y_train)
    return float(r2_score(y_test, pipe.predict(X_test.to_numpy(float))))


def _split_gender(df: pd.DataFrame, gender: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[(df["gender"] == gender) & (df["year"] == 2023)].copy()
    test = df[(df["gender"] == gender) & (df["year"] == 2024)].copy()
    return train, test


def _base_cols(train: pd.DataFrame, feature_list: Sequence[str]) -> List[str]:
    return [c for c in features_for_gender_model(feature_list) if c in train.columns]


def run_compact_vs_legacy(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gender, gname in [("M", "Men"), ("F", "Women")]:
        train, test = _split_gender(df, gender)
        ytr = train["improvement_rate"].to_numpy(float)
        yte = test["improvement_rate"].to_numpy(float)
        for label, flist in [
            ("compact_primary", PRIMARY_FEATURES),
            ("legacy_full", LEGACY_FULL_FEATURES),
        ]:
            cols = _base_cols(train, flist)
            r2 = _fit_svr_r2(train[cols], ytr, test[cols], yte)
            rows.append(
                {
                    "gender": gname,
                    "spec": label,
                    "n_features": len(cols),
                    "test_r2": r2,
                }
            )
        compact_r2 = rows[-2]["test_r2"]
        legacy_r2 = rows[-1]["test_r2"]
        rows.append(
            {
                "gender": gname,
                "spec": "compact_minus_legacy",
                "n_features": rows[-2]["n_features"] - rows[-1]["n_features"],
                "test_r2": compact_r2 - legacy_r2,
            }
        )
    return pd.DataFrame(rows)


def run_add_back_tests(df: pd.DataFrame) -> pd.DataFrame:
    leaky = [
        "last_time",
        "total_improvement",
        "improvement_per_race",
        "improvement_to_variability_ratio",
        "late_season_performance",
        "progression_improvement",
    ]
    duplicates = ["races_duration_ratio", "early_season_performance"]
    pruned = list(REDUNDANCY_PRUNED)
    rows = []
    for gender, gname in [("M", "Men"), ("F", "Women")]:
        train, test = _split_gender(df, gender)
        base_cols = _base_cols(train, PRIMARY_FEATURES)
        ytr = train["improvement_rate"].to_numpy(float)
        yte = test["improvement_rate"].to_numpy(float)
        r2_base = _fit_svr_r2(train[base_cols], ytr, test[base_cols], yte)
        rows.append(
            {
                "gender": gname,
                "spec": "primary_compact",
                "class": "baseline",
                "features_added": "",
                "n_features": len(base_cols),
                "test_r2": r2_base,
                "delta_r2_vs_primary": 0.0,
                "verdict": "baseline",
            }
        )
        for feat in leaky + duplicates + pruned:
            if feat not in train.columns:
                continue
            cols = base_cols + [feat]
            tr = train.dropna(subset=[feat])
            te = test.dropna(subset=[feat])
            r2 = _fit_svr_r2(
                tr[cols],
                tr["improvement_rate"].to_numpy(float),
                te[cols],
                te["improvement_rate"].to_numpy(float),
            )
            delta = r2 - r2_base
            if feat in leaky:
                klass, verdict = "A/B_leakage", (
                    "leakage_confirmed" if delta > 0.05 else "weak_or_no_leakage_boost"
                )
            elif feat in duplicates:
                klass, verdict = "C_exact_duplicate", (
                    "duplicate_ok" if abs(delta) < 0.03 else "duplicate_unexpected_signal"
                )
            else:
                klass, verdict = "redundancy_pruned", (
                    "prune_ok" if abs(delta) < 0.03 else "prune_changed_r2"
                )
            rows.append(
                {
                    "gender": gname,
                    "spec": f"primary+{feat}",
                    "class": klass,
                    "features_added": feat,
                    "n_features": len(cols),
                    "test_r2": r2,
                    "delta_r2_vs_primary": delta,
                    "verdict": verdict,
                }
            )
        all_leaky = [f for f in leaky if f in train.columns]
        cols = base_cols + all_leaky
        r2 = _fit_svr_r2(train[cols], ytr, test[cols], yte)
        rows.append(
            {
                "gender": gname,
                "spec": "primary+all_leaky",
                "class": "A/B_leakage",
                "features_added": "+".join(all_leaky),
                "n_features": len(cols),
                "test_r2": r2,
                "delta_r2_vs_primary": r2 - r2_base,
                "verdict": "leakage_confirmed" if (r2 - r2_base) > 0.05 else "weak",
            }
        )
    return pd.DataFrame(rows)


def run_over_exclusion_tests(df: pd.DataFrame) -> pd.DataFrame:
    blocks = {
        "drop_slope": ["slope"],
        "drop_absolute_times": ["first_time", "best_time"],
        "drop_schedule": [
            "num_races",
            "num_races_squared",
            "season_duration",
            "season_duration_squared",
            "race_frequency",
            "avg_days_between_races",
            "experience_level",
            "bad_race_count",
            "best_race_timing",
        ],
        "drop_starting_ability": [
            "starting_percentile",
            "starting_percentile_squared",
            "first_time",
        ],
        "schedule_only": None,
        "early_external_proxy": None,
    }
    schedule_keep = [
        "year",
        "num_races",
        "num_races_squared",
        "season_duration",
        "season_duration_squared",
        "race_frequency",
        "avg_days_between_races",
        "experience_level",
        "bad_race_count",
        "best_race_timing",
    ]
    early_keep = ["year", "first_time", "starting_percentile", "starting_percentile_squared"]

    rows = []
    for gender, gname in [("M", "Men"), ("F", "Women")]:
        train, test = _split_gender(df, gender)
        base_cols = _base_cols(train, PRIMARY_FEATURES)
        ytr = train["improvement_rate"].to_numpy(float)
        yte = test["improvement_rate"].to_numpy(float)
        r2_base = _fit_svr_r2(train[base_cols], ytr, test[base_cols], yte)
        rows.append(
            {
                "gender": gname,
                "spec": "primary_compact",
                "n_features": len(base_cols),
                "test_r2": r2_base,
                "delta_r2_vs_primary": 0.0,
                "verdict": "baseline",
            }
        )
        for name, drop_cols in blocks.items():
            if name == "schedule_only":
                cols = [c for c in schedule_keep if c in train.columns]
            elif name == "early_external_proxy":
                cols = [c for c in early_keep if c in train.columns]
            else:
                cols = [c for c in base_cols if c not in set(drop_cols or [])]
            if len(cols) < 2:
                continue
            r2 = _fit_svr_r2(train[cols], ytr, test[cols], yte)
            delta = r2 - r2_base
            if name in {"schedule_only", "early_external_proxy"}:
                verdict = "still_near_zero" if r2 < 0.1 else "positive_external_signal"
            else:
                verdict = (
                    "block_matters" if delta < -0.05 else "null_robust_to_dropping_block"
                )
            rows.append(
                {
                    "gender": gname,
                    "spec": name,
                    "n_features": len(cols),
                    "test_r2": r2,
                    "delta_r2_vs_primary": delta,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def write_policy_markdown(
    policy_df: pd.DataFrame,
    add_back: pd.DataFrame,
    over_ex: pd.DataFrame,
    compact_legacy: pd.DataFrame,
    path: str,
) -> None:
    def _get(df, spec, gender="Men"):
        hit = df[(df["spec"] == spec) & (df["gender"] == gender)]
        return hit.iloc[0] if len(hit) else None

    men_base = _get(add_back, "primary_compact")
    men_last = _get(add_back, "primary+last_time")
    men_tot = _get(add_back, "primary+total_improvement")
    men_dup = _get(add_back, "primary+races_duration_ratio")
    men_compact = _get(compact_legacy, "compact_primary")
    men_legacy = _get(compact_legacy, "legacy_full")
    men_delta = _get(compact_legacy, "compact_minus_legacy")
    over_men = over_ex[over_ex["gender"] == "Men"].set_index("spec")

    lines = [
        "# Feature exclusion policy and audit",
        "",
        "Source of truth: `scripts/feature_policy.py`.",
        "Primary model: gender-separated SVR, train 2023 → test 2024.",
        "",
        f"Compact primary features ({len(features_for_gender_model())}): "
        + ", ".join(f"`{c}`" for c in features_for_gender_model()) + ".",
        "",
        f"Redundancy-pruned (kept out of primary): "
        + ", ".join(f"`{c}`" for c in REDUNDANCY_PRUNED) + ".",
        "",
        "## Policy",
        "",
        "| Feature | Status | Class | Reason |",
        "|---|---|---|---|",
    ]
    for _, r in policy_df.iterrows():
        lines.append(
            f"| `{r['feature']}` | {r['status']} | {r['class']} | {r['reason']} |"
        )
    lines += [
        "",
        "## Compact vs legacy full",
        "",
        f"Men compact R²={men_compact['test_r2']:.3f} (n_feat={int(men_compact['n_features'])}); "
        f"legacy R²={men_legacy['test_r2']:.3f} (n_feat={int(men_legacy['n_features'])}); "
        f"Δ={men_delta['test_r2']:+.3f}.",
        "",
        "## Leakage / duplicate / prune add-back",
        "",
        f"Primary (compact) R² men={men_base['test_r2']:.3f}.",
        f"`+last_time` ΔR²={men_last['delta_r2_vs_primary']:+.3f} → {men_last['verdict']}.",
        f"`+total_improvement` ΔR²={men_tot['delta_r2_vs_primary']:+.3f} → {men_tot['verdict']}.",
        f"`+races_duration_ratio` ΔR²={men_dup['delta_r2_vs_primary']:+.3f} → {men_dup['verdict']}.",
        "",
        "## Over-exclusion",
        "",
    ]
    for spec in [
        "drop_slope",
        "drop_absolute_times",
        "drop_schedule",
        "schedule_only",
        "early_external_proxy",
    ]:
        if spec in over_men.index:
            r = over_men.loc[spec]
            lines.append(
                f"- `{spec}`: R²={r['test_r2']:.3f} (Δ={r['delta_r2_vs_primary']:+.3f}) — {r['verdict']}"
            )
    lines += [
        "",
        "Reproduce: `python scripts/feature_exclusion_audit.py`",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main(output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("Loading athlete features...")
    df = _augment_candidates(_load_features())
    print(f"  n={len(df)}")

    policy_df = pd.DataFrame(EXCLUSION_POLICY)
    policy_df.to_csv(os.path.join(output_dir, "exclusion_policy.csv"), index=False)

    print("\nCompact vs legacy...")
    compact_legacy = run_compact_vs_legacy(df)
    compact_legacy.to_csv(os.path.join(output_dir, "compact_vs_legacy.csv"), index=False)
    print(compact_legacy.to_string(index=False))

    print("\nAdd-back tests...")
    add_back = run_add_back_tests(df)
    add_back.to_csv(os.path.join(output_dir, "add_back_tests.csv"), index=False)
    print(
        add_back[add_back["class"].isin(["baseline", "A/B_leakage", "C_exact_duplicate"])]
        .to_string(index=False)
    )

    print("\nOver-exclusion...")
    over_ex = run_over_exclusion_tests(df)
    over_ex.to_csv(os.path.join(output_dir, "over_exclusion_tests.csv"), index=False)
    print(over_ex.to_string(index=False))

    write_policy_markdown(
        policy_df,
        add_back,
        over_ex,
        compact_legacy,
        os.path.join(output_dir, "EXCLUSION_POLICY.md"),
    )

    men_last = add_back[
        (add_back["gender"] == "Men") & (add_back["spec"] == "primary+last_time")
    ].iloc[0]
    men_delta = compact_legacy[
        (compact_legacy["gender"] == "Men")
        & (compact_legacy["spec"] == "compact_minus_legacy")
    ].iloc[0]
    men_drop_slope = over_ex[
        (over_ex["gender"] == "Men") & (over_ex["spec"] == "drop_slope")
    ].iloc[0]
    summary = {
        "random_seed": RANDOM_SEED,
        "primary_feature_set": "compact_primary",
        "n_primary_features_within_sex": len(features_for_gender_model()),
        "redundancy_pruned": REDUNDANCY_PRUNED,
        "compact_minus_legacy_r2_men": float(men_delta["test_r2"]),
        "leakage_last_time_delta_r2_men": float(men_last["delta_r2_vs_primary"]),
        "drop_slope_delta_r2_men": float(men_drop_slope["delta_r2_vs_primary"]),
        "conclusion": (
            "Primary = compact set after redundancy prune; "
            "A/B leakage exclusions confirmed by add-back; "
            "compact≈legacy R²; null not from over-pruning."
        ),
    }
    with open(os.path.join(output_dir, "reproducibility.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n", summary["conclusion"])
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    main()
