"""
Sensitivity analysis sweep for key modeling choices.

This script runs a small grid of "reasonable alternatives" on the saved
modeling table `output/rq1/raw_data_athlete_features.csv` and reports
how conclusions change (or do not change) under those alternatives.

Outputs are written under: output/rq1/sensitivity_sweep/
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_FEATURE_COLUMNS = [
    "gender_encoded",
    "year",
    "num_races",
    "season_duration",
    "first_time",
    "last_time",
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


BASE_FEATURES_NO_SQUARED = [c for c in DEFAULT_FEATURE_COLUMNS if not c.endswith("_squared")]
BASE_FEATURES_NO_LAST_TIME = [c for c in DEFAULT_FEATURE_COLUMNS if c != "last_time"]
BASE_FEATURES_NO_TIMES = [c for c in DEFAULT_FEATURE_COLUMNS if c not in {"first_time", "last_time", "best_time", "worst_time", "avg_time"}]


def _feature_columns_for_gender(
    feature_columns: Tuple[str, ...], gender_filter: Optional[str]
) -> Tuple[str, ...]:
    if gender_filter is None:
        return feature_columns
    g = str(gender_filter).upper()
    if g not in {"M", "F"}:
        return feature_columns
    return tuple(c for c in feature_columns if c not in {"gender_encoded", "gender_year"})


@dataclass(frozen=True)
class SplitSpec:
    name: str
    train_years: Tuple[int, ...]
    test_year: int


SPLITS: List[SplitSpec] = [
    SplitSpec(name="train_2023_test_2024", train_years=(2023,), test_year=2024),
    SplitSpec(name="train_2023_test_2025", train_years=(2023,), test_year=2025),
    SplitSpec(name="train_2023_2024_test_2025", train_years=(2023, 2024), test_year=2025),
]


@dataclass(frozen=True)
class Scenario:
    scenario: str
    feature_set: str
    feature_columns: Tuple[str, ...]
    improvement_rate_range: Tuple[float, float]
    model_n_estimators: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_features_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _prepare_work(
    df: pd.DataFrame,
    feature_columns: List[str],
    improvement_rate_range: Tuple[float, float],
    gender_filter: Optional[str] = None,
) -> pd.DataFrame:
    # Avoid duplicate column selection (e.g., 'year' is both a feature and metadata)
    needed = list(dict.fromkeys(["athlete_id", "gender", "year", "improvement_rate"] + list(feature_columns)))
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in features table: {missing}")

    work = df[needed].copy()
    if gender_filter is not None:
        g = str(gender_filter).upper()
        work = work.loc[work["gender"] == g].copy()
    mask = ~work[feature_columns + ["improvement_rate"]].isna().any(axis=1)
    lo, hi = improvement_rate_range
    mask &= (work["improvement_rate"] >= lo) & (work["improvement_rate"] <= hi)
    return work.loc[mask].copy()


def _fit_eval(
    work: pd.DataFrame,
    feature_columns: List[str],
    split: SplitSpec,
    n_estimators: int,
    random_state: int = 42,
) -> Dict[str, float]:
    train_mask = work["year"].isin(split.train_years)
    test_mask = work["year"] == split.test_year
    train = work.loc[train_mask]
    test = work.loc[test_mask]

    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"Split {split.name} has empty train/test (train={len(train)} test={len(test)})")

    X_train = train[feature_columns].to_numpy()
    y_train = train["improvement_rate"].to_numpy()
    X_test = test[feature_columns].to_numpy()
    y_test = test["improvement_rate"].to_numpy()

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))
    return {"r2": r2, "rmse": rmse, "mae": mae, "n_train": float(len(train)), "n_test": float(len(test))}


def _scenario_grid() -> List[Scenario]:
    return [
        Scenario(
            scenario="baseline",
            feature_set="all_features",
            feature_columns=tuple(DEFAULT_FEATURE_COLUMNS),
            improvement_rate_range=(-50, 50),
            model_n_estimators=100,
        ),
        Scenario(
            scenario="tighter_outlier_filter",
            feature_set="all_features",
            feature_columns=tuple(DEFAULT_FEATURE_COLUMNS),
            improvement_rate_range=(-20, 20),
            model_n_estimators=100,
        ),
        Scenario(
            scenario="wider_outlier_filter",
            feature_set="all_features",
            feature_columns=tuple(DEFAULT_FEATURE_COLUMNS),
            improvement_rate_range=(-100, 100),
            model_n_estimators=100,
        ),
        Scenario(
            scenario="remove_last_time",
            feature_set="no_last_time",
            feature_columns=tuple(BASE_FEATURES_NO_LAST_TIME),
            improvement_rate_range=(-50, 50),
            model_n_estimators=100,
        ),
        Scenario(
            scenario="remove_squared_terms",
            feature_set="no_squared_terms",
            feature_columns=tuple(BASE_FEATURES_NO_SQUARED),
            improvement_rate_range=(-50, 50),
            model_n_estimators=100,
        ),
        Scenario(
            scenario="remove_all_time_markers",
            feature_set="no_time_markers",
            feature_columns=tuple(BASE_FEATURES_NO_TIMES),
            improvement_rate_range=(-50, 50),
            model_n_estimators=100,
        ),
        Scenario(
            scenario="smaller_forest",
            feature_set="all_features",
            feature_columns=tuple(DEFAULT_FEATURE_COLUMNS),
            improvement_rate_range=(-50, 50),
            model_n_estimators=50,
        ),
        Scenario(
            scenario="larger_forest",
            feature_set="all_features",
            feature_columns=tuple(DEFAULT_FEATURE_COLUMNS),
            improvement_rate_range=(-50, 50),
            model_n_estimators=200,
        ),
    ]


def run_sweep(
    features_csv: str,
    out_dir: str,
    gender_filter: Optional[str] = None,
    file_suffix: str = "",
) -> pd.DataFrame:
    _ensure_dir(out_dir)
    df = _load_features_table(features_csv)

    scenarios: List[Scenario] = _scenario_grid()

    rows = []
    for sc in scenarios:
        feature_columns = list(_feature_columns_for_gender(sc.feature_columns, gender_filter))
        work = _prepare_work(
            df,
            feature_columns=feature_columns,
            improvement_rate_range=sc.improvement_rate_range,
            gender_filter=gender_filter,
        )
        for split in SPLITS:
            metrics = _fit_eval(
                work,
                feature_columns=feature_columns,
                split=split,
                n_estimators=sc.model_n_estimators,
            )
            rows.append(
                {
                    "scenario": sc.scenario,
                    "feature_set": sc.feature_set,
                    "improvement_rate_lo": sc.improvement_rate_range[0],
                    "improvement_rate_hi": sc.improvement_rate_range[1],
                    "n_estimators": sc.model_n_estimators,
                    "split": split.name,
                    **metrics,
                }
            )

    results = pd.DataFrame(rows)

    # Compute deltas relative to baseline, per split
    baseline = (
        results[results["scenario"] == "baseline"][["split", "r2", "rmse", "mae"]]
        .set_index("split")
        .rename(columns={"r2": "baseline_r2", "rmse": "baseline_rmse", "mae": "baseline_mae"})
    )
    results = results.join(baseline, on="split")
    results["delta_r2"] = results["r2"] - results["baseline_r2"]
    results["delta_rmse"] = results["rmse"] - results["baseline_rmse"]
    results["delta_mae"] = results["mae"] - results["baseline_mae"]

    results.to_csv(os.path.join(out_dir, f"sensitivity_sweep_results{file_suffix}.csv"), index=False)

    # Plot: ΔR² by scenario for primary split
    primary = results[results["split"] == "train_2023_test_2024"].copy()
    primary = primary.sort_values("delta_r2")
    gender_note = f" ({gender_filter} only)" if gender_filter else ""
    plt.figure(figsize=(10, 5))
    sns.barplot(data=primary, y="scenario", x="delta_r2", color="#72B7B2")
    plt.axvline(0, color="black", linewidth=1)
    plt.title(
        f"Sensitivity sweep: ΔR² vs baseline{gender_note}\n(Train 2023 → Test 2024, Random Forest)"
    )
    plt.xlabel("ΔR² vs baseline")
    plt.ylabel("Scenario")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sensitivity_sweep_delta_r2_train_2023_test_2024{file_suffix}.pdf"))
    plt.close()

    # Plot: R² across splits for each scenario
    plt.figure(figsize=(10, 5))
    sns.pointplot(data=results, x="split", y="r2", hue="scenario", dodge=0.5)
    plt.xticks(rotation=20, ha="right")
    plt.title(f"Sensitivity sweep: R² across temporal splits{gender_note}")
    plt.xlabel("Split")
    plt.ylabel("R²")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sensitivity_sweep_r2_across_splits{file_suffix}.pdf"))
    plt.close()

    with open(os.path.join(out_dir, f"README_sensitivity_sweep{file_suffix}.txt"), "w") as f:
        f.write("Sensitivity sweep\n")
        f.write("=================\n\n")
        f.write(f"Input table: {features_csv}\n")
        if gender_filter:
            f.write(f"Gender subset: {gender_filter} (gender_encoded / gender_year excluded)\n")
        f.write("Model: RandomForestRegressor with varying n_estimators (see CSV)\n")
        f.write("Splits: temporal (2023→2024, 2023→2025, 2023+2024→2025)\n\n")
        f.write("Outputs:\n")
        f.write(f"- sensitivity_sweep_results{file_suffix}.csv (all scenarios/splits + deltas vs baseline)\n")
        f.write(f"- sensitivity_sweep_delta_r2_train_2023_test_2024{file_suffix}.pdf\n")
        f.write(f"- sensitivity_sweep_r2_across_splits{file_suffix}.pdf\n")

    return results


def main(output_dir: str = "output/rq1") -> None:
    out_dir = os.path.join(output_dir, "sensitivity_sweep")
    _ensure_dir(out_dir)

    features_csv = os.path.join(output_dir, "raw_data_athlete_features.csv")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(
            f"Expected features table not found at {features_csv}. "
            "Run `python scripts/ml_improvement_prediction.py` (or `python scripts/rq1.py`) first."
        )

    for gender, suffix in [("M", "_men"), ("F", "_women")]:
        run_sweep(features_csv=features_csv, out_dir=out_dir, gender_filter=gender, file_suffix=suffix)
    print(f"\nSensitivity sweep complete (men/women). Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()

