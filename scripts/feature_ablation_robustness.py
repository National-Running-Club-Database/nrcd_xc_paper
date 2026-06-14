"""
Feature ablation / feature removal robustness checks.

This script is designed to be reproducible without the raw dataset by
operating on the saved modeling table produced by `ml_improvement_prediction.py`:
`output/rq1/raw_data_athlete_features.csv`.

Outputs are written under: output/rq1/robustness_feature_ablation/
(file suffix _men / _women only; pooled models omitted.)
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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_features_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _feature_columns_for_gender(
    feature_columns: List[str], gender_filter: Optional[str]
) -> List[str]:
    """Drop gender dummy features when fitting a single-gender subset."""
    if gender_filter is None:
        return list(feature_columns)
    g = str(gender_filter).upper()
    if g not in {"M", "F"}:
        return list(feature_columns)
    return [c for c in feature_columns if c not in {"gender_encoded", "gender_year"}]


def _prepare_xy(
    df: pd.DataFrame,
    feature_columns: List[str],
    improvement_rate_range: Tuple[float, float] = (-50, 50),
    gender_filter: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    # Avoid duplicate column selection (e.g., 'year' is both a feature and metadata)
    cols = list(dict.fromkeys(["athlete_id", "gender", "year", "improvement_rate"] + list(feature_columns)))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in features table: {missing}")

    work = df[cols].copy()
    if gender_filter is not None:
        g = str(gender_filter).upper()
        work = work.loc[work["gender"] == g].copy()
    mask = ~work[feature_columns + ["improvement_rate"]].isna().any(axis=1)
    lo, hi = improvement_rate_range
    mask &= (work["improvement_rate"] >= lo) & (work["improvement_rate"] <= hi)
    work = work.loc[mask].copy()

    X = work[feature_columns].copy()
    y = work["improvement_rate"].copy()
    return X, y, work


def _fit_eval(
    work: pd.DataFrame,
    feature_columns: List[str],
    split: SplitSpec,
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

    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    return {"r2": r2, "rmse": rmse, "mae": mae, "n_train": float(len(train)), "n_test": float(len(test))}


def _read_feature_importance(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        return None
    return df


def run_feature_ablation(
    features_csv: str,
    out_dir: str,
    feature_columns: Optional[List[str]] = None,
    top_k_groups: Tuple[int, ...] = (3, 5, 10),
    gender_filter: Optional[str] = None,
    file_suffix: str = "",
) -> None:
    _ensure_dir(out_dir)
    base_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)
    feature_columns = _feature_columns_for_gender(base_columns, gender_filter)

    df = _load_features_table(features_csv)
    _, _, work = _prepare_xy(
        df, feature_columns=feature_columns, gender_filter=gender_filter
    )

    importance_path = os.path.join(os.path.dirname(os.path.dirname(out_dir)), "raw_data_feature_importance.csv")
    # out_dir is output/rq1/robustness_feature_ablation/, so go up one level to output/rq1/
    importance_path = os.path.join(os.path.dirname(out_dir), "raw_data_feature_importance.csv")
    importance_df = _read_feature_importance(importance_path)

    rows = []

    # Baseline for each split
    for split in SPLITS:
        metrics = _fit_eval(work, feature_columns, split)
        rows.append(
            {
                "split": split.name,
                "ablation_type": "baseline",
                "removed": "",
                **metrics,
            }
        )

    # Single-feature ablations (train 2023 -> test 2024 is primary; we do all three splits to show robustness)
    for feat in feature_columns:
        cols = [c for c in feature_columns if c != feat]
        for split in SPLITS:
            metrics = _fit_eval(work, cols, split)
            rows.append(
                {
                    "split": split.name,
                    "ablation_type": "single_feature",
                    "removed": feat,
                    **metrics,
                }
            )

    # Group ablations based on feature importance rankings, if available
    if importance_df is not None:
        ranked = [f for f in importance_df["feature"].tolist() if f in feature_columns]
        for k in top_k_groups:
            removed = ranked[:k]
            cols = [c for c in feature_columns if c not in removed]
            for split in SPLITS:
                metrics = _fit_eval(work, cols, split)
                rows.append(
                    {
                        "split": split.name,
                        "ablation_type": "drop_topk_by_importance",
                        "removed": ",".join(removed),
                        "k": k,
                        **metrics,
                    }
                )

    results = pd.DataFrame(rows)

    # Add deltas relative to baseline per split
    baseline = (
        results[results["ablation_type"] == "baseline"][["split", "r2", "rmse", "mae"]]
        .set_index("split")
        .rename(columns={"r2": "baseline_r2", "rmse": "baseline_rmse", "mae": "baseline_mae"})
    )
    results = results.join(baseline, on="split")
    results["delta_r2"] = results["r2"] - results["baseline_r2"]
    results["delta_rmse"] = results["rmse"] - results["baseline_rmse"]
    results["delta_mae"] = results["mae"] - results["baseline_mae"]

    results_path = os.path.join(out_dir, f"feature_ablation_results{file_suffix}.csv")
    results.to_csv(results_path, index=False)

    # Plot: single-feature deltas on primary split
    primary = results[(results["ablation_type"] == "single_feature") & (results["split"] == "train_2023_test_2024")].copy()
    primary = primary.sort_values("delta_r2")

    gender_note = f" ({gender_filter} only)" if gender_filter else ""
    plt.figure(figsize=(10, max(6, 0.28 * len(primary))))
    sns.barplot(data=primary, y="removed", x="delta_r2", color="#4C78A8")
    plt.axvline(0, color="black", linewidth=1)
    plt.title(
        f"Feature ablation: ΔR² after removing one feature{gender_note}\n"
        "(Train 2023 → Test 2024, Random Forest)"
    )
    plt.xlabel("ΔR² vs baseline (negative = worse)")
    plt.ylabel("Removed feature")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"feature_ablation_delta_r2_train_2023_test_2024{file_suffix}.pdf")
    )
    plt.close()

    # Plot: highlight top-k group drops on primary split
    if "k" in results.columns:
        topk = results[(results["ablation_type"] == "drop_topk_by_importance") & (results["split"] == "train_2023_test_2024")].copy()
        if len(topk) > 0:
            topk = topk.sort_values("k")
            plt.figure(figsize=(7, 4))
            sns.lineplot(data=topk, x="k", y="r2", marker="o")
            base_r2 = float(baseline.loc["train_2023_test_2024", "baseline_r2"])
            plt.axhline(base_r2, color="black", linestyle="--", linewidth=1, label="baseline")
            plt.title(
                f"Drop top-k important features: R²{gender_note}\n"
                "(Train 2023 → Test 2024, Random Forest)"
            )
            plt.xlabel("k (number of top-importance features removed)")
            plt.ylabel("R²")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_dir, f"feature_ablation_drop_topk_r2_train_2023_test_2024{file_suffix}.pdf")
            )
            plt.close()

    # Short text summary
    with open(os.path.join(out_dir, f"README_feature_ablation{file_suffix}.txt"), "w") as f:
        f.write("Feature ablation robustness checks\n")
        f.write("=================================\n\n")
        f.write(f"Input table: {features_csv}\n")
        if gender_filter:
            f.write(f"Gender subset: {gender_filter} (gender_encoded / gender_year excluded)\n")
        f.write("Model: RandomForestRegressor(n_estimators=100, random_state=42)\n")
        f.write("Primary split: train 2023, test 2024\n\n")

        base = results[(results["ablation_type"] == "baseline") & (results["split"] == "train_2023_test_2024")].iloc[0]
        f.write(f"Baseline (train 2023 → test 2024): R²={base['r2']:.4f}, RMSE={base['rmse']:.4f}, MAE={base['mae']:.4f}\n\n")

        worst = primary.head(10)[["removed", "delta_r2", "r2", "rmse", "mae"]]
        f.write("Largest performance drops (single-feature removal; sorted by ΔR² ascending):\n")
        f.write(worst.to_string(index=False))
        f.write("\n")


def main(output_dir: str = "output/rq1") -> None:
    out_dir = os.path.join(output_dir, "robustness_feature_ablation")
    _ensure_dir(out_dir)

    features_csv = os.path.join(output_dir, "raw_data_athlete_features.csv")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(
            f"Expected features table not found at {features_csv}. "
            "Run `python scripts/ml_improvement_prediction.py` (or `python scripts/rq1.py`) first."
        )

    # Men and women only (no pooled run): avoids encoding issues in combined models.
    for gender, suffix in [("M", "_men"), ("F", "_women")]:
        run_feature_ablation(
            features_csv=features_csv,
            out_dir=out_dir,
            gender_filter=gender,
            file_suffix=suffix,
        )
    print(f"\nFeature ablation robustness complete (men/women). Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()

