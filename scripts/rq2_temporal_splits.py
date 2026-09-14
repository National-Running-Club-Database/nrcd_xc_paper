"""Consolidated, reproducible RQ2 multi-season temporal-split evaluation.

Purpose: resolve the inconsistency between the multi-season prediction figure
(which plotted the 2025 generalization/extended splits) and the main-text claim
(which reported only the 2023->2024 split). This script evaluates all six models
on all three temporal splits, separately by sex, with test n and 2,000-bootstrap
95% CIs (seed 42), so the figure and prose can be made consistent.

Splits:
  - train 2023 -> test 2024 (primary; reported in text)
  - train 2023 -> test 2025 (generalization)
  - train 2023+2024 -> test 2025 (extended)

Output: output/rq2/temporal_splits/
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from _setup_paths import setup_paths

setup_paths()

from ml_improvement_prediction import (
    BOOTSTRAP_REPLICATES,
    RANDOM_SEED,
    bootstrap_confidence_interval,
    calculate_athlete_features,
    create_advanced_features,
    load_raw_data,
    prepare_model_data,
)
from rq2 import filter_athletes_by_race_count_diff

SCALED = {"Linear Regression", "Ridge Regression", "Lasso Regression", "SVR"}


def _models() -> Dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        "SVR": SVR(kernel="rbf", C=1.0, gamma="scale"),
    }


def _pipeline(name: str, estimator) -> Pipeline:
    steps = [("model", estimator)]
    if name in SCALED:
        steps.insert(0, ("scaler", StandardScaler()))
    return Pipeline(steps)


def build_rq2_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = load_raw_data(mode="standardized")
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["year"] = df["start_date"].dt.year
    valid_ids, _ = filter_athletes_by_race_count_diff(df, max_diff=2)
    df = df[df["athlete_id"].isin(valid_ids)].copy()

    training_df = df[df["year"] == 2023].copy()
    per_year = []
    for year in (2023, 2024, 2025):
        year_data = df[df["year"] == year].copy()
        if len(year_data) == 0:
            continue
        feats = calculate_athlete_features(year_data, training_df=training_df)
        feats = feats[feats["athlete_id"].isin(valid_ids)].copy()
        per_year.append(feats)
    features_df = create_advanced_features(pd.concat(per_year, ignore_index=True))
    X, y, features_df_filtered = prepare_model_data(features_df)
    if "last_time" in X.columns:
        X = X.drop(columns=["last_time"])
    return X, y, features_df_filtered


SPLITS: List[Tuple[str, Tuple[int, ...], int]] = [
    ("train 2023 -> test 2024", (2023,), 2024),
    ("train 2023 -> test 2025", (2023,), 2025),
    ("train 2023+2024 -> test 2025", (2023, 2024), 2025),
]


def evaluate(X, y, features_df) -> pd.DataFrame:
    year = features_df["year"].to_numpy()
    gender = features_df["gender"].to_numpy()
    rows = []
    for split_name, train_years, test_year in SPLITS:
        train_mask = np.isin(year, train_years)
        for g, glabel in [("M", "Men"), ("F", "Women")]:
            test_mask = (year == test_year) & (gender == g)
            tr_mask = train_mask & (gender == g)
            if tr_mask.sum() < 30 or test_mask.sum() < 20:
                continue
            X_tr = X.iloc[tr_mask]
            y_tr = y.iloc[tr_mask]
            X_te = X.iloc[test_mask]
            y_te = y.iloc[test_mask]
            for name, estimator in _models().items():
                pipe = _pipeline(name, estimator)
                pipe.fit(X_tr, y_tr)
                pred = pipe.predict(X_te)
                r2 = r2_score(y_te, pred)
                _, lo, hi = bootstrap_confidence_interval(
                    y_te.to_numpy(), pred, r2_score
                )
                rows.append(
                    {
                        "split": split_name,
                        "gender": glabel,
                        "model": name,
                        "n_train": int(tr_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "test_r2": r2,
                        "r2_ci_low": lo,
                        "r2_ci_high": hi,
                        "test_mae": mean_absolute_error(y_te, pred),
                        "random_seed": RANDOM_SEED,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    }
                )
    return pd.DataFrame(rows)


def plot_r2(results: pd.DataFrame, out_path: str) -> None:
    splits = [s for s, _, _ in SPLITS]
    genders = ["Men", "Women"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    models = list(_models().keys())
    x = np.arange(len(models))
    width = 0.26
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for ax, g in zip(axes, genders):
        for j, split in enumerate(splits):
            sub = results[(results["gender"] == g) & (results["split"] == split)]
            vals = [
                sub.loc[sub["model"] == m, "test_r2"].mean() if (sub["model"] == m).any() else np.nan
                for m in models
            ]
            lo = [
                sub.loc[sub["model"] == m, "r2_ci_low"].mean() if (sub["model"] == m).any() else np.nan
                for m in models
            ]
            hi = [
                sub.loc[sub["model"] == m, "r2_ci_high"].mean() if (sub["model"] == m).any() else np.nan
                for m in models
            ]
            vals = np.array(vals, dtype=float)
            err_low = vals - np.array(lo, dtype=float)
            err_high = np.array(hi, dtype=float) - vals
            ax.bar(
                x + (j - 1) * width,
                vals,
                width,
                label=split,
                color=colors[j],
                yerr=[err_low, err_high],
                capsize=2,
                error_kw={"elinewidth": 0.8},
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(g)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Test $R^2$")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Multi-season held-out $R^2$ by temporal split (95% bootstrap CIs)"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(output_dir: str = "output/rq2") -> None:
    out_dir = os.path.join(output_dir, "temporal_splits")
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"RQ2 temporal splits: seed={RANDOM_SEED}, "
        f"bootstrap={BOOTSTRAP_REPLICATES}; SVR deterministic."
    )
    X, y, features_df = build_rq2_features()
    results = evaluate(X, y, features_df)
    results.to_csv(os.path.join(out_dir, "temporal_splits_r2.csv"), index=False)
    plot_r2(results, os.path.join(out_dir, "multi_season_temporal_splits.pdf"))
    with open(os.path.join(out_dir, "reproducibility.json"), "w") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "splits": [s for s, _, _ in SPLITS],
            },
            f,
            indent=2,
        )
    print(results.to_string(index=False))
    print(f"\nSaved RQ2 temporal-split diagnostics to {out_dir}/")


if __name__ == "__main__":
    main()
