"""Reproducible diagnostics for the RQ1 individual-prediction null result.

Runs gender-separated temporal learning curves (2023 -> 2024) for SVR,
Random Forest, and Gradient Boosting, plus outcome-outlier sensitivity under
four prespecified trimming rules. All stochastic operations use seed 42.

Outputs: output/rq1/prediction_diagnostics/
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from sensitivity_analysis_sweep import DEFAULT_FEATURE_COLUMNS
from feature_policy import features_for_gender_model


RANDOM_SEED = 42
LEARNING_CURVE_REPEATS = 20
TRAIN_FRACTIONS = (0.20, 0.40, 0.60, 0.80, 1.00)


def _models() -> Dict[str, object]:
    return {
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, gamma="scale")),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_SEED
        ),
    }


def _gender_features(gender: str) -> List[str]:
    del gender
    return features_for_gender_model(DEFAULT_FEATURE_COLUMNS)


def _complete_cases(df: pd.DataFrame, gender: str) -> pd.DataFrame:
    features = _gender_features(gender)
    needed = list(
        dict.fromkeys(["gender", "year", "improvement_rate"] + features)
    )
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in athlete feature table: {missing}")
    return (
        df.loc[df["gender"] == gender, needed]
        .dropna(subset=["improvement_rate"] + features)
        .copy()
    )


def _trim_bounds(
    train: pd.DataFrame, rule: str
) -> Tuple[float, float]:
    if rule == "absolute_20":
        return -20.0, 20.0
    if rule == "absolute_50_primary":
        return -50.0, 50.0
    if rule == "absolute_100":
        return -100.0, 100.0
    if rule == "training_iqr_1.5":
        q1, q3 = train["improvement_rate"].quantile([0.25, 0.75])
        iqr = q3 - q1
        return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
    raise ValueError(f"Unknown trimming rule: {rule}")


def run_learning_curves(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gender, label in [("M", "Men"), ("F", "Women")]:
        work = _complete_cases(df, gender)
        work = work.loc[work["improvement_rate"].between(-50, 50)].copy()
        train = work.loc[work["year"] == 2023]
        test = work.loc[work["year"] == 2024]
        features = _gender_features(gender)

        for fraction in TRAIN_FRACTIONS:
            n_train = max(50, int(round(len(train) * fraction)))
            n_train = min(n_train, len(train))
            repeats = 1 if fraction == 1.0 else LEARNING_CURVE_REPEATS
            for repeat in range(repeats):
                subset_seed = RANDOM_SEED + repeat
                subset = (
                    train
                    if n_train == len(train)
                    else train.sample(n=n_train, random_state=subset_seed)
                )
                for model_name, model in _models().items():
                    model.fit(subset[features], subset["improvement_rate"])
                    pred = model.predict(test[features])
                    rows.append(
                        {
                            "gender": label,
                            "model": model_name,
                            "train_fraction": fraction,
                            "n_train": n_train,
                            "n_test": len(test),
                            "repeat": repeat,
                            "subset_seed": subset_seed,
                            "model_seed": (
                                "deterministic"
                                if model_name == "SVR"
                                else RANDOM_SEED
                            ),
                            "test_r2": r2_score(test["improvement_rate"], pred),
                            "test_mae": mean_absolute_error(
                                test["improvement_rate"], pred
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def run_outlier_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = (
        "absolute_20",
        "absolute_50_primary",
        "absolute_100",
        "training_iqr_1.5",
    )
    for gender, label in [("M", "Men"), ("F", "Women")]:
        work = _complete_cases(df, gender)
        raw_train = work.loc[work["year"] == 2023]
        raw_test = work.loc[work["year"] == 2024]
        features = _gender_features(gender)
        for rule in rules:
            lo, hi = _trim_bounds(raw_train, rule)
            train = raw_train.loc[raw_train["improvement_rate"].between(lo, hi)]
            test = raw_test.loc[raw_test["improvement_rate"].between(lo, hi)]
            for model_name, model in _models().items():
                model.fit(train[features], train["improvement_rate"])
                pred = model.predict(test[features])
                rows.append(
                    {
                        "gender": label,
                        "trim_rule": rule,
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "model": model_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        "test_r2": r2_score(test["improvement_rate"], pred),
                        "test_mae": mean_absolute_error(
                            test["improvement_rate"], pred
                        ),
                        "random_seed": RANDOM_SEED,
                    }
                )
    return pd.DataFrame(rows)


def _plot_learning_curves(results: pd.DataFrame, out_path: str) -> None:
    summary = (
        results.groupby(["gender", "model", "train_fraction"], as_index=False)
        .agg(mean_r2=("test_r2", "mean"), sd_r2=("test_r2", "std"))
        .fillna({"sd_r2": 0.0})
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (gender, data) in zip(axes, summary.groupby("gender")):
        for model, model_data in data.groupby("model"):
            x = model_data["train_fraction"] * 100
            y = model_data["mean_r2"]
            sd = model_data["sd_r2"]
            ax.plot(x, y, marker="o", label=model)
            ax.fill_between(x, y - sd, y + sd, alpha=0.15)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_title(gender)
        ax.set_xlabel("2023 training data used (%)")
        ax.set_ylabel("2024 test $R^2$")
        ax.legend(fontsize=8)
    fig.suptitle("Temporal learning curves (mean ± 1 SD across fixed-seed subsets)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(output_dir: str = "output/rq1") -> None:
    out_dir = os.path.join(output_dir, "prediction_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    features_path = os.path.join(output_dir, "raw_data_athlete_features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Missing {features_path}; run ml_improvement_prediction.py first."
        )

    print(
        f"Prediction diagnostics: seed={RANDOM_SEED}, "
        f"learning-curve repeats={LEARNING_CURVE_REPEATS}; SVR deterministic."
    )
    df = pd.read_csv(features_path)
    learning = run_learning_curves(df)
    sensitivity = run_outlier_sensitivity(df)
    learning.to_csv(os.path.join(out_dir, "learning_curves.csv"), index=False)
    sensitivity.to_csv(
        os.path.join(out_dir, "outlier_sensitivity.csv"), index=False
    )
    _plot_learning_curves(
        learning, os.path.join(out_dir, "learning_curves.pdf")
    )

    summary = (
        learning.groupby(["gender", "model", "train_fraction"], as_index=False)
        .agg(
            mean_test_r2=("test_r2", "mean"),
            sd_test_r2=("test_r2", "std"),
            n_train=("n_train", "first"),
        )
        .fillna({"sd_test_r2": 0.0})
    )
    summary.to_csv(
        os.path.join(out_dir, "learning_curves_summary.csv"), index=False
    )
    with open(os.path.join(out_dir, "reproducibility.json"), "w") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "learning_curve_repeats": LEARNING_CURVE_REPEATS,
                "train_fractions": TRAIN_FRACTIONS,
                "random_forest_random_state": RANDOM_SEED,
                "gradient_boosting_random_state": RANDOM_SEED,
                "svr_randomness": "deterministic",
                "outlier_bounds": (
                    "IQR bounds estimated from 2023 training outcomes only"
                ),
            },
            f,
            indent=2,
        )
    print(f"Prediction diagnostics saved to {out_dir}/")


if __name__ == "__main__":
    main()
