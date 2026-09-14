"""Re-evaluate RQ1 models and key appendix diagnostics on the compact primary set.

Writes:
  output/rq1/feature_exclusion_audit/compact_six_models.csv
  output/rq1/feature_exclusion_audit/compact_learning_curves.csv
  output/rq1/feature_exclusion_audit/compact_outlier_sensitivity.csv
  output/rq1/feature_exclusion_audit/compact_permutation_r2.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path

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

from feature_policy import PRIMARY_FEATURES, features_for_gender_model

RANDOM_SEED = 42
BOOTSTRAP = 2000
PERMUTATIONS = 1000
RATE_RANGE = (-50.0, 50.0)
OUT = Path(__file__).resolve().parents[1] / "output" / "rq1" / "feature_exclusion_audit"


def _load():
    path = Path(__file__).resolve().parents[1] / "output" / "rq1" / "raw_data_athlete_features.csv"
    df = pd.read_csv(path)
    return df[df["improvement_rate"].between(*RATE_RANGE)].copy()


def _models():
    return {
        "SVR": Pipeline(
            [("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=1.0, gamma="scale"))]
        ),
        "Lasso": Pipeline(
            [("scaler", StandardScaler()), ("model", Lasso(alpha=0.01, max_iter=10000, random_state=RANDOM_SEED))]
        ),
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=RANDOM_SEED))]
        ),
        "OLS": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }


def _boot_r2(y, pred, rng):
    n = len(y)
    scores = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, n, n)
        scores.append(r2_score(y[idx], pred[idx]))
    return float(np.mean(scores)), float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))


def six_models(df):
    cols = features_for_gender_model(PRIMARY_FEATURES)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for gender, gname in [("M", "Men"), ("F", "Women")]:
        tr = df[(df.gender == gender) & (df.year == 2023)]
        te = df[(df.gender == gender) & (df.year == 2024)]
        Xtr, ytr = tr[cols], tr["improvement_rate"].to_numpy(float)
        Xte, yte = te[cols], te["improvement_rate"].to_numpy(float)
        for name, model in _models().items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            r2 = float(r2_score(yte, pred))
            mae = float(mean_absolute_error(yte, pred))
            _, lo, hi = _boot_r2(yte, pred, rng)
            rows.append(
                {
                    "gender": gname,
                    "model": name,
                    "train_n": len(tr),
                    "test_n": len(te),
                    "r2": r2,
                    "r2_ci_lo": lo,
                    "r2_ci_hi": hi,
                    "mae": mae,
                    "n_features": len(cols),
                }
            )
            print(f"{gname:6} {name:20} R2={r2:7.3f} [{lo:6.3f},{hi:6.3f}] MAE={mae:.2f}")
    return pd.DataFrame(rows)


def learning_curves(df):
    cols = features_for_gender_model(PRIMARY_FEATURES)
    rows = []
    for gender, gname in [("M", "Men"), ("F", "Women")]:
        tr_full = df[(df.gender == gender) & (df.year == 2023)].reset_index(drop=True)
        te = df[(df.gender == gender) & (df.year == 2024)]
        Xte = te[cols]
        yte = te["improvement_rate"].to_numpy(float)
        for model_name in ["SVR", "Gradient Boosting", "Random Forest"]:
            for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                scores = []
                if frac < 1.0:
                    for seed in range(RANDOM_SEED, RANDOM_SEED + 20):
                        rng = np.random.default_rng(seed)
                        n = max(20, int(round(frac * len(tr_full))))
                        idx = rng.choice(len(tr_full), size=n, replace=False)
                        tr = tr_full.iloc[idx]
                        model = _models()[model_name]
                        model.fit(tr[cols], tr["improvement_rate"])
                        scores.append(r2_score(yte, model.predict(Xte)))
                    mean_r2 = float(np.mean(scores))
                else:
                    model = _models()[model_name]
                    model.fit(tr_full[cols], tr_full["improvement_rate"])
                    mean_r2 = float(r2_score(yte, model.predict(Xte)))
                rows.append(
                    {
                        "gender": gname,
                        "model": model_name,
                        "train_frac": frac,
                        "mean_test_r2": mean_r2,
                    }
                )
    return pd.DataFrame(rows)


def outlier_sensitivity(df):
    cols = features_for_gender_model(PRIMARY_FEATURES)
    rows = []
    raw = pd.read_csv(
        Path(__file__).resolve().parents[1] / "output" / "rq1" / "raw_data_athlete_features.csv"
    )
    for rule, lo, hi in [
        ("pm20", -20, 20),
        ("pm50_primary", -50, 50),
        ("pm100", -100, 100),
    ]:
        for gender, gname in [("M", "Men"), ("F", "Women")]:
            tr = raw[(raw.gender == gender) & (raw.year == 2023)]
            te = raw[(raw.gender == gender) & (raw.year == 2024)]
            # IQR from train only
            if rule == "iqr":
                q1, q3 = tr["improvement_rate"].quantile([0.25, 0.75])
                iqr = q3 - q1
                lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
            tr = tr[tr["improvement_rate"].between(lo, hi)]
            te = te[te["improvement_rate"].between(lo, hi)]
            model = _models()["SVR"]
            model.fit(tr[cols], tr["improvement_rate"])
            r2 = float(r2_score(te["improvement_rate"], model.predict(te[cols])))
            rows.append(
                {
                    "rule": rule,
                    "gender": gname,
                    "test_n": len(te),
                    "r2": r2,
                    "lo": lo,
                    "hi": hi,
                }
            )
    # training 1.5-IQR
    for gender, gname in [("M", "Men"), ("F", "Women")]:
        tr = raw[(raw.gender == gender) & (raw.year == 2023)]
        te = raw[(raw.gender == gender) & (raw.year == 2024)]
        q1, q3 = tr["improvement_rate"].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        tr = tr[tr["improvement_rate"].between(lo, hi)]
        te = te[te["improvement_rate"].between(lo, hi)]
        model = _models()["SVR"]
        model.fit(tr[cols], tr["improvement_rate"])
        r2 = float(r2_score(te["improvement_rate"], model.predict(te[cols])))
        rows.append(
            {
                "rule": "train_1.5_IQR",
                "gender": gname,
                "test_n": len(te),
                "r2": r2,
                "lo": lo,
                "hi": hi,
            }
        )
    return pd.DataFrame(rows)


def permutation_r2(df):
    cols = features_for_gender_model(PRIMARY_FEATURES)
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for gender, gname in [("M", "Men"), ("F", "Women")]:
        tr = df[(df.gender == gender) & (df.year == 2023)]
        te = df[(df.gender == gender) & (df.year == 2024)]
        ytr = tr["improvement_rate"].to_numpy(float)
        yte = te["improvement_rate"].to_numpy(float)
        model = _models()["SVR"]
        model.fit(tr[cols], ytr)
        obs = float(r2_score(yte, model.predict(te[cols])))
        nulls = []
        for _ in range(PERMUTATIONS):
            y_perm = rng.permutation(ytr)
            model.fit(tr[cols], y_perm)
            nulls.append(r2_score(yte, model.predict(te[cols])))
        nulls = np.asarray(nulls)
        p = float((nulls >= obs).mean())
        rows.append(
            {
                "gender": gname,
                "observed_r2": obs,
                "null_mean": float(nulls.mean()),
                "permutation_p": max(p, 1.0 / PERMUTATIONS),
            }
        )
        print(f"perm {gname}: obs={obs:.3f} null_mean={nulls.mean():.3f} p={rows[-1]['permutation_p']:.3f}")
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = _load()
    print("=== Six models (compact) ===")
    six = six_models(df)
    six.to_csv(OUT / "compact_six_models.csv", index=False)

    print("\n=== Learning curves ===")
    lc = learning_curves(df)
    lc.to_csv(OUT / "compact_learning_curves.csv", index=False)
    print(lc.to_string(index=False))

    print("\n=== Outlier sensitivity ===")
    out = outlier_sensitivity(df)
    out.to_csv(OUT / "compact_outlier_sensitivity.csv", index=False)
    print(out.to_string(index=False))

    print("\n=== Permutation ===")
    perm = permutation_r2(df)
    perm.to_csv(OUT / "compact_permutation_r2.csv", index=False)

    with open(OUT / "compact_suite_reproducibility.json", "w") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "bootstrap": BOOTSTRAP,
                "permutations": PERMUTATIONS,
                "feature_set": "compact_primary",
                "n_features_within_sex": len(features_for_gender_model()),
            },
            f,
            indent=2,
        )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
