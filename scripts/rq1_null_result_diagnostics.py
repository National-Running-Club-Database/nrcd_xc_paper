"""Diagnostics that strengthen the RQ1 individual-prediction null result.

Three complementary analyses, all gender-separated and seed-controlled:

1. Noise ceiling / reliability of the outcome. Distinguishes "no signal exists"
   from "signal is swamped by race-to-race noise" by estimating (a) the
   intraclass correlation (ICC) of standardized race times within an
   athlete-season, and (b) the split-half reliability of the season improvement
   slope (odd/even races, Spearman-Brown corrected). The split-half reliability
   is an upper bound on the test R^2 any predictor of that slope could attain.

2. Permutation null for the held-out R^2. Shuffles the training outcome, refits
   the primary gender-separated SVR (train 2023 -> test 2024), and reports the
   fraction of permuted test-R^2 values at least as large as observed.

3. Classification robustness. Recasts the task as "top-quartile improver vs.
   not" (threshold from the training distribution) and reports held-out AUC with
   bootstrap CIs, in case tail structure carries weak signal that R^2 misses.

Outputs: output/rq1/null_result_diagnostics/
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from _setup_paths import setup_paths

setup_paths()

from prediction_diagnostics import _complete_cases, _gender_features

RANDOM_SEED = 42
PERMUTATIONS = 1000
BOOTSTRAP = 2000
RATE_RANGE = (-50.0, 50.0)


# ---------------------------------------------------------------------------
# 1. Noise ceiling / reliability
# ---------------------------------------------------------------------------
def _icc_one_way(values: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1,1): reliability of a single measurement within a group."""
    df = pd.DataFrame({"y": values, "g": groups})
    grand = df["y"].mean()
    group_means = df.groupby("g")["y"]
    k = group_means.ngroups
    n_total = len(df)
    sizes = group_means.size().to_numpy()
    n0 = (n_total - (sizes**2).sum() / n_total) / (k - 1)
    ss_between = (group_means.mean().to_numpy() - grand) ** 2
    ms_between = (sizes * ss_between).sum() / (k - 1)
    ss_within = df.groupby("g")["y"].transform(lambda s: ((s - s.mean()) ** 2)).sum()
    ms_within = ss_within / (n_total - k)
    var_between = (ms_between - ms_within) / n0
    icc = var_between / (var_between + ms_within)
    return float(icc)


def noise_ceiling(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    raw = raw.copy()
    raw["start_date"] = pd.to_datetime(raw["start_date"], errors="coerce")
    raw["year"] = raw["start_date"].dt.year
    raw = raw.dropna(subset=["standardized_to_target", "start_date"])
    raw = raw[raw["year"].isin([2023, 2024, 2025])]
    for gender, label in [("M", "Men"), ("F", "Women")]:
        g = raw[raw["gender"] == gender].copy()
        g["season"] = g["athlete_id"].astype(str) + "_" + g["year"].astype(str)
        counts = g.groupby("season").size()
        # ICC of single race time using athlete-seasons with >= 2 races
        multi = g[g["season"].isin(counts[counts >= 2].index)]
        icc = _icc_one_way(
            multi["standardized_to_target"].to_numpy(),
            multi["season"].to_numpy(),
        )
        # Split-half reliability of the season slope (odd/even races)
        slopes_a, slopes_b = [], []
        for season, sub in g[g["season"].isin(counts[counts >= 4].index)].groupby(
            "season"
        ):
            sub = sub.sort_values("start_date")
            day = (sub["start_date"] - sub["start_date"].min()).dt.days.to_numpy(float)
            y = sub["standardized_to_target"].to_numpy(float)
            idx = np.arange(len(sub))
            a, b = idx[idx % 2 == 0], idx[idx % 2 == 1]
            if len(a) >= 2 and len(b) >= 2 and np.ptp(day[a]) > 0 and np.ptp(day[b]) > 0:
                slopes_a.append(np.polyfit(day[a], y[a], 1)[0])
                slopes_b.append(np.polyfit(day[b], y[b], 1)[0])
        n_sh = len(slopes_a)
        if n_sh >= 10:
            r_half = float(np.corrcoef(slopes_a, slopes_b)[0, 1])
            r_sb = 2 * r_half / (1 + r_half) if (1 + r_half) != 0 else np.nan
        else:
            r_half, r_sb = np.nan, np.nan
        rows.append(
            {
                "gender": label,
                "single_race_icc": icc,
                "n_seasons_ge2": int((counts >= 2).sum()),
                "split_half_r": r_half,
                "split_half_reliability_sb": r_sb,
                "r2_ceiling_estimate": r_sb,
                "n_seasons_ge4_used": n_sh,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Permutation null for held-out R^2
# ---------------------------------------------------------------------------
def _svr() -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=1.0, gamma="scale"))]
    )


def permutation_r2(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for gender, label in [("M", "Men"), ("F", "Women")]:
        work = _complete_cases(df, gender)
        work = work[work["improvement_rate"].between(*RATE_RANGE)]
        feats = _gender_features(gender)
        train = work[work["year"] == 2023]
        test = work[work["year"] == 2024]
        Xtr, ytr = train[feats], train["improvement_rate"].to_numpy()
        Xte, yte = test[feats], test["improvement_rate"].to_numpy()
        model = _svr()
        model.fit(Xtr, ytr)
        observed = r2_score(yte, model.predict(Xte))
        perm_r2 = np.empty(PERMUTATIONS)
        for i in range(PERMUTATIONS):
            yperm = rng.permutation(ytr)
            m = _svr().fit(Xtr, yperm)
            perm_r2[i] = r2_score(yte, m.predict(Xte))
        p = float((np.sum(perm_r2 >= observed) + 1) / (PERMUTATIONS + 1))
        rows.append(
            {
                "gender": label,
                "observed_test_r2": observed,
                "perm_r2_mean": float(perm_r2.mean()),
                "perm_r2_p95": float(np.percentile(perm_r2, 95)),
                "permutation_p": p,
                "n_permutations": PERMUTATIONS,
                "n_train": len(train),
                "n_test": len(test),
                "random_seed": RANDOM_SEED,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Classification: top-quartile improver vs. not
# ---------------------------------------------------------------------------
# Features that are direct functions of the same race times used to build the
# first-to-last change-score outcome; including them couples baseline to change
# (regression to the mean). The "schedule_only" set drops them to test whether
# any tail signal survives beyond that mechanical coupling.
_BASELINE_COUPLED = {
    "first_time",
    "best_time",
    "worst_time",
    "avg_time",
    "time_std",
    "time_range",
    "cv_time",
    "total_improvement",
    "slope",
    "starting_percentile",
    "starting_percentile_squared",
    "best_to_avg_ratio",
    "worst_to_avg_ratio",
    "variability_score",
    "consistency_score",
    "race_to_race_improvement_std",
}


def classification_auc(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    classifiers = {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }
    for gender, label in [("M", "Men"), ("F", "Women")]:
        work = _complete_cases(df, gender)
        work = work[work["improvement_rate"].between(*RATE_RANGE)]
        all_feats = _gender_features(gender)
        schedule_feats = [c for c in all_feats if c not in _BASELINE_COUPLED]
        train = work[work["year"] == 2023].copy()
        test = work[work["year"] == 2024].copy()
        # "Top-quartile improver" = fastest improvement = lowest improvement_rate.
        thresh = train["improvement_rate"].quantile(0.25)
        ytr = (train["improvement_rate"] <= thresh).astype(int).to_numpy()
        yte = (test["improvement_rate"] <= thresh).astype(int).to_numpy()
        for feat_set, feats in [
            ("all_features", all_feats),
            ("schedule_only", schedule_feats),
        ]:
            Xtr, Xte = train[feats], test[feats]
            for name, clf in classifiers.items():
                clf.fit(Xtr, ytr)
                score = clf.predict_proba(Xte)[:, 1]
                auc = roc_auc_score(yte, score)
                boots = np.empty(BOOTSTRAP)
                n = len(yte)
                for i in range(BOOTSTRAP):
                    idx = rng.integers(0, n, n)
                    if len(np.unique(yte[idx])) < 2:
                        boots[i] = np.nan
                        continue
                    boots[i] = roc_auc_score(yte[idx], score[idx])
                boots = boots[~np.isnan(boots)]
                rows.append(
                    {
                        "gender": label,
                        "feature_set": feat_set,
                        "classifier": name,
                        "n_features": len(feats),
                        "test_auc": auc,
                        "auc_ci_low": float(np.percentile(boots, 2.5)),
                        "auc_ci_high": float(np.percentile(boots, 97.5)),
                        "positive_rate_test": float(yte.mean()),
                        "n_train": len(train),
                        "n_test": len(test),
                        "bootstrap_replicates": BOOTSTRAP,
                        "random_seed": RANDOM_SEED,
                    }
                )
    return pd.DataFrame(rows)


def main(output_dir: str = "output/rq1") -> None:
    from ml_improvement_prediction import load_raw_data

    out_dir = os.path.join(output_dir, "null_result_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"RQ1 null-result diagnostics: seed={RANDOM_SEED}, "
        f"permutations={PERMUTATIONS}, bootstrap={BOOTSTRAP}."
    )

    features_path = os.path.join(output_dir, "raw_data_athlete_features.csv")
    df = pd.read_csv(features_path)

    print("  [1/3] Noise ceiling / reliability...")
    raw = load_raw_data(mode="standardized")
    ceiling = noise_ceiling(raw)
    ceiling.to_csv(os.path.join(out_dir, "noise_ceiling.csv"), index=False)
    print(ceiling.to_string(index=False))

    print("  [2/3] Permutation null for held-out R^2...")
    perm = permutation_r2(df)
    perm.to_csv(os.path.join(out_dir, "permutation_r2.csv"), index=False)
    print(perm.to_string(index=False))

    print("  [3/3] Classification (top-quartile improver) AUC...")
    auc = classification_auc(df)
    auc.to_csv(os.path.join(out_dir, "classification_auc.csv"), index=False)
    print(auc.to_string(index=False))

    with open(os.path.join(out_dir, "reproducibility.json"), "w") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "permutations": PERMUTATIONS,
                "bootstrap_replicates": BOOTSTRAP,
                "rate_range": RATE_RANGE,
            },
            f,
            indent=2,
        )
    print(f"\nSaved RQ1 null-result diagnostics to {out_dir}/")


if __name__ == "__main__":
    main()
