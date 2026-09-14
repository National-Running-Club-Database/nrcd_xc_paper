"""Audit why CIKM reports R²≈0.72 while this paper reports R²≈0.05.

CIKM (data_paper/scripts/validation_ml_r2.py) is an illustrative paired
comparison of Converted-Only vs Standardized labels — not a forecasting
benchmark. Its absolute R² is nevertheless inflated by a near-tautology:
the feature set includes first_time, last_time, and season_duration, while
the label is exactly (last_time - first_time) / season_duration.

This script reproduces four regimes on the same Standardized XC data
(train 2023 → test 2024, gender-stratified Random Forest, seed 42):

  A. Algebraic reconstruction: predict y from {first_time, last_time,
     season_duration} only (OLS). Shows the ceiling from the identity.
  B. CIKM-style: full-season summaries + last_time in X (as in CIKM).
  C. Drop last_time only: full-season best/avg/etc., but no last_time.
  D. JQAS leakage-controlled: pre-final-race summaries, no last_time
     (this repository's primary protocol).

Outputs: output/rq1/cikm_r2_discrepancy_audit/
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from _setup_paths import setup_paths

setup_paths()

from ml_improvement_prediction import (  # noqa: E402
    RANDOM_SEED,
    calculate_athlete_features,
    create_advanced_features,
    load_raw_data,
    prepare_model_data,
)

OUT_DIR = Path("output/rq1/cikm_r2_discrepancy_audit")


def _cikm_style_features(df: pd.DataFrame, training_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Reproduce CIKM athlete-level feature construction (full-season + last_time).

    Matches data_paper/scripts/validation_ml_r2.py closely enough for the
    leakage audit: group by athlete_id (not athlete-season), include last_time
    in the returned frame, and compute time summaries over all races.
    """
    percentile_df = training_df if training_df is not None else df
    records = []

    for athlete_id, athlete_races in df.groupby("athlete_id"):
        athlete_races = athlete_races.sort_values("start_date")
        if len(athlete_races) < 2:
            continue

        first_time = athlete_races.iloc[0]["standardized_to_target"]
        last_time = athlete_races.iloc[-1]["standardized_to_target"]
        first_date = athlete_races.iloc[0]["start_date"]
        last_date = athlete_races.iloc[-1]["start_date"]
        days_diff = (last_date - first_date).days
        if days_diff < 7 or pd.isna(first_time) or pd.isna(last_time):
            continue

        times = athlete_races["standardized_to_target"].values
        improvement_rate = (last_time - first_time) / days_diff
        num_races = len(athlete_races)
        season_duration = days_diff
        best_time = float(np.min(times))
        worst_time = float(np.max(times))
        avg_time = float(np.mean(times))
        time_std = float(np.std(times))
        time_range = worst_time - best_time
        cv_time = time_std / avg_time if avg_time > 0 else 0.0

        if len(times) >= 3:
            xs = np.arange(len(times) - 1).reshape(-1, 1)
            slope = float(LinearRegression().fit(xs, times[:-1]).coef_[0])
        elif len(times) == 2:
            slope = float(times[1] - times[0])
        else:
            slope = 0.0

        race_frequency = num_races / season_duration
        gaps = athlete_races["start_date"].diff().dropna()
        avg_days_between_races = float(gaps.dt.days.mean()) if num_races > 1 else 0.0
        diffs = np.diff(times)
        race_to_race_improvement_std = float(np.std(diffs)) if len(times) >= 2 else 0.0
        bad_race_count = int(np.sum(diffs > 0)) if len(times) >= 2 else 0

        best_idx = int(np.argmin(times))
        if best_idx == 0:
            best_race_timing = 0.0
        elif best_idx == len(times) - 1:
            best_race_timing = float(season_duration)
        else:
            best_race_timing = float(
                (athlete_races.iloc[best_idx]["start_date"] - first_date).days
            )

        gender = athlete_races.iloc[0]["gender"]
        year = int(first_date.year)
        pg = percentile_df[
            (percentile_df["start_date"].dt.year < year)
            & (percentile_df["gender"] == gender)
        ]
        if len(pg) == 0:
            pg = percentile_df[percentile_df["gender"] == gender]
        starting_percentile = (
            float((pg["standardized_to_target"] <= first_time).mean() * 100) if len(pg) else 50.0
        )

        records.append(
            {
                "athlete_id": athlete_id,
                "gender": gender,
                "year": year,
                "num_races": num_races,
                "season_duration": season_duration,
                "first_time": first_time,
                "last_time": last_time,
                "best_time": best_time,
                "worst_time": worst_time,
                "avg_time": avg_time,
                "time_std": time_std,
                "time_range": time_range,
                "cv_time": cv_time,
                "improvement_rate": improvement_rate,
                "slope": slope,
                "race_frequency": race_frequency,
                "starting_percentile": starting_percentile,
                "avg_days_between_races": avg_days_between_races,
                "race_to_race_improvement_std": race_to_race_improvement_std,
                "best_race_timing": best_race_timing,
                "bad_race_count": bad_race_count,
            }
        )

    return pd.DataFrame(records)


CIKM_FEATURE_COLUMNS = [
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


def _rf_r2(X_train, y_train, X_test, y_test) -> float:
    model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X_train, y_train)
    return float(r2_score(y_test, model.predict(X_test)))


def _eval_gender(features_df: pd.DataFrame, feature_cols: list[str], gender: str) -> dict:
    f = features_df.copy()
    mask = (
        ~f[feature_cols + ["improvement_rate"]].isnull().any(axis=1)
        & (f["improvement_rate"] >= -50)
        & (f["improvement_rate"] <= 50)
        & (f["gender"] == gender)
    )
    f = f.loc[mask]
    train = f["year"] == 2023
    test = f["year"] == 2024
    if train.sum() < 50 or test.sum() < 50:
        return {"r2": None, "train_n": int(train.sum()), "test_n": int(test.sum())}
    X_tr, y_tr = f.loc[train, feature_cols], f.loc[train, "improvement_rate"]
    X_te, y_te = f.loc[test, feature_cols], f.loc[test, "improvement_rate"]
    return {
        "r2": _rf_r2(X_tr, y_tr, X_te, y_te),
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
    }


def _algebraic_r2(features_df: pd.DataFrame, gender: str) -> dict:
    """OLS on the three features that algebraically define the label."""
    cols = ["first_time", "last_time", "season_duration"]
    f = features_df.copy()
    mask = (
        ~f[cols + ["improvement_rate"]].isnull().any(axis=1)
        & (f["improvement_rate"] >= -50)
        & (f["improvement_rate"] <= 50)
        & (f["gender"] == gender)
    )
    f = f.loc[mask]
    train = f["year"] == 2023
    test = f["year"] == 2024
    model = LinearRegression()
    model.fit(f.loc[train, cols], f.loc[train, "improvement_rate"])
    r2 = float(r2_score(f.loc[test, "improvement_rate"], model.predict(f.loc[test, cols])))
    # Also report exact reconstruction error of the identity
    y_exact = (f.loc[test, "last_time"] - f.loc[test, "first_time"]) / f.loc[
        test, "season_duration"
    ]
    identity_r2 = float(r2_score(f.loc[test, "improvement_rate"], y_exact))
    return {
        "r2_ols": r2,
        "r2_identity": identity_r2,
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "pct_best_equals_last": float(
            (np.isclose(f.loc[test, "best_time"], f.loc[test, "last_time"])).mean()
        ),
    }


def main(output_dir: str | Path = OUT_DIR) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Standardized XC data...")
    df = load_raw_data()
    df = df.copy()
    df["year"] = df["start_date"].dt.year
    training = df[df["year"] == 2023]

    print("Building CIKM-style (leaky) features...")
    cikm_raw = _cikm_style_features(df, training_df=training)
    cikm_feat = create_advanced_features(cikm_raw)

    print("Building JQAS leakage-controlled features...")
    jqas_raw = calculate_athlete_features(df, training_df=training)
    jqas_feat = create_advanced_features(jqas_raw)
    X_jqas, y_jqas, jqas_filt = prepare_model_data(jqas_feat)
    jqas_filt = jqas_filt.copy()
    jqas_filt[X_jqas.columns] = X_jqas
    jqas_filt["improvement_rate"] = y_jqas
    jqas_cols = list(X_jqas.columns)

    rows = []
    for gender, label in (("M", "Men"), ("F", "Women")):
        print(f"\n=== {label} ===")
        alg = _algebraic_r2(cikm_feat, gender)
        print(
            f"  A. Algebraic identity R²={alg['r2_identity']:.3f}; "
            f"OLS on {{first,last,duration}} R²={alg['r2_ols']:.3f}; "
            f"best==last on {100*alg['pct_best_equals_last']:.1f}% of test"
        )
        rows.append(
            {
                "regime": "A_algebraic_identity",
                "gender": label,
                "model": "identity",
                "r2": alg["r2_identity"],
                "train_n": alg["train_n"],
                "test_n": alg["test_n"],
                "notes": f"pct_best_equals_last={alg['pct_best_equals_last']:.3f}",
            }
        )
        rows.append(
            {
                "regime": "A_algebraic_ols",
                "gender": label,
                "model": "OLS",
                "r2": alg["r2_ols"],
                "train_n": alg["train_n"],
                "test_n": alg["test_n"],
                "notes": "features={first_time,last_time,season_duration}",
            }
        )

        # RF can learn the nonlinear ratio that OLS cannot
        alg_rf = _eval_gender(
            cikm_feat, ["first_time", "last_time", "season_duration"], gender
        )
        print(f"  A'. RF on {{first,last,duration}} only R²={alg_rf['r2']:.3f}")
        rows.append(
            {
                "regime": "A_algebraic_rf",
                "gender": label,
                "model": "RandomForest",
                "r2": alg_rf["r2"],
                "train_n": alg_rf["train_n"],
                "test_n": alg_rf["test_n"],
                "notes": "RF learns y≈(last-first)/duration from three inputs",
            }
        )

        b = _eval_gender(cikm_feat, CIKM_FEATURE_COLUMNS, gender)
        print(f"  B. CIKM-style (incl. last_time) RF R²={b['r2']:.3f} (n_test={b['test_n']})")
        rows.append(
            {
                "regime": "B_cikm_style_with_last_time",
                "gender": label,
                "model": "RandomForest",
                "r2": b["r2"],
                "train_n": b["train_n"],
                "test_n": b["test_n"],
                "notes": "full-season summaries + last_time in X",
            }
        )

        cols_no_last = [c for c in CIKM_FEATURE_COLUMNS if c != "last_time"]
        c = _eval_gender(cikm_feat, cols_no_last, gender)
        print(f"  C. Drop last_time only RF R²={c['r2']:.3f}")
        rows.append(
            {
                "regime": "C_drop_last_time_only",
                "gender": label,
                "model": "RandomForest",
                "r2": c["r2"],
                "train_n": c["train_n"],
                "test_n": c["test_n"],
                "notes": "full-season best/avg still include outcome race",
            }
        )

        # JQAS path uses gender labels Men/Women in some frames; normalize
        gmask_vals = {gender, "Men" if gender == "M" else "Women", label}
        jf = jqas_filt[jqas_filt["gender"].isin(gmask_vals)].copy()
        if "gender" in jf.columns and jf["gender"].dtype == object:
            # prepare_model_data keeps original gender codes
            pass
        d = _eval_gender(jqas_filt, jqas_cols, gender)
        # gender column in jqas may be M/F
        if d["r2"] is None:
            # try Men/Women encoding
            jqas_filt2 = jqas_filt.copy()
            if jqas_filt2["gender"].isin(["Men", "Women"]).any():
                g2 = "Men" if gender == "M" else "Women"
                d = _eval_gender(jqas_filt2, jqas_cols, g2)
        print(f"  D. JQAS leakage-controlled RF R²={d['r2']}")
        rows.append(
            {
                "regime": "D_jqas_leakage_controlled",
                "gender": label,
                "model": "RandomForest",
                "r2": d["r2"],
                "train_n": d["train_n"],
                "test_n": d["test_n"],
                "notes": "pre-final races only; last_time excluded from X",
            }
        )

    out_csv = output_dir / "regime_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    manifest = {
        "random_seed": RANDOM_SEED,
        "protocol": "train 2023 → test 2024, Standardized times",
        "cikm_github": "https://github.com/National-Running-Club-Database/data_paper",
        "cikm_script": "scripts/validation_ml_r2.py",
        "verdict": (
            "CIKM absolute R² is not wrong as a paired Converted vs Standardized "
            "demonstration, but is not forecasting skill: y=(last-first)/duration "
            "is nearly recoverable from features that include last_time."
        ),
        "output_csv": str(out_csv),
    }
    with open(output_dir / "reproducibility.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nWrote {out_csv}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
