"""
Enrichment analyses for the manuscript after leakage-controlled prediction
proved weak. These focus on scientifically reportable descriptive and
associational evidence:

1. Race-count dose-response on Standardized first-to-last improvement
2. Weather inflation: Standardized vs Converted Only mean improvement
3. Starting-ability quartile × race-count interaction
4. Season-to-season retention by gender and prior race count
5. Early-window prediction (first race / first two races only)

Run from repo root:
  python scripts/paper_enrichment_analyses.py
"""

from __future__ import annotations

import os
import sys
import json

from _setup_paths import setup_paths

setup_paths()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from utils import standardize_both_tiers

OUTPUT_DIR = "output/rq1/enrichment"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 2000
RNG = np.random.default_rng(RANDOM_SEED)


def _bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = BOOTSTRAP_REPLICATES, alpha: float = 0.05
):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = []
    n = len(x)
    for _ in range(n_boot):
        sample = RNG.choice(x, size=n, replace=True)
        means.append(sample.mean())
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(x)), float(lo), float(hi)


def _bootstrap_median_ci(
    x: np.ndarray, n_boot: int = BOOTSTRAP_REPLICATES, alpha: float = 0.05
):
    """Point median with bootstrap percentile CI for the median."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    medians = []
    n = len(x)
    for _ in range(n_boot):
        sample = RNG.choice(x, size=n, replace=True)
        medians.append(np.median(sample))
    lo, hi = np.percentile(medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.median(x)), float(lo), float(hi)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled == 0:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def build_athlete_season_table(df: pd.DataFrame, time_col: str = "standardized_to_target") -> pd.DataFrame:
    """One row per athlete-season with first/last Standardized times."""
    work = df.copy()
    work["start_date"] = pd.to_datetime(work["start_date"], errors="coerce")
    work = work.dropna(subset=[time_col, "start_date", "gender", "athlete_id"])
    work["year"] = work["start_date"].dt.year

    rows = []
    for (athlete_id, year), races in work.groupby(["athlete_id", "year"], sort=False):
        races = races.sort_values("start_date")
        if len(races) < 2:
            continue
        first = races.iloc[0]
        last = races.iloc[-1]
        days = (last["start_date"] - first["start_date"]).days
        if days < 7:
            continue
        first_t = float(first[time_col])
        last_t = float(last[time_col])
        if not np.isfinite(first_t) or not np.isfinite(last_t):
            continue
        # Positive = faster (seconds improved)
        total_improve = first_t - last_t
        improve_rate = (last_t - first_t) / days  # negative = getting faster
        race_bin = str(len(races)) if len(races) < 5 else "5+"
        rows.append(
            {
                "athlete_id": athlete_id,
                "year": int(year),
                "gender": first["gender"],
                "num_races": int(len(races)),
                "race_bin": race_bin,
                "season_duration": int(days),
                "first_time": first_t,
                "last_time": last_t,
                "total_improvement_sec": total_improve,
                "improvement_rate": improve_rate,
                "improved": total_improve > 0,
            }
        )
    return pd.DataFrame(rows)


def race_count_dose_response(as_df: pd.DataFrame) -> pd.DataFrame:
    """Gender-separated dose-response of improvement vs race count (median primary)."""
    records = []
    for gender, gdf in as_df.groupby("gender"):
        baseline = gdf.loc[gdf["race_bin"] == "2", "total_improvement_sec"].values
        for race_bin in ["2", "3", "4", "5+"]:
            sub = gdf[gdf["race_bin"] == race_bin]
            vals = sub["total_improvement_sec"].values
            med, lo, hi = _bootstrap_median_ci(vals)
            mean, m_lo, m_hi = _bootstrap_mean_ci(vals)
            rate_vals = sub["improvement_rate"].values
            rate_med, rate_lo, rate_hi = _bootstrap_median_ci(rate_vals)
            d = _cohens_d(vals, baseline) if race_bin != "2" else 0.0
            if race_bin == "2" or len(vals) < 5 or len(baseline) < 5:
                mw_p = np.nan
            else:
                mw_p = stats.mannwhitneyu(vals, baseline, alternative="two-sided").pvalue
            records.append(
                {
                    "gender": "Men" if gender == "M" else "Women",
                    "race_bin": race_bin,
                    "n": int(len(sub)),
                    "median_improvement_sec": med,
                    "ci_low": lo,
                    "ci_high": hi,
                    "mean_improvement_sec": mean,
                    "mean_ci_low": m_lo,
                    "mean_ci_high": m_hi,
                    "pct_improved": 100.0 * float(sub["improved"].mean()) if len(sub) else np.nan,
                    "median_improvement_rate": rate_med,
                    "rate_ci_low": rate_lo,
                    "rate_ci_high": rate_hi,
                    "cohens_d_vs_2": d,
                    "mannwhitney_p_vs_2": mw_p,
                }
            )
        groups = [
            gdf.loc[gdf["race_bin"] == b, "total_improvement_sec"].values
            for b in ["2", "3", "4", "5+"]
            if (gdf["race_bin"] == b).any()
        ]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
        else:
            h, p = np.nan, np.nan
        records.append(
            {
                "gender": "Men" if gender == "M" else "Women",
                "race_bin": "Kruskal-Wallis",
                "n": int(len(gdf)),
                "median_improvement_sec": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "mean_improvement_sec": np.nan,
                "mean_ci_low": np.nan,
                "mean_ci_high": np.nan,
                "pct_improved": np.nan,
                "median_improvement_rate": np.nan,
                "rate_ci_low": np.nan,
                "rate_ci_high": np.nan,
                "cohens_d_vs_2": h,
                "mannwhitney_p_vs_2": p,
            }
        )
    return pd.DataFrame(records)


def weather_inflation(std_as: pd.DataFrame, conv_as: pd.DataFrame) -> pd.DataFrame:
    """Compare Standardized vs Converted Only mean improvement by gender."""
    merged = std_as.merge(
        conv_as[
            [
                "athlete_id",
                "year",
                "total_improvement_sec",
                "improvement_rate",
                "num_races",
            ]
        ],
        on=["athlete_id", "year"],
        suffixes=("_std", "_conv"),
        how="inner",
    )
    rows = []
    for gender, gdf in merged.groupby("gender"):
        label = "Men" if gender == "M" else "Women"
        for method, col in [
            ("Standardized", "total_improvement_sec_std"),
            ("Converted Only", "total_improvement_sec_conv"),
        ]:
            mean, lo, hi = _bootstrap_mean_ci(gdf[col].values)
            rows.append(
                {
                    "gender": label,
                    "method": method,
                    "n": int(len(gdf)),
                    "mean_improvement_sec": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "pct_improved": 100.0
                    * float((gdf[col] > 0).mean()),
                }
            )
        delta = gdf["total_improvement_sec_conv"] - gdf["total_improvement_sec_std"]
        d_mean, d_lo, d_hi = _bootstrap_mean_ci(delta.values)
        # Paired Wilcoxon: is Converted larger (more apparent improvement)?
        try:
            w_stat, w_p = stats.wilcoxon(
                gdf["total_improvement_sec_conv"],
                gdf["total_improvement_sec_std"],
                alternative="greater",
            )
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        rows.append(
            {
                "gender": label,
                "method": "Converted − Standardized",
                "n": int(len(gdf)),
                "mean_improvement_sec": d_mean,
                "ci_low": d_lo,
                "ci_high": d_hi,
                "pct_improved": w_p,
                "wilcoxon_stat": w_stat,
            }
        )
    out = pd.DataFrame(rows)
    # also save paired athlete-seasons for reproducibility
    merged.to_csv(os.path.join(OUTPUT_DIR, "std_vs_converted_paired.csv"), index=False)
    return out


def starting_ability_by_race_count(as_df: pd.DataFrame) -> pd.DataFrame:
    """Median improvement by gender-year starting quartile and race bin."""
    work = as_df.copy()
    work["start_quartile"] = work.groupby(["gender", "year"])["first_time"].transform(
        lambda s: pd.qcut(
            s, 4, labels=["Q1_fastest", "Q2", "Q3", "Q4_slowest"], duplicates="drop"
        )
    )
    rows = []
    for (gender, q, race_bin), sub in work.groupby(
        ["gender", "start_quartile", "race_bin"], observed=True
    ):
        if pd.isna(q) or len(sub) < 10:
            continue
        med, lo, hi = _bootstrap_median_ci(sub["total_improvement_sec"].values)
        mean, m_lo, m_hi = _bootstrap_mean_ci(sub["total_improvement_sec"].values)
        rows.append(
            {
                "gender": "Men" if gender == "M" else "Women",
                "start_quartile": str(q),
                "race_bin": race_bin,
                "n": int(len(sub)),
                "median_improvement_sec": med,
                "ci_low": lo,
                "ci_high": hi,
                "mean_improvement_sec": mean,
                "mean_ci_low": m_lo,
                "mean_ci_high": m_hi,
                "pct_improved": 100.0 * float(sub["improved"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["gender", "start_quartile", "race_bin"])


def retention_analysis(as_df: pd.DataFrame, all_results: pd.DataFrame) -> pd.DataFrame:
    """Probability of racing next season by gender and prior race count."""
    # Any race next year counts as retained (even 1 race)
    all_results = all_results.copy()
    all_results["start_date"] = pd.to_datetime(all_results["start_date"], errors="coerce")
    all_results["year"] = all_results["start_date"].dt.year
    raced = (
        all_results.dropna(subset=["athlete_id", "year", "gender"])
        .groupby(["athlete_id", "year", "gender"])
        .size()
        .reset_index(name="races_any")
    )

    base = as_df[as_df["year"].isin([2023, 2024])].copy()
    base["next_year"] = base["year"] + 1
    merged = base.merge(
        raced.rename(columns={"year": "next_year", "races_any": "next_races"}),
        on=["athlete_id", "next_year", "gender"],
        how="left",
    )
    merged["retained"] = merged["next_races"].fillna(0) > 0

    rows = []
    for gender, gdf in merged.groupby("gender"):
        label = "Men" if gender == "M" else "Women"
        for race_bin in ["2", "3", "4", "5+"]:
            sub = gdf[gdf["race_bin"] == race_bin]
            if len(sub) == 0:
                continue
            p = float(sub["retained"].mean())
            # Wilson-ish bootstrap CI
            boots = []
            idx = sub["retained"].values
            for _ in range(2000):
                boots.append(RNG.choice(idx, size=len(idx), replace=True).mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(
                {
                    "gender": label,
                    "prior_race_bin": race_bin,
                    "n": int(len(sub)),
                    "retention_rate": p,
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                }
            )
        # Overall
        p = float(gdf["retained"].mean())
        boots = [
            RNG.choice(gdf["retained"].values, size=len(gdf), replace=True).mean()
            for _ in range(2000)
        ]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(
            {
                "gender": label,
                "prior_race_bin": "All (≥2 races)",
                "n": int(len(gdf)),
                "retention_rate": p,
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
    return pd.DataFrame(rows)


def early_window_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prospective-style check: predict improvement rate using only information
    available after the first race, or after the first two races.
    """
    work = df.copy()
    work["start_date"] = pd.to_datetime(work["start_date"], errors="coerce")
    work = work.dropna(
        subset=["standardized_to_target", "start_date", "gender", "athlete_id"]
    )
    work["year"] = work["start_date"].dt.year

    rows = []
    for (athlete_id, year), races in work.groupby(["athlete_id", "year"], sort=False):
        races = races.sort_values("start_date")
        if len(races) < 2:
            continue
        times = races["standardized_to_target"].astype(float).values
        dates = races["start_date"].values
        days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
        if days < 7:
            continue
        y = (times[-1] - times[0]) / days
        if not (-50 <= y <= 50):
            continue
        gender = races.iloc[0]["gender"]
        # Window A: first race only
        rows.append(
            {
                "athlete_id": athlete_id,
                "year": int(year),
                "gender": gender,
                "window": "first_race_only",
                "first_time": times[0],
                "early_slope": 0.0,
                "early_n": 1,
                "num_races_total": len(races),
                "season_duration": days,
                "y": y,
            }
        )
        # Window B: first two races (need ≥3 races so last is still held out)
        if len(races) >= 3:
            early_slope = times[1] - times[0]
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "year": int(year),
                    "gender": gender,
                    "window": "first_two_races",
                    "first_time": times[0],
                    "early_slope": early_slope,
                    "early_n": 2,
                    "num_races_total": len(races),
                    "season_duration": days,
                    "y": y,
                }
            )

    feat = pd.DataFrame(rows)
    results = []
    models = {
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]
        ),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, gamma="scale")),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=1
        ),
    }

    for gender in ["M", "F"]:
        label = "Men" if gender == "M" else "Women"
        for window in ["first_race_only", "first_two_races"]:
            sub = feat[(feat["gender"] == gender) & (feat["window"] == window)].copy()
            train = sub[sub["year"] == 2023]
            test = sub[sub["year"] == 2024]
            if len(train) < 50 or len(test) < 30:
                continue
            xcols = ["first_time", "early_slope", "early_n", "season_duration"]
            # For first_race_only, season_duration includes future schedule length —
            # that is schedule knowledge, not race performance. Report both with
            # and without season_duration.
            for schedule_flag, cols in [
                ("with_schedule_length", xcols),
                ("performance_only", ["first_time", "early_slope", "early_n"]),
            ]:
                Xtr, ytr = train[cols], train["y"]
                Xte, yte = test[cols], test["y"]
                for name, model in models.items():
                    est = model
                    if name != "Random Forest":
                        # Pipeline objects need fresh clones
                        from sklearn.base import clone

                        est = clone(model)
                    else:
                        from sklearn.base import clone

                        est = clone(model)
                    est.fit(Xtr, ytr)
                    pred = est.predict(Xte)
                    results.append(
                        {
                            "gender": label,
                            "window": window,
                            "features": schedule_flag,
                            "model": name,
                            "n_train": int(len(train)),
                            "n_test": int(len(test)),
                            "r2": float(r2_score(yte, pred)),
                            "mae": float(mean_absolute_error(yte, pred)),
                        }
                    )
    return pd.DataFrame(results)


# Split-half reliability of improvement (from null diagnostics); used for SER.
_EARLY_RELIABILITY = {"Men": 0.22822087419717996, "Women": 0.2787539600162913}


def _early_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Athlete-season rows with early-window features and improvement target."""
    work = df.copy()
    work["start_date"] = pd.to_datetime(work["start_date"], errors="coerce")
    work = work.dropna(
        subset=["standardized_to_target", "start_date", "gender", "athlete_id"]
    )
    work["year"] = work["start_date"].dt.year

    rows = []
    for (athlete_id, year), races in work.groupby(["athlete_id", "year"], sort=False):
        races = races.sort_values("start_date")
        if len(races) < 2:
            continue
        times = races["standardized_to_target"].astype(float).values
        dates = races["start_date"].values
        days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days
        if days < 7:
            continue
        y = (times[-1] - times[0]) / days
        if not (-50 <= y <= 50):
            continue
        gender = races.iloc[0]["gender"]
        rows.append(
            {
                "athlete_id": athlete_id,
                "year": int(year),
                "gender": gender,
                "window": "first_race_only",
                "first_time": times[0],
                "early_slope": 0.0,
                "early_n": 1,
                "num_races_total": len(races),
                "season_duration": days,
                "y": y,
            }
        )
        if len(races) >= 3:
            rows.append(
                {
                    "athlete_id": athlete_id,
                    "year": int(year),
                    "gender": gender,
                    "window": "first_two_races",
                    "first_time": times[0],
                    "early_slope": times[1] - times[0],
                    "early_n": 2,
                    "num_races_total": len(races),
                    "season_duration": days,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


def early_window_decision_analysis(
    df: pd.DataFrame, output_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coach-facing early-info check: skill vs mean baseline, SER, calibration.

    Uses performance-only features (no future schedule length) under temporal
    validation (train 2023 → test 2024). Writes
    ``early_window_decision.csv``, ``early_window_calibration.csv``, and
    ``early_window_decision.pdf``.
    """
    from sklearn.base import clone

    feat = _early_feature_frame(df)
    models = {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "SVR": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0, gamma="scale")),
            ]
        ),
    }
    cols = ["first_time", "early_slope", "early_n"]
    summary_rows = []
    calib_rows = []

    for gender_code, label in [("M", "Men"), ("F", "Women")]:
        rho = _EARLY_RELIABILITY[label]
        for window in ["first_race_only", "first_two_races"]:
            sub = feat[(feat["gender"] == gender_code) & (feat["window"] == window)]
            train = sub[sub["year"] == 2023]
            test = sub[sub["year"] == 2024]
            if len(train) < 50 or len(test) < 30:
                continue
            ytr = train["y"].to_numpy(dtype=float)
            yte = test["y"].to_numpy(dtype=float)
            baseline_pred = np.full_like(yte, fill_value=float(np.mean(ytr)))
            baseline_mae = float(mean_absolute_error(yte, baseline_pred))
            # Constant predictor has R² ≤ 0 by definition vs test mean; report MAE skill.
            for name, model in models.items():
                est = clone(model)
                est.fit(train[cols], ytr)
                pred = est.predict(test[cols])
                r2 = float(r2_score(yte, pred))
                mae = float(mean_absolute_error(yte, pred))
                ser = r2 / rho if rho > 0 else np.nan
                summary_rows.append(
                    {
                        "gender": label,
                        "window": window,
                        "model": name,
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "r2": r2,
                        "mae": mae,
                        "baseline_mae": baseline_mae,
                        "mae_skill_vs_baseline": float(1.0 - mae / baseline_mae)
                        if baseline_mae > 0
                        else np.nan,
                        "reliability_rho": rho,
                        "SER": ser,
                        "SER_clipped": max(ser, 0.0) if np.isfinite(ser) else np.nan,
                    }
                )
                # Calibration: deciles of predicted improvement vs mean observed.
                order = np.argsort(pred)
                n = len(pred)
                n_bins = min(10, max(4, n // 40))
                edges = np.array_split(order, n_bins)
                for b, idx in enumerate(edges, start=1):
                    if len(idx) == 0:
                        continue
                    calib_rows.append(
                        {
                            "gender": label,
                            "window": window,
                            "model": name,
                            "bin": b,
                            "n": int(len(idx)),
                            "mean_predicted": float(np.mean(pred[idx])),
                            "mean_observed": float(np.mean(yte[idx])),
                        }
                    )

    summary = pd.DataFrame(summary_rows)
    calib = pd.DataFrame(calib_rows)
    summary.to_csv(os.path.join(output_dir, "early_window_decision.csv"), index=False)
    calib.to_csv(os.path.join(output_dir, "early_window_calibration.csv"), index=False)

    # Figure: R² / SER by window (SVR) + calibration for men first-two SVR if present.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor("white")
    ax0 = axes[0]
    svr = summary[summary["model"] == "SVR"].copy()
    if not svr.empty:
        xlabels = []
        r2s = []
        sers = []
        for _, r in svr.iterrows():
            xlabels.append(f"{r['gender']}\n{r['window'].replace('_', ' ')}")
            r2s.append(r["r2"])
            sers.append(r["SER_clipped"])
        x = np.arange(len(xlabels))
        w = 0.35
        ax0.bar(x - w / 2, r2s, w, label="Held-out R²", color="#4C72B0")
        ax0.bar(x + w / 2, sers, w, label="SER (clipped)", color="#55A868")
        ax0.axhline(0, color="gray", lw=0.8)
        ax0.set_xticks(x)
        ax0.set_xticklabels(xlabels, fontsize=8)
        ax0.set_ylabel("Score")
        ax0.set_title("Early-window forecast skill (SVR, performance-only)")
        ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    cal = calib[
        (calib["gender"] == "Men")
        & (calib["window"] == "first_two_races")
        & (calib["model"] == "SVR")
    ]
    if cal.empty:
        cal = calib[
            (calib["gender"] == "Men")
            & (calib["window"] == "first_race_only")
            & (calib["model"] == "SVR")
        ]
    if not cal.empty:
        ax1.plot(cal["mean_predicted"], cal["mean_observed"], "o-", color="#4C72B0")
        lims = [
            min(cal["mean_predicted"].min(), cal["mean_observed"].min()),
            max(cal["mean_predicted"].max(), cal["mean_observed"].max()),
        ]
        ax1.plot(lims, lims, "--", color="gray", lw=1, label="Perfect calibration")
        ax1.set_xlabel("Mean predicted improvement (s/day)")
        ax1.set_ylabel("Mean observed improvement (s/day)")
        ax1.set_title("Calibration (Men SVR; early window)")
        ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "early_window_decision.pdf")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary, calib


def plot_dose_response(dose_df: pd.DataFrame, path: str) -> None:
    plot_df = dose_df[dose_df["race_bin"] != "Kruskal-Wallis"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    order = ["2", "3", "4", "5+"]
    ycol = (
        "median_improvement_sec"
        if "median_improvement_sec" in plot_df.columns
        else "mean_improvement_sec"
    )
    for ax, gender in zip(axes, ["Men", "Women"]):
        sub = plot_df[plot_df["gender"] == gender].set_index("race_bin").loc[order]
        x = np.arange(len(order))
        y = sub[ycol].values
        yerr = np.vstack(
            [y - sub["ci_low"].values, sub["ci_high"].values - y]
        )
        ax.bar(x, y, color="#2E86AB" if gender == "Men" else "#A23B72", alpha=0.85)
        ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=4)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(order)
        ax.set_xlabel("Regular-season races")
        ax.set_title(gender)
        for i, n in enumerate(sub["n"].values):
            ax.text(i, max(y[i], 0) + 5, f"n={n}", ha="center", fontsize=8)
    axes[0].set_ylabel("Median first→last improvement (s)\n(positive = faster)")
    fig.suptitle(
        "Standardized improvement by race count (median)\n(95% bootstrap CI; nationals excluded)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dose_by_starting_quartile(start_df: pd.DataFrame, path: str) -> None:
    """Facet: median improvement by race count within starting-time quartile."""
    if start_df.empty or "median_improvement_sec" not in start_df.columns:
        return
    q_order = ["Q1_fastest", "Q2", "Q3", "Q4_slowest"]
    r_order = ["2", "3", "4", "5+"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=True)
    fig.patch.set_facecolor("white")
    for row, gender in enumerate(["Men", "Women"]):
        for col, q in enumerate(q_order):
            ax = axes[row, col]
            sub = start_df[
                (start_df["gender"] == gender) & (start_df["start_quartile"] == q)
            ].set_index("race_bin")
            xs, ys, ylo, yhi, ns = [], [], [], [], []
            for r in r_order:
                if r not in sub.index:
                    continue
                xs.append(r)
                ys.append(float(sub.loc[r, "median_improvement_sec"]))
                ylo.append(float(sub.loc[r, "ci_low"]))
                yhi.append(float(sub.loc[r, "ci_high"]))
                ns.append(int(sub.loc[r, "n"]))
            if not xs:
                ax.set_visible(False)
                continue
            x = np.arange(len(xs))
            y = np.asarray(ys)
            yerr = np.vstack([y - np.asarray(ylo), np.asarray(yhi) - y])
            ax.bar(x, y, color="#2E86AB" if gender == "Men" else "#A23B72", alpha=0.85)
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(xs)
            title_q = q.replace("_", " ")
            ax.set_title(f"{gender}: {title_q}", fontsize=9)
            for i, n in enumerate(ns):
                ax.text(i, max(y[i], 0) + 3, f"n={n}", ha="center", fontsize=6)
            if col == 0:
                ax.set_ylabel("Median improve (s)")
            if row == 1:
                ax.set_xlabel("Races")
    fig.suptitle(
        "Median Standardized first→last improvement by starting-time quartile × race count",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_weather_inflation(infl_df: pd.DataFrame, path: str) -> None:
    plot_df = infl_df[infl_df["method"].isin(["Standardized", "Converted Only"])]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    genders = ["Men", "Women"]
    methods = ["Standardized", "Converted Only"]
    width = 0.35
    x = np.arange(len(genders))
    colors = {"Standardized": "#2E86AB", "Converted Only": "#E9C46A"}
    for i, method in enumerate(methods):
        vals = []
        errs = []
        for g in genders:
            row = plot_df[(plot_df["gender"] == g) & (plot_df["method"] == method)].iloc[0]
            vals.append(row["mean_improvement_sec"])
            errs.append(
                [
                    row["mean_improvement_sec"] - row["ci_low"],
                    row["ci_high"] - row["mean_improvement_sec"],
                ]
            )
        errs = np.array(errs).T
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=method,
            color=colors[method],
            alpha=0.9,
            yerr=errs,
            capsize=4,
            error_kw={"ecolor": "black"},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(genders)
    ax.set_ylabel("Mean first→last improvement (s)")
    ax.set_title("Weather inflation of apparent improvement\n(Converted Only vs Standardized)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("PAPER ENRICHMENT ANALYSES")
    print("=" * 60)
    print(
        f"Reproducibility: random_seed={RANDOM_SEED}, "
        f"bootstrap_replicates={BOOTSTRAP_REPLICATES}; SVR is deterministic."
    )
    with open(os.path.join(output_dir, "reproducibility.json"), "w") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "random_forest_random_state": RANDOM_SEED,
                "svr_randomness": "deterministic",
            },
            f,
            indent=2,
        )

    print("\nLoading Standardized and Converted Only data (nrcd batch)...")
    df_conv, df_std = standardize_both_tiers()

    print("Building athlete-season tables...")
    as_std = build_athlete_season_table(df_std, "standardized_to_target")
    as_conv = build_athlete_season_table(df_conv, "standardized_to_target")
    as_std.to_csv(os.path.join(output_dir, "athlete_season_improvement.csv"), index=False)
    print(f"  Standardized athlete-seasons: {len(as_std)}")
    print(f"  Converted athlete-seasons: {len(as_conv)}")

    print("\n1. Race-count dose-response...")
    dose = race_count_dose_response(as_std)
    dose_path = os.path.join(output_dir, "race_count_dose_response.csv")
    dose.to_csv(dose_path, index=False)
    plot_dose_response(dose, os.path.join(output_dir, "race_count_dose_response.pdf"))
    print(dose[dose["race_bin"] != "Kruskal-Wallis"].to_string(index=False))

    print("\n2. Weather inflation (Standardized vs Converted)...")
    infl = weather_inflation(as_std, as_conv)
    infl.to_csv(os.path.join(output_dir, "weather_inflation.csv"), index=False)
    plot_weather_inflation(infl, os.path.join(output_dir, "weather_inflation.pdf"))
    print(infl.to_string(index=False))

    print("\n3. Starting ability × race count...")
    start = starting_ability_by_race_count(as_std)
    start.to_csv(os.path.join(output_dir, "starting_ability_by_race_count.csv"), index=False)
    plot_dose_by_starting_quartile(
        start, os.path.join(output_dir, "race_count_dose_by_starting_quartile.pdf")
    )
    print(f"  Wrote {len(start)} strata (≥10 athlete-seasons each)")

    print("\n4. Season-to-season retention...")
    ret = retention_analysis(as_std, df_std)
    ret.to_csv(os.path.join(output_dir, "season_retention.csv"), index=False)
    print(ret.to_string(index=False))

    print("\n5. Early-window prediction...")
    early = early_window_prediction(df_std)
    early.to_csv(os.path.join(output_dir, "early_window_prediction.csv"), index=False)
    print(early.sort_values(["gender", "window", "features", "r2"], ascending=[True, True, True, False]).to_string(index=False))

    print("\n5b. Early-window decision analysis (baseline skill, SER, calibration)...")
    decision, calib = early_window_decision_analysis(df_std, output_dir)
    print(decision.to_string(index=False))
    print(f"  Calibration bins: {len(calib)}; figure early_window_decision.pdf")

    # Compact summary markdown for findings sync
    summary_lines = [
        "# Enrichment analysis summary",
        "",
        f"Generated under `{output_dir}/`.",
        "",
        "## Dose-response (Standardized first→last improvement, seconds, median)",
        "",
    ]
    for _, row in dose[dose["race_bin"] != "Kruskal-Wallis"].iterrows():
        summary_lines.append(
            f"- {row['gender']} {row['race_bin']} races: "
            f"median={row['median_improvement_sec']:.1f}s "
            f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}], "
            f"mean={row['mean_improvement_sec']:.1f}s, "
            f"n={row['n']}, %improved={row['pct_improved']:.1f}%, "
            f"d vs 2={row['cohens_d_vs_2']:.2f}"
        )
    kw = dose[dose["race_bin"] == "Kruskal-Wallis"]
    for _, row in kw.iterrows():
        summary_lines.append(
            f"- {row['gender']} Kruskal-Wallis H={row['cohens_d_vs_2']:.2f}, "
            f"p={row['mannwhitney_p_vs_2']:.2e}"
        )
    summary_lines += ["", "## Weather inflation", ""]
    for _, row in infl.iterrows():
        if row["method"] == "Converted − Standardized":
            summary_lines.append(
                f"- {row['gender']} Converted−Std: "
                f"{row['mean_improvement_sec']:.1f}s "
                f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}], "
                f"Wilcoxon p={row['pct_improved']:.2e}"
            )
        else:
            summary_lines.append(
                f"- {row['gender']} {row['method']}: "
                f"{row['mean_improvement_sec']:.1f}s "
                f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}]"
            )
    with open(os.path.join(output_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nWrote {os.path.join(output_dir, 'SUMMARY.md')}")
    print("Done.")


if __name__ == "__main__":
    main()
