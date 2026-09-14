"""Robustness / sensitivity checks (ability match, team confounders, reliability, weather).

1. Ability-matched / IPW individual volume (2 vs 4+ races).
2. Team confounder proxies: roster size, years-active, mean meet travel,
   and within-team demeaned associations.
4. Full-sample reliability alternatives (test-retest; simulation noise floor
   on the >=2-race prediction estimand).
5. Weather-coefficient holdout: re-fit environment residual a=t_conv-t_std.

Explicitly does NOT use historical-era race counts as experience covariates.
See WHY_NOT_HISTORICAL_EXPERIENCE.md written by this script.

Outputs: output/rq1/robustness_checks/
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from _setup_paths import setup_paths

setup_paths()

from load_nrcd_data import (
    COMPREHENSIVE_ERA_START,
    get_data_dir,
    load_cross_country_results,
    load_tables,
)
from paper_enrichment_analyses import build_athlete_season_table
from underexplored_mechanisms import _load_race_frame, build_team_roster_metrics
from utils import standardize_both_tiers, _prepare_nrcd_frame, _ensure_join_columns

RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(
    Path(__file__).resolve().parents[1], "output", "rq1", "robustness_checks"
)
SPLIT_HALF_RHO = {"M": 0.22822087419717996, "F": 0.2787539600162913}
PRIMARY_R2 = {"M": 0.043, "F": -0.029}
EARTH_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    rlat1, rlon1, rlat2, rlon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def write_historical_experience_caveat(out_dir: str) -> dict:
    """Document why pre-2023 race counts are not usable experience covariates."""
    tables = load_tables()
    hist = load_cross_country_results(era="historical", tables=tables)
    comp = load_cross_country_results(era="comprehensive", tables=tables)
    hist = hist.copy()
    hist["start_date"] = pd.to_datetime(hist["start_date"], errors="coerce")
    hist["year"] = hist["start_date"].dt.year
    hy = hist.groupby(["athlete_id", "year"]).size()
    ha = set(hist["athlete_id"].dropna().unique())
    ca = set(comp["athlete_id"].dropna().unique())
    both = ha & ca

    meet = tables["meet"].copy()
    meet["start_date"] = pd.to_datetime(meet["start_date"], errors="coerce")
    hist_meets = set(meet.loc[meet["start_date"] < COMPREHENSIVE_ERA_START, "meet_id"])
    cd = tables["course_details"]
    hcd = cd[cd["meet_id"].isin(hist_meets)]
    temp_cov = float(hcd["temperature"].notna().mean()) if len(hcd) else 0.0

    stats_d = {
        "historical_results": int(len(hist)),
        "comprehensive_results": int(len(comp)),
        "historical_athletes": int(len(ha)),
        "comprehensive_athletes": int(len(ca)),
        "athletes_in_both_eras": int(len(both)),
        "historical_athlete_years": int(len(hy)),
        "hist_median_races_per_athlete_year": float(hy.median()) if len(hy) else None,
        "hist_mean_races_per_athlete_year": float(hy.mean()) if len(hy) else None,
        "hist_pct_athlete_years_with_exactly_1_race": float((hy == 1).mean())
        if len(hy)
        else None,
        "hist_course_details_rows": int(len(hcd)),
        "hist_course_details_temp_coverage": temp_cov,
        "decision": "skip_historical_experience_covariates",
    }
    with open(os.path.join(out_dir, "historical_era_fragmentation.json"), "w") as f:
        json.dump(stats_d, f, indent=2)

    pct1 = 100 * stats_d["hist_pct_athlete_years_with_exactly_1_race"]
    pct_both = 100 * stats_d["athletes_in_both_eras"] / max(stats_d["comprehensive_athletes"], 1)
    lines = [
        "# Why we do not add historical-era experience covariates",
        "",
        "Reviewers often ask for pre-2023 racing experience. We **deliberately",
        "omit** historical race counts as predictors of comprehensive-era",
        "improvement because the historical export is a **fragmentary** record,",
        "not a complete season log.",
        "",
        "## Empirical fragmentation (this export)",
        "",
        f"- Historical XC results: **{stats_d['historical_results']:,}**; "
        f"comprehensive: **{stats_d['comprehensive_results']:,}**.",
        f"- Athletes in both eras: only **{stats_d['athletes_in_both_eras']:,}** "
        f"of {stats_d['comprehensive_athletes']:,} comprehensive athletes "
        f"(~{pct_both:.1f}%).",
        f"- Among historical athlete-years, median races/season = "
        f"**{stats_d['hist_median_races_per_athlete_year']:.1f}**, and "
        f"**{pct1:.1f}%** have exactly one recorded race.",
        f"- Historical course-details temperature coverage on joined rows ~"
        f"**{100 * temp_cov:.1f}%** (vs ~97.7% comprehensive).",
        "",
        "## Why that breaks an experience covariate",
        "",
        "A recorded historical race count of 2 may mean the athlete truly raced",
        "twice -- or that half (or more) of their meets were never entered.",
        "Using that count as experience would systematically **undercount** true",
        "volume and invent a spurious low-experience class. Metadata needed to",
        "standardize those races is also mostly missing.",
        "",
        "Therefore experience / grade-year / pre-2023 load remain Limitations,",
        "not silent omissions we paper over with a bad proxy.",
        "",
        "Artifact: `historical_era_fragmentation.json`.",
        "",
    ]
    with open(os.path.join(out_dir, "WHY_NOT_HISTORICAL_EXPERIENCE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return stats_d


def ability_matched_volume(as_std: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1:1 NN match on first_time within gender x year: 4+ vs 2 races."""
    work = as_std[as_std["num_races"].isin([2]) | (as_std["num_races"] >= 4)].copy()
    work["treated"] = (work["num_races"] >= 4).astype(int)
    match_rows = []
    summary_rows = []

    for (gender, year), sub in work.groupby(["gender", "year"]):
        treated = sub[sub["treated"] == 1]
        control = sub[sub["treated"] == 0]
        if len(treated) < 15 or len(control) < 15:
            continue
        sd = float(sub["first_time"].std(ddof=0))
        caliper = 0.25 * sd if sd > 0 else 30.0
        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(control[["first_time"]].to_numpy())
        dist, idx = nn.kneighbors(treated[["first_time"]].to_numpy())
        used = set()
        pairs = []
        for i, (d, j) in enumerate(zip(dist.ravel(), idx.ravel())):
            if d > caliper:
                continue
            c_id = control.index[j]
            if c_id in used:
                continue
            used.add(c_id)
            t_row = treated.iloc[i]
            c_row = control.loc[c_id]
            pairs.append(
                {
                    "gender": gender,
                    "year": int(year),
                    "treated_athlete": t_row["athlete_id"],
                    "control_athlete": c_row["athlete_id"],
                    "treated_first": float(t_row["first_time"]),
                    "control_first": float(c_row["first_time"]),
                    "first_time_diff_sec": float(t_row["first_time"] - c_row["first_time"]),
                    "treated_improve_sec": float(t_row["total_improvement_sec"]),
                    "control_improve_sec": float(c_row["total_improvement_sec"]),
                    "improve_diff_sec": float(
                        t_row["total_improvement_sec"] - c_row["total_improvement_sec"]
                    ),
                    "treated_n_races": int(t_row["num_races"]),
                    "control_n_races": int(c_row["num_races"]),
                }
            )
        if len(pairs) < 10:
            continue
        pdf = pd.DataFrame(pairs)
        match_rows.append(pdf)
        diff = pdf["improve_diff_sec"].to_numpy()
        wstat = stats.wilcoxon(diff, alternative="two-sided")
        summary_rows.append(
            {
                "method": "1NN_caliper_0.25SD",
                "gender": "Men" if gender == "M" else "Women",
                "year": int(year),
                "n_pairs": int(len(pdf)),
                "mean_improve_diff_sec": float(diff.mean()),
                "median_improve_diff_sec": float(np.median(diff)),
                "ci_low": float(np.percentile(diff, 2.5)),
                "ci_high": float(np.percentile(diff, 97.5)),
                "wilcoxon_p": float(wstat.pvalue),
                "cohens_d": float(diff.mean() / diff.std(ddof=1))
                if diff.std(ddof=1) > 0
                else np.nan,
                "mean_abs_first_match_sec": float(pdf["first_time_diff_sec"].abs().mean()),
            }
        )

    ipw_rows = []
    for gender, sub in work.groupby("gender"):
        sub = sub.dropna(subset=["first_time", "total_improvement_sec"]).copy()
        if sub["treated"].nunique() < 2 or len(sub) < 80:
            continue
        X = np.column_stack(
            [sub["first_time"].to_numpy(float), sub["year"].to_numpy(float)]
        )
        y = sub["treated"].to_numpy(int)
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        clf.fit(Xs, y)
        p = np.clip(clf.predict_proba(Xs)[:, 1], 0.05, 0.95)
        p_t = y.mean()
        w = np.where(y == 1, p_t / p, (1 - p_t) / (1 - p))
        w = w / w.mean()
        outcome = sub["total_improvement_sec"].to_numpy(float)
        mu1 = np.average(outcome[y == 1], weights=w[y == 1])
        mu0 = np.average(outcome[y == 0], weights=w[y == 0])
        ipw_rows.append(
            {
                "method": "IPW_ATE",
                "gender": "Men" if gender == "M" else "Women",
                "year": "2023-2025",
                "n_pairs": int(len(sub)),
                "mean_improve_diff_sec": float(mu1 - mu0),
                "median_improve_diff_sec": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "wilcoxon_p": np.nan,
                "cohens_d": np.nan,
                "mean_abs_first_match_sec": np.nan,
                "ate_treated_mean": float(mu1),
                "ate_control_mean": float(mu0),
            }
        )

    matches = pd.concat(match_rows, ignore_index=True) if match_rows else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if ipw_rows:
        summary = pd.concat([summary, pd.DataFrame(ipw_rows)], ignore_index=True)
    if not matches.empty:
        pool_rows = []
        for gender, sub in matches.groupby("gender"):
            diff = sub["improve_diff_sec"].to_numpy()
            wstat = stats.wilcoxon(diff, alternative="two-sided")
            pool_rows.append(
                {
                    "method": "1NN_caliper_pooled_years",
                    "gender": "Men" if gender == "M" else "Women",
                    "year": "2023-2025",
                    "n_pairs": int(len(sub)),
                    "mean_improve_diff_sec": float(diff.mean()),
                    "median_improve_diff_sec": float(np.median(diff)),
                    "ci_low": float(np.percentile(diff, 2.5)),
                    "ci_high": float(np.percentile(diff, 97.5)),
                    "wilcoxon_p": float(wstat.pvalue),
                    "cohens_d": float(diff.mean() / diff.std(ddof=1))
                    if diff.std(ddof=1) > 0
                    else np.nan,
                    "mean_abs_first_match_sec": float(
                        sub["first_time_diff_sec"].abs().mean()
                    ),
                }
            )
        summary = pd.concat([summary, pd.DataFrame(pool_rows)], ignore_index=True)
    return matches, summary


def _team_year_travel(df: pd.DataFrame, meet: pd.DataFrame) -> pd.DataFrame:
    m = meet[["meet_id", "meet_latitude", "meet_longitude", "start_date"]].copy()
    m["start_date"] = pd.to_datetime(m["start_date"], errors="coerce")
    races = df[["team_id", "meet_id", "gender", "start_date"]].dropna().copy()
    races["start_date"] = pd.to_datetime(races["start_date"], errors="coerce")
    races["year"] = races["start_date"].dt.year
    races = races.merge(m[["meet_id", "meet_latitude", "meet_longitude"]], on="meet_id")
    races = races.dropna(subset=["meet_latitude", "meet_longitude", "team_id"])

    rows = []
    for (team_id, year, gender), sub in races.groupby(["team_id", "year", "gender"]):
        coords = sub.drop_duplicates("meet_id")[
            ["meet_latitude", "meet_longitude"]
        ].to_numpy(float)
        if len(coords) < 2:
            mean_d = 0.0
        else:
            lat0, lon0 = coords.mean(axis=0)
            d = _haversine_km(coords[:, 0], coords[:, 1], lat0, lon0)
            mean_d = float(np.mean(d))
        rows.append(
            {
                "team_id": int(team_id),
                "year": int(year),
                "gender": gender,
                "n_meets": int(len(coords)),
                "mean_travel_from_centroid_km": mean_d,
            }
        )
    return pd.DataFrame(rows)


def team_confounder_models(roster: pd.DataFrame, df: pd.DataFrame):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    meet = pd.read_csv(os.path.join(get_data_dir(), "meet.csv"))
    travel = _team_year_travel(df, meet)
    # Time-varying longevity: team appeared in the prior comprehensive season.
    keys = roster[["team_id", "gender", "year"]].drop_duplicates()
    prior_keys = keys.copy()
    prior_keys["year"] = prior_keys["year"] + 1
    prior_keys["appeared_prior_year"] = 1
    data = roster.merge(travel, on=["team_id", "year", "gender"], how="left")
    data = data.merge(
        prior_keys[["team_id", "gender", "year", "appeared_prior_year"]],
        on=["team_id", "gender", "year"],
        how="left",
    )
    data["appeared_prior_year"] = data["appeared_prior_year"].fillna(0).astype(int)
    data["mean_travel_from_centroid_km"] = data["mean_travel_from_centroid_km"].fillna(0)
    data["top15"] = data["top15"].astype(int)
    data = data.sort_values(["team_id", "gender", "year"])
    data["years_active_to_date"] = data.groupby(["team_id", "gender"]).cumcount() + 1

    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0

    data["z_maxrace"] = _z(data["max_race_count"])
    data["z_depth"] = _z(data["n_athletes_ge3"])
    data["z_size"] = _z(data["n_athletes"])
    data["z_years"] = _z(data["years_active_to_date"])
    data["z_travel"] = _z(data["mean_travel_from_centroid_km"])
    data["prior"] = data["appeared_prior_year"]

    specs = [
        ("maxrace", "top15 ~ z_maxrace + C(gender) + C(year)"),
        (
            "maxrace_size_travel",
            "top15 ~ z_maxrace + z_size + z_travel + C(gender) + C(year)",
        ),
        (
            "depth_size_travel",
            "top15 ~ z_depth + z_size + z_travel + C(gender) + C(year)",
        ),
        (
            "full",
            "top15 ~ z_maxrace + z_depth + z_size + z_travel + C(gender) + C(year)",
        ),
    ]
    rows = []
    for name, formula in specs:
        gee = smf.gee(
            formula,
            groups="team_id",
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        ).fit()
        for term in gee.params.index:
            if term == "Intercept" or str(term).startswith("C("):
                continue
            est = float(gee.params[term])
            ci = gee.conf_int().loc[term]
            rows.append(
                {
                    "model": name,
                    "term": term,
                    "odds_ratio": float(np.exp(est)),
                    "or_ci_low": float(np.exp(ci[0])),
                    "or_ci_high": float(np.exp(ci[1])),
                    "p_value": float(gee.pvalues[term]),
                    "n_obs": int(len(data)),
                    "n_teams": int(data["team_id"].nunique()),
                    "note": "OR per SD",
                }
            )

    dem = data.copy()
    for col in ["z_maxrace", "z_depth", "z_size", "z_travel", "top15"]:
        dem[f"{col}_dm"] = dem[col] - dem.groupby("team_id")[col].transform("mean")
    y = dem["top15_dm"].to_numpy(float)
    X = dem[["z_maxrace_dm", "z_depth_dm", "z_size_dm", "z_travel_dm"]].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() > 50:
        Xo = sm.add_constant(X[ok])
        try:
            fit = sm.OLS(y[ok], Xo).fit(
                cov_type="cluster", cov_kwds={"groups": dem.loc[ok, "team_id"]}
            )
        except Exception:
            fit = sm.OLS(y[ok], Xo).fit(cov_type="HC1")
        names = ["const", "z_maxrace_dm", "z_depth_dm", "z_size_dm", "z_travel_dm"]
        ci = np.asarray(fit.conf_int())
        for i, term in enumerate(names):
            if term == "const":
                continue
            rows.append(
                {
                    "model": "within_team_demean_LPM",
                    "term": term,
                    "odds_ratio": float(fit.params[i]),
                    "or_ci_low": float(ci[i, 0]),
                    "or_ci_high": float(ci[i, 1]),
                    "p_value": float(fit.pvalues[i]),
                    "n_obs": int(ok.sum()),
                    "n_teams": int(dem.loc[ok, "team_id"].nunique()),
                    "note": "LPM coef on demeaned predictors (not OR)",
                }
            )

    # Descriptive longevity (avoid logistic separation on nearly-perfect predictors).
    for label, mask in (
        ("top15", data["top15"] == 1),
        ("not_top15", data["top15"] == 0),
    ):
        sub = data.loc[mask]
        rows.append(
            {
                "model": "descriptive_longevity",
                "term": f"share_appeared_prior_{label}",
                "odds_ratio": float(sub["appeared_prior_year"].mean()),
                "or_ci_low": np.nan,
                "or_ci_high": np.nan,
                "p_value": np.nan,
                "n_obs": int(len(sub)),
                "n_teams": int(sub["team_id"].nunique()),
                "note": "share with prior-year presence (not an OR)",
            }
        )
    return pd.DataFrame(rows), data


def full_sample_reliability(as_std: pd.DataFrame, race_std: pd.DataFrame) -> pd.DataFrame:
    rows = []
    as_std = as_std.copy()
    race_std = race_std.copy()
    race_std["start_date"] = pd.to_datetime(race_std["start_date"], errors="coerce")
    race_std["year"] = race_std["start_date"].dt.year

    for gender, label in [("M", "Men"), ("F", "Women")]:
        g = as_std[as_std["gender"] == gender].copy()
        pairs = []
        for _, sub in g.groupby("athlete_id"):
            sub = sub.sort_values("year")
            years = sub["year"].to_numpy()
            rates = sub["improvement_rate"].to_numpy(float)
            for i in range(len(sub) - 1):
                if years[i + 1] - years[i] == 1 and np.isfinite(rates[i]) and np.isfinite(
                    rates[i + 1]
                ):
                    pairs.append((rates[i], rates[i + 1]))
        if len(pairs) >= 30:
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            r, p = stats.pearsonr(a, b)
            rho_s, p_s = stats.spearmanr(a, b)
        else:
            r = p = rho_s = p_s = np.nan
            a = np.array([])

        g_r = race_std[race_std["gender"] == gender]
        resid_vars = []
        for (_, _), races in g_r.groupby(["athlete_id", "year"]):
            races = races.sort_values("start_date")
            yv = races["standardized_to_target"].to_numpy(float)
            if len(yv) < 3 or not np.all(np.isfinite(yv)):
                continue
            x = np.arange(len(yv), dtype=float)
            beta = np.polyfit(x, yv, 1)
            resid = yv - np.polyval(beta, x)
            resid_vars.append(float(resid.var(ddof=1)))
        sigma_eps2 = float(np.nanmean(resid_vars)) if resid_vars else np.nan
        deltas = g["total_improvement_sec"].to_numpy(float)
        deltas = deltas[np.isfinite(deltas)]
        var_delta = float(np.var(deltas, ddof=1)) if len(deltas) > 2 else np.nan
        phi = (2.0 * sigma_eps2 / var_delta) if var_delta and var_delta > 0 else np.nan
        rho_delta = float(np.clip(1.0 - phi, 0.0, 1.0)) if np.isfinite(phi) else np.nan

        r2 = PRIMARY_R2[gender]
        rho_ge4 = SPLIT_HALF_RHO[gender]
        rate_ceiling = max(float(r), 0.0) if np.isfinite(r) else np.nan
        if np.isfinite(rho_delta) and np.isfinite(rho_ge4):
            conservative = float(min(rho_ge4, rho_delta))
        elif np.isfinite(rho_ge4):
            conservative = float(rho_ge4)
        else:
            conservative = np.nan

        ser_ge4 = r2 / rho_ge4 if rho_ge4 else np.nan
        ser_cons = r2 / conservative if conservative and conservative > 0 else np.nan

        rows.append(
            {
                "gender": label,
                "n_athlete_seasons_ge2": int(len(g)),
                "n_test_retest_pairs": int(len(a)),
                "test_retest_pearson_r": float(r) if np.isfinite(r) else np.nan,
                "test_retest_pearson_p": float(p) if np.isfinite(p) else np.nan,
                "test_retest_spearman_rho": float(rho_s) if np.isfinite(rho_s) else np.nan,
                "test_retest_spearman_p": float(p_s) if np.isfinite(p_s) else np.nan,
                "rate_ceiling_from_retest": rate_ceiling,
                "sigma_eps2_trend_resid": sigma_eps2,
                "var_delta_ge2": var_delta,
                "noise_floor_phi_ge2": phi,
                "reliability_delta_approx": rho_delta,
                "split_half_rho_ge4": rho_ge4,
                "conservative_r2_ceiling": conservative,
                "heldout_r2": r2,
                "SER_vs_split_half_ge4": ser_ge4,
                "SER_vs_conservative_ceiling": ser_cons,
            }
        )
    return pd.DataFrame(rows)


def weather_coefficient_holdout(df_conv: pd.DataFrame, df_std: pd.DataFrame):
    key = ["athlete_id", "meet_id", "gender"]
    a = df_conv[key + ["start_date", "standardized_to_target"]].merge(
        df_std[key + ["standardized_to_target"]],
        on=key,
        suffixes=("_conv", "_std"),
    )
    weather_cols = [
        c
        for c in (
            "temperature",
            "dew_point",
            "elevation_gain",
            "elevation_loss",
            "meet_elevation",
            "barometric_pressure",
        )
        if c in df_std.columns
    ]
    if weather_cols:
        a = a.merge(df_std[key + weather_cols].drop_duplicates(key), on=key, how="left")
    a["a"] = a["standardized_to_target_conv"] - a["standardized_to_target_std"]
    a["start_date"] = pd.to_datetime(a["start_date"], errors="coerce")
    a["year"] = a["start_date"].dt.year
    a["doy"] = a["start_date"].dt.dayofyear
    feats = [c for c in weather_cols if c in a.columns] + ["doy"]
    for c in feats:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    a = a.dropna(subset=["a"] + feats)
    a = a[np.isfinite(a["a"])]

    hold_rows = []
    years = sorted(y for y in a["year"].dropna().unique() if y in (2023, 2024, 2025))
    for hold in years:
        train = a[a["year"] != hold]
        test = a[a["year"] == hold]
        if len(train) < 500 or len(test) < 200:
            continue
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[feats])
        Xte = scaler.transform(test[feats])
        model = Ridge(alpha=1.0)
        model.fit(Xtr, train["a"])
        pred = model.predict(Xte)
        resid = test["a"].to_numpy() - pred
        corr = float(np.corrcoef(test["a"], pred)[0, 1])
        ss_tot = np.sum((test["a"] - test["a"].mean()) ** 2)
        hold_rows.append(
            {
                "holdout_year": int(hold),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "features": ",".join(feats),
                "corr_pred_vs_actual_a": corr,
                "mae_a_sec": float(np.mean(np.abs(resid))),
                "rmse_a_sec": float(np.sqrt(np.mean(resid**2))),
                "r2_a": float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 0 else np.nan,
                "mean_actual_a": float(test["a"].mean()),
                "mean_pred_a": float(pred.mean()),
            }
        )

    season_rows = []
    if "temperature" in a.columns:
        for year, sub in a.groupby("year"):
            sub = sub.dropna(subset=["temperature", "a"])
            if len(sub) < 50:
                continue
            r, pval = stats.pearsonr(sub["temperature"], sub["a"])
            season_rows.append(
                {
                    "year": int(year),
                    "n": int(len(sub)),
                    "corr_temp_vs_a": float(r),
                    "p": float(pval),
                }
            )
    return pd.DataFrame(hold_rows), pd.DataFrame(season_rows)


def _plot_summary(matched_sum, reliability, weather_hold, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    fig.patch.set_facecolor("white")

    ax = axes[0]
    pool = matched_sum[matched_sum["method"] == "1NN_caliper_pooled_years"]
    if not pool.empty:
        yerr = np.vstack(
            [
                pool["mean_improve_diff_sec"] - pool["ci_low"],
                pool["ci_high"] - pool["mean_improve_diff_sec"],
            ]
        )
        ax.bar(
            pool["gender"],
            pool["mean_improve_diff_sec"],
            color=["#4C72B0", "#C44E52"][: len(pool)],
            yerr=yerr,
            capsize=4,
        )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel("Matched improve diff (s)\n(4+ minus 2 races)")
        ax.set_title("Ability-matched volume")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if not reliability.empty:
        x = np.arange(len(reliability))
        ax.bar(x - 0.2, reliability["split_half_rho_ge4"], 0.4, label="Split-half >=4")
        ax.bar(
            x + 0.2,
            reliability["conservative_r2_ceiling"],
            0.4,
            label="Conservative >=2",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(reliability["gender"])
        ax.set_ylabel("Reliability / R2 ceiling")
        ax.set_title("Reliability bounds")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if not weather_hold.empty:
        ax.bar(
            weather_hold["holdout_year"].astype(str),
            weather_hold["corr_pred_vs_actual_a"],
            color="#55A868",
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("corr(pred a, actual a)")
        ax.set_title("Weather residual holdout")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "robustness_checks_summary.pdf"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary_md(
    out_dir, hist_stats, matched_sum, team_models, reliability, weather_hold
) -> None:
    pct1 = 100 * hist_stats.get("hist_pct_athlete_years_with_exactly_1_race", 0)
    pct_temp = 100 * hist_stats.get("hist_course_details_temp_coverage", 0)
    lines = [
        "# Robustness / sensitivity checks -- summary",
        "",
        "Script: `scripts/robustness_checks.py`.",
        "",
        "## Skipped: historical experience covariates",
        "",
        "See `WHY_NOT_HISTORICAL_EXPERIENCE.md`. Historical seasons are fragmentary",
        f"(**{pct1:.0f}%** of historical athlete-years have only one recorded race;",
        f"weather coverage ~{pct_temp:.0f}% on course-detail rows).",
        "Race counts would understate true volume -- often by a large fraction.",
        "",
        "## 1. Ability-matched volume (4+ vs 2 races)",
        "",
    ]
    pool = matched_sum[
        matched_sum["method"].isin(["1NN_caliper_pooled_years", "IPW_ATE"])
    ]
    if not pool.empty:
        lines += [
            "| Method | Gender | n | Mean d improve (s) | p / note |",
            "|--------|--------|---|--------------------|----------|",
        ]
        for _, r in pool.iterrows():
            pnote = (
                f"Wilcoxon p={r['wilcoxon_p']:.3g}"
                if pd.notna(r.get("wilcoxon_p"))
                else "IPW ATE"
            )
            lines.append(
                f"| {r['method']} | {r['gender']} | {int(r['n_pairs'])} | "
                f"{r['mean_improve_diff_sec']:.1f} | {pnote} |"
            )
    lines += ["", "## 2. Team confounder proxies", ""]
    if not team_models.empty:
        show = team_models[
            team_models["model"].isin(
                [
                    "maxrace",
                    "maxrace_size_travel",
                    "full",
                    "within_team_demean_LPM",
                    "descriptive_longevity",
                ]
            )
        ]
        lines += [
            "| Model | Term | OR or LPM | 95% CI | p |",
            "|-------|------|-----------|--------|---|",
        ]
        for _, r in show.iterrows():
            lines.append(
                f"| {r['model']} | {r['term']} | {r['odds_ratio']:.3f} | "
                f"[{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}] | {r['p_value']:.3g} |"
            )
    lines += ["", "## 4. Full-sample reliability", ""]
    if not reliability.empty:
        lines += [
            "| Gender | Test-retest r | phi (>=2) | Cons. ceiling | SER (cons.) |",
            "|--------|---------------|-----------|---------------|-------------|",
        ]
        for _, r in reliability.iterrows():
            lines.append(
                f"| {r['gender']} | {r['test_retest_pearson_r']:.3f} | "
                f"{r['noise_floor_phi_ge2']:.2f} | {r['conservative_r2_ceiling']:.3f} | "
                f"{r['SER_vs_conservative_ceiling']:.3f} |"
            )
    lines += ["", "## 5. Weather residual holdout", ""]
    if not weather_hold.empty:
        lines += [
            "| Holdout year | corr(pred,a) | MAE (s) | R2 |",
            "|--------------|--------------|---------|----|",
        ]
        for _, r in weather_hold.iterrows():
            lines.append(
                f"| {int(r['holdout_year'])} | {r['corr_pred_vs_actual_a']:.3f} | "
                f"{r['mae_a_sec']:.2f} | {r['r2_a']:.3f} |"
            )
    lines.append("")
    with open(os.path.join(out_dir, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("REVIEWER GAP ANALYSES (1, 2, 4, 5; skip historical experience)")
    print("=" * 60)

    print("\n0. Historical fragmentation caveat...")
    hist_stats = write_historical_experience_caveat(output_dir)
    print(
        "  hist 1-race athlete-years: "
        f"{100 * hist_stats['hist_pct_athlete_years_with_exactly_1_race']:.1f}%"
    )

    print("\nLoading Standardized + Converted...")
    df_conv, df_std = standardize_both_tiers()
    tables = load_tables()
    prepared = _prepare_nrcd_frame(
        _ensure_join_columns(df_std, meet_df=tables["meet"]),
        tables["course_details"],
    )
    for c in (
        "temperature",
        "dew_point",
        "elevation_gain",
        "elevation_loss",
        "meet_elevation",
        "barometric_pressure",
    ):
        if c in prepared.columns and c not in df_std.columns:
            df_std[c] = prepared[c].to_numpy()
        elif c in prepared.columns:
            df_std[c] = prepared[c].to_numpy()

    as_std = build_athlete_season_table(df_std)
    race_df = _load_race_frame()
    roster = build_team_roster_metrics(race_df)

    print("\n1. Ability-matched volume...")
    matches, matched_sum = ability_matched_volume(as_std)
    matches.to_csv(os.path.join(output_dir, "ability_matched_pairs.csv"), index=False)
    matched_sum.to_csv(os.path.join(output_dir, "ability_matched_summary.csv"), index=False)
    print(matched_sum.to_string(index=False))

    print("\n2. Team confounder proxies...")
    team_models, team_panel = team_confounder_models(roster, race_df)
    team_models.to_csv(os.path.join(output_dir, "team_confounder_models.csv"), index=False)
    team_panel[
        [
            "year",
            "gender",
            "team_id",
            "n_athletes",
            "max_race_count",
            "n_athletes_ge3",
            "appeared_prior_year",
            "years_active_to_date",
            "mean_travel_from_centroid_km",
            "top15",
        ]
    ].to_csv(os.path.join(output_dir, "team_confounder_panel.csv"), index=False)
    print(team_models.to_string(index=False))

    print("\n4. Full-sample reliability...")
    reliability = full_sample_reliability(as_std, df_std)
    reliability.to_csv(os.path.join(output_dir, "full_sample_reliability.csv"), index=False)
    print(reliability.to_string(index=False))

    print("\n5. Weather coefficient holdout...")
    weather_hold, weather_season = weather_coefficient_holdout(df_conv, df_std)
    weather_hold.to_csv(os.path.join(output_dir, "weather_holdout.csv"), index=False)
    weather_season.to_csv(os.path.join(output_dir, "weather_temp_vs_a.csv"), index=False)
    print(weather_hold.to_string(index=False))

    _plot_summary(matched_sum, reliability, weather_hold, output_dir)
    write_summary_md(
        output_dir, hist_stats, matched_sum, team_models, reliability, weather_hold
    )
    with open(os.path.join(output_dir, "reproducibility.json"), "w") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "primary_r2": PRIMARY_R2,
                "split_half_rho": SPLIT_HALF_RHO,
                "skipped": "historical_experience_covariates",
            },
            f,
            indent=2,
        )
    print(f"\nWrote {output_dir}/")


if __name__ == "__main__":
    main()
