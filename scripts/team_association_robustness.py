"""Robustness analyses for the team race-frequency / nationals association.

1. Race-count threshold sensitivity (k in {3,4,5} vs top-15).
2. Cluster-robust continuous race count → top-15.
3. Partial pooling of per-cell risk ratios.
4. Placement-threshold sensitivity: top-k for k in {5,10,15,20,25} at the
   primary >=4 race cut — a curve showing the association is not an artifact
   of choosing top-15 specifically.
5. Selection-robust checks: within-team first differences across consecutive
   seasons, and roster-size-adjusted GEE models (depth + max races + n athletes).
6. Non-top-7 (bench) depth vs roster size: athletes outside the season-best
   top 7 with ≥3 starts — is the “depth” signal independent of club size?

Output: output/rq3/team_association_robustness/
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from _setup_paths import setup_paths

setup_paths()

from load_nrcd_data import get_data_dir
from utils import standardize_convert_exclude_nationals_df

# Top-15 nationals rosters, imported from the primary overlap module so the two
# analyses stay in sync.
from nationals_overlap_analysis import (
    analyze_2023_mens_overlap,
    analyze_2023_womens_overlap,
    analyze_2024_mens_overlap,
    analyze_2024_womens_overlap,
    analyze_2025_mens_overlap,
    analyze_2025_womens_overlap,
    get_total_teams_with_3plus_athletes,
)

CELLS = [
    (2023, "M", analyze_2023_mens_overlap),
    (2023, "F", analyze_2023_womens_overlap),
    (2024, "M", analyze_2024_mens_overlap),
    (2024, "F", analyze_2024_womens_overlap),
    (2025, "M", analyze_2025_mens_overlap),
    (2025, "F", analyze_2025_womens_overlap),
]


def team_year_maxrace(year: int, gender: str) -> pd.DataFrame:
    """One row per qualifying team (>=3 participating athletes of `gender`) with
    that team's maximum single-athlete regular-season race count."""
    df = standardize_convert_exclude_nationals_df()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    ata = pd.read_csv(os.path.join(get_data_dir(), "athlete_team_association.csv"))
    ath = pd.read_csv(os.path.join(get_data_dir(), "athlete.csv"))[
        ["athlete_id", "gender"]
    ]
    ata = ata.merge(ath, on="athlete_id", how="left").dropna(
        subset=["athlete_id", "team_id", "gender"]
    )
    df = df.dropna(subset=["athlete_id", "start_date", "gender"])
    start = pd.Timestamp(year=year, month=8, day=1)
    end = pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59)
    yr = df[(df["start_date"] >= start) & (df["start_date"] <= end)]
    yr = yr[yr["gender"] == gender]
    counts = yr.groupby("athlete_id").size().reset_index(name="race_count")
    merged = counts.merge(
        ata[ata["gender"] == gender][["athlete_id", "team_id"]],
        on="athlete_id",
        how="left",
    ).dropna(subset=["team_id"])
    natl = merged.groupby("team_id")["athlete_id"].nunique()
    qualifying = natl[natl >= 3].index
    merged = merged[merged["team_id"].isin(qualifying)]
    out = (
        merged.groupby("team_id")["race_count"]
        .max()
        .reset_index(name="max_race_count")
    )
    out["year"] = year
    out["gender"] = gender
    return out


def _rr_2x2(a: int, b: int, c: int, d: int) -> Tuple[float, float, float]:
    """Risk ratio of top-15 for (exposed vs not); returns RR, logRR, var(logRR)."""
    a_, b_, c_, d_ = (max(a, 0), max(b, 0), max(c, 0), max(d, 0))
    # Haldane-Anscombe correction if any zero cell.
    if min(a_, b_, c_, d_) == 0:
        a_, b_, c_, d_ = a_ + 0.5, b_ + 0.5, c_ + 0.5, d_ + 0.5
    r1 = a_ / (a_ + b_)
    r0 = c_ / (c_ + d_)
    rr = r1 / r0 if r0 > 0 else np.nan
    log_rr = np.log(rr)
    var = 1 / a_ - 1 / (a_ + b_) + 1 / c_ - 1 / (c_ + d_)
    return rr, log_rr, var


def threshold_sensitivity() -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_cell_rows = []
    team_tables = {}
    top15_ids: Dict[Tuple[int, str], set] = {}
    for year, gender, analyze in CELLS:
        team_tables[(year, gender)] = team_year_maxrace(year, gender)
        res = analyze()
        top15_ids[(year, gender)] = set(res["nationals_teams"])

    for k in (3, 4, 5):
        for year, gender, _ in CELLS:
            tt = team_tables[(year, gender)].copy()
            total = get_total_teams_with_3plus_athletes(year, gender)
            top15 = top15_ids[(year, gender)]
            tt["exposed"] = tt["max_race_count"] >= k
            tt["top15"] = tt["team_id"].isin(top15)
            a = int((tt["exposed"] & tt["top15"]).sum())
            exposed_total = int(tt["exposed"].sum())
            b = exposed_total - a
            top15_total = len(top15)
            c = top15_total - a
            d = total - exposed_total - c
            rr, log_rr, var = _rr_2x2(a, b, c, d)
            chi2, p, _, _ = stats.chi2_contingency([[a, b], [c, max(d, 0)]])
            per_cell_rows.append(
                {
                    "threshold": f">= {k}",
                    "year": year,
                    "gender": gender,
                    "total_teams": total,
                    "exposed_teams": exposed_total,
                    "top15_exposed": a,
                    "risk_ratio": rr,
                    "log_rr": log_rr,
                    "var_log_rr": var,
                    "chi2_p": p,
                }
            )
    per_cell = pd.DataFrame(per_cell_rows)

    # Pooled (DL random effects) RR per threshold across the six cells.
    pooled_rows = []
    for k in (3, 4, 5):
        sub = per_cell[per_cell["threshold"] == f">= {k}"].dropna(
            subset=["log_rr", "var_log_rr"]
        )
        pooled_rows.append(_dersimonian_laird(sub, f">= {k}"))
    pooled = pd.DataFrame(pooled_rows)
    return per_cell, pooled


def _dersimonian_laird(sub: pd.DataFrame, label: str) -> Dict[str, float]:
    y = sub["log_rr"].to_numpy()
    v = sub["var_log_rr"].to_numpy()
    w = 1 / v
    y_fixed = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - y_fixed) ** 2)
    dfree = len(y) - 1
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    tau2 = max(0.0, (q - dfree) / c) if c > 0 else 0.0
    w_star = 1 / (v + tau2)
    y_re = np.sum(w_star * y) / np.sum(w_star)
    se_re = np.sqrt(1 / np.sum(w_star))
    return {
        "threshold": label,
        "k_cells": len(y),
        "pooled_rr": float(np.exp(y_re)),
        "ci_low": float(np.exp(y_re - 1.96 * se_re)),
        "ci_high": float(np.exp(y_re + 1.96 * se_re)),
        "tau2": float(tau2),
        "Q": float(q),
        "Q_p": float(1 - stats.chi2.cdf(q, dfree)) if dfree > 0 else np.nan,
    }


def shrunken_cell_rr(per_cell: pd.DataFrame, threshold: str = ">= 4") -> pd.DataFrame:
    """Empirical-Bayes shrunken per-cell RRs at a given threshold."""
    sub = per_cell[per_cell["threshold"] == threshold].dropna(
        subset=["log_rr", "var_log_rr"]
    ).copy()
    dl = _dersimonian_laird(sub, threshold)
    mu = np.log(dl["pooled_rr"])
    tau2 = dl["tau2"]
    rows = []
    for _, r in sub.iterrows():
        v = r["var_log_rr"]
        shrink = tau2 / (tau2 + v) if (tau2 + v) > 0 else 0.0
        post = mu + shrink * (r["log_rr"] - mu)
        rows.append(
            {
                "year": int(r["year"]),
                "gender": r["gender"],
                "raw_rr": float(np.exp(r["log_rr"])),
                "shrunken_rr": float(np.exp(post)),
                "shrinkage_weight": float(shrink),
            }
        )
    return pd.DataFrame(rows)


def continuous_model(team_tables: Dict = None) -> pd.DataFrame:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    frames = []
    top15_ids = {}
    for year, gender, analyze in CELLS:
        tt = team_year_maxrace(year, gender)
        top15_ids[(year, gender)] = set(analyze()["nationals_teams"])
        tt["top15"] = tt["team_id"].isin(top15_ids[(year, gender)]).astype(int)
        frames.append(tt)
    data = pd.concat(frames, ignore_index=True)
    data["z_maxrace"] = (
        data["max_race_count"] - data["max_race_count"].mean()
    ) / data["max_race_count"].std(ddof=0)
    data["year_c"] = data["year"].astype("category")

    rows = []
    # Cluster-robust logistic (GEE) with team clusters.
    gee = smf.gee(
        "top15 ~ z_maxrace + C(gender) + C(year)",
        groups="team_id",
        data=data,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Independence(),
    ).fit()
    est = gee.params["z_maxrace"]
    ci = gee.conf_int().loc["z_maxrace"]
    rows.append(
        {
            "model": "GEE logistic (cluster-robust by team)",
            "term": "z_maxrace",
            "odds_ratio": float(np.exp(est)),
            "or_ci_low": float(np.exp(ci[0])),
            "or_ci_high": float(np.exp(ci[1])),
            "p_value": float(gee.pvalues["z_maxrace"]),
            "n_obs": int(data.shape[0]),
            "n_teams": int(data["team_id"].nunique()),
        }
    )
    # Bayesian mixed GLM with team random intercept (best-effort).
    try:
        mix = sm.BinomialBayesMixedGLM.from_formula(
            "top15 ~ z_maxrace + C(gender) + C(year)",
            {"team": "0 + C(team_id)"},
            data,
        ).fit_vb()
        idx = list(mix.model.exog_names).index("z_maxrace")
        est_m = mix.fe_mean[idx]
        sd_m = mix.fe_sd[idx]
        rows.append(
            {
                "model": "Bayesian mixed GLM (team random intercept)",
                "term": "z_maxrace",
                "odds_ratio": float(np.exp(est_m)),
                "or_ci_low": float(np.exp(est_m - 1.96 * sd_m)),
                "or_ci_high": float(np.exp(est_m + 1.96 * sd_m)),
                "p_value": np.nan,
                "n_obs": int(data.shape[0]),
                "n_teams": int(data["team_id"].nunique()),
            }
        )
    except Exception as e:  # pragma: no cover
        print(f"  (mixed GLM skipped: {e})")
    return pd.DataFrame(rows)


def _load_ranked_nationals(
    metrics_path: str = "output/rq1/top25_teams/top25_teams_metrics.csv",
) -> pd.DataFrame:
    """Nationals finish ranks 1..25 from the top-25 metrics table."""
    m = pd.read_csv(metrics_path)
    m["gender"] = m["gender"].map({"Men": "M", "Women": "F"}).fillna(m["gender"])
    m["team_id"] = m["team_id"].astype(int)
    return m[["year", "gender", "rank", "team_id", "team_name", "max_races"]]


def _team_year_maxrace_from_df(df: pd.DataFrame, year: int, gender: str) -> pd.DataFrame:
    """Like team_year_maxrace but reuses an already-standardized frame."""
    ata = pd.read_csv(os.path.join(get_data_dir(), "athlete_team_association.csv"))
    ath = pd.read_csv(os.path.join(get_data_dir(), "athlete.csv"))[
        ["athlete_id", "gender"]
    ]
    ata = ata.merge(ath, on="athlete_id", how="left").dropna(
        subset=["athlete_id", "team_id", "gender"]
    )
    start, end = (
        pd.Timestamp(year=year, month=8, day=1),
        pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59),
    )
    yr = df[(df["start_date"] >= start) & (df["start_date"] <= end) & (df["gender"] == gender)]
    counts = yr.groupby("athlete_id").size().reset_index(name="race_count")
    merged = counts.merge(
        ata[ata["gender"] == gender][["athlete_id", "team_id"]],
        on="athlete_id",
        how="left",
    ).dropna(subset=["team_id"])
    natl = merged.groupby("team_id")["athlete_id"].nunique()
    qualifying = natl[natl >= 3].index
    merged = merged[merged["team_id"].isin(qualifying)]
    out = (
        merged.groupby("team_id")["race_count"].max().reset_index(name="max_race_count")
    )
    out["year"] = year
    out["gender"] = gender
    return out


def topk_placement_curve(
    race_threshold: int = 4,
    ks: Tuple[int, ...] = (5, 10, 15, 20, 25),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Placement-threshold sensitivity: top-k for k in ks, parallel to race-count dose.

    Why: top-15 is a conventional but arbitrary cut. If the race-frequency
    association is real, pooled RR / continuous OR should remain elevated across
    a range of placement thresholds rather than spike only at k=15.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    ranked = _load_ranked_nationals()
    print("  Loading standardized race frame once...", flush=True)
    df = standardize_convert_exclude_nationals_df()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    print("  Building team-year max-race tables...", flush=True)
    team_tables = {}
    totals = {}
    for year, gender, _ in CELLS:
        tt = _team_year_maxrace_from_df(df, year, gender)
        team_tables[(year, gender)] = tt
        totals[(year, gender)] = int(tt["team_id"].nunique())

    per_cell_rows = []
    for k in ks:
        for year, gender, _ in CELLS:
            tt = team_tables[(year, gender)].copy()
            topk = set(
                ranked.loc[
                    (ranked["year"] == year)
                    & (ranked["gender"] == gender)
                    & (ranked["rank"] <= k),
                    "team_id",
                ]
            )
            if len(topk) == 0:
                continue
            total = totals[(year, gender)]
            tt["exposed"] = tt["max_race_count"] >= race_threshold
            tt["in_topk"] = tt["team_id"].isin(topk)
            a = int((tt["exposed"] & tt["in_topk"]).sum())
            exposed_total = int(tt["exposed"].sum())
            b = exposed_total - a
            topk_total = len(topk)
            c = topk_total - a
            d = total - exposed_total - c
            rr, log_rr, var = _rr_2x2(a, b, c, d)
            chi2, p, _, _ = stats.chi2_contingency([[a, b], [c, max(d, 0)]])
            per_cell_rows.append(
                {
                    "k": k,
                    "race_threshold": f">= {race_threshold}",
                    "year": year,
                    "gender": gender,
                    "total_teams": total,
                    "exposed_teams": exposed_total,
                    "topk_exposed": a,
                    "topk_n": topk_total,
                    "risk_ratio": rr,
                    "log_rr": log_rr,
                    "var_log_rr": var,
                    "chi2_p": p,
                }
            )
    per_cell = pd.DataFrame(per_cell_rows)

    pooled_rows = []
    for k in ks:
        sub = per_cell[per_cell["k"] == k].dropna(subset=["log_rr", "var_log_rr"])
        if len(sub) < 2:
            continue
        dl = _dersimonian_laird(sub, f"top-{k}")
        dl["k"] = k
        dl["race_threshold"] = f">= {race_threshold}"
        pooled_rows.append(dl)
    pooled = pd.DataFrame(pooled_rows)

    print("  Fitting GEE OR by placement k...", flush=True)
    cont_rows = []
    frames = []
    for year, gender, _ in CELLS:
        tt = team_tables[(year, gender)].copy()
        ranks = ranked[(ranked["year"] == year) & (ranked["gender"] == gender)][
            ["team_id", "rank"]
        ]
        tt = tt.merge(ranks, on="team_id", how="left")
        frames.append(tt)
    data = pd.concat(frames, ignore_index=True)
    data["z_maxrace"] = (
        data["max_race_count"] - data["max_race_count"].mean()
    ) / data["max_race_count"].std(ddof=0)

    for k in ks:
        data[f"top{k}"] = ((data["rank"].notna()) & (data["rank"] <= k)).astype(int)
        gee = smf.gee(
            f"top{k} ~ z_maxrace + C(gender) + C(year)",
            groups="team_id",
            data=data,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        ).fit()
        est = gee.params["z_maxrace"]
        ci = gee.conf_int().loc["z_maxrace"]
        cont_rows.append(
            {
                "k": k,
                "model": "GEE logistic (cluster-robust by team)",
                "odds_ratio": float(np.exp(est)),
                "or_ci_low": float(np.exp(ci[0])),
                "or_ci_high": float(np.exp(ci[1])),
                "p_value": float(gee.pvalues["z_maxrace"]),
                "n_obs": int(data.shape[0]),
                "n_in_topk": int(data[f"top{k}"].sum()),
            }
        )
    continuous = pd.DataFrame(cont_rows)

    rank_panel = ranked.copy()
    rho, p_rho = stats.spearmanr(rank_panel["rank"], rank_panel["max_races"])
    continuous = pd.concat(
        [
            continuous,
            pd.DataFrame(
                [
                    {
                        "k": "rank_1_to_25",
                        "model": "Spearman(rank, max_races) among top-25 panel",
                        "odds_ratio": float(rho),
                        "or_ci_low": np.nan,
                        "or_ci_high": np.nan,
                        "p_value": float(p_rho),
                        "n_obs": int(len(rank_panel)),
                        "n_in_topk": int(len(rank_panel)),
                        "note": "rho (not OR); negative = more races ↔ better place",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return per_cell, pooled, continuous


def _roster_panel_with_ero() -> pd.DataFrame:
    """Team–year–gender panel with depth, ERO, and top-15 from roster metrics."""
    from underexplored_mechanisms import build_team_roster_metrics, _load_race_frame

    df = _load_race_frame()
    roster = build_team_roster_metrics(df)
    roster = roster.copy()
    roster["effective_n_athletes"] = np.exp(roster["shannon_entropy"].clip(lower=0))
    roster["ERO"] = roster["effective_n_athletes"] * roster["mean_race_count"]
    ranked = _load_ranked_nationals()
    roster = roster.merge(
        ranked[["year", "gender", "team_id", "rank"]],
        on=["year", "gender", "team_id"],
        how="left",
    )
    return roster


def within_team_first_differences(roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """Consecutive-season within-team changes in schedule metrics → Δ top-15 / rank.

    For each (team_id, gender) observed in years t and t+1, compute first
    differences of max_race_count, n_athletes_ge3, ERO, n_athletes, and outcomes
    Δtop15 and Δrank (rank only when both years have a top-25 finish).
    """
    if roster is None:
        roster = _roster_panel_with_ero()
    rows = []
    for (team_id, gender), sub in roster.groupby(["team_id", "gender"]):
        sub = sub.sort_values("year")
        years = sub["year"].to_numpy()
        for i in range(len(sub) - 1):
            if years[i + 1] - years[i] != 1:
                continue
            a = sub.iloc[i]
            b = sub.iloc[i + 1]
            d_rank = (
                float(b["rank"] - a["rank"])
                if pd.notna(a["rank"]) and pd.notna(b["rank"])
                else np.nan
            )
            rows.append(
                {
                    "team_id": int(team_id),
                    "gender": gender,
                    "year_from": int(a["year"]),
                    "year_to": int(b["year"]),
                    "d_max_race_count": float(b["max_race_count"] - a["max_race_count"]),
                    "d_n_athletes_ge3": float(b["n_athletes_ge3"] - a["n_athletes_ge3"]),
                    "d_ERO": float(b["ERO"] - a["ERO"]),
                    "d_n_athletes": float(b["n_athletes"] - a["n_athletes"]),
                    "d_top15": int(b["top15"]) - int(a["top15"]),
                    "top15_from": int(a["top15"]),
                    "top15_to": int(b["top15"]),
                    "d_rank": d_rank,
                    "rank_from": a["rank"] if pd.notna(a["rank"]) else np.nan,
                    "rank_to": b["rank"] if pd.notna(b["rank"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_within_team_diffs(diffs: pd.DataFrame) -> pd.DataFrame:
    """Association of Δ predictors with Δtop15 (and Spearman with Δrank)."""
    import statsmodels.api as sm

    rows = []
    if diffs.empty:
        return pd.DataFrame(rows)

    y = diffs["d_top15"].to_numpy(dtype=float)
    for col, label in (
        ("d_max_race_count", "Δ max_race_count"),
        ("d_n_athletes_ge3", "Δ n_athletes_ge3"),
        ("d_ERO", "Δ ERO"),
        ("d_n_athletes", "Δ n_athletes"),
    ):
        x = diffs[col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 10:
            continue
        # OLS on first differences (within-team linear probability change).
        X = sm.add_constant(x[ok])
        fit = sm.OLS(y[ok], X).fit(cov_type="HC1")
        rho, p_rho = stats.spearmanr(x[ok], y[ok])
        rows.append(
            {
                "outcome": "d_top15",
                "predictor": label,
                "n_pairs": int(ok.sum()),
                "ols_coef": float(fit.params[1]),
                "ols_se": float(fit.bse[1]),
                "ols_p": float(fit.pvalues[1]),
                "spearman_rho": float(rho),
                "spearman_p": float(p_rho),
                "mean_d_predictor": float(np.mean(x[ok])),
                "mean_d_top15": float(np.mean(y[ok])),
            }
        )

    # Rank change among pairs with both ranks observed (negative Δrank = improved).
    rank_ok = diffs["d_rank"].notna()
    if int(rank_ok.sum()) >= 10:
        y_r = diffs.loc[rank_ok, "d_rank"].to_numpy(dtype=float)
        for col, label in (
            ("d_max_race_count", "Δ max_race_count"),
            ("d_n_athletes_ge3", "Δ n_athletes_ge3"),
            ("d_ERO", "Δ ERO"),
        ):
            x = diffs.loc[rank_ok, col].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y_r)
            if ok.sum() < 10:
                continue
            X = sm.add_constant(x[ok])
            fit = sm.OLS(y_r[ok], X).fit(cov_type="HC1")
            rho, p_rho = stats.spearmanr(x[ok], y_r[ok])
            rows.append(
                {
                    "outcome": "d_rank (lower=better)",
                    "predictor": label,
                    "n_pairs": int(ok.sum()),
                    "ols_coef": float(fit.params[1]),
                    "ols_se": float(fit.bse[1]),
                    "ols_p": float(fit.pvalues[1]),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_rho),
                    "mean_d_predictor": float(np.mean(x[ok])),
                    "mean_d_top15": float(np.mean(y_r[ok])),
                }
            )
    return pd.DataFrame(rows)


def roster_size_adjusted_models(roster: pd.DataFrame | None = None) -> pd.DataFrame:
    """GEE logistic: top15 ~ z_maxrace + z_depth + z_n_athletes (+ year, gender)."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    if roster is None:
        roster = _roster_panel_with_ero()
    data = roster.copy()
    data["top15"] = data["top15"].astype(int)

    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 0 else s * 0.0

    data["z_maxrace"] = _z(data["max_race_count"])
    data["z_depth"] = _z(data["n_athletes_ge3"])
    data["z_n_athletes"] = _z(data["n_athletes"])
    data["z_ERO"] = _z(data["ERO"])

    specs = [
        ("maxrace_only", "top15 ~ z_maxrace + C(gender) + C(year)"),
        ("depth_only", "top15 ~ z_depth + C(gender) + C(year)"),
        ("maxrace_plus_size", "top15 ~ z_maxrace + z_n_athletes + C(gender) + C(year)"),
        (
            "depth_plus_maxrace_plus_size",
            "top15 ~ z_maxrace + z_depth + z_n_athletes + C(gender) + C(year)",
        ),
        (
            "ERO_plus_size",
            "top15 ~ z_ERO + z_n_athletes + C(gender) + C(year)",
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
                    "n_obs": int(data.shape[0]),
                    "n_teams": int(data["team_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def beyond_top7_depth_analysis(
    roster: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bench depth (athletes outside season-best top 7 with ≥3 starts) vs size.

    Returns (descriptives, gee_models). Non-top-7 depth is highly correlated
    with roster size; controlling for n_athletes typically nulls the OR.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from utils import standardize_both_tiers

    if roster is None:
        roster = _roster_panel_with_ero()

    _, df = standardize_both_tiers()
    df = df.copy()
    df["year"] = pd.to_datetime(df["start_date"]).dt.year
    sub = df[~df["nationals"].fillna(False)].copy()
    g = (
        sub.groupby(["team_id", "year", "gender", "athlete_id"], as_index=False)
        .agg(n_races=("result_id", "count"), best_time=("standardized_to_target", "min"))
    )
    rows = []
    for (team, year, gender), t in g.groupby(["team_id", "year", "gender"]):
        t = t.sort_values("best_time")
        n = len(t)
        top7, rest = t.head(7), t.iloc[7:]
        rows.append(
            {
                "team_id": team,
                "year": int(year),
                "gender": gender,
                "n_beyond_7": max(0, n - 7),
                "n_ge3_top7": int((top7["n_races"] >= 3).sum()),
                "n_ge3_beyond7": int((rest["n_races"] >= 3).sum()) if len(rest) else 0,
            }
        )
    bench = pd.DataFrame(rows)
    data = roster.merge(bench, on=["team_id", "year", "gender"], how="inner")
    data["top15"] = data["top15"].astype(int)

    def _z(s: pd.Series) -> pd.Series:
        sd = float(s.std(ddof=0))
        return (s - s.mean()) / sd if sd > 0 else s * 0.0

    for col in [
        "n_ge3_beyond7",
        "n_beyond_7",
        "n_athletes",
        "n_athletes_ge3",
        "n_ge3_top7",
        "max_race_count",
    ]:
        data[f"z_{col}"] = _z(data[col])

    desc_rows = []
    for gender, subg in data.groupby("gender"):
        t15 = subg[subg["top15"] == 1]
        nt = subg[subg["top15"] == 0]
        desc_rows.append(
            {
                "gender": gender,
                "top15_mean_n_ge3_beyond7": float(t15["n_ge3_beyond7"].mean()),
                "not_top15_mean_n_ge3_beyond7": float(nt["n_ge3_beyond7"].mean()),
                "top15_mean_n_athletes": float(t15["n_athletes"].mean()),
                "not_top15_mean_n_athletes": float(nt["n_athletes"].mean()),
                "corr_beyond7_ge3_vs_size": float(
                    subg["n_ge3_beyond7"].corr(subg["n_athletes"])
                ),
                "corr_overall_ge3_vs_size": float(
                    subg["n_athletes_ge3"].corr(subg["n_athletes"])
                ),
                "n_top15": int(len(t15)),
                "n_not": int(len(nt)),
            }
        )
    descriptives = pd.DataFrame(desc_rows)

    specs = [
        ("beyond7_depth_only", "top15 ~ z_n_ge3_beyond7 + C(gender) + C(year)"),
        (
            "beyond7_plus_size",
            "top15 ~ z_n_ge3_beyond7 + z_n_athletes + C(gender) + C(year)",
        ),
        (
            "beyond7_plus_maxrace",
            "top15 ~ z_n_ge3_beyond7 + z_max_race_count + C(gender) + C(year)",
        ),
        (
            "full",
            "top15 ~ z_n_ge3_beyond7 + z_n_ge3_top7 + z_n_athletes + "
            "z_max_race_count + C(gender) + C(year)",
        ),
        ("size_only", "top15 ~ z_n_athletes + C(gender) + C(year)"),
        ("overall_depth_only", "top15 ~ z_n_athletes_ge3 + C(gender) + C(year)"),
        (
            "overall_depth_plus_size",
            "top15 ~ z_n_athletes_ge3 + z_n_athletes + C(gender) + C(year)",
        ),
    ]
    model_rows = []
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
            model_rows.append(
                {
                    "model": name,
                    "term": term,
                    "odds_ratio": float(np.exp(est)),
                    "or_ci_low": float(np.exp(ci[0])),
                    "or_ci_high": float(np.exp(ci[1])),
                    "p_value": float(gee.pvalues[term]),
                    "n_obs": int(data.shape[0]),
                }
            )
    return descriptives, pd.DataFrame(model_rows)


def write_selection_robustness_markdown(
    out_dir: str,
    summary: pd.DataFrame,
    adjusted: pd.DataFrame,
    diffs: pd.DataFrame,
    beyond7_models: pd.DataFrame | None = None,
    beyond7_desc: pd.DataFrame | None = None,
) -> None:
    """Human-readable summary for the appendix / Discussion hedge."""
    lines = [
        "# Selection robustness for team race-frequency → top-15",
        "",
        "Cross-sectional team associations (pooled RR, GEE OR, ERO AUC) can",
        "reflect stable program quality rather than schedule choices. These",
        "checks ask whether **within-team year-to-year changes** in racing",
        "volume/depth still track placement, and whether controlling for",
        "**roster size** absorbs the association.",
        "",
        f"Within-team consecutive pairs: **{len(diffs)}**.",
        "",
        "## Within-team first differences (Δtop15)",
        "",
    ]
    if summary.empty:
        lines.append("No estimable pairs.")
    else:
        sub = summary[summary["outcome"] == "d_top15"]
        lines.append("| Predictor | n | OLS coef | p | Spearman ρ |")
        lines.append("|-----------|---|----------|---|------------|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['predictor']} | {int(r['n_pairs'])} | "
                f"{r['ols_coef']:.4f} | {r['ols_p']:.3g} | {r['spearman_rho']:.3f} |"
            )
        rank_sub = summary[summary["outcome"].astype(str).str.startswith("d_rank")]
        if not rank_sub.empty:
            lines.extend(
                [
                    "",
                    "## Within-team Δrank (among top-25 both years; lower = better)",
                    "",
                    "| Predictor | n | OLS coef | p | Spearman ρ |",
                    "|-----------|---|----------|---|------------|",
                ]
            )
            for _, r in rank_sub.iterrows():
                lines.append(
                    f"| {r['predictor']} | {int(r['n_pairs'])} | "
                    f"{r['ols_coef']:.4f} | {r['ols_p']:.3g} | {r['spearman_rho']:.3f} |"
                )

    lines.extend(
        [
            "",
            "## Roster-size-adjusted GEE (cluster-robust by team)",
            "",
            "| Model | Term | OR/SD | 95% CI | p |",
            "|-------|------|-------|--------|---|",
        ]
    )
    for _, r in adjusted.iterrows():
        lines.append(
            f"| {r['model']} | {r['term']} | {r['odds_ratio']:.3f} | "
            f"[{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}] | {r['p_value']:.3g} |"
        )
    if beyond7_models is not None and not beyond7_models.empty:
        lines.extend(
            [
                "",
                "## Non-top-7 (bench) depth vs roster size",
                "",
                "Athletes outside the season-best top 7 with ≥3 starts.",
                "",
            ]
        )
        if beyond7_desc is not None and not beyond7_desc.empty:
            lines.append(
                "| Gender | Top-15 mean beyond-7≥3 | Not | Top-15 size | Not | "
                "corr(beyond7, size) |"
            )
            lines.append("|--------|------------------------|-----|-------------|-----|---------------------|")
            for _, r in beyond7_desc.iterrows():
                lines.append(
                    f"| {r['gender']} | {r['top15_mean_n_ge3_beyond7']:.1f} | "
                    f"{r['not_top15_mean_n_ge3_beyond7']:.1f} | "
                    f"{r['top15_mean_n_athletes']:.1f} | "
                    f"{r['not_top15_mean_n_athletes']:.1f} | "
                    f"{r['corr_beyond7_ge3_vs_size']:.3f} |"
                )
            lines.append("")
        lines.extend(
            [
                "| Model | Term | OR/SD | 95% CI | p |",
                "|-------|------|-------|--------|---|",
            ]
        )
        for _, r in beyond7_models.iterrows():
            lines.append(
                f"| {r['model']} | {r['term']} | {r['odds_ratio']:.3f} | "
                f"[{r['or_ci_low']:.3f}, {r['or_ci_high']:.3f}] | {r['p_value']:.3g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation (associational)",
            "",
            "If within-team Δvolume/Δdepth coefficients are near zero while",
            "cross-sectional ORs remain large, treat the primary team finding as",
            "**between-program selection** rather than evidence that changing a",
            "given team's schedule moves nationals placement. Roster-size",
            "adjustment speaks to whether depth/volume proxies for program size.",
            "Non-top-7 depth likewise collapses once size is controlled — it is",
            "largely a club-size channel, not an independent schedule effect.",
            "",
            "Artifacts: `within_team_first_diff.csv`,",
            "`within_team_first_diff_summary.csv`,",
            "`roster_size_adjusted_models.csv`,",
            "`beyond_top7_depth_descriptives.csv`,",
            "`beyond_top7_depth_models.csv`.",
            "",
        ]
    )
    path = os.path.join(out_dir, "SELECTION_ROBUSTNESS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(output_dir: str = "output/rq3") -> None:
    out_dir = os.path.join(output_dir, "team_association_robustness")
    os.makedirs(out_dir, exist_ok=True)

    print("Threshold sensitivity (k=3/4/5 races)...")
    per_cell, pooled = threshold_sensitivity()
    per_cell.to_csv(os.path.join(out_dir, "threshold_sensitivity_cells.csv"), index=False)
    pooled.to_csv(os.path.join(out_dir, "threshold_pooled_rr.csv"), index=False)
    print(pooled.to_string(index=False))

    print("Partial pooling of cell RRs at >=4...")
    shrunk = shrunken_cell_rr(per_cell, threshold=">= 4")
    shrunk.to_csv(os.path.join(out_dir, "shrunken_cell_rr_k4.csv"), index=False)

    print("Continuous race-count models (top-15)...")
    cont = continuous_model()
    cont.to_csv(os.path.join(out_dir, "continuous_race_count_model.csv"), index=False)
    print(cont.to_string(index=False))

    print("Top-k placement curve (k=5..25) at race threshold >=4...")
    topk_cells, topk_pooled, topk_cont = topk_placement_curve(race_threshold=4)
    topk_cells.to_csv(os.path.join(out_dir, "topk_placement_cells.csv"), index=False)
    topk_pooled.to_csv(os.path.join(out_dir, "topk_placement_pooled_rr.csv"), index=False)
    topk_cont.to_csv(os.path.join(out_dir, "topk_placement_continuous_or.csv"), index=False)
    print(topk_pooled.to_string(index=False))
    print(topk_cont.to_string(index=False))

    print("Selection robustness: within-team first differences + roster-size GEE...")
    roster = _roster_panel_with_ero()
    diffs = within_team_first_differences(roster)
    diffs.to_csv(os.path.join(out_dir, "within_team_first_diff.csv"), index=False)
    summary = summarize_within_team_diffs(diffs)
    summary.to_csv(os.path.join(out_dir, "within_team_first_diff_summary.csv"), index=False)
    print(summary.to_string(index=False))
    adjusted = roster_size_adjusted_models(roster)
    adjusted.to_csv(os.path.join(out_dir, "roster_size_adjusted_models.csv"), index=False)
    print(adjusted.to_string(index=False))

    print("Non-top-7 (bench) depth vs roster size...")
    beyond7_desc, beyond7_models = beyond_top7_depth_analysis(roster)
    beyond7_desc.to_csv(
        os.path.join(out_dir, "beyond_top7_depth_descriptives.csv"), index=False
    )
    beyond7_models.to_csv(
        os.path.join(out_dir, "beyond_top7_depth_models.csv"), index=False
    )
    print(beyond7_desc.to_string(index=False))
    print(beyond7_models.to_string(index=False))
    write_selection_robustness_markdown(
        out_dir,
        summary,
        adjusted,
        diffs,
        beyond7_models=beyond7_models,
        beyond7_desc=beyond7_desc,
    )

    print(f"Wrote team-association robustness outputs to {out_dir}/")


if __name__ == "__main__":
    main()
