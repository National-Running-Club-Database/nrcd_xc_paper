"""Underexplored mechanisms bridging the individual null and team positive finding.

Three analyses that the paper's spine invites but does not yet report:

1. Roster depth / Shannon entropy of starts → nationals placement
   Separates "one workhorse racing 5×" from deep programs. Conditions on the
   existing max-race-count association so depth is not just a restatement of
   volume.

2. Within-team peer co-start density → individual improvement / retention
   Mean number of same-team, same-gender teammates present at each of an
   athlete's meets. Tests a social/organizational channel between the
   individual null and the team positive finding.

3. Quantile regression: who benefits from race volume
   τ ∈ {0.25, 0.50, 0.75} of Standardized first→last improvement on race
   count (with starting ability control). Makes the "slow starters improve
   more" pattern a formal quantile claim rather than cell means alone.

Output: output/rq1/underexplored_mechanisms/
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from _setup_paths import setup_paths

setup_paths()

from load_nrcd_data import get_data_dir
from nationals_overlap_analysis import (
    analyze_2023_mens_overlap,
    analyze_2023_womens_overlap,
    analyze_2024_mens_overlap,
    analyze_2024_womens_overlap,
    analyze_2025_mens_overlap,
    analyze_2025_womens_overlap,
)
from paper_enrichment_analyses import build_athlete_season_table
from utils import standardize_convert_exclude_nationals_df

RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 2000

CELLS = [
    (2023, "M", analyze_2023_mens_overlap),
    (2023, "F", analyze_2023_womens_overlap),
    (2024, "M", analyze_2024_mens_overlap),
    (2024, "F", analyze_2024_womens_overlap),
    (2025, "M", analyze_2025_mens_overlap),
    (2025, "F", analyze_2025_womens_overlap),
]


def _season_window(year: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return (
        pd.Timestamp(year=year, month=8, day=1),
        pd.Timestamp(year=year, month=11, day=28, hour=23, minute=59, second=59),
    )


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a count distribution (nats). 0 if empty/degenerate."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _gini(counts: np.ndarray) -> float:
    """Gini coefficient of race-count inequality (0 = equal, 1 = one athlete)."""
    x = np.sort(np.asarray(counts, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    n = x.size
    i = np.arange(1, n + 1)
    return float((2 * (i * x).sum()) / (n * x.sum()) - (n + 1) / n)


def _load_race_frame() -> pd.DataFrame:
    df = standardize_convert_exclude_nationals_df()
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["athlete_id", "team_id", "meet_id", "start_date", "gender"])
    df["athlete_id"] = df["athlete_id"].astype(int)
    df["team_id"] = df["team_id"].astype(int)
    df["meet_id"] = df["meet_id"].astype(int)
    df["year"] = df["start_date"].dt.year
    # Map Aug–Nov races: Dec/Jan edge cases already excluded by nationals filter.
    # Use calendar year of race date (matches existing team scripts).
    return df


# ---------------------------------------------------------------------------
# 1. Roster depth / entropy
# ---------------------------------------------------------------------------


def build_team_roster_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team–year–gender with depth/entropy descriptors."""
    rows = []
    for year, gender, analyze in CELLS:
        start, end = _season_window(year)
        yr = df[
            (df["start_date"] >= start)
            & (df["start_date"] <= end)
            & (df["gender"] == gender)
        ]
        # Athlete race counts on this team (use result.team_id for the meet).
        ath_counts = (
            yr.groupby(["team_id", "athlete_id"])
            .size()
            .reset_index(name="race_count")
        )
        top15 = set(analyze()["nationals_teams"])

        for team_id, sub in ath_counts.groupby("team_id"):
            n_athletes = int(sub["athlete_id"].nunique())
            if n_athletes < 3:
                continue
            counts = sub["race_count"].to_numpy()
            H = _shannon_entropy(counts)
            H_norm = H / np.log(n_athletes) if n_athletes > 1 else 0.0
            rows.append(
                {
                    "year": year,
                    "gender": gender,
                    "team_id": int(team_id),
                    "n_athletes": n_athletes,
                    "total_starts": int(counts.sum()),
                    "max_race_count": int(counts.max()),
                    "mean_race_count": float(counts.mean()),
                    "n_athletes_ge3": int((counts >= 3).sum()),
                    "n_athletes_ge4": int((counts >= 4).sum()),
                    "share_ge3": float((counts >= 3).mean()),
                    "shannon_entropy": H,
                    "shannon_evenness": H_norm,  # 1 = equal starts across roster
                    "gini_race_count": _gini(counts),
                    "top15": int(team_id) in top15,
                }
            )
    return pd.DataFrame(rows)


def roster_depth_models(roster: pd.DataFrame) -> pd.DataFrame:
    """Cluster-robust logistic: top15 ~ z_maxrace + depth/entropy predictors."""
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except ImportError as exc:
        raise ImportError("statsmodels required for GEE models") from exc

    results = []
    specs = [
        ("maxrace_only", ["max_race_count"]),
        ("maxrace_plus_evenness", ["max_race_count", "shannon_evenness"]),
        ("maxrace_plus_depth_ge3", ["max_race_count", "n_athletes_ge3"]),
        ("maxrace_plus_share_ge3", ["max_race_count", "share_ge3"]),
        ("maxrace_plus_gini", ["max_race_count", "gini_race_count"]),
        ("depth_ge3_only", ["n_athletes_ge3"]),
        ("evenness_only", ["shannon_evenness"]),
    ]

    data = roster.dropna().copy()
    data["team_id"] = data["team_id"].astype(int)
    # Gender indicators (women as reference)
    data["male"] = (data["gender"] == "M").astype(float)
    # Year fixed effects
    for y in sorted(data["year"].unique()):
        data[f"year_{y}"] = (data["year"] == y).astype(float)

    for name, preds in specs:
        use = data.copy()
        zcols = []
        for p in preds:
            zname = f"z_{p}"
            use[zname] = StandardScaler().fit_transform(use[[p]])
            zcols.append(zname)
        fe = ["male"] + [f"year_{y}" for y in sorted(use["year"].unique())[1:]]
        X = sm.add_constant(use[zcols + fe])
        y = use["top15"].astype(float)
        try:
            model = GEE(
                y,
                X,
                groups=use["team_id"],
                family=Binomial(),
                cov_struct=Exchangeable(),
            )
            fit = model.fit()
            for z in zcols:
                or_ = float(np.exp(fit.params[z]))
                ci = fit.conf_int().loc[z]
                results.append(
                    {
                        "spec": name,
                        "term": z,
                        "odds_ratio": or_,
                        "or_ci_low": float(np.exp(ci[0])),
                        "or_ci_high": float(np.exp(ci[1])),
                        "p_value": float(fit.pvalues[z]),
                        "n_obs": int(len(use)),
                        "n_teams": int(use["team_id"].nunique()),
                        "top15_rate": float(use["top15"].mean()),
                    }
                )
        except Exception as e:
            results.append(
                {
                    "spec": name,
                    "term": "FAILED",
                    "odds_ratio": np.nan,
                    "or_ci_low": np.nan,
                    "or_ci_high": np.nan,
                    "p_value": np.nan,
                    "n_obs": int(len(use)),
                    "n_teams": int(use["team_id"].nunique()),
                    "top15_rate": float(use["top15"].mean()),
                    "error": str(e),
                }
            )
    return pd.DataFrame(results)


def roster_depth_descriptives(roster: pd.DataFrame) -> pd.DataFrame:
    """Mean depth/entropy among top-15 vs not, by gender (pooled years)."""
    rows = []
    for gender, label in [("M", "Men"), ("F", "Women")]:
        sub = roster[roster["gender"] == gender]
        for metric in [
            "shannon_evenness",
            "n_athletes_ge3",
            "share_ge3",
            "gini_race_count",
            "max_race_count",
            "n_athletes",
        ]:
            a = sub.loc[sub["top15"], metric].dropna().values
            b = sub.loc[~sub["top15"], metric].dropna().values
            if len(a) < 5 or len(b) < 5:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            pooled_sd = np.sqrt(
                ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                / (len(a) + len(b) - 2)
            )
            d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            rows.append(
                {
                    "gender": label,
                    "metric": metric,
                    "top15_mean": float(a.mean()),
                    "not_top15_mean": float(b.mean()),
                    "diff": float(a.mean() - b.mean()),
                    "cohens_d": float(d),
                    "mannwhitney_p": float(p),
                    "n_top15": int(len(a)),
                    "n_not": int(len(b)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Peer co-start density
# ---------------------------------------------------------------------------


def build_peer_exposure(df: pd.DataFrame) -> pd.DataFrame:
    """Athlete–season mean # of same-team same-gender teammates at each meet."""
    # Teammates at each (team, meet, gender)
    meet_team = (
        df.groupby(["year", "gender", "team_id", "meet_id"])["athlete_id"]
        .nunique()
        .reset_index(name="n_teammates_at_meet")
    )
    # Join back so each result row knows how many teammates shared that meet
    merged = df.merge(
        meet_team, on=["year", "gender", "team_id", "meet_id"], how="left"
    )
    # Peers = teammates including self minus 1
    merged["n_peers"] = (merged["n_teammates_at_meet"] - 1).clip(lower=0)
    # Athlete-season aggregates
    ag = (
        merged.groupby(["athlete_id", "year", "gender", "team_id"])
        .agg(
            num_races=("meet_id", "nunique"),
            mean_peers=("n_peers", "mean"),
            median_peers=("n_peers", "median"),
            pct_meets_with_ge3_peers=("n_peers", lambda s: float((s >= 3).mean())),
            pct_solo_meets=("n_peers", lambda s: float((s == 0).mean())),
            first_time=("standardized_to_target", "first"),
        )
        .reset_index()
    )
    return ag


def peer_improvement_model(
    peer: pd.DataFrame, as_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Associate peer exposure with improvement rate and retention."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels required") from exc

    # Merge peer with athlete-season improvement
    m = as_df.merge(
        peer[
            [
                "athlete_id",
                "year",
                "gender",
                "team_id",
                "mean_peers",
                "pct_meets_with_ge3_peers",
                "pct_solo_meets",
            ]
        ],
        on=["athlete_id", "year", "gender"],
        how="inner",
    )
    m = m[(m["improvement_rate"] >= -50) & (m["improvement_rate"] <= 50)].copy()
    m["male"] = (m["gender"] == "M").astype(float)
    # Starting ability within gender-year
    m["starting_pct"] = m.groupby(["gender", "year"])["first_time"].rank(pct=True)
    for col in ["mean_peers", "num_races", "starting_pct", "season_duration"]:
        m[f"z_{col}"] = StandardScaler().fit_transform(m[[col]])

    # MixedLM: improvement_rate ~ peers + races + starting + male + year FE
    # Random intercept for athlete (and we cluster conceptually by team via FE)
    rows = []
    for label, formula in [
        (
            "peers_plus_controls",
            "improvement_rate ~ z_mean_peers + z_num_races + z_starting_pct + z_season_duration + male + C(year)",
        ),
        (
            "peers_only_plus_starting",
            "improvement_rate ~ z_mean_peers + z_starting_pct + male + C(year)",
        ),
        (
            "solo_share",
            "improvement_rate ~ pct_solo_meets + z_num_races + z_starting_pct + male + C(year)",
        ),
    ]:
        try:
            fit = smf.mixedlm(formula, m, groups=m["athlete_id"]).fit(reml=True, method="lbfgs")
            for term in fit.fe_params.index:
                if term == "Intercept" or term.startswith("C("):
                    continue
                rows.append(
                    {
                        "outcome": "improvement_rate",
                        "spec": label,
                        "term": term,
                        "coef": float(fit.fe_params[term]),
                        "se": float(fit.bse_fe[term]),
                        "p_value": float(fit.pvalues[term]),
                        "n_obs": int(fit.nobs),
                        "n_groups": int(fit.model.n_groups),
                    }
                )
        except Exception as e:
            rows.append(
                {
                    "outcome": "improvement_rate",
                    "spec": label,
                    "term": "FAILED",
                    "coef": np.nan,
                    "se": np.nan,
                    "p_value": np.nan,
                    "n_obs": len(m),
                    "n_groups": m["athlete_id"].nunique(),
                    "error": str(e),
                }
            )

    coef_df = pd.DataFrame(rows)

    # Retention: raced next calendar year?
    # Build next-year presence from as_df years
    presence = as_df[["athlete_id", "year", "gender"]].drop_duplicates()
    presence["returned"] = False
    for y in [2023, 2024]:
        cur = presence[presence["year"] == y][["athlete_id"]].copy()
        nxt = set(presence.loc[presence["year"] == y + 1, "athlete_id"])
        cur["returned"] = cur["athlete_id"].isin(nxt)
        cur["year"] = y
        presence.loc[presence["year"] == y, "returned"] = presence.loc[
            presence["year"] == y, "athlete_id"
        ].isin(nxt)

    ret = m[m["year"].isin([2023, 2024])].merge(
        presence[["athlete_id", "year", "returned"]],
        on=["athlete_id", "year"],
        how="left",
    )
    ret = ret.dropna(subset=["returned"])
    ret_rows = []
    if len(ret) > 50:
        try:
            import statsmodels.api as sm
            from statsmodels.genmod.cov_struct import Exchangeable
            from statsmodels.genmod.families import Binomial
            from statsmodels.genmod.generalized_estimating_equations import GEE

            ret["returned"] = ret["returned"].astype(float)
            for name, preds in [
                ("peers", ["z_mean_peers", "z_num_races", "z_starting_pct", "male"]),
                ("solo", ["pct_solo_meets", "z_num_races", "z_starting_pct", "male"]),
            ]:
                X = sm.add_constant(ret[preds])
                fit = GEE(
                    ret["returned"],
                    X,
                    groups=ret["team_id"],
                    family=Binomial(),
                    cov_struct=Exchangeable(),
                ).fit()
                for p in preds:
                    ret_rows.append(
                        {
                            "outcome": "retention",
                            "spec": name,
                            "term": p,
                            "odds_ratio": float(np.exp(fit.params[p])),
                            "or_ci_low": float(np.exp(fit.conf_int().loc[p, 0])),
                            "or_ci_high": float(np.exp(fit.conf_int().loc[p, 1])),
                            "p_value": float(fit.pvalues[p]),
                            "n_obs": int(len(ret)),
                        }
                    )
        except Exception as e:
            ret_rows.append(
                {
                    "outcome": "retention",
                    "spec": "FAILED",
                    "term": "FAILED",
                    "odds_ratio": np.nan,
                    "or_ci_low": np.nan,
                    "or_ci_high": np.nan,
                    "p_value": np.nan,
                    "n_obs": len(ret),
                    "error": str(e),
                }
            )
    ret_df = pd.DataFrame(ret_rows)
    return coef_df, ret_df


def peer_descriptives(peer: pd.DataFrame, as_df: pd.DataFrame) -> pd.DataFrame:
    m = as_df.merge(
        peer[["athlete_id", "year", "gender", "mean_peers", "pct_solo_meets"]],
        on=["athlete_id", "year", "gender"],
        how="inner",
    )
    m = m[(m["improvement_rate"] >= -50) & (m["improvement_rate"] <= 50)]
    rows = []
    for gender, label in [("M", "Men"), ("F", "Women")]:
        sub = m[m["gender"] == gender]
        # Tertiles of mean_peers
        try:
            sub = sub.copy()
            sub["peer_tert"] = pd.qcut(
                sub["mean_peers"], 3, labels=["low", "mid", "high"], duplicates="drop"
            )
        except ValueError:
            continue
        for tert, tsub in sub.groupby("peer_tert"):
            vals = tsub["total_improvement_sec"].values
            mean, lo, hi = _bootstrap_mean_ci(vals)
            rows.append(
                {
                    "gender": label,
                    "peer_tertile": tert,
                    "n": len(tsub),
                    "mean_peers": float(tsub["mean_peers"].mean()),
                    "mean_improvement_sec": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "mean_num_races": float(tsub["num_races"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    vals: np.ndarray, n_boot: int = BOOTSTRAP_REPLICATES, seed: int = RANDOM_SEED
) -> Tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(vals, size=len(vals), replace=True).mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(
        np.percentile(means, 97.5)
    )


# ---------------------------------------------------------------------------
# 3. Quantile regression
# ---------------------------------------------------------------------------


def quantile_race_volume(as_df: pd.DataFrame) -> pd.DataFrame:
    """Quantile regression of first→last improvement on race count by gender."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise ImportError("statsmodels required") from exc

    rows = []
    for gender, label in [("M", "Men"), ("F", "Women")]:
        sub = as_df[as_df["gender"] == gender].copy()
        sub = sub[np.isfinite(sub["total_improvement_sec"]) & (sub["num_races"] >= 2)]
        sub["starting_pct"] = sub.groupby("year")["first_time"].rank(pct=True)
        sub["male"] = 1.0 if gender == "M" else 0.0
        for tau in (0.25, 0.50, 0.75):
            try:
                # QuantReg via formula
                fit = smf.quantreg(
                    "total_improvement_sec ~ num_races + starting_pct + C(year)",
                    sub,
                ).fit(q=tau)
                for term in ["num_races", "starting_pct"]:
                    rows.append(
                        {
                            "gender": label,
                            "tau": tau,
                            "term": term,
                            "coef": float(fit.params[term]),
                            "se": float(fit.bse[term]),
                            "p_value": float(fit.pvalues[term]),
                            "ci_low": float(fit.conf_int().loc[term, 0]),
                            "ci_high": float(fit.conf_int().loc[term, 1]),
                            "n": int(fit.nobs),
                        }
                    )
            except Exception as e:
                rows.append(
                    {
                        "gender": label,
                        "tau": tau,
                        "term": "FAILED",
                        "coef": np.nan,
                        "se": np.nan,
                        "p_value": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n": len(sub),
                        "error": str(e),
                    }
                )
    return pd.DataFrame(rows)


def starting_ability_cate(as_df: pd.DataFrame) -> pd.DataFrame:
    """4+ vs 2-race contrast of improvement within starting-ability quartiles, BH-corrected."""
    from statsmodels.stats.multitest import multipletests

    rows = []
    for gender, label in [("M", "Men"), ("F", "Women")]:
        sub = as_df[as_df["gender"] == gender].copy()
        sub = sub[np.isfinite(sub["total_improvement_sec"])]
        sub["start_q"] = pd.qcut(
            sub["first_time"], 4, labels=["Q1_fastest", "Q2", "Q3", "Q4_slowest"]
        )
        for q, qsub in sub.groupby("start_q"):
            a = qsub.loc[qsub["num_races"] >= 4, "total_improvement_sec"].values
            b = qsub.loc[qsub["num_races"] == 2, "total_improvement_sec"].values
            if len(a) < 10 or len(b) < 10:
                continue
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            pooled_sd = np.sqrt(
                ((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2)
                / max(len(a) + len(b) - 2, 1)
            )
            d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            rows.append(
                {
                    "gender": label,
                    "start_quartile": q,
                    "n_4plus": int(len(a)),
                    "n_2": int(len(b)),
                    "mean_4plus": float(a.mean()),
                    "mean_2": float(b.mean()),
                    "diff": float(a.mean() - b.mean()),
                    "cohens_d": float(d),
                    "mannwhitney_p": float(p),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        reject, p_bh, _, _ = multipletests(out["mannwhitney_p"], method="fdr_bh")
        out["p_bh"] = p_bh
        out["bh_significant"] = reject
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: str = "output/rq1/underexplored_mechanisms") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng_note = {
        "random_seed": RANDOM_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }

    print("Loading race frame...")
    df = _load_race_frame()
    print(f"  {len(df)} results, {df.athlete_id.nunique()} athletes, {df.team_id.nunique()} teams")

    print("\n1. Roster depth / Shannon entropy...")
    roster = build_team_roster_metrics(df)
    roster.to_csv(out / "team_roster_metrics.csv", index=False)
    desc = roster_depth_descriptives(roster)
    desc.to_csv(out / "roster_depth_descriptives.csv", index=False)
    models = roster_depth_models(roster)
    models.to_csv(out / "roster_depth_gee_models.csv", index=False)
    print(desc.to_string(index=False))
    print(models.to_string(index=False))

    print("\n2. Peer co-start density...")
    peer = build_peer_exposure(df)
    peer.to_csv(out / "athlete_peer_exposure.csv", index=False)
    as_df = build_athlete_season_table(df)
    peer_desc = peer_descriptives(peer, as_df)
    peer_desc.to_csv(out / "peer_improvement_descriptives.csv", index=False)
    peer_coefs, peer_ret = peer_improvement_model(peer, as_df)
    peer_coefs.to_csv(out / "peer_improvement_mixedlm.csv", index=False)
    peer_ret.to_csv(out / "peer_retention_gee.csv", index=False)
    print(peer_desc.to_string(index=False))
    print(peer_coefs.to_string(index=False))
    if len(peer_ret):
        print(peer_ret.to_string(index=False))

    print("\n3. Quantile regression + BH starting-ability CATE...")
    qreg = quantile_race_volume(as_df)
    qreg.to_csv(out / "quantile_race_volume.csv", index=False)
    cate = starting_ability_cate(as_df)
    cate.to_csv(out / "starting_ability_cate_bh.csv", index=False)
    print(qreg.to_string(index=False))
    print(cate.to_string(index=False))

    with open(out / "reproducibility.json", "w") as fh:
        json.dump(rng_note, fh, indent=2)

    # Short SUMMARY.md for findings sync
    summary_lines = [
        "# Underexplored mechanisms — summary",
        "",
        "## Roster depth (conditional on max races)",
        "",
    ]
    key = models[models["spec"] == "maxrace_plus_evenness"]
    if len(key):
        for _, r in key.iterrows():
            summary_lines.append(
                f"- {r['spec']} / {r['term']}: OR={r['odds_ratio']:.2f} "
                f"[{r['or_ci_low']:.2f}, {r['or_ci_high']:.2f}], p={r['p_value']:.4g}"
            )
    summary_lines += ["", "## Peer co-start → improvement", ""]
    for _, r in peer_coefs[
        peer_coefs["term"].astype(str).str.contains("peer|solo", case=False, na=False)
    ].iterrows():
        summary_lines.append(
            f"- {r['spec']} / {r['term']}: coef={r['coef']:.3f}, p={r['p_value']:.4g}"
        )
    summary_lines += ["", "## Quantile race-volume (num_races coef)", ""]
    for _, r in qreg[qreg["term"] == "num_races"].iterrows():
        summary_lines.append(
            f"- {r['gender']} τ={r['tau']}: β={r['coef']:.2f} "
            f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}], p={r['p_value']:.4g}"
        )
    summary_lines += ["", "## Starting-ability CATE (4+ vs 2), BH", ""]
    for _, r in cate.iterrows():
        summary_lines.append(
            f"- {r['gender']} {r['start_quartile']}: Δ={r['diff']:.1f}s "
            f"(d={r['cohens_d']:.2f}), p={r['mannwhitney_p']:.4g}, "
            f"BH sig={r['bh_significant']}"
        )
    (out / "SUMMARY.md").write_text("\n".join(summary_lines) + "\n")
    print(f"\nWrote outputs to {out}/")


if __name__ == "__main__":
    main()
