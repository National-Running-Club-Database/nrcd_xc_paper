"""Relative-finish and club field-relative course-factor analyses.

Complements NRCD metadata standardization (distance / weather / elevation)
with a LACCTiC-motivated field-relative path that does **not** use weather:

1. Derive within-meet finish place and relative percentile from clock times
2. First→last season change in time vs relative finish for Raw / Converted /
   Standardized clocks
3. Lightweight LACCTiC-style course factor α (alternating athlete fitness ↔
   meet factors on overlapping athletes; no track-PR calibration)
4. Concordance: field-adjusted Δ vs Standardized Δ; α vs weather residual
5. Sanity: within-meet order agreement across Raw / Converted / Standardized
6. Head-to-head vs actual NRCD course factors (next-race MAE, holdout residual
   SD, split-half reliability, temperature alignment) → ``VERDICT.md``

Related motivation: https://www.lacctic.com/ (Mazaheri). This is a transparent
club analogue, not a bit-for-bit reimplementation.

Run from repo root:
  python scripts/relative_finish_course_factors.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from _setup_paths import setup_paths

setup_paths()

from paper_enrichment_analyses import build_athlete_season_table
from utils import parse_time, standardize_both_tiers

OUTPUT_DIR = os.path.join(
    Path(__file__).resolve().parents[1],
    "output",
    "rq1",
    "relative_finish_course_factors",
)
RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 2000
RNG = np.random.default_rng(RANDOM_SEED)

# Meets with tiny fields give unstable places / α estimates.
MIN_FIELD_SIZE = 5
# EM for course factors
EM_MAX_ITERS = 300
EM_TOL = 5e-4  # relative |Δα|; ~0.05% is ample for course factors
EM_DAMP = 0.5  # under-relaxation for stability
MIN_ATHLETE_RACES_FOR_FITNESS = 2
# Cap extreme ratios when updating α (robustness to outliers / tiny fields)
ALPHA_RATIO_CLIP = (0.5, 2.0)


def _bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = BOOTSTRAP_REPLICATES, alpha: float = 0.05
) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = [float(RNG.choice(x, size=len(x), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(x)), float(lo), float(hi)


def _spearman(a: Iterable[float], b: Iterable[float]) -> Tuple[float, float, int]:
    aa = np.asarray(list(a), dtype=float)
    bb = np.asarray(list(b), dtype=float)
    mask = np.isfinite(aa) & np.isfinite(bb)
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(aa[mask], bb[mask])
    return float(rho), float(p), n


def _gender_label(g: str) -> str:
    return "Men" if g == "M" else "Women"


# ---------------------------------------------------------------------------
# Race-level relative finishes
# ---------------------------------------------------------------------------


def annotate_relative_finishes(df: pd.DataFrame, time_col: str, prefix: str) -> pd.DataFrame:
    """Add place / field size / finish percentile for a clock column.

    finish_pct ∈ [0, 1]: 1 = won the field, 0 = last (undefined if field_size=1).
    Place is 1-indexed dense rank of ascending time (ties: min rank).
    Field key: meet_id × gender (club fields are scored within gender).
    """
    out = df.copy()
    t = out[time_col].astype(float)
    out[f"{prefix}_time_sec"] = t
    # rank within meet×gender; NaN times get NaN place
    out[f"{prefix}_place"] = (
        t.groupby([out["meet_id"], out["gender"]])
        .rank(method="min", ascending=True)
        .astype(float)
    )
    field = out.groupby(["meet_id", "gender"])[time_col].transform(
        lambda s: float(np.isfinite(s.astype(float)).sum())
    )
    out[f"{prefix}_field_size"] = field
    denom = (field - 1.0).replace(0, np.nan)
    out[f"{prefix}_finish_pct"] = (field - out[f"{prefix}_place"]) / denom
    return out


def build_paired_race_frame() -> pd.DataFrame:
    """Merge Raw / Converted / Standardized clocks and relative finishes."""
    df_conv, df_std = standardize_both_tiers()
    key = ["result_id", "athlete_id", "meet_id", "gender", "running_event_id"]

    std = df_std[
        key
        + ["start_date", "team_id", "result_time", "standardized_to_target", "nationals"]
    ].copy()
    std["start_date"] = pd.to_datetime(std["start_date"], errors="coerce")
    std["year"] = std["start_date"].dt.year
    std["raw_time_sec"] = std["result_time"].map(parse_time)
    std = std.rename(columns={"standardized_to_target": "std_time_sec"})

    conv = df_conv[key + ["standardized_to_target"]].rename(
        columns={"standardized_to_target": "conv_time_sec"}
    )
    races = std.merge(conv, on=key, how="inner")
    races = races.dropna(
        subset=["raw_time_sec", "conv_time_sec", "std_time_sec", "start_date", "athlete_id"]
    )
    # Drop pathological clock parses
    races = races[
        (races["raw_time_sec"] > 60)
        & (races["conv_time_sec"] > 60)
        & (races["std_time_sec"] > 60)
    ].copy()

    for prefix, col in [
        ("raw", "raw_time_sec"),
        ("conv", "conv_time_sec"),
        ("std", "std_time_sec"),
    ]:
        races = annotate_relative_finishes(races, col, prefix)

    # Environment residual used elsewhere: a = t_conv - t_std
    races["env_residual_sec"] = races["conv_time_sec"] - races["std_time_sec"]
    return races.reset_index(drop=True)


def within_meet_order_agreement(races: pd.DataFrame) -> pd.DataFrame:
    """Fraction of meets where within-field order agrees across clocks."""
    rows = []
    for (meet_id, gender), g in races.groupby(["meet_id", "gender"], sort=False):
        if len(g) < MIN_FIELD_SIZE:
            continue
        # Pairwise Spearman of places (or times — monotone identical)
        pairs = [
            ("raw_vs_conv", "raw_time_sec", "conv_time_sec"),
            ("raw_vs_std", "raw_time_sec", "std_time_sec"),
            ("conv_vs_std", "conv_time_sec", "std_time_sec"),
        ]
        rec = {
            "meet_id": int(meet_id),
            "gender": gender,
            "year": int(g["year"].iloc[0]) if g["year"].notna().any() else np.nan,
            "field_size": int(len(g)),
        }
        for name, a, b in pairs:
            rho, _, n = _spearman(g[a], g[b])
            rec[f"spearman_{name}"] = rho
            # Exact place agreement rate
            pa = g[a].rank(method="min", ascending=True)
            pb = g[b].rank(method="min", ascending=True)
            rec[f"exact_place_agree_{name}"] = float((pa == pb).mean())
            rec[f"n_{name}"] = n
        rows.append(rec)
    meet_df = pd.DataFrame(rows)

    summary_rows = []
    for gender, gdf in meet_df.groupby("gender"):
        label = _gender_label(gender)
        for name in ["raw_vs_conv", "raw_vs_std", "conv_vs_std"]:
            agree = gdf[f"exact_place_agree_{name}"].values
            mean_a, lo, hi = _bootstrap_mean_ci(agree)
            rho = gdf[f"spearman_{name}"].values
            mean_r, r_lo, r_hi = _bootstrap_mean_ci(rho)
            summary_rows.append(
                {
                    "gender": label,
                    "comparison": name,
                    "n_meets": int(len(gdf)),
                    "mean_exact_place_agree": mean_a,
                    "agree_ci_low": lo,
                    "agree_ci_high": hi,
                    "mean_spearman": mean_r,
                    "spearman_ci_low": r_lo,
                    "spearman_ci_high": r_hi,
                    "pct_meets_perfect_agree": 100.0 * float((agree >= 1.0 - 1e-12).mean()),
                }
            )
    return meet_df, pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Athlete-season: time Δ and relative-finish Δ
# ---------------------------------------------------------------------------


def build_relative_athlete_season(races: pd.DataFrame) -> pd.DataFrame:
    """One row per athlete-season with Raw/Conv/Std time and finish-pct change."""
    rows = []
    for (athlete_id, year), g in races.groupby(["athlete_id", "year"], sort=False):
        g = g.sort_values("start_date")
        if len(g) < 2:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        days = (last["start_date"] - first["start_date"]).days
        if days < 7:
            continue
        # Require usable fields at both ends
        if (
            first["raw_field_size"] < MIN_FIELD_SIZE
            or last["raw_field_size"] < MIN_FIELD_SIZE
        ):
            continue
        if not all(
            np.isfinite(first[c]) and np.isfinite(last[c])
            for c in [
                "raw_time_sec",
                "conv_time_sec",
                "std_time_sec",
                "raw_finish_pct",
                "conv_finish_pct",
                "std_finish_pct",
            ]
        ):
            continue

        rec = {
            "athlete_id": athlete_id,
            "year": int(year),
            "gender": first["gender"],
            "num_races": int(len(g)),
            "season_duration": int(days),
            "first_meet_id": int(first["meet_id"]),
            "last_meet_id": int(last["meet_id"]),
            "first_field_size": int(first["raw_field_size"]),
            "last_field_size": int(last["raw_field_size"]),
        }
        for tier, tcol, pcol in [
            ("raw", "raw_time_sec", "raw_finish_pct"),
            ("conv", "conv_time_sec", "conv_finish_pct"),
            ("std", "std_time_sec", "std_finish_pct"),
        ]:
            t1, tL = float(first[tcol]), float(last[tcol])
            p1, pL = float(first[pcol]), float(last[pcol])
            rec[f"{tier}_first_time"] = t1
            rec[f"{tier}_last_time"] = tL
            # Positive = faster
            rec[f"{tier}_time_improve_sec"] = t1 - tL
            rec[f"{tier}_first_finish_pct"] = p1
            rec[f"{tier}_last_finish_pct"] = pL
            # Positive = better relative finish (higher pct)
            rec[f"{tier}_finish_pct_improve"] = pL - p1
            rec[f"{tier}_time_improved"] = (t1 - tL) > 0
            rec[f"{tier}_place_improved"] = (pL - p1) > 0
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_tier_improvements(as_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gender, gdf in as_df.groupby("gender"):
        label = _gender_label(gender)
        for tier in ["raw", "conv", "std"]:
            tcol = f"{tier}_time_improve_sec"
            pcol = f"{tier}_finish_pct_improve"
            t_mean, t_lo, t_hi = _bootstrap_mean_ci(gdf[tcol].values)
            p_mean, p_lo, p_hi = _bootstrap_mean_ci(gdf[pcol].values)
            rows.append(
                {
                    "gender": label,
                    "tier": tier,
                    "n": int(len(gdf)),
                    "mean_time_improve_sec": t_mean,
                    "time_ci_low": t_lo,
                    "time_ci_high": t_hi,
                    "pct_time_improved": 100.0 * float(gdf[f"{tier}_time_improved"].mean()),
                    "mean_finish_pct_improve": p_mean,
                    "pct_ci_low": p_lo,
                    "pct_ci_high": p_hi,
                    "pct_place_improved": 100.0 * float(gdf[f"{tier}_place_improved"].mean()),
                }
            )
        # Paired weather inflation on this (field-filtered) sample
        delta = gdf["conv_time_improve_sec"] - gdf["std_time_improve_sec"]
        d_mean, d_lo, d_hi = _bootstrap_mean_ci(delta.values)
        try:
            _, w_p = stats.wilcoxon(
                gdf["conv_time_improve_sec"],
                gdf["std_time_improve_sec"],
                alternative="greater",
            )
        except ValueError:
            w_p = np.nan
        rows.append(
            {
                "gender": label,
                "tier": "conv_minus_std",
                "n": int(len(gdf)),
                "mean_time_improve_sec": d_mean,
                "time_ci_low": d_lo,
                "time_ci_high": d_hi,
                "pct_time_improved": w_p,
                "mean_finish_pct_improve": np.nan,
                "pct_ci_low": np.nan,
                "pct_ci_high": np.nan,
                "pct_place_improved": np.nan,
            }
        )
    return pd.DataFrame(rows)


def improvement_concordance(as_df: pd.DataFrame) -> pd.DataFrame:
    """Rank agreement between time Δ and finish-pct Δ across tiers."""
    rows = []
    for gender, gdf in as_df.groupby("gender"):
        label = _gender_label(gender)
        comparisons = [
            ("raw_time_vs_raw_place", "raw_time_improve_sec", "raw_finish_pct_improve"),
            ("conv_time_vs_conv_place", "conv_time_improve_sec", "conv_finish_pct_improve"),
            ("std_time_vs_std_place", "std_time_improve_sec", "std_finish_pct_improve"),
            ("raw_time_vs_std_time", "raw_time_improve_sec", "std_time_improve_sec"),
            ("conv_time_vs_std_time", "conv_time_improve_sec", "std_time_improve_sec"),
            ("raw_place_vs_std_place", "raw_finish_pct_improve", "std_finish_pct_improve"),
            ("raw_place_vs_conv_place", "raw_finish_pct_improve", "conv_finish_pct_improve"),
        ]
        for name, a, b in comparisons:
            rho, p, n = _spearman(gdf[a], gdf[b])
            # Sign agreement: both improved or both not
            sa = gdf[a].values > 0
            sb = gdf[b].values > 0
            sign_agree = float(np.mean(sa == sb))
            rows.append(
                {
                    "gender": label,
                    "comparison": name,
                    "n": n,
                    "spearman_rho": rho,
                    "spearman_p": p,
                    "sign_agreement": sign_agree,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LACCTiC-style course factors (time-ratio EM, no track PR calibration)
# ---------------------------------------------------------------------------


def estimate_course_factors(
    races: pd.DataFrame,
    time_col: str = "raw_time_sec",
    gender: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Alternating athlete fitness f_i and meet factor α_j.

    Model (LACCTiC-style): α_j ≈ median_i (f_i / t_ij); f_i ≈ median_j (α_j t_ij).
    Slower / harder meets → larger times → smaller α. Adjusted time = α · t.
    Scale pinned so median(α) = 1 over meets used. Updates are damped and
    ratio-clipped for club-field sparsity.

    Returns (meet_factors, athlete_fitness, diagnostics).
    """
    work = races.copy()
    if gender is not None:
        work = work[work["gender"] == gender]
    field = work.groupby("meet_id")[time_col].transform("count")
    work = work[field >= MIN_FIELD_SIZE].copy()
    n_meets_per = work.groupby("athlete_id")["meet_id"].transform("nunique")
    work = work[n_meets_per >= MIN_ATHLETE_RACES_FOR_FITNESS].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), {"converged": False, "iters": 0}

    athletes = work["athlete_id"].unique()
    meets = work["meet_id"].unique()
    a_index = {aid: i for i, aid in enumerate(athletes)}
    m_index = {mid: j for j, mid in enumerate(meets)}

    obs: List[Tuple[int, int, float]] = []
    for row in work.itertuples(index=False):
        t = float(getattr(row, time_col))
        if not np.isfinite(t) or t <= 0:
            continue
        obs.append((a_index[row.athlete_id], m_index[row.meet_id], t))

    n_a, n_m = len(athletes), len(meets)
    log_sums = np.zeros(n_a)
    counts = np.zeros(n_a)
    for ai, _, t in obs:
        log_sums[ai] += np.log(t)
        counts[ai] += 1
    f = np.exp(log_sums / np.maximum(counts, 1))

    alpha = np.ones(n_m, dtype=float)
    history = []
    converged = False
    f_den = counts.copy()
    lo_clip, hi_clip = ALPHA_RATIO_CLIP

    by_meet: List[List[Tuple[int, float]]] = [[] for _ in range(n_m)]
    by_ath: List[List[Tuple[int, float]]] = [[] for _ in range(n_a)]
    for ai, mj, t in obs:
        by_meet[mj].append((ai, t))
        by_ath[ai].append((mj, t))

    for _it in range(EM_MAX_ITERS):
        new_alpha = np.ones(n_m, dtype=float)
        for mj in range(n_m):
            ratios = []
            for ai, t in by_meet[mj]:
                r = f[ai] / t
                ratios.append(float(np.clip(r, lo_clip * alpha[mj], hi_clip * alpha[mj])))
            if ratios:
                new_alpha[mj] = float(np.median(ratios))
        med = np.median(new_alpha)
        if med > 0:
            new_alpha = new_alpha / med
        new_alpha = EM_DAMP * new_alpha + (1.0 - EM_DAMP) * alpha
        med = np.median(new_alpha)
        if med > 0:
            new_alpha = new_alpha / med

        new_f = np.ones(n_a, dtype=float)
        f_den = np.zeros(n_a)
        for ai in range(n_a):
            vals = [new_alpha[mj] * t for mj, t in by_ath[ai]]
            f_den[ai] = float(len(vals))
            if vals:
                new_f[ai] = float(np.median(vals))
        new_f = EM_DAMP * new_f + (1.0 - EM_DAMP) * f

        rel = float(np.max(np.abs(new_alpha - alpha) / np.maximum(np.abs(alpha), 1e-8)))
        history.append(rel)
        alpha, f = new_alpha, new_f
        if rel < EM_TOL:
            converged = True
            break

    meet_rows = []
    for mid, j in m_index.items():
        sub = work[work["meet_id"] == mid]
        meet_rows.append(
            {
                "meet_id": int(mid),
                "gender": gender if gender is not None else "pooled",
                "alpha": float(alpha[j]),
                "n_results": int(len(sub)),
                "n_athletes": int(sub["athlete_id"].nunique()),
                "mean_time_sec": float(sub[time_col].mean()),
                "mean_env_residual_sec": float(sub["env_residual_sec"].mean())
                if "env_residual_sec" in sub.columns
                else np.nan,
                "year": int(sub["year"].mode().iloc[0]) if sub["year"].notna().any() else np.nan,
            }
        )
    athlete_rows = []
    for aid, i in a_index.items():
        athlete_rows.append(
            {
                "athlete_id": int(aid) if not isinstance(aid, str) else aid,
                "gender": gender if gender is not None else "pooled",
                "fitness_sec": float(f[i]),
                "n_races_in_em": int(f_den[i]),
            }
        )

    diag = {
        "gender": gender if gender is not None else "pooled",
        "time_col": time_col,
        "n_athletes": int(n_a),
        "n_meets": int(n_m),
        "n_obs": int(len(obs)),
        "converged": converged,
        "iters": int(len(history)),
        "final_rel_delta_alpha": float(history[-1]) if history else np.nan,
        "min_field_size": MIN_FIELD_SIZE,
        "em_damp": EM_DAMP,
        "alpha_ratio_clip": list(ALPHA_RATIO_CLIP),
        "update": "median_damped",
    }
    return pd.DataFrame(meet_rows), pd.DataFrame(athlete_rows), diag



def apply_field_adjusted_times(
    races: pd.DataFrame, meet_factors: pd.DataFrame, gender: str
) -> pd.DataFrame:
    """Attach α and field-adjusted raw time for one gender."""
    mf = meet_factors[meet_factors["gender"] == gender][["meet_id", "alpha"]]
    out = races[races["gender"] == gender].merge(mf, on="meet_id", how="left")
    out["field_adj_time_sec"] = out["alpha"] * out["raw_time_sec"]
    return out


def field_adjusted_athlete_season(races_adj: pd.DataFrame) -> pd.DataFrame:
    """First→last using field-adjusted times (requires α on first and last)."""
    rows = []
    for (athlete_id, year), g in races_adj.groupby(["athlete_id", "year"], sort=False):
        g = g.sort_values("start_date")
        if len(g) < 2:
            continue
        first, last = g.iloc[0], g.iloc[-1]
        days = (last["start_date"] - first["start_date"]).days
        if days < 7:
            continue
        if not (
            np.isfinite(first.get("field_adj_time_sec", np.nan))
            and np.isfinite(last.get("field_adj_time_sec", np.nan))
            and np.isfinite(first["std_time_sec"])
            and np.isfinite(last["std_time_sec"])
        ):
            continue
        fa1, faL = float(first["field_adj_time_sec"]), float(last["field_adj_time_sec"])
        s1, sL = float(first["std_time_sec"]), float(last["std_time_sec"])
        r1, rL = float(first["raw_time_sec"]), float(last["raw_time_sec"])
        rows.append(
            {
                "athlete_id": athlete_id,
                "year": int(year),
                "gender": first["gender"],
                "num_races": int(len(g)),
                "season_duration": int(days),
                "alpha_first": float(first["alpha"]),
                "alpha_last": float(last["alpha"]),
                "field_adj_improve_sec": fa1 - faL,
                "std_improve_sec": s1 - sL,
                "raw_improve_sec": r1 - rL,
                "conv_improve_sec": float(first["conv_time_sec"]) - float(last["conv_time_sec"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_field_vs_std(fa_as: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gender, gdf in fa_as.groupby("gender"):
        label = _gender_label(gender)
        for name, col in [
            ("field_adj", "field_adj_improve_sec"),
            ("standardized", "std_improve_sec"),
            ("raw", "raw_improve_sec"),
            ("converted", "conv_improve_sec"),
        ]:
            mean, lo, hi = _bootstrap_mean_ci(gdf[col].values)
            rows.append(
                {
                    "gender": label,
                    "method": name,
                    "n": int(len(gdf)),
                    "mean_improve_sec": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "pct_improved": 100.0 * float((gdf[col] > 0).mean()),
                }
            )
        rho, p, n = _spearman(gdf["field_adj_improve_sec"], gdf["std_improve_sec"])
        delta = gdf["field_adj_improve_sec"] - gdf["std_improve_sec"]
        d_mean, d_lo, d_hi = _bootstrap_mean_ci(delta.values)
        try:
            _, w_p = stats.wilcoxon(
                gdf["field_adj_improve_sec"],
                gdf["std_improve_sec"],
                alternative="two-sided",
            )
        except ValueError:
            w_p = np.nan
        rows.append(
            {
                "gender": label,
                "method": "field_adj_minus_std",
                "n": n,
                "mean_improve_sec": d_mean,
                "ci_low": d_lo,
                "ci_high": d_hi,
                "pct_improved": w_p,
                "spearman_field_vs_std": rho,
                "spearman_p": p,
            }
        )
        rho_r, p_r, _ = _spearman(gdf["field_adj_improve_sec"], gdf["raw_improve_sec"])
        rows.append(
            {
                "gender": label,
                "method": "spearman_field_vs_raw",
                "n": n,
                "mean_improve_sec": rho_r,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "pct_improved": p_r,
            }
        )
    return pd.DataFrame(rows)


def course_factor_vs_weather(meet_factors: pd.DataFrame) -> pd.DataFrame:
    """Correlate meet α with mean environment residual (conv − std)."""
    rows = []
    for gender, gdf in meet_factors.groupby("gender"):
        if gender in ("pooled",):
            continue
        label = _gender_label(gender)
        # Harder course → smaller α; larger positive env residual → Converted slower
        # than Standardized (weather/elev made clock look slow). Expect negative
        # corr(α, mean_env_residual) if field difficulty tracks weather residual.
        rho, p, n = _spearman(gdf["alpha"], gdf["mean_env_residual_sec"])
        rows.append(
            {
                "gender": label,
                "n_meets": n,
                "spearman_alpha_vs_env_residual": rho,
                "spearman_p": p,
                "mean_alpha": float(gdf["alpha"].mean()),
                "std_alpha": float(gdf["alpha"].std(ddof=1)),
                "mean_env_residual": float(gdf["mean_env_residual_sec"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_tier_time_vs_place(tier_sum: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    tiers = ["raw", "conv", "std"]
    tier_labels = {"raw": "Raw", "conv": "Converted", "std": "Standardized"}
    colors = {"raw": "#4C78A8", "conv": "#F58518", "std": "#54A24B"}
    for ax, gender in zip(axes, ["Men", "Women"]):
        sub = tier_sum[(tier_sum["gender"] == gender) & (tier_sum["tier"].isin(tiers))]
        x = np.arange(len(tiers))
        means = [float(sub.loc[sub["tier"] == t, "mean_time_improve_sec"].iloc[0]) for t in tiers]
        los = [float(sub.loc[sub["tier"] == t, "time_ci_low"].iloc[0]) for t in tiers]
        his = [float(sub.loc[sub["tier"] == t, "time_ci_high"].iloc[0]) for t in tiers]
        yerr = np.vstack([np.array(means) - np.array(los), np.array(his) - np.array(means)])
        ax.bar(
            x,
            means,
            yerr=yerr,
            color=[colors[t] for t in tiers],
            capsize=4,
            width=0.7,
            label="Time Δ (s)",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([tier_labels[t] for t in tiers])
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set_ylabel("Mean first→last time improvement (s)")
        ax.set_title(gender)
    fig.suptitle("Club first→last improvement by clock tier\n(field size ≥ 5 at first and last)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_finish_pct_improve(tier_sum: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    tiers = ["raw", "conv", "std"]
    tier_labels = {"raw": "Raw", "conv": "Converted", "std": "Standardized"}
    colors = {"raw": "#4C78A8", "conv": "#F58518", "std": "#54A24B"}
    for ax, gender in zip(axes, ["Men", "Women"]):
        sub = tier_sum[(tier_sum["gender"] == gender) & (tier_sum["tier"].isin(tiers))]
        x = np.arange(len(tiers))
        means = [float(sub.loc[sub["tier"] == t, "mean_finish_pct_improve"].iloc[0]) for t in tiers]
        los = [float(sub.loc[sub["tier"] == t, "pct_ci_low"].iloc[0]) for t in tiers]
        his = [float(sub.loc[sub["tier"] == t, "pct_ci_high"].iloc[0]) for t in tiers]
        yerr = np.vstack([np.array(means) - np.array(los), np.array(his) - np.array(means)])
        ax.bar(x, means, yerr=yerr, color=[colors[t] for t in tiers], capsize=4, width=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([tier_labels[t] for t in tiers])
        ax.axhline(0, color="gray", lw=1, ls="--")
        ax.set_ylabel("Mean finish-percentile gain (last − first)")
        ax.set_title(gender)
    fig.suptitle(
        "Relative-finish change (1 = won field, 0 = last)\n"
        "Near-identical across tiers if env adjustment is meet-level",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_field_vs_std_scatter(fa_as: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, gender_code, title in zip(axes, ["M", "F"], ["Men", "Women"]):
        g = fa_as[fa_as["gender"] == gender_code]
        ax.scatter(
            g["std_improve_sec"],
            g["field_adj_improve_sec"],
            alpha=0.25,
            s=12,
            edgecolors="none",
        )
        q_lo = np.nanpercentile(
            np.concatenate([g["std_improve_sec"], g["field_adj_improve_sec"]]), 1
        )
        q_hi = np.nanpercentile(
            np.concatenate([g["std_improve_sec"], g["field_adj_improve_sec"]]), 99
        )
        ax.plot([q_lo, q_hi], [q_lo, q_hi], "k--", lw=1, label="y = x")
        ax.set_xlim(q_lo, q_hi)
        ax.set_ylim(q_lo, q_hi)
        rho, p, n = _spearman(g["std_improve_sec"], g["field_adj_improve_sec"])
        ax.set_xlabel("Standardized time improvement (s)")
        ax.set_ylabel("Field-adjusted (α·raw) improvement (s)")
        ax.set_title(f"{title} (n={n}, ρ={rho:.2f})")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Club LACCTiC-style field adjustment vs NRCD Standardized", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_vs_env(meet_factors: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, gender_code, title in zip(axes, ["M", "F"], ["Men", "Women"]):
        g = meet_factors[meet_factors["gender"] == gender_code]
        ax.scatter(g["mean_env_residual_sec"], g["alpha"], alpha=0.7, s=28)
        rho, p, n = _spearman(g["alpha"], g["mean_env_residual_sec"])
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Mean env residual (conv − std), seconds")
        ax.set_ylabel("Course factor α (median-pinned to 1)")
        ax.set_title(f"{title} (n={n}, ρ={rho:.2f})")
    fig.suptitle("Field-inferred α vs weather/elevation residual", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("RELATIVE FINISH + CLUB COURSE FACTORS")
    print("=" * 60)
    print(
        f"Reproducibility: seed={RANDOM_SEED}, bootstrap={BOOTSTRAP_REPLICATES}, "
        f"min_field={MIN_FIELD_SIZE}, EM_tol={EM_TOL}"
    )
    with open(os.path.join(output_dir, "reproducibility.json"), "w") as f:
        json.dump(
            {
                "random_seed": RANDOM_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "min_field_size": MIN_FIELD_SIZE,
                "em_max_iters": EM_MAX_ITERS,
                "em_tol": EM_TOL,
                "em_damp": EM_DAMP,
                "alpha_ratio_clip": list(ALPHA_RATIO_CLIP),
                "min_athlete_races_for_fitness": MIN_ATHLETE_RACES_FOR_FITNESS,
                "notes": (
                    "LACCTiC-motivated club course factors on raw times; "
                    "no track-PR calibration. Place derived within meet×gender."
                ),
                "lacctic_url": "https://www.lacctic.com/",
            },
            f,
            indent=2,
        )

    print("\n1. Building paired race frame (raw / conv / std + places)...")
    races = build_paired_race_frame()
    races.to_csv(os.path.join(output_dir, "race_relative_finish.csv"), index=False)
    print(f"  Races: {len(races):,}")

    print("\n2. Within-meet order agreement across clocks...")
    meet_agree, agree_sum = within_meet_order_agreement(races)
    meet_agree.to_csv(os.path.join(output_dir, "within_meet_order_by_meet.csv"), index=False)
    agree_sum.to_csv(os.path.join(output_dir, "within_meet_order_agreement.csv"), index=False)
    print(agree_sum.to_string(index=False))

    print("\n3. Athlete-season time Δ and finish-pct Δ...")
    as_rel = build_relative_athlete_season(races)
    as_rel.to_csv(os.path.join(output_dir, "athlete_season_relative.csv"), index=False)
    print(f"  Athlete-seasons (field≥{MIN_FIELD_SIZE} at ends): {len(as_rel):,}")

    tier_sum = summarize_tier_improvements(as_rel)
    tier_sum.to_csv(os.path.join(output_dir, "tier_improvement_summary.csv"), index=False)
    print(tier_sum.to_string(index=False))

    conc = improvement_concordance(as_rel)
    conc.to_csv(os.path.join(output_dir, "improvement_concordance.csv"), index=False)
    print("\n  Concordance (Spearman):")
    print(conc.to_string(index=False))

    plot_tier_time_vs_place(
        tier_sum, os.path.join(output_dir, "tier_time_improvement.pdf")
    )
    plot_finish_pct_improve(
        tier_sum, os.path.join(output_dir, "tier_finish_pct_improvement.pdf")
    )

    print("\n4. LACCTiC-style course factors (gender-separated EM on raw times)...")
    meet_parts = []
    athlete_parts = []
    diags = []
    adj_parts = []
    fa_parts = []
    for gender in ["M", "F"]:
        mf, af, diag = estimate_course_factors(races, time_col="raw_time_sec", gender=gender)
        print(
            f"  { _gender_label(gender) }: meets={diag.get('n_meets')}, "
            f"athletes={diag.get('n_athletes')}, iters={diag.get('iters')}, "
            f"converged={diag.get('converged')}"
        )
        meet_parts.append(mf)
        athlete_parts.append(af)
        diags.append(diag)
        if mf.empty:
            continue
        adj = apply_field_adjusted_times(races, mf, gender)
        adj_parts.append(adj)
        fa_parts.append(field_adjusted_athlete_season(adj))

    meet_factors = pd.concat(meet_parts, ignore_index=True) if meet_parts else pd.DataFrame()
    athlete_fit = (
        pd.concat(athlete_parts, ignore_index=True) if athlete_parts else pd.DataFrame()
    )
    meet_factors.to_csv(os.path.join(output_dir, "course_factors.csv"), index=False)
    athlete_fit.to_csv(os.path.join(output_dir, "athlete_fitness.csv"), index=False)
    with open(os.path.join(output_dir, "course_factor_em_diagnostics.json"), "w") as f:
        json.dump(diags, f, indent=2)

    alpha_weather = course_factor_vs_weather(meet_factors)
    alpha_weather.to_csv(
        os.path.join(output_dir, "course_factor_vs_weather_residual.csv"), index=False
    )
    print(alpha_weather.to_string(index=False))
    plot_alpha_vs_env(meet_factors, os.path.join(output_dir, "alpha_vs_env_residual.pdf"))

    fa_as = pd.concat(fa_parts, ignore_index=True) if fa_parts else pd.DataFrame()
    fa_as.to_csv(os.path.join(output_dir, "field_adjusted_athlete_season.csv"), index=False)
    fa_sum = summarize_field_vs_std(fa_as)
    fa_sum.to_csv(os.path.join(output_dir, "field_adjusted_vs_standardized.csv"), index=False)
    print("\n5. Field-adjusted vs Standardized first→last:")
    print(fa_sum.to_string(index=False))
    plot_field_vs_std_scatter(
        fa_as, os.path.join(output_dir, "field_adjusted_vs_standardized.pdf")
    )

    # Also keep a convenience merge with enrichment-style std table for cross-checks
    as_std = build_athlete_season_table(
        races.rename(columns={"std_time_sec": "standardized_to_target"}),
        "standardized_to_target",
    )
    as_std.to_csv(os.path.join(output_dir, "athlete_season_std_reference.csv"), index=False)

    # Head-to-head: metadata course factors vs field α
    from _relative_finish_comparisons import run_comparisons

    cmp_tables = run_comparisons(races, meet_factors, output_dir)

    # SUMMARY.md
    lines = [
        "# Relative finish & club course factors",
        "",
        f"Generated under `{output_dir}/`.",
        "",
        "Motivation: [LACCTiC](https://www.lacctic.com/) adjusts XC times for course",
        "difficulty from overlapping athletes' relative performances (not weather).",
        "This club analysis derives within-meet place/percentile, compares Raw /",
        "Converted / Standardized season change, and fits a transparent",
        "LACCTiC-style course factor α (EM on raw times; median α pinned to 1;",
        "no track-PR calibration).",
        "",
        "**Head-to-head verdict** (metadata Standardized vs field α): see",
        "[`VERDICT.md`](VERDICT.md).",
        "",
        f"Inclusion: field size ≥ {MIN_FIELD_SIZE} at first and last race for",
        "athlete-season relative metrics; same floor for α estimation.",
        "",
        "## Within-meet order agreement",
        "",
    ]
    for _, row in agree_sum.iterrows():
        lines.append(
            f"- {row['gender']} {row['comparison']}: "
            f"exact place agree={row['mean_exact_place_agree']:.3f} "
            f"[{row['agree_ci_low']:.3f}, {row['agree_ci_high']:.3f}], "
            f"perfect meets={row['pct_meets_perfect_agree']:.1f}%, "
            f"mean Spearman={row['mean_spearman']:.4f}, n_meets={row['n_meets']}"
        )

    lines += ["", "## First→last by clock tier (time and finish percentile)", ""]
    for _, row in tier_sum.iterrows():
        if row["tier"] == "conv_minus_std":
            lines.append(
                f"- {row['gender']} Converted−Std time inflation: "
                f"{row['mean_time_improve_sec']:.1f}s "
                f"[{row['time_ci_low']:.1f}, {row['time_ci_high']:.1f}], "
                f"Wilcoxon p={row['pct_time_improved']:.2e}, n={row['n']}"
            )
        else:
            lines.append(
                f"- {row['gender']} {row['tier']}: time Δ={row['mean_time_improve_sec']:.1f}s "
                f"[{row['time_ci_low']:.1f}, {row['time_ci_high']:.1f}] "
                f"({row['pct_time_improved']:.1f}% faster); "
                f"finish-pct Δ={row['mean_finish_pct_improve']:.3f} "
                f"[{row['pct_ci_low']:.3f}, {row['pct_ci_high']:.3f}] "
                f"({row['pct_place_improved']:.1f}% better place), n={row['n']}"
            )

    lines += ["", "## Concordance (Spearman)", ""]
    for _, row in conc.iterrows():
        lines.append(
            f"- {row['gender']} {row['comparison']}: ρ={row['spearman_rho']:.3f} "
            f"(p={row['spearman_p']:.2e}), sign agree={row['sign_agreement']:.3f}, "
            f"n={row['n']}"
        )

    lines += ["", "## Course factor α vs weather residual", ""]
    for _, row in alpha_weather.iterrows():
        lines.append(
            f"- {row['gender']}: Spearman(α, mean env residual)="
            f"{row['spearman_alpha_vs_env_residual']:.3f} "
            f"(p={row['spearman_p']:.2e}), n_meets={row['n_meets']}, "
            f"sd(α)={row['std_alpha']:.4f}"
        )

    lines += ["", "## Field-adjusted (α·raw) vs Standardized improvement", ""]
    for _, row in fa_sum.iterrows():
        if row["method"] == "field_adj_minus_std":
            lines.append(
                f"- {row['gender']} field_adj−Std: "
                f"{row['mean_improve_sec']:.1f}s "
                f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}], "
                f"Wilcoxon p={row['pct_improved']:.2e}, "
                f"Spearman ρ={row.get('spearman_field_vs_std', np.nan):.3f}"
            )
        elif row["method"] in ("field_adj", "standardized", "raw", "converted"):
            lines.append(
                f"- {row['gender']} {row['method']}: "
                f"{row['mean_improve_sec']:.1f}s "
                f"[{row['ci_low']:.1f}, {row['ci_high']:.1f}], "
                f"%improved={row['pct_improved']:.1f}, n={row['n']}"
            )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- If weather/elevation adjustments are mostly meet-level, within-meet",
        "  place order is nearly identical for Raw / Converted / Standardized;",
        "  relative-finish Δ then mainly reflects field composition change, not",
        "  environmental standardization.",
        "- Field-inferred α is a *complement* to NRCD Standardized: agreement",
        "  with Standardized Δ supports robustness; disagreement highlights meets",
        "  where field strength and weather residuals diverge.",
        "",
        "## Head-to-head (see VERDICT.md)",
        "",
    ]
    if cmp_tables.get("verdict_text"):
        # Append compact win counts from verdict CSV
        v = cmp_tables["verdict"]
        if v is not None and not v.empty:
            wins = v[v["is_best"]].groupby("method").size().sort_values(ascending=False)
            for method, n in wins.items():
                lines.append(f"- {method}: {int(n)} criterion×gender wins")
        lines.append("")
        lines.append("Full write-up: `VERDICT.md`.")
        lines.append("")

    with open(os.path.join(output_dir, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {os.path.join(output_dir, 'SUMMARY.md')}")
    print("Done.")


if __name__ == "__main__":
    main()
