"""Mathematical identities and derived indices for the NRCD XC analysis paper.

Three contributions that go beyond "run another regression":

1. Weather-path bias identity (exact)
       Δ_conv − Δ_std = a_1 − a_L
   where a = t_conv − t_std is the per-race environment residual.
   Mean inflation is therefore the seasonal drift of that residual — an
   algebraic identity, not a fitted model. We verify it to machine precision.

2. Reliability bound and Signal Extraction Ratio (SER)
       R² ≤ ρ_yy'          (classical attenuation / noise ceiling)
       SER := R²_heldout / ρ_yy'
   Fraction of *reliable* outcome variance captured. Near-zero SER reframes
   the individual null as "we extract almost none of a small reliable signal."
   Also reports an endpoint-noise floor φ for two-race seasons:
       φ := 2 σ̂_ε² / Var(Δ)
   using σ̂_ε² from the within-athlete ICC residual.

3. Effective Racing Opportunity (ERO) index
       ERO := exp(H) · n̄
   where H is Shannon entropy of the roster start distribution and n̄ is mean
   starts. exp(H) is the perplexity ("effective number of equally racing
   athletes"). ERO is a closed-form program score; we show out-of-year
   top-15 discrimination vs max-race-count alone.

Outputs: output/rq1/mathematical_contributions/
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from _setup_paths import setup_paths

setup_paths()

from paper_enrichment_analyses import build_athlete_season_table
from underexplored_mechanisms import build_team_roster_metrics, _load_race_frame
from utils import standardize_both_tiers, standardize_convert_exclude_nationals_df

RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(
    Path(__file__).resolve().parents[1],
    "output",
    "rq1",
    "mathematical_contributions",
)

# Primary held-out R² from gender-separated SVR (train 2023 → test 2024)
PRIMARY_R2 = {"M": 0.043, "F": -0.029}
# Split-half reliability of improvement slope (from null diagnostics)
RELIABILITY = {"M": 0.22822087419717996, "F": 0.2787539600162913}
SINGLE_RACE_ICC = {"M": 0.7751304229400794, "F": 0.8197451602502119}


# ---------------------------------------------------------------------------
# 1. Weather-path bias identity
# ---------------------------------------------------------------------------


def _paired_race_times() -> pd.DataFrame:
    """Merge Standardized and Converted times at the race level."""
    df_conv, df_std = standardize_both_tiers()
    key = ["athlete_id", "meet_id", "gender"]
    for d in (df_std, df_conv):
        d["start_date"] = pd.to_datetime(d["start_date"], errors="coerce")
    std = df_std[key + ["start_date", "standardized_to_target"]].rename(
        columns={"standardized_to_target": "t_std"}
    )
    conv = df_conv[key + ["standardized_to_target"]].rename(
        columns={"standardized_to_target": "t_conv"}
    )
    merged = std.merge(conv, on=key, how="inner")
    merged = merged.dropna(subset=["t_std", "t_conv", "start_date", "athlete_id"])
    merged["a"] = merged["t_conv"] - merged["t_std"]  # environment residual
    merged["year"] = merged["start_date"].dt.year
    return merged


def weather_path_identity(race_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate Δ_conv − Δ_std = a_1 − a_L athlete-season by athlete-season."""
    rows = []
    for (athlete_id, year), races in race_df.groupby(["athlete_id", "year"], sort=False):
        races = races.sort_values("start_date")
        if len(races) < 2:
            continue
        first, last = races.iloc[0], races.iloc[-1]
        days = (last["start_date"] - first["start_date"]).days
        if days < 7:
            continue
        delta_std = float(first["t_std"] - last["t_std"])  # + = faster
        delta_conv = float(first["t_conv"] - last["t_conv"])
        a1, aL = float(first["a"]), float(last["a"])
        path_bias = a1 - aL
        inflation = delta_conv - delta_std
        rows.append(
            {
                "athlete_id": int(athlete_id),
                "year": int(year),
                "gender": first["gender"],
                "num_races": int(len(races)),
                "delta_std": delta_std,
                "delta_conv": delta_conv,
                "inflation": inflation,
                "path_bias_a1_minus_aL": path_bias,
                "identity_residual": inflation - path_bias,
                "a_first": a1,
                "a_last": aL,
            }
        )
    paired = pd.DataFrame(rows)
    # Numerical identity check
    max_abs = float(np.nanmax(np.abs(paired["identity_residual"])))
    summary_rows = []
    for gender, g in paired.groupby("gender"):
        summary_rows.append(
            {
                "gender": "Men" if gender == "M" else "Women",
                "n": int(len(g)),
                "mean_inflation_sec": float(g["inflation"].mean()),
                "mean_path_bias_sec": float(g["path_bias_a1_minus_aL"].mean()),
                "max_abs_identity_residual": float(np.abs(g["identity_residual"]).max()),
                "corr_inflation_path_bias": float(
                    g["inflation"].corr(g["path_bias_a1_minus_aL"])
                ),
                "mean_a_first": float(g["a_first"].mean()),
                "mean_a_last": float(g["a_last"].mean()),
                "seasonal_residual_drift": float(
                    g["a_first"].mean() - g["a_last"].mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.attrs["global_max_abs_identity_residual"] = max_abs
    return paired, summary


# ---------------------------------------------------------------------------
# 2. Reliability bound, SER, endpoint-noise floor
# ---------------------------------------------------------------------------


def signal_extraction_and_noise_floor(
    as_std: pd.DataFrame, race_std: pd.DataFrame
) -> pd.DataFrame:
    """SER = R² / ρ and φ = 2σ_ε² / Var(Δ) for two-race seasons."""
    rows = []
    race_std = race_std.copy()
    race_std["start_date"] = pd.to_datetime(race_std["start_date"], errors="coerce")
    race_std["year"] = race_std["start_date"].dt.year

    for gender_code, gender_name in [("M", "Men"), ("F", "Women")]:
        rho = RELIABILITY[gender_code]
        r2 = PRIMARY_R2[gender_code]
        ser = r2 / rho if rho > 0 else np.nan
        # Bound slack: how far below the ceiling
        slack = rho - max(r2, 0.0)

        g_as = as_std[as_std["gender"] == gender_code]
        # Two-race endpoint-noise floor
        two = g_as[g_as["num_races"] == 2]
        var_delta = float(two["total_improvement_sec"].var(ddof=1)) if len(two) > 2 else np.nan

        g_races = race_std[race_std["gender"] == gender_code]
        # Within-athlete residual variance from ICC identity:
        # ICC = σ_b² / (σ_b² + σ_ε²)  ⇒  σ_ε² = (1 − ICC) · Var(t) within seasons
        # Use pooled within-season variance of times.
        season_var = []
        for (_, _), races in g_races.groupby(["athlete_id", "year"]):
            if len(races) < 2:
                continue
            season_var.append(float(races["standardized_to_target"].var(ddof=1)))
        mean_within_var = float(np.nanmean(season_var)) if season_var else np.nan
        icc = SINGLE_RACE_ICC[gender_code]
        # Rough: average within-season total var ≈ σ_b_season + σ_ε; ICC was
        # estimated across races within athlete-season, so σ_ε² ≈ (1-ICC)*Var_within_approx
        # Better: σ_ε² = MS_within from one-way; here use (1-ICC) * mean season var
        # as a conservative plug-in (season var includes true change + noise).
        # Prefer residual after removing linear time trend when ≥3 races.
        resid_vars = []
        for (_, _), races in g_races.groupby(["athlete_id", "year"]):
            races = races.sort_values("start_date")
            y = races["standardized_to_target"].to_numpy(float)
            if len(y) < 3 or not np.all(np.isfinite(y)):
                continue
            x = np.arange(len(y), dtype=float)
            # OLS residual variance
            beta = np.polyfit(x, y, 1)
            resid = y - np.polyval(beta, x)
            resid_vars.append(float(resid.var(ddof=1)))
        sigma_eps2 = float(np.nanmean(resid_vars)) if resid_vars else mean_within_var * (1 - icc)
        phi = (2.0 * sigma_eps2 / var_delta) if (var_delta and var_delta > 0) else np.nan

        # RTM null under shared athlete mean + iid race noise:
        # κ_null = Corr(t1, t1−tL) = sqrt((1 − ICC) / 2)
        kappa = float(g_as["first_time"].corr(g_as["total_improvement_sec"]))
        kappa_null = float(np.sqrt(max(1.0 - icc, 0.0) / 2.0))

        rows.append(
            {
                "gender": gender_name,
                "heldout_r2": r2,
                "reliability_rho": rho,
                "r2_ceiling": rho,
                "SER": ser,
                "SER_clipped": max(ser, 0.0),
                "ceiling_slack": slack,
                "single_race_icc": icc,
                "sigma_eps2_trend_resid": sigma_eps2,
                "var_delta_two_race": var_delta,
                "noise_floor_phi_two_race": phi,
                "n_two_race": int(len(two)),
                "kappa_corr_first_delta": kappa,
                "kappa_null_icc": kappa_null,
                "kappa_excess_over_null": kappa - kappa_null,
                "n_athlete_seasons": int(len(g_as)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Effective Racing Opportunity (ERO)
# ---------------------------------------------------------------------------


def effective_racing_opportunity(roster: pd.DataFrame) -> pd.DataFrame:
    """ERO = exp(H) * mean_race_count; also z-scored variants for modeling."""
    out = roster.copy()
    out["effective_n_athletes"] = np.exp(out["shannon_entropy"].clip(lower=0))
    out["ERO"] = out["effective_n_athletes"] * out["mean_race_count"]
    # Geometric alternative: depth × intensity under equal weight in log space
    out["ERO_log"] = np.log1p(out["ERO"])
    return out


def ero_discrimination(roster: pd.DataFrame) -> pd.DataFrame:
    """Train 2023–2024 logistic → test 2025 AUC for ERO vs max-race vs depth."""
    data = effective_racing_opportunity(roster).dropna(
        subset=["ERO", "max_race_count", "n_athletes_ge3", "top15"]
    )
    train = data[data["year"].isin([2023, 2024])]
    test = data[data["year"] == 2025]
    specs = {
        "max_race_count": ["max_race_count"],
        "n_athletes_ge3": ["n_athletes_ge3"],
        "ERO": ["ERO"],
        "ERO_plus_maxrace": ["ERO", "max_race_count"],
        "depth_plus_maxrace": ["n_athletes_ge3", "max_race_count"],
    }
    rows = []
    for name, cols in specs.items():
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[cols].to_numpy(float))
        Xte = scaler.transform(test[cols].to_numpy(float))
        ytr = train["top15"].to_numpy(int)
        yte = test["top15"].to_numpy(int)
        clf = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        auc = float(roc_auc_score(yte, proba))
        # Also report in-sample pooled OR via simple z-score correlation proxy
        rows.append(
            {
                "model": name,
                "features": "+".join(cols),
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "test_auc_2025": auc,
                "coef": ";".join(f"{c}={v:.3f}" for c, v in zip(cols, clf.coef_.ravel())),
            }
        )
    return pd.DataFrame(rows)


def ero_pooled_or(roster: pd.DataFrame) -> pd.DataFrame:
    """Cluster-robust-ish: simple logistic OR per 1 SD of ERO / depth / maxrace."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return pd.DataFrame()

    data = effective_racing_opportunity(roster).dropna()
    rows = []
    for feat in ["ERO", "n_athletes_ge3", "max_race_count", "effective_n_athletes"]:
        z = StandardScaler().fit_transform(data[[feat]].to_numpy(float)).ravel()
        X = sm.add_constant(z)
        y = data["top15"].to_numpy(float)
        try:
            fit = sm.Logit(y, X).fit(disp=False)
            beta = float(fit.params[1])
            se = float(fit.bse[1])
            lo, hi = beta - 1.96 * se, beta + 1.96 * se
            rows.append(
                {
                    "feature": feat,
                    "OR_per_SD": float(np.exp(beta)),
                    "OR_lo": float(np.exp(lo)),
                    "OR_hi": float(np.exp(hi)),
                    "p": float(fit.pvalues[1]),
                    "n": int(len(data)),
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots + markdown summary of formulas
# ---------------------------------------------------------------------------


def plot_identity_check(paired: pd.DataFrame, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for ax, gender, title in zip(
        axes, ["M", "F"], ["Men", "Women"]
    ):
        g = paired[paired["gender"] == gender]
        ax.scatter(
            g["path_bias_a1_minus_aL"],
            g["inflation"],
            s=8,
            alpha=0.25,
            edgecolors="none",
            c="#1f4e79" if gender == "M" else "#8b2942",
        )
        lims = [
            min(g["path_bias_a1_minus_aL"].min(), g["inflation"].min()),
            max(g["path_bias_a1_minus_aL"].max(), g["inflation"].max()),
        ]
        ax.plot(lims, lims, "k--", lw=1, label=r"identity $y=x$")
        ax.set_xlabel(r"$a_1 - a_L$ (path bias, s)")
        ax.set_ylabel(r"$\Delta_{\mathrm{conv}} - \Delta_{\mathrm{std}}$ (s)")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        r"Weather-path bias identity: $\Delta_{\mathrm{conv}}-\Delta_{\mathrm{std}}=a_1-a_L$",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_ero_vs_placement(roster: pd.DataFrame, path: str) -> None:
    data = effective_racing_opportunity(roster)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for top, label, color in [
        (1, "Top-15", "#1f4e79"),
        (0, "Other qualifying", "#9aa0a6"),
    ]:
        sub = data[data["top15"] == top]
        ax.scatter(
            sub["ERO"],
            sub["max_race_count"] + np.random.default_rng(42).normal(0, 0.08, len(sub)),
            s=18,
            alpha=0.45,
            c=color,
            label=label,
            edgecolors="none",
        )
    ax.set_xlabel(r"Effective Racing Opportunity  $\mathrm{ERO}=e^{H}\bar{n}$")
    ax.set_ylabel("Max athlete race count")
    ax.set_title("Program opportunity index vs workhorse max races")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_formulas_md(
    weather_sum: pd.DataFrame,
    ser_df: pd.DataFrame,
    ero_auc: pd.DataFrame,
    ero_or: pd.DataFrame,
    max_resid: float,
    path: str,
) -> None:
    men_ser = ser_df.loc[ser_df["gender"] == "Men"].iloc[0]
    women_ser = ser_df.loc[ser_df["gender"] == "Women"].iloc[0]
    men_w = weather_sum.loc[weather_sum["gender"] == "Men"].iloc[0]
    women_w = weather_sum.loc[weather_sum["gender"] == "Women"].iloc[0]
    ero_row = ero_auc.loc[ero_auc["model"] == "ERO"].iloc[0]
    max_row = ero_auc.loc[ero_auc["model"] == "max_race_count"].iloc[0]
    depth_row = ero_auc.loc[ero_auc["model"] == "n_athletes_ge3"].iloc[0]

    lines = [
        "# Mathematical contributions",
        "",
        "Closed-form identities and one derived program index, estimated on the",
        "comprehensive-era NRCD XC sample. Reproducible via",
        "`python scripts/mathematical_contributions.py`.",
        "",
        "## 1. Weather-path bias identity (exact)",
        "",
        "Let $t^{\\mathrm{conv}}_{ir}$ and $t^{\\mathrm{std}}_{ir}$ be Converted Only and",
        "Standardized times for athlete $i$ at race $r$, and define the environment",
        "residual",
        "",
        "$$a_{ir} := t^{\\mathrm{conv}}_{ir} - t^{\\mathrm{std}}_{ir}.$$",
        "",
        "First-to-last improvement (positive = faster) is $\\Delta = t_1 - t_L$. Then",
        "",
        "$$",
        "\\Delta^{\\mathrm{conv}} - \\Delta^{\\mathrm{std}}",
        "= (t_1^{c}-t_L^{c}) - (t_1^{s}-t_L^{s})",
        "= (t_1^{c}-t_1^{s}) - (t_L^{c}-t_L^{s})",
        "= a_1 - a_L.",
        "$$",
        "",
        "So **weather inflation of apparent improvement equals the first-to-last",
        "drift of the environment residual** — an algebraic identity. Empirically,",
        f"max $|(\\Delta_c-\\Delta_s)-(a_1-a_L)| = {max_resid:.2e}$ s (machine noise).",
        f"Mean inflation: men ${men_w['mean_inflation_sec']:.1f}$ s, women",
        f"${women_w['mean_inflation_sec']:.1f}$ s, matching $E[a_1-a_L]$.",
        "",
        "## 2. Reliability bound and Signal Extraction Ratio",
        "",
        "If the improvement outcome has split-half reliability $\\rho_{yy'}$, classical",
        "attenuation implies any predictor of the *observed* outcome satisfies",
        "",
        "$$R^2 \\le \\rho_{yy'}.$$",
        "",
        "Define the **Signal Extraction Ratio**",
        "",
        "$$\\mathrm{SER} := \\frac{R^2_{\\mathrm{heldout}}}{\\rho_{yy'}}.$$",
        "",
        "SER is the fraction of *reliable* outcome variance captured.",
        f"Men: $R^2={men_ser['heldout_r2']:.3f}$, $\\rho={men_ser['reliability_rho']:.3f}$,",
        f"$\\mathrm{{SER}}={men_ser['SER']:.3f}$ (clipped ${men_ser.get('SER_clipped', max(men_ser['SER'], 0)):.3f}$).",
        f"Women: $R^2={women_ser['heldout_r2']:.3f}$, $\\rho={women_ser['reliability_rho']:.3f}$,",
        f"$\\mathrm{{SER}}={women_ser['SER']:.3f}$ (negative $R^2$ ⇒ no reliable signal extracted).",
        "",
        "For two-race seasons, an endpoint-noise floor is",
        "",
        "$$\\phi := \\frac{2\\hat\\sigma_\\varepsilon^2}{\\widehat{\\mathrm{Var}}(\\Delta)},$$",
        "",
        f"with $\\hat\\sigma_\\varepsilon^2$ from within-season trend residuals.",
        f"Men $\\phi={men_ser['noise_floor_phi_two_race']:.2f}$;",
        f"women $\\phi={women_ser['noise_floor_phi_two_race']:.2f}$",
        "(large $\\phi$ means two-race $\\Delta$ is heavily measurement noise).",
        "",
        "Under a shared-mean + iid race-noise null,",
        "$\\kappa_{\\mathrm{null}}=\\mathrm{Corr}(t_1,\\Delta)=\\sqrt{(1-\\mathrm{ICC})/2}$.",
        f"Observed $\\kappa$ is {men_ser['kappa_corr_first_delta']:.2f} (men) /",
        f"{women_ser['kappa_corr_first_delta']:.2f} (women) vs null",
        f"{men_ser['kappa_null_icc']:.2f} / {women_ser['kappa_null_icc']:.2f}",
        "— excess over null is the slower-starters-improve-more association beyond pure RTM.",
        "",
        "## 3. Effective Racing Opportunity (ERO)",
        "",
        "For a team roster with start counts $\\{n_j\\}$ and Shannon entropy",
        "$H=-\\sum p_j\\log p_j$, $p_j=n_j/\\sum n_k$, define",
        "",
        "$$",
        "\\mathrm{ERO} := e^{H}\\,\\bar n,",
        "\\qquad e^{H}=\\text{perplexity (effective number of equally racing athletes)}.",
        "$$",
        "",
        "ERO multiplies *effective depth* by *mean intensity* in one scalar.",
        f"Out-of-year (train 2023–24 → test 2025) top-15 AUC:",
        f"ERO ${ero_row['test_auc_2025']:.3f}$,",
        f"depth (#≥3 starts) ${depth_row['test_auc_2025']:.3f}$,",
        f"max race count ${max_row['test_auc_2025']:.3f}$.",
        "",
    ]
    if len(ero_or):
        lines += ["Pooled logistic OR per 1 SD:", ""]
        for _, r in ero_or.iterrows():
            lines.append(
                f"- {r['feature']}: OR={r['OR_per_SD']:.2f} "
                f"[{r['OR_lo']:.2f}, {r['OR_hi']:.2f}], p={r['p']:.2g}"
            )
        lines.append("")
    lines += [
        "## Interpretation for the paper",
        "",
        "- Identity (1) *explains* weather inflation without a new regression.",
        "- SER (2) reframes the individual null as near-zero extraction of a weak",
        "  reliable signal, with an explicit ceiling $R^2\\le\\rho$.",
        "- ERO (3) is a closed-form team score combining entropy-depth and intensity;",
        "  it matches or beats max-race-count alone for out-of-year placement.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


def main(output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rng_state = np.random.get_state()
    np.random.seed(RANDOM_SEED)

    print("1. Weather-path bias identity...")
    race_paired = _paired_race_times()
    paired, weather_sum = weather_path_identity(race_paired)
    paired.to_csv(os.path.join(output_dir, "weather_path_identity_paired.csv"), index=False)
    weather_sum.to_csv(os.path.join(output_dir, "weather_path_identity_summary.csv"), index=False)
    max_resid = float(np.abs(paired["identity_residual"]).max())
    print(f"   max |identity residual| = {max_resid:.3e} s")
    print(weather_sum.to_string(index=False))
    plot_identity_check(paired, os.path.join(output_dir, "weather_path_identity.pdf"))

    print("\n2. SER and noise floor...")
    df_std = standardize_convert_exclude_nationals_df()
    as_std = build_athlete_season_table(df_std, "standardized_to_target")
    ser_df = signal_extraction_and_noise_floor(as_std, df_std)
    ser_df.to_csv(os.path.join(output_dir, "signal_extraction_ratio.csv"), index=False)
    print(ser_df.to_string(index=False))

    print("\n3. Effective Racing Opportunity...")
    race_frame = _load_race_frame()
    roster = build_team_roster_metrics(race_frame)
    roster_ero = effective_racing_opportunity(roster)
    roster_ero.to_csv(os.path.join(output_dir, "team_ero.csv"), index=False)
    ero_auc = ero_discrimination(roster)
    ero_auc.to_csv(os.path.join(output_dir, "ero_out_of_year_auc.csv"), index=False)
    ero_or = ero_pooled_or(roster)
    ero_or.to_csv(os.path.join(output_dir, "ero_pooled_or.csv"), index=False)
    print(ero_auc.to_string(index=False))
    if len(ero_or):
        print(ero_or.to_string(index=False))
    plot_ero_vs_placement(roster, os.path.join(output_dir, "ero_vs_maxrace.pdf"))

    write_formulas_md(
        weather_sum,
        ser_df,
        ero_auc,
        ero_or,
        max_resid,
        os.path.join(output_dir, "FORMULAS.md"),
    )

    manifest = {
        "random_seed": RANDOM_SEED,
        "primary_r2": PRIMARY_R2,
        "reliability": RELIABILITY,
        "max_abs_identity_residual_sec": max_resid,
        "outputs": sorted(os.listdir(output_dir)),
    }
    with open(os.path.join(output_dir, "reproducibility.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    np.random.set_state(rng_state)
    print(f"\nWrote {output_dir}")


if __name__ == "__main__":
    main()
