"""
Mixed-effects explanatory model (complementary to ML).

Goal: provide an interpretable, repeated-measures-aware model of improvement_rate
using random effects for athletes (athlete_id), trained on the same derived
feature table used by the ML analysis.

Input:  output/rq1/raw_data_athlete_features.csv
Output: output/rq1/mixed_effects/ (men and women only; no pooled run)
  - mixedlm_fixed_effects_{men,women}.csv
  - mixedlm_variance_components_{men,women}.csv
  - mixedlm_model_summary_{men,women}.txt
  - mixedlm_coefficients_forest_{men,women}.pdf
  - ols_comparison_{men,women}.csv
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _prepare_df(
    df: pd.DataFrame,
    gender_filter: str = "all",
    improvement_rate_range: Tuple[float, float] = (-50, 50),
) -> pd.DataFrame:
    required = ["athlete_id", "gender", "year", "improvement_rate", "num_races", "season_duration", "starting_percentile"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df[required].copy()
    work = work.dropna()
    lo, hi = improvement_rate_range
    work = work[(work["improvement_rate"] >= lo) & (work["improvement_rate"] <= hi)].copy()

    # Encode gender for fixed effects (keep original too)
    work["gender"] = work["gender"].astype(str)
    work["gender_M"] = (work["gender"] == "M").astype(int)

    gender_filter = str(gender_filter).lower()
    if gender_filter in {"m", "men"}:
        work = work[work["gender"] == "M"].copy()
    elif gender_filter in {"f", "women", "w"}:
        work = work[work["gender"] == "F"].copy()
    elif gender_filter in {"all", "both"}:
        pass
    else:
        raise ValueError(f"Unknown gender_filter={gender_filter!r}. Use 'all', 'M', or 'F'.")

    # Center/scale continuous predictors for stability and interpretability
    for col in ["num_races", "season_duration", "starting_percentile", "year"]:
        mu = work[col].mean()
        sd = work[col].std(ddof=0) or 1.0
        work[f"z_{col}"] = (work[col] - mu) / sd

    return work


def _fit_models(work: pd.DataFrame, include_gender_term: bool):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    # Fixed effects: basic training/experience proxies + gender + year trend
    # Random intercept for athlete_id captures repeated measures across years (if present).
    if include_gender_term:
        formula = "improvement_rate ~ z_num_races + z_season_duration + z_starting_percentile + z_year + gender_M"
    else:
        formula = "improvement_rate ~ z_num_races + z_season_duration + z_starting_percentile + z_year"

    # OLS baseline (ignores repeated measures)
    ols = smf.ols(formula=formula, data=work).fit(cov_type="HC3")

    # MixedLM (random intercept by athlete)
    model = smf.mixedlm(formula=formula, data=work, groups=work["athlete_id"])
    try:
        mixed = model.fit(reml=False, method="lbfgs")
    except Exception:
        # Fallback for occasional optimizer instability
        mixed = model.fit(reml=False, method="powell", maxiter=2000, disp=False)
    return ols, mixed, formula


def _coef_table(result, model_name: str) -> pd.DataFrame:
    params = result.params
    conf = result.conf_int()
    out = pd.DataFrame(
        {
            "term": params.index,
            "estimate": params.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
        }
    )
    out["model"] = model_name
    # p-values if present (MixedLM provides pvalues; robust OLS does as well)
    if hasattr(result, "pvalues"):
        out["p_value"] = result.pvalues.values
    return out


def _variance_table(mixed_result) -> pd.DataFrame:
    rows = []
    # Random intercept variance
    try:
        re_var = float(mixed_result.cov_re.iloc[0, 0])
        rows.append({"component": "random_intercept_athlete_id", "variance": re_var})
    except Exception:
        pass
    # Residual variance
    try:
        rows.append({"component": "residual", "variance": float(mixed_result.scale)})
    except Exception:
        pass
    return pd.DataFrame(rows)


def _plot_forest(coefs: pd.DataFrame, out_path: str) -> None:
    # Drop intercept for plot
    plot = coefs[coefs["term"] != "Intercept"].copy()
    plot = plot.sort_values("estimate")

    plt.figure(figsize=(7, max(3.5, 0.5 * len(plot))))
    plt.errorbar(plot["estimate"], plot["term"], xerr=[plot["estimate"] - plot["ci_low"], plot["ci_high"] - plot["estimate"]], fmt="o")
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Mixed-effects model coefficients (95% CI)\nOutcome: improvement_rate (sec/day)")
    plt.xlabel("Coefficient (sec/day per 1 SD increase in predictor)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(output_dir: str = "output/rq1") -> None:
    out_dir = os.path.join(output_dir, "mixed_effects")
    _ensure_dir(out_dir)

    features_csv = os.path.join(output_dir, "raw_data_athlete_features.csv")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(
            f"Expected features table not found at {features_csv}. "
            "Run `python scripts/ml_improvement_prediction.py` (or `python scripts/rq1.py`) first."
        )

    df = _load_table(features_csv)

    def run_one(label: str, gender_filter: str) -> None:
        suffix = label.lower()
        include_gender = gender_filter.lower() in {"all", "both"}
        work = _prepare_df(df, gender_filter=gender_filter)

        if len(work) == 0:
            raise ValueError(f"No rows remaining after filtering for {label}.")

        try:
            ols, mixed, formula = _fit_models(work, include_gender_term=include_gender)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"{e}\n\nMissing dependency. Install with:\n  pip install statsmodels\n"
            )

        # Save summaries
        with open(os.path.join(out_dir, f"mixedlm_model_summary_{suffix}.txt"), "w") as f:
            f.write("Mixed-effects explanatory model\n")
            f.write("===============================\n\n")
            f.write(f"Subset: {label}\n")
            f.write(f"Formula: {formula}\n")
            f.write(f"Rows used: {len(work)}\n")
            f.write(f"Unique athletes: {work['athlete_id'].nunique()}\n\n")
            f.write("OLS (robust SE, HC3):\n")
            f.write(str(ols.summary()))
            f.write("\n\nMixedLM (random intercept by athlete_id, ML fit):\n")
            f.write(str(mixed.summary()))
            f.write("\n")

        mixed_coefs = _coef_table(mixed, f"MixedLM_{label}")
        ols_coefs = _coef_table(ols, f"OLS_HC3_{label}")

        mixed_coefs.to_csv(os.path.join(out_dir, f"mixedlm_fixed_effects_{suffix}.csv"), index=False)
        ols_coefs.to_csv(os.path.join(out_dir, f"ols_fixed_effects_{suffix}.csv"), index=False)

        var_df = _variance_table(mixed)
        var_df.to_csv(os.path.join(out_dir, f"mixedlm_variance_components_{suffix}.csv"), index=False)

        # Simple fit comparison on the used data (in-sample): R², RMSE, MAE
        y = work["improvement_rate"].to_numpy()
        ols_pred = ols.predict(work).to_numpy()
        mixed_pred = mixed.predict(work).to_numpy()

        def metrics(y_true, y_pred):
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))
            return r2, rmse, mae

        rows = []
        for name, pred in [(f"OLS_HC3_{label}", ols_pred), (f"MixedLM_{label}", mixed_pred)]:
            r2, rmse, mae = metrics(y, pred)
            rows.append({"model": name, "r2_in_sample": r2, "rmse_in_sample": rmse, "mae_in_sample": mae, "n": len(work)})
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"ols_comparison_{suffix}.csv"), index=False)

        # Coefficient forest plot (MixedLM)
        _plot_forest(mixed_coefs, os.path.join(out_dir, f"mixedlm_coefficients_forest_{suffix}.pdf"))

    # Men and women only: skip pooled "All" models (gender encoding / heterogeneity).
    run_one("Men", "M")
    run_one("Women", "F")

    print(f"\nMixed-effects modeling complete (Men/Women). Outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()

