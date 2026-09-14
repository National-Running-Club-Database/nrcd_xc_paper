"""Single mixed-effects model with gender x race-count interactions.

Turns the informal "significant for men, imprecise for women" comparison of two
separately-fit models into one testable model: fit both sexes jointly with
gender x z_num_races and gender x z_num_races^2 interaction terms (random
intercept for athlete). The interaction coefficients directly test whether the
linear and quadratic race-count associations differ by sex.

Input:  output/rq1/raw_data_athlete_features.csv
Output: output/rq1/mixed_effects/interaction_fixed_effects.csv
        output/rq1/mixed_effects/interaction_model_summary.txt
"""

import os

import numpy as np
import pandas as pd

from _setup_paths import setup_paths

setup_paths()


def _prepare(df: pd.DataFrame, rate_range=(-50, 50)) -> pd.DataFrame:
    cols = [
        "athlete_id",
        "gender",
        "year",
        "improvement_rate",
        "num_races",
        "season_duration",
        "starting_percentile",
    ]
    work = df[cols].dropna().copy()
    lo, hi = rate_range
    work = work[(work["improvement_rate"] >= lo) & (work["improvement_rate"] <= hi)]
    work["gender"] = work["gender"].astype(str)
    # Female is the reference level; the interaction terms are the male offsets.
    work["male"] = (work["gender"] == "M").astype(int)
    for col in ["num_races", "season_duration", "starting_percentile", "year"]:
        mu = work[col].mean()
        sd = work[col].std(ddof=0) or 1.0
        work[f"z_{col}"] = (work[col] - mu) / sd
    work["z_num_races_sq"] = work["z_num_races"] ** 2
    return work


def main(output_dir: str = "output/rq1") -> None:
    import statsmodels.formula.api as smf

    out_dir = os.path.join(output_dir, "mixed_effects")
    os.makedirs(out_dir, exist_ok=True)
    features_csv = os.path.join(output_dir, "raw_data_athlete_features.csv")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(
            f"Missing {features_csv}; run ml_improvement_prediction.py first."
        )

    work = _prepare(pd.read_csv(features_csv))
    formula = (
        "improvement_rate ~ z_num_races + z_num_races_sq "
        "+ z_season_duration + z_starting_percentile + z_year "
        "+ male + male:z_num_races + male:z_num_races_sq"
    )
    model = smf.mixedlm(formula, data=work, groups=work["athlete_id"])
    try:
        fit = model.fit(reml=False, method="lbfgs")
    except Exception:
        fit = model.fit(reml=False, method="powell", maxiter=2000, disp=False)

    params = fit.params
    conf = fit.conf_int()
    out = pd.DataFrame(
        {
            "term": params.index,
            "estimate": params.values,
            "ci_low": conf[0].values,
            "ci_high": conf[1].values,
            "p_value": fit.pvalues.reindex(params.index).values,
        }
    )
    out.to_csv(os.path.join(out_dir, "interaction_fixed_effects.csv"), index=False)
    with open(os.path.join(out_dir, "interaction_model_summary.txt"), "w") as f:
        f.write("Gender x race-count interaction mixed model (Female = reference)\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Rows: {len(work)}  Athletes: {work['athlete_id'].nunique()}\n\n")
        f.write(str(fit.summary()))
        f.write("\n")

    interaction_terms = [
        "male:z_num_races",
        "male:z_num_races_sq",
    ]
    print("Gender x race-count interaction terms (male offset vs female):")
    print(out[out["term"].isin(interaction_terms)].to_string(index=False))
    print(f"\nSaved interaction model to {out_dir}/")


if __name__ == "__main__":
    main()
