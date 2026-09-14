"""Generate sample-size and metadata-coverage summary for the XC paper Method section.

Writes:
  output/sample_summary.csv
  output/sample_summary_by_year_gender.csv
  output/sample_summary.md
"""

from __future__ import annotations

import os
import sys

from _setup_paths import setup_paths

setup_paths()

import pandas as pd

from load_nrcd_data import (
    COMPREHENSIVE_ERA_START,
    get_data_dir,
    load_analysis_tables,
    load_cross_country_results,
    load_tables,
)


def main(output_dir: str = "output") -> None:
    os.makedirs(output_dir, exist_ok=True)
    data_dir = get_data_dir()
    tables = load_tables(data_dir)
    meet = tables["meet"].copy()
    meet["start_date"] = pd.to_datetime(meet["start_date"], errors="coerce")
    sport = tables["sport"]
    result = tables["result"]
    course = tables["course_details"]
    athlete = tables["athlete"]

    m = meet.merge(sport, on="sport_id", how="left")
    r = result.merge(m[["meet_id", "sport_name", "start_date", "nationals"]], on="meet_id")
    r = r.merge(athlete[["athlete_id", "gender"]], on="athlete_id", how="left")

    hist = r[
        (r["sport_name"] == "Cross Country")
        & (r["start_date"] < COMPREHENSIVE_ERA_START)
    ]
    comp = r[
        (r["sport_name"] == "Cross Country")
        & (r["start_date"] >= COMPREHENSIVE_ERA_START)
    ]

    def coverage(df: pd.DataFrame) -> dict:
        keys = ["meet_id", "running_event_id"]
        cd_keys = course[keys].drop_duplicates().assign(_cd=1)
        weather_keys = (
            course.dropna(subset=["weather_conditions"])[keys]
            .drop_duplicates()
            .assign(_w=1)
        )
        elev_keys = (
            course.dropna(subset=["elevation_gain"])[keys]
            .drop_duplicates()
            .assign(_e=1)
        )
        joined = df.merge(cd_keys, on=keys, how="left")
        joined = joined.merge(weather_keys, on=keys, how="left")
        joined = joined.merge(elev_keys, on=keys, how="left")
        n = len(df)
        return {
            "n_results": n,
            "n_athletes": df["athlete_id"].nunique(),
            "n_meets": df["meet_id"].nunique(),
            "pct_course_details": 100.0 * joined["_cd"].fillna(0).mean() if n else 0.0,
            "pct_weather": 100.0 * joined["_w"].fillna(0).mean() if n else 0.0,
            "pct_elevation_gain": 100.0 * joined["_e"].fillna(0).mean() if n else 0.0,
        }

    rows = [
        {"era": "historical", **coverage(hist)},
        {"era": "comprehensive", **coverage(comp)},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "sample_summary.csv"), index=False)

    xc = load_cross_country_results(tables=tables, era="comprehensive")
    xc["start_date"] = pd.to_datetime(xc["start_date"], errors="coerce")
    xc["year"] = xc["start_date"].dt.year
    by_yg = (
        xc.groupby(["year", "gender"])
        .agg(n_results=("result_id", "count"), n_athletes=("athlete_id", "nunique"),
             n_meets=("meet_id", "nunique"))
        .reset_index()
    )
    by_yg.to_csv(os.path.join(output_dir, "sample_summary_by_year_gender.csv"), index=False)

    # Nationals-excluded ML-style cohort (≥2 races implied later in ML)
    xc_nn = load_cross_country_results(
        tables=tables, era="comprehensive", exclude_nationals=True
    )
    analysis = load_analysis_tables(data_dir, era="comprehensive")

    md_path = os.path.join(output_dir, "sample_summary.md")
    with open(md_path, "w") as f:
        f.write("# NRCD XC analysis sample summary\n\n")
        f.write(f"Data directory: `{data_dir}`\n\n")
        f.write("## Era comparison (Cross Country)\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n## Comprehensive XC by year × gender\n\n")
        f.write(by_yg.to_string(index=False))
        f.write("\n\n## Analysis filters\n\n")
        f.write(f"- Comprehensive XC results: **{len(xc):,}**\n")
        f.write(f"- Excluding nationals: **{len(xc_nn):,}**\n")
        f.write(f"- Filtered meet rows in analysis tables: **{len(analysis['meet']):,}**\n")
        f.write(f"- Filtered result rows in analysis tables: **{len(analysis['result']):,}**\n")

    print(summary.to_string(index=False))
    print()
    print(by_yg.to_string(index=False))
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output")
