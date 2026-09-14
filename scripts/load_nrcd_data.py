"""Load the National Running Club Database (NRCD) public export and filter to
cross country (comprehensive / current era).

The dataset itself is published separately (NRCD resource paper / Zenodo); this
module resolves the public CSV export and exposes the cross country subset used
by this analysis. The public export spans two eras:

* **Historical** (2004 – July 2023): partial metadata
* **Comprehensive / current** (August 2023 onward): full course/weather metadata

By default we keep only the comprehensive era (and Cross Country).
Standardization is delegated to the ``nrcd`` package (see ``utils.py``).

Data source resolution order:
  1. ``$NRCD_DATA_DIR``     -- explicit directory containing the CSVs.
  2. ``<repo>/data_public`` -- git submodule (public GitHub export).
  3. ``git submodule update --init --recursive data_public`` if needed.
  4. ``<repo>/data``        -- legacy fallback (e.g. a local symlink).
"""

from __future__ import annotations

import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

if sys.version_info < (3, 10):  # pragma: no cover
    raise RuntimeError(
        f"nrcd-xc-paper requires Python >= 3.10 (running {sys.version.split()[0]})."
    )

import pandas as pd

PUBLIC_REPO_URL = (
    "https://github.com/National-Running-Club-Database/"
    "national_running_club_database_public_dataset.git"
)
XC_SPORT_NAME = "Cross Country"
# Meets on or after this date belong to the comprehensive (current) era.
COMPREHENSIVE_ERA_START = pd.Timestamp("2023-08-01")

REQUIRED_FILES = (
    "result.csv",
    "meet.csv",
    "sport.csv",
    "athlete.csv",
    "running_event.csv",
    "course_details.csv",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _has_dataset(path: Path) -> bool:
    return path.is_dir() and all((path / f).exists() for f in REQUIRED_FILES)


def _init_data_submodule(dest: Path) -> None:
    """Initialize/update the ``data_public`` git submodule."""
    root = _repo_root()
    print(
        f"Initializing NRCD public dataset submodule at {dest} ...",
        file=sys.stderr,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "submodule",
            "update",
            "--init",
            "--recursive",
            "data_public",
        ],
        check=True,
    )
    if not _has_dataset(dest):
        raise FileNotFoundError(
            f"Submodule checked out at {dest} but required CSVs are missing. "
            f"Expected files: {', '.join(REQUIRED_FILES)}"
        )


@lru_cache(maxsize=1)
def get_data_dir(refresh: bool = False) -> str:
    """Return a directory containing the NRCD CSV export, fetching if needed."""
    env_dir = os.environ.get("NRCD_DATA_DIR")
    if env_dir and _has_dataset(Path(env_dir)):
        return str(Path(env_dir))

    cache = _repo_root() / "data_public"
    if _has_dataset(cache) and not refresh:
        return str(cache)

    if refresh or not _has_dataset(cache):
        try:
            _init_data_submodule(cache)
        except Exception as exc:  # noqa: BLE001 - fall back to any local copy
            legacy = _repo_root() / "data"
            if _has_dataset(legacy):
                print(
                    f"WARNING: could not init data submodule ({exc}); "
                    f"falling back to {legacy}",
                    file=sys.stderr,
                )
                return str(legacy)
            raise FileNotFoundError(
                "NRCD dataset not found. Initialize the git submodule with:\n"
                "  git submodule update --init --recursive\n"
                f"or set $NRCD_DATA_DIR to a directory containing the CSVs "
                f"(see {PUBLIC_REPO_URL})."
            ) from exc

    if _has_dataset(cache):
        return str(cache)

    legacy = _repo_root() / "data"
    if _has_dataset(legacy):
        return str(legacy)
    raise FileNotFoundError(
        "NRCD dataset not found. Initialize the git submodule with:\n"
        "  git submodule update --init --recursive\n"
        f"or set $NRCD_DATA_DIR (see {PUBLIC_REPO_URL})."
    )


@lru_cache(maxsize=2)
def _load_tables_cached(data_dir: str) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    return {
        "result": pd.read_csv(d / "result.csv", low_memory=False),
        "meet": pd.read_csv(d / "meet.csv"),
        "sport": pd.read_csv(d / "sport.csv"),
        "athlete": pd.read_csv(d / "athlete.csv"),
        "running_event": pd.read_csv(d / "running_event.csv"),
        "course_details": pd.read_csv(d / "course_details.csv", low_memory=False),
    }


def load_tables(data_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """Load the core NRCD CSV tables (cached per data directory in-process)."""
    return _load_tables_cached(str(Path(data_dir or get_data_dir())))


def cross_country_meet_ids(
    meet: pd.DataFrame,
    sport: pd.DataFrame,
    *,
    era: str = "comprehensive",
    exclude_nationals: bool = False,
) -> set:
    """Return meet_ids for Cross Country rows, optionally era-/nationals-filtered."""
    m = meet.merge(sport, on="sport_id", how="left")
    m["start_date"] = pd.to_datetime(m["start_date"], errors="coerce")
    mask = m["sport_name"] == XC_SPORT_NAME
    if era == "comprehensive":
        mask &= m["start_date"] >= COMPREHENSIVE_ERA_START
    elif era == "historical":
        mask &= m["start_date"] < COMPREHENSIVE_ERA_START
    elif era in ("all", None):
        pass
    else:
        raise ValueError(
            f"Unknown era={era!r}; expected 'comprehensive', 'historical', or 'all'"
        )
    if exclude_nationals and "nationals" in m.columns:
        mask &= m["nationals"].fillna(False).astype(bool) == False  # noqa: E712
    return set(m.loc[mask, "meet_id"].tolist())


def load_cross_country_results(
    *,
    tables: dict[str, pd.DataFrame] | None = None,
    data_dir: str | None = None,
    era: str = "comprehensive",
    exclude_nationals: bool = False,
) -> pd.DataFrame:
    """Return Cross Country result rows joined with meet/athlete/event fields."""
    tables = tables or load_tables(data_dir)
    meet_ids = cross_country_meet_ids(
        tables["meet"],
        tables["sport"],
        era=era,
        exclude_nationals=exclude_nationals,
    )
    result = tables["result"]
    df = result[result["meet_id"].isin(meet_ids)].copy()

    athlete = tables["athlete"]
    if "gender" not in df.columns and "athlete_id" in df.columns:
        df = df.merge(athlete[["athlete_id", "gender"]], on="athlete_id", how="left")

    meet = tables["meet"].copy()
    meet["start_date"] = pd.to_datetime(meet["start_date"], errors="coerce")
    meet_cols = ["meet_id", "start_date"]
    for optional in ("nationals", "altitude", "end_date", "name"):
        if optional in meet.columns and optional not in df.columns:
            meet_cols.append(optional)
    df = df.merge(meet[meet_cols], on="meet_id", how="left")

    running_event = tables["running_event"]
    if "event_name" not in df.columns and "running_event_id" in df.columns:
        df = df.merge(
            running_event[["running_event_id", "event_name"]],
            on="running_event_id",
            how="left",
        )
    return df


def load_analysis_tables(
    data_dir: str | None = None,
    *,
    era: str = "comprehensive",
    exclude_nationals: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load core tables filtered to the analysis Cross Country meet set.

    Returns the same keys as ``load_tables``, plus filtered ``meet`` / ``result``
    rows for the requested era. Athlete / event / course tables are left full so
    joins remain valid.
    """
    tables = load_tables(data_dir)
    meet_ids = cross_country_meet_ids(
        tables["meet"],
        tables["sport"],
        era=era,
        exclude_nationals=exclude_nationals,
    )
    meet = tables["meet"][tables["meet"]["meet_id"].isin(meet_ids)].copy()
    result = tables["result"][tables["result"]["meet_id"].isin(meet_ids)].copy()
    return {
        "result": result,
        "meet": meet,
        "sport": tables["sport"],
        "athlete": tables["athlete"],
        "running_event": tables["running_event"],
        "course_details": tables["course_details"],
    }
