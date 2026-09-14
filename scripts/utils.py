"""Cross country data preparation for the NRCD XC analysis.

Standardization is delegated to the ``nrcd`` package (PyPI, ``nrcd[data]>=0.1.5``).
This module loads the public dataset (git submodule ``data_public/``), filters
to cross country, and applies the package's two tiers via **parallel batch**
conversion (`standardize_dataframe` over row chunks):

* ``standardized`` — distance + course accuracy + weather + elevation
* ``converted``    — distance + course accuracy only

Prepared frames are cached under ``output/.cache/`` so repeated script runs
skip re-standardization when the dataset commit and nrcd version are unchanged.

Requires **Python >= 3.10**.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

if sys.version_info < (3, 10):  # pragma: no cover
    raise RuntimeError(
        f"nrcd-xc-paper requires Python >= 3.10 (running {sys.version.split()[0]}). "
        "See pyproject.toml requires-python."
    )

import numpy as np
import pandas as pd

from load_nrcd_data import (
    get_data_dir,
    load_cross_country_results,
    load_tables,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

try:
    import nrcd
    from nrcd.standardize import (
        standardize_dataframe,
        standardize_xc,
        xc_target_distance_m,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'nrcd' package (>=0.1.5) is required for standardization "
        "(Python >= 3.10). Install with:\n"
        "  pip install 'nrcd[data]>=0.1.5'"
    ) from exc

# Environment columns nulled for the Converted Only tier.
_ENV_COLS = (
    "temperature",
    "dew_point",
    "elevation_gain",
    "elevation_loss",
    "meet_elevation",
    "barometric_pressure",
    "altitude",
)

# Columns passed into nrcd.standardize_dataframe (keeps pickling light).
_STD_COLS = (
    "gender",
    "result_time",
    "sport_name",
    "event_name",
    "reported_distance_m",
    "actual_distance_m",
    "target_distance_m",
    "temperature",
    "dew_point",
    "elevation_gain",
    "elevation_loss",
    "meet_elevation",
    "barometric_pressure",
    "grade_input",
    "course_distance_m",
)

_CHUNK_ROWS = 2500
_MAX_WORKERS = 8
_CACHE_ENV = "NRCD_XC_DISABLE_CACHE"
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
_POOL_WARNED = False


def _pool_initializer(scripts_dir: str) -> None:
    """Ensure worker processes can import ``_nrcd_std_worker`` (macOS spawn)."""
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)


def parse_time(time_str):
    """Parse a clock string to seconds; invalid values become NaN (analysis-safe)."""
    if pd.isna(time_str):
        return float("nan")
    parts = str(time_str).split(":")
    try:
        if len(parts) == 3:
            h, m, s = map(float, parts)
            return h * 3600 + m * 60 + s
        if len(parts) == 2:
            m, s = map(float, parts)
            return m * 60 + s
        return float(time_str)
    except Exception:
        return float("nan")


def format_parsed_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:05.2f}"
    if minutes > 0:
        return f"{minutes}:{seconds:05.2f}"
    return f"{seconds:05.2f}"


def get_course_details(row, course_details_df):
    """Legacy single-row course lookup (kept for callers / debugging)."""
    match = course_details_df[
        (course_details_df["meet_id"] == row["meet_id"])
        & (course_details_df["running_event_id"] == row["running_event_id"])
        & (course_details_df["gender"] == row["gender"])
    ]
    if not match.empty:
        return match.iloc[0].to_dict()
    match = course_details_df[
        (course_details_df["meet_id"] == row["meet_id"])
        & (course_details_df["running_event_id"] == row["running_event_id"])
    ]
    if not match.empty:
        return match.iloc[0].to_dict()
    return {}


def get_event_dist(event_name):
    if pd.isna(event_name) or event_name is None:
        return None
    event_name = str(event_name).strip()
    if event_name.endswith("m"):
        try:
            return float(event_name.replace("m", "").strip())
        except ValueError:
            return None
    if event_name.endswith("k"):
        try:
            return float(event_name.replace("k", "").strip()) * 1000.0
        except ValueError:
            return None
    if event_name == "4 Mile":
        return 4 * 1609.34
    if event_name == "5 Mile":
        return 5 * 1609.34
    if event_name == "3 Mile":
        return 3 * 1609.34
    return None


def _event_dist_series(event_names: pd.Series) -> pd.Series:
    """Vectorized event-name → meters (NaN if unknown)."""
    s = event_names.astype("string")
    out = pd.Series(np.nan, index=event_names.index, dtype="float64")
    m_mask = s.str.endswith("m", na=False)
    k_mask = s.str.endswith("k", na=False) & ~m_mask
    out.loc[m_mask] = pd.to_numeric(s.loc[m_mask].str[:-1], errors="coerce")
    out.loc[k_mask] = pd.to_numeric(s.loc[k_mask].str[:-1], errors="coerce") * 1000.0
    mile_map = {"4 Mile": 4 * 1609.34, "5 Mile": 5 * 1609.34, "3 Mile": 3 * 1609.34}
    for label, meters in mile_map.items():
        out.loc[s == label] = meters
    return out


def _valid_time_mask(result_time: pd.Series) -> pd.Series:
    """Fast validity check; avoids per-row parse for the common well-formed case."""
    s = result_time.astype("string")
    ok = result_time.notna() & s.str.len().gt(0) & ~s.isin(["nan", "None", "<NA>"])
    # Reject obvious non-times; nrcd still validates remaining strings.
    return ok & ~s.str.contains(r"[A-Za-z]", na=True)


def _merge_course_details(results_df: pd.DataFrame, course_details_df: pd.DataFrame) -> pd.DataFrame:
    """Attach course_details fields (prefer gender-specific rows)."""
    if course_details_df is None or course_details_df.empty:
        return results_df.copy()

    detail_cols = [
        c
        for c in (
            "estimated_course_distance",
            "temperature",
            "dew_point",
            "elevation_gain",
            "elevation_loss",
            "barometric_pressure",
            "altitude",
            "weather_conditions",
        )
        if c in course_details_df.columns
    ]
    keys_g = ["meet_id", "running_event_id", "gender"]
    keys_e = ["meet_id", "running_event_id"]
    out = results_df

    if detail_cols and all(k in course_details_df.columns for k in keys_g) and all(
        k in out.columns for k in keys_g
    ):
        cd_g = course_details_df[keys_g + detail_cols].drop_duplicates(subset=keys_g, keep="first")
        out = out.merge(cd_g, on=keys_g, how="left", suffixes=("", "_drop"))
        out = out[[c for c in out.columns if not str(c).endswith("_drop")]]

    if detail_cols and all(k in course_details_df.columns for k in keys_e):
        cd_e = (
            course_details_df[keys_e + detail_cols]
            .drop_duplicates(subset=keys_e, keep="first")
            .rename(columns={c: f"{c}_fb" for c in detail_cols})
        )
        out = out.merge(cd_e, on=keys_e, how="left")
        for c in detail_cols:
            fb = f"{c}_fb"
            if c not in out.columns:
                out[c] = out[fb]
            else:
                out[c] = out[c].where(out[c].notna(), out[fb])
            out.drop(columns=[fb], inplace=True, errors="ignore")

    return out


def _prepare_nrcd_frame(results_df: pd.DataFrame, course_details_df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame ready for ``standardize_dataframe`` (both tiers)."""
    df = _merge_course_details(results_df, course_details_df)

    if "sport_name" not in df.columns:
        df["sport_name"] = "Cross Country"

    target_m = xc_target_distance_m("M")
    target_f = xc_target_distance_m("F")
    targets = np.where(
        df["gender"].to_numpy() == "M",
        target_m,
        np.where(df["gender"].to_numpy() == "F", target_f, np.nan),
    )
    targets = pd.Series(targets, index=df.index, dtype="float64")

    if "event_name" in df.columns:
        reported = _event_dist_series(df["event_name"]).fillna(targets)
    else:
        reported = targets
    df["reported_distance_m"] = reported
    df["target_distance_m"] = targets

    if "estimated_course_distance" in df.columns:
        actual = pd.to_numeric(df["estimated_course_distance"], errors="coerce")
        lo = 0.5 * df["reported_distance_m"]
        hi = 1.5 * df["reported_distance_m"]
        ok = actual.notna() & (actual >= lo) & (actual <= hi)
        df["actual_distance_m"] = actual.where(ok, df["reported_distance_m"])
    else:
        df["actual_distance_m"] = df["reported_distance_m"]

    df["grade_input"] = "feet"
    df["course_distance_m"] = df["actual_distance_m"]

    # Vectorized meet altitude (ft): result/meet altitude, else course_details altitude.
    elev = None
    if "altitude" in df.columns:
        elev = pd.to_numeric(df["altitude"], errors="coerce")
    if "elevation" in df.columns:
        elev2 = pd.to_numeric(df["elevation"], errors="coerce")
        elev = elev2 if elev is None else elev.fillna(elev2)
    df["meet_elevation"] = elev

    return df


def _split_dataframe(df: pd.DataFrame, n_chunks: int) -> list[pd.DataFrame]:
    if n_chunks <= 1 or len(df) == 0:
        return [df]
    size = math.ceil(len(df) / n_chunks)
    return [df.iloc[i : i + size].copy() for i in range(0, len(df), size)]


def _worker_count(n_rows: int) -> int:
    if n_rows < _CHUNK_ROWS:
        return 1
    cpu = os.cpu_count() or 4
    return max(1, min(_MAX_WORKERS, cpu, math.ceil(n_rows / _CHUNK_ROWS)))


def _run_chunk_pool(chunks: list[pd.DataFrame]) -> list[list[float]]:
    """Map ``standardize_chunk`` over DataFrame chunks; fall back to serial."""
    global _POOL_WARNED
    if len(chunks) <= 1:
        from _nrcd_std_worker import standardize_chunk

        return [standardize_chunk(chunks[0])] if chunks else []

    from _nrcd_std_worker import standardize_chunk

    n_workers = min(_MAX_WORKERS, len(chunks), os.cpu_count() or 4)
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_pool_initializer,
            initargs=(_SCRIPTS_DIR,),
        ) as pool:
            return list(pool.map(standardize_chunk, chunks, chunksize=1))
    except Exception as exc:
        if not _POOL_WARNED:
            warnings.warn(
                f"Parallel nrcd standardization unavailable ({type(exc).__name__}: {exc}); "
                "using serial standardize_dataframe. Cache still applies on subsequent runs.",
                RuntimeWarning,
                stacklevel=2,
            )
            _POOL_WARNED = True
        return [standardize_chunk(c) for c in chunks]


def _parallel_standardize(slim: pd.DataFrame) -> np.ndarray:
    """Run ``standardize_dataframe`` over chunks in parallel processes."""
    n_workers = _worker_count(len(slim))
    if n_workers == 1:
        return standardize_dataframe(slim, std_col="_std_sec")["_std_sec"].to_numpy()

    chunks = _split_dataframe(slim.reset_index(drop=True), n_workers)
    parts = _run_chunk_pool(chunks)
    flat: list[float] = [x for part in parts for x in part]
    return np.asarray(flat, dtype="float64")


def _slim_for_nrcd(prepared: pd.DataFrame, *, tier: str) -> pd.DataFrame:
    cols = [c for c in _STD_COLS if c in prepared.columns]
    slim = prepared.loc[:, cols].copy()
    if tier == "converted":
        for c in _ENV_COLS:
            if c in slim.columns:
                slim[c] = np.nan
    return slim


def _eligibility_mask(prepared: pd.DataFrame) -> pd.Series:
    raw_ok = (
        _valid_time_mask(prepared["result_time"])
        if "result_time" in prepared.columns
        else pd.Series(False, index=prepared.index)
    )
    gender_ok = prepared["gender"].isin(["M", "F"])
    dist_ok = prepared["reported_distance_m"].notna() & prepared["target_distance_m"].notna()
    return raw_ok & gender_ok & dist_ok


def _standardize_prepared(prepared: pd.DataFrame, *, tier: str) -> pd.Series:
    """Run parallel nrcd batch standardization; align to ``prepared.index``."""
    mask = _eligibility_mask(prepared)
    out = pd.Series(np.nan, index=prepared.index, dtype="float64")
    if not mask.any():
        return out

    slim = _slim_for_nrcd(prepared.loc[mask], tier=tier)
    values = _parallel_standardize(slim)
    out.loc[mask] = values
    return out


def _standardize_prepared_both(prepared: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Standardize Converted + Standardized tiers in one process-pool wave."""
    mask = _eligibility_mask(prepared)
    empty = pd.Series(np.nan, index=prepared.index, dtype="float64")
    if not mask.any():
        return empty, empty.copy()

    slim_conv = _slim_for_nrcd(prepared.loc[mask], tier="converted")
    slim_std = _slim_for_nrcd(prepared.loc[mask], tier="standardized")
    n_workers = _worker_count(len(slim_std))
    if n_workers == 1:
        conv_vals = standardize_dataframe(slim_conv, std_col="_std_sec")["_std_sec"].to_numpy()
        std_vals = standardize_dataframe(slim_std, std_col="_std_sec")["_std_sec"].to_numpy()
    else:
        # One pool for both tiers: up to 2 * n_workers chunks, capped by CPU.
        n_chunks = max(1, min(n_workers, math.ceil(len(slim_std) / _CHUNK_ROWS)))
        chunks_conv = _split_dataframe(slim_conv.reset_index(drop=True), n_chunks)
        chunks_std = _split_dataframe(slim_std.reset_index(drop=True), n_chunks)
        n_conv = len(chunks_conv)
        parts = _run_chunk_pool(chunks_conv + chunks_std)
        conv_vals = np.asarray([x for part in parts[:n_conv] for x in part], dtype="float64")
        std_vals = np.asarray([x for part in parts[n_conv:] for x in part], dtype="float64")

    out_conv = empty.copy()
    out_std = empty.copy()
    out_conv.loc[mask] = conv_vals
    out_std.loc[mask] = std_vals
    return out_conv, out_std


def _cache_dir() -> Path:
    d = _repo_root() / "output" / ".cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _data_fingerprint() -> str:
    data_dir = Path(get_data_dir())
    git_dir = data_dir / ".git"
    commit = "nogit"
    if git_dir.exists():
        try:
            import subprocess

            commit = (
                subprocess.check_output(
                    ["git", "-C", str(data_dir), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()[:12]
            )
        except Exception:
            commit = "unknown"
    # Include key CSV mtimes so non-git NRCD_DATA_DIR still invalidates.
    mtimes = []
    for name in ("result.csv", "meet.csv", "course_details.csv"):
        p = data_dir / name
        if p.exists():
            mtimes.append(str(int(p.stat().st_mtime)))
    blob = "|".join([commit, getattr(nrcd, "__version__", "?"), *mtimes])
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _tier_cache_path(tier: str, *, exclude_nationals: bool) -> Path:
    key = f"xc_{tier}_{'nn' if exclude_nationals else 'all'}_{_data_fingerprint()}.parquet"
    return _cache_dir() / key


def _write_tier_cache(df: pd.DataFrame, tier: str, *, exclude_nationals: bool) -> None:
    if os.environ.get(_CACHE_ENV):
        return
    path = _tier_cache_path(tier, exclude_nationals=exclude_nationals).with_suffix(".pkl")
    df.to_pickle(path)


def _read_tier_cache(tier: str, *, exclude_nationals: bool) -> pd.DataFrame | None:
    if os.environ.get(_CACHE_ENV):
        return None
    path = _tier_cache_path(tier, exclude_nationals=exclude_nationals)
    pkl = path.with_suffix(".pkl")
    for candidate in (pkl, path):
        if not candidate.exists():
            continue
        try:
            if candidate.suffix == ".pkl":
                return pd.read_pickle(candidate)
            return pd.read_parquet(candidate)
        except Exception:
            continue
    return None


def _standardize_df(results_df, course_details_df, tier):
    """Add ``standardized_to_target`` using parallel nrcd batch standardization."""
    results_df = results_df.copy()
    prepared = _prepare_nrcd_frame(results_df, course_details_df)
    results_df["standardized_to_target"] = _standardize_prepared(prepared, tier=tier).to_numpy()
    return results_df


def _ensure_join_columns(results_df, athlete_df=None, running_event_df=None, meet_df=None):
    results_df = results_df.copy()
    if "gender" not in results_df.columns and athlete_df is not None:
        results_df = results_df.merge(
            athlete_df[["athlete_id", "gender"]], on="athlete_id", how="left"
        )
    if "event_name" not in results_df.columns and running_event_df is not None:
        results_df = results_df.merge(
            running_event_df[["running_event_id", "event_name"]],
            on="running_event_id",
            how="left",
        )
    if "start_date" not in results_df.columns and meet_df is not None:
        results_df = results_df.merge(
            meet_df[["meet_id", "start_date"]], on="meet_id", how="left"
        )
    if "altitude" not in results_df.columns and meet_df is not None and "altitude" in meet_df.columns:
        results_df = results_df.merge(
            meet_df[["meet_id", "altitude"]], on="meet_id", how="left"
        )
    return results_df


def standardize_both_tiers(results_df=None, course_details_df=None,
                           athlete_df=None, running_event_df=None, meet_df=None,
                           *, exclude_nationals: bool = True, use_cache: bool = True):
    """Prepare once; run Converted Only and Standardized in parallel batches.

    Returns ``(df_converted, df_standardized)`` each with ``standardized_to_target``.
    """
    cached_conv = cached_std = None
    if use_cache and results_df is None:
        cached_conv = _read_tier_cache("converted", exclude_nationals=exclude_nationals)
        cached_std = _read_tier_cache("standardized", exclude_nationals=exclude_nationals)
        if cached_conv is not None and cached_std is not None:
            return cached_conv, cached_std

    tables = None
    if results_df is None:
        tables = load_tables(get_data_dir())
        results_df = load_cross_country_results(
            exclude_nationals=exclude_nationals, tables=tables
        )
    if course_details_df is None:
        tables = tables or load_tables(get_data_dir())
        course_details_df = tables["course_details"]
    results_df = _ensure_join_columns(
        results_df, athlete_df=athlete_df, running_event_df=running_event_df, meet_df=meet_df
    )
    prepared = _prepare_nrcd_frame(results_df, course_details_df)

    df_conv = results_df.copy()
    df_std = results_df.copy()
    conv_s, std_s = _standardize_prepared_both(prepared)
    df_conv["standardized_to_target"] = conv_s.to_numpy()
    df_std["standardized_to_target"] = std_s.to_numpy()

    if use_cache and tables is not None:
        _write_tier_cache(df_conv, "converted", exclude_nationals=exclude_nationals)
        _write_tier_cache(df_std, "standardized", exclude_nationals=exclude_nationals)
    return df_conv, df_std


def standardize_and_convert_to_6k_8k(results_df=None, course_details_df=None,
                                     athlete_df=None, running_event_df=None,
                                     meet_df=None, tier="standardized",
                                     adjust_terrain=None, adjust_weather=None):
    """Add a ``standardized_to_target`` column using nrcd batch standardization."""
    if adjust_terrain is False or adjust_weather is False:
        tier = "converted"

    tables = None
    if results_df is None:
        tables = load_tables(get_data_dir())
        results_df = load_cross_country_results(tables=tables)
    if course_details_df is None:
        tables = tables or load_tables(get_data_dir())
        course_details_df = tables["course_details"]

    results_df = _ensure_join_columns(
        results_df, athlete_df=athlete_df, running_event_df=running_event_df, meet_df=meet_df
    )
    return _standardize_df(results_df, course_details_df, tier)


@lru_cache(maxsize=1)
def _cached_both_exclude_nationals() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process-local memo: building one tier fills both (disk + RAM)."""
    return standardize_both_tiers(exclude_nationals=True, use_cache=True)


def standardize_convert_exclude_nationals_df(results_df=None, course_details_df=None,
                                             meet_df=None, athlete_df=None,
                                             running_event_df=None):
    """Cross country, nationals excluded, fully standardized (weather + elevation)."""
    if results_df is None and course_details_df is None:
        return _cached_both_exclude_nationals()[1].copy()
    tables = load_tables(get_data_dir()) if course_details_df is None else None
    df = results_df if results_df is not None else load_cross_country_results(
        exclude_nationals=True, tables=tables or load_tables(get_data_dir())
    )
    cd = course_details_df if course_details_df is not None else (tables or load_tables(get_data_dir()))["course_details"]
    return _standardize_df(df, cd, tier="standardized")


def convert_exclude_nationals(results_df=None, meet_df=None, athlete_df=None,
                              running_event_df=None):
    """Cross country, nationals excluded, distance-converted only."""
    if results_df is None:
        return _cached_both_exclude_nationals()[0].copy()
    tables = load_tables(get_data_dir())
    df = results_df
    return _standardize_df(df, tables["course_details"], tier="converted")


def standardize_one_xc(time, *, gender, tier="standardized", **kwargs):
    """Thin wrapper around ``nrcd.standardize.standardize_xc`` for single results."""
    flags = {}
    if tier == "converted":
        flags = dict(
            apply_weather=False,
            apply_elevation_grade=False,
            apply_meet_altitude_correction=False,
        )
    if "target_distance_m" not in kwargs and "target_distance" not in kwargs:
        kwargs["target_distance_m"] = xc_target_distance_m(gender)
    return standardize_xc(time, gender=gender, **flags, **kwargs)
