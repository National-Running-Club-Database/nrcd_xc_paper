"""Picklable worker for parallel nrcd.standardize_dataframe chunks."""

from __future__ import annotations

from typing import Any


def standardize_chunk(chunk: Any) -> list[float]:
    """Standardize one DataFrame chunk; returns std seconds as a list."""
    from nrcd.standardize import standardize_dataframe

    return standardize_dataframe(chunk, std_col="_std_sec")["_std_sec"].tolist()
