# Why we do not add historical-era experience covariates

Reviewers often ask for pre-2023 racing experience. We **deliberately
omit** historical race counts as predictors of comprehensive-era
improvement because the historical export is a **fragmentary** record,
not a complete season log.

## Empirical fragmentation (this export)

- Historical XC results: **52,129**; comprehensive: **23,360**.
- Athletes in both eras: only **1,356** of 7,056 comprehensive athletes (~19.2%).
- Among historical athlete-years, median races/season = **2.0**, and **48.3%** have exactly one recorded race.
- Historical course-details temperature coverage on joined rows ~**52.9%** (vs ~97.7% comprehensive).

## Why that breaks an experience covariate

A recorded historical race count of 2 may mean the athlete truly raced
twice -- or that half (or more) of their meets were never entered.
Using that count as experience would systematically **undercount** true
volume and invent a spurious low-experience class. Metadata needed to
standardize those races is also mostly missing.

Therefore experience / grade-year / pre-2023 load remain Limitations,
not silent omissions we paper over with a bad proxy.

Artifact: `historical_era_fragmentation.json`.
