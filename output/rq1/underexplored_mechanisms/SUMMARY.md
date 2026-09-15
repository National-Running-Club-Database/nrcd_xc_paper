# Underexplored mechanisms — summary

## Roster depth (conditional on max races)

- maxrace_plus_evenness / z_max_race_count: OR=2.44 [1.51, 3.92], p=0.0002417
- maxrace_plus_evenness / z_shannon_evenness: OR=1.84 [1.10, 3.09], p=0.02035

## Peer co-start → improvement

- peers_plus_controls / z_mean_peers: coef=-0.387, p=2.2e-05
- peers_only_plus_starting / z_mean_peers: coef=-0.391, p=1.867e-05
- solo_share / pct_solo_meets: coef=1.214, p=0.4388

## Quantile race-volume (num_races coef)

- Men τ=0.25: β=12.60 [7.96, 17.25], p=1.103e-07
- Men τ=0.5: β=7.70 [4.10, 11.31], p=2.884e-05
- Men τ=0.75: β=5.60 [2.15, 9.06], p=0.00147
- Women τ=0.25: β=8.01 [2.06, 13.97], p=0.008392
- Women τ=0.5: β=5.92 [0.51, 11.33], p=0.03202
- Women τ=0.75: β=4.43 [-0.34, 9.20], p=0.0687

## Starting-ability CATE (4+ vs 2), BH

- Men Q1_fastest: Δ=23.4s (d=0.29), p=0.0011, BH sig=True
- Men Q2: Δ=12.8s (d=0.13), p=0.08769, BH sig=False
- Men Q3: Δ=26.9s (d=0.27), p=0.0001974, BH sig=True
- Men Q4_slowest: Δ=40.6s (d=0.19), p=0.0002095, BH sig=True
- Women Q1_fastest: Δ=14.6s (d=0.21), p=0.09775, BH sig=False
- Women Q2: Δ=8.7s (d=0.10), p=0.3927, BH sig=False
- Women Q3: Δ=30.4s (d=0.34), p=0.006936, BH sig=True
- Women Q4_slowest: Δ=34.3s (d=0.22), p=0.04192, BH sig=False
