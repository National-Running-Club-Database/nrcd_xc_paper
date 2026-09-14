# Underexplored mechanisms — summary

## Roster depth (conditional on max races)

- maxrace_plus_evenness / z_max_race_count: OR=2.48 [1.54, 3.98], p=0.0001703
- maxrace_plus_evenness / z_shannon_evenness: OR=1.91 [1.13, 3.20], p=0.01482

## Peer co-start → improvement

- peers_plus_controls / z_mean_peers: coef=-0.365, p=7.715e-05
- peers_only_plus_starting / z_mean_peers: coef=-0.370, p=6.479e-05
- solo_share / pct_solo_meets: coef=1.052, p=0.5101

## Quantile race-volume (num_races coef)

- Men τ=0.25: β=11.83 [7.16, 16.51], p=7.402e-07
- Men τ=0.5: β=7.68 [4.01, 11.35], p=4.128e-05
- Men τ=0.75: β=6.55 [3.06, 10.05], p=0.000239
- Women τ=0.25: β=7.89 [1.98, 13.81], p=0.00893
- Women τ=0.5: β=5.58 [0.14, 11.02], p=0.04437
- Women τ=0.75: β=3.26 [-1.61, 8.12], p=0.1892

## Starting-ability CATE (4+ vs 2), BH

- Men Q1_fastest: Δ=25.1s (d=0.28), p=0.0001752, BH sig=True
- Men Q2: Δ=9.3s (d=0.10), p=0.1938, BH sig=False
- Men Q3: Δ=28.9s (d=0.29), p=0.0001223, BH sig=True
- Men Q4_slowest: Δ=39.2s (d=0.19), p=0.0003016, BH sig=True
- Women Q1_fastest: Δ=14.7s (d=0.21), p=0.08058, BH sig=False
- Women Q2: Δ=5.7s (d=0.06), p=0.5438, BH sig=False
- Women Q3: Δ=32.6s (d=0.36), p=0.004942, BH sig=True
- Women Q4_slowest: Δ=41.3s (d=0.26), p=0.01998, BH sig=True
