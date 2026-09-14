# Mathematical contributions

Closed-form identities and one derived program index, estimated on the
comprehensive-era NRCD XC sample. Reproducible via
`python scripts/mathematical_contributions.py`.

## 1. Weather-path bias identity (exact)

Let $t^{\mathrm{conv}}_{ir}$ and $t^{\mathrm{std}}_{ir}$ be Converted Only and
Standardized times for athlete $i$ at race $r$, and define the environment
residual

$$a_{ir} := t^{\mathrm{conv}}_{ir} - t^{\mathrm{std}}_{ir}.$$

First-to-last improvement (positive = faster) is $\Delta = t_1 - t_L$. Then

$$
\Delta^{\mathrm{conv}} - \Delta^{\mathrm{std}}
= (t_1^{c}-t_L^{c}) - (t_1^{s}-t_L^{s})
= (t_1^{c}-t_1^{s}) - (t_L^{c}-t_L^{s})
= a_1 - a_L.
$$

So **weather inflation of apparent improvement equals the first-to-last
drift of the environment residual** — an algebraic identity. Empirically,
max $|(\Delta_c-\Delta_s)-(a_1-a_L)| = 0.00e+00$ s (machine noise).
Mean inflation: men $20.1$ s, women
$14.2$ s, matching $E[a_1-a_L]$.

## 2. Reliability bound and Signal Extraction Ratio

If the improvement outcome has split-half reliability $\rho_{yy'}$, classical
attenuation implies any predictor of the *observed* outcome satisfies

$$R^2 \le \rho_{yy'}.$$

Define the **Signal Extraction Ratio**

$$\mathrm{SER} := \frac{R^2_{\mathrm{heldout}}}{\rho_{yy'}}.$$

SER is the fraction of *reliable* outcome variance captured.
Men: $R^2=0.043$, $\rho=0.228$,
$\mathrm{SER}=0.188$ (clipped $0.188$).
Women: $R^2=-0.029$, $\rho=0.279$,
$\mathrm{SER}=-0.104$ (negative $R^2$ ⇒ no reliable signal extracted).

For two-race seasons, an endpoint-noise floor is

$$\phi := \frac{2\hat\sigma_\varepsilon^2}{\widehat{\mathrm{Var}}(\Delta)},$$

with $\hat\sigma_\varepsilon^2$ from within-season trend residuals.
Men $\phi=0.47$;
women $\phi=0.65$
(large $\phi$ means two-race $\Delta$ is heavily measurement noise).

Under a shared-mean + iid race-noise null,
$\kappa_{\mathrm{null}}=\mathrm{Corr}(t_1,\Delta)=\sqrt{(1-\mathrm{ICC})/2}$.
Observed $\kappa$ is 0.58 (men) /
0.38 (women) vs null
0.34 / 0.30
— excess over null is the slower-starters-improve-more association beyond pure RTM.

## 3. Effective Racing Opportunity (ERO)

For a team roster with start counts $\{n_j\}$ and Shannon entropy
$H=-\sum p_j\log p_j$, $p_j=n_j/\sum n_k$, define

$$
\mathrm{ERO} := e^{H}\,\bar n,
\qquad e^{H}=\text{perplexity (effective number of equally racing athletes)}.
$$

ERO multiplies *effective depth* by *mean intensity* in one scalar.
Out-of-year (train 2023–24 → test 2025) top-15 AUC:
ERO $0.837$,
depth (#≥3 starts) $0.782$,
max race count $0.742$.

Pooled logistic OR per 1 SD:

- ERO: OR=2.70 [2.12, 3.44], p=6.3e-16
- n_athletes_ge3: OR=2.40 [1.91, 3.02], p=9.5e-14
- max_race_count: OR=2.51 [1.89, 3.35], p=2.5e-10
- effective_n_athletes: OR=2.55 [2.02, 3.22], p=5.4e-15

## Interpretation for the paper

- Identity (1) *explains* weather inflation without a new regression.
- SER (2) reframes the individual null as near-zero extraction of a weak
  reliable signal, with an explicit ceiling $R^2\le\rho$.
- ERO (3) is a closed-form team score combining entropy-depth and intensity;
  it matches or beats max-race-count alone for out-of-year placement.
