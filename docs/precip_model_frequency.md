# How the model's exceedance frequency is computed

This documents the "Model pred (P(Y ≥ t mm))" column of
`spatial_wetday_freq_thresholds.png` and the model category probabilities in
`precip_category_spatial.png` / `precip_category_distribution.png`
(code: `eval_precip_figures._wetfreq_at_threshold` / `_precip_category_eval`).

## Step 1 — the model's daily output is a distribution, not a value

For each day `d`, the network predicts three parameters at target point `p`:

- `ρ(p,d)` — probability the day is wet,
- `α(p,d), β(p,d)` — shape and **rate** of a Gamma for the wet-day amount.

So its full law for the daily accumulation is `Y = 0` with probability `1−ρ`,
and `Y ~ Gamma(α, rate=β)` with probability `ρ` (mixed Bernoulli-Gamma).

## Step 2 — analytic daily exceedance probability

For each day separately, the probability of exceeding the threshold is computed
in closed form:

```
P(p,d)(Y ≥ t) = ρ(p,d) · [1 − F_Gamma(t; shape=α(p,d), scale=1/β(p,d))]
```

(`F_Gamma` = `scipy.stats.gamma.cdf`; the `ρ` factor handles the dry atom — a
dry day can never exceed `t>0`.) In code:

```python
pexc = rho * (1 - scipy_stats.gamma.cdf(t, a=alpha, scale=1.0 / beta))
```

## Step 3 — average over days

```
freq_model(p) = (1/D) · Σ_d P(p,d)(Y ≥ t)
```

That mean predicted probability **is** the model's expected fraction of
exceedance days — by linearity of expectation, `E[#days with Y≥t]/D` under the
model's own law. That is the number shown in the map.

## Properties

- **No sampling, no thresholding, no classification.** It is the exact
  expectation under the predictive distribution — zero Monte-Carlo noise
  (verified against a 200k-sample MC estimate to ~4 decimals), and no
  `ρ≥0.5`-style cutoff anywhere.
- **Directly comparable to the observed column by construction**: obs shows the
  *realized* fraction of days with `Y≥t` (MeteoSwiss RhiresD); the model shows
  its *expected* fraction. For a perfectly calibrated model the two differ only
  by observation-sampling noise. The ERA5-Land column, by contrast, is a
  realized 0/1 count of the bilinearly interpolated field — same nature as obs.
- **Which parameters enter on each day**: in the CV regime (2020–2023), day
  `d`'s `(ρ,α,β)` come from the one fold model that held that day out; in the
  2024 holdout regime they are the fold-averaged parameters (caveat: averaging
  parameters is not the same as averaging the per-fold probabilities).
- Same formula and `scale = 1/β` convention as the pooled metrics
  (`eval_precip._bg_exceedance_prob`) and the PIT computation, so the figures,
  the metrics JSON, and the calibration diagnostics agree by construction.
- Because each day contributes a *probability* rather than a 0/1 decision, the
  model's frequency map is smooth and can never be exactly zero (`ρ` is clamped
  ≥ 1e-5) — which is why the model column never shows the true-zero grey that
  the observed column can show at high thresholds.

Category probabilities are the same construction on disjoint bins:
`P(e_lo ≤ Y < e_hi) = P(Y ≥ e_lo) − P(Y ≥ e_hi)`, with `P(Y ≥ 0) = 1` so the
first bin absorbs the dry atom; the per-point category probabilities sum to
exactly 1 by construction.
