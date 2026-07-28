# Statistical method

## Benchmark

For every frozen split, a lab-only model is fitted to its `N` in-lab labels and
evaluated on the held-out formulations. Each literature-only model is fitted
once per endpoint to all eligible literature rows and evaluated on the same
held-out formulations. Regression reports RMSE and R²; UL-94 reports ROC AUC.

Backbone selection is performed from this benchmark. One lab backbone and one
literature backbone are then fixed per regression endpoint across every `N`.
The final comparison uses exactly those same backbones, rather than the
post-hoc best method in each table cell.

## Commensurability-weighted pooling expert

Let `s_i` be the literature HistGB prediction at a labeled lab formulation and
let `sigma_s` be the literature HistGB five-fold OOF RMSE. We compare

\[
  (y_i-s_i)/\sigma_s \sim N(0,1)
  \quad\text{versus}\quad
  (y_i-s_i)/\sigma_s \sim N(0,\kappa^2).
\]

With equal prior model probabilities, the log Bayes factor and posterior
commensurability power are

\[
  \log BF = N\log\kappa-
  \frac12(1-\kappa^{-2})\sum_i[(y_i-s_i)/\sigma_s]^2,
  \qquad \alpha=\operatorname{logit}^{-1}(\log BF).
\]

A HistGB regressor is fitted to the union of literature and lab rows, assigning
weight `alpha` to every literature row and weight one to every lab row. `alpha`
therefore follows evidence of domain commensurability, not `N` directly.

## Bayesian positive-affine expert

Let `z_lit` be the fixed literature-backbone score and `z_lab` a strictly
cross-fitted lab-backbone score, both expressed on a common response scale. We
fit

\[
 \tilde y_i=\beta_0+bz_{lit,i}+c(z_{lab,i}-z_{lit,i})+\epsilon_i,
\]

subject to `b>0` and `0<=c<=b`. Equivalently, the literature coefficient is
`b-c>=0` and the lab coefficient is `c>=0`: calibration may re-scale and blend
the two sources but cannot reverse either expert.

The proper conjugate prior is

\[
 \theta\mid\sigma^2\sim N(m_0,\sigma^2K_0^{-1}),\qquad
 \sigma^2\sim IG(a_0,b_0),
\]

with `theta=(beta0,b,c)`, `m0=(0,1,0)`,
`K0=diag(0.25,4,4)`, and `(a0,b0)=(2,1)`. The reported estimator is the
posterior mean conditioned on the coefficient constraints, evaluated by a
fixed scrambled Sobol design. The initial paper setting uses `2^16` points;
the public implementation deterministically enlarges the same design only if
fewer than the declared minimum constrained draws survive. Reported fits met
the primary rule.

## Bayesian synthesis

For the pooling and positive-affine experts, outer-training predictions are
constructed by leave-one-out refitting; the affine expert itself performs the
required inner leave-one-out lab refits. On the fixed common scale,

\[
 y_i\mid w,\sigma^2\sim N((1-w)p_i+wa_i,\sigma^2),\quad
 w\sim U(0,1),\quad \sigma^2\sim IG(2,1).
\]

After integrating out `sigma²`, the posterior mean `E[w|data]` is computed by
deterministic Gauss-Legendre quadrature. A 256-point result is audited against
512 points. The outer-test prediction is

\[
 \hat y=(1-E[w\mid data])\hat y_{pool}+E[w\mid data]\hat y_{affine}.
\]

No held-out outcome is available to any fit, weight, or method choice.

## What is selected versus numerically fixed

The pooling/backend families and endpoint backbones are selected in the
development benchmark and then frozen across `N`. The method-level sensitivity
study concerns `kappa` and the affine prior precision `K0`. Sobol sample count,
minimum acceptance, inverse-gamma constants, and quadrature point counts are
fixed numerical/proper-prior specifications; they are recorded for exact
reproduction and are not re-selected per endpoint or shot count.
