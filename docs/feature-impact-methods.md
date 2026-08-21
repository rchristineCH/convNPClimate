# Feature-Impact Verification Methods: A Comprehensive Comparison

This document surveys the main families of methods used to verify and quantify
the impact of input features on a model's predictions, with references and a
detailed cross-method comparison. It is written with this project's context in
mind — a spatial **ConvCNP** that ingests gridded ERA5-Land channels plus
per-target topographic/seasonal features and outputs a (mean, std) predictive —
but the comparison is general.

> **TL;DR for this project.** The ablation study already in the report
> (leave-one-feature-out retraining) is the *most faithful* answer to "does this
> feature matter to the trained system," but it is expensive and answers a
> different question than post-hoc attributions. Permutation importance is the
> cheap global complement; SHAP gives consistent local + global attributions
> with the soundest theory; LIME is the quickest local sanity check but the
> least stable. Use them in combination, not as substitutes — they measure
> different things (see §1, §3). A second tier — variance-based sensitivity
> analysis, controlled selection (knockoffs), and concept/counterfactual
> explanations — is in §5; these matter especially for correlated climate inputs.

---

## 1. What "feature impact" actually means

Before comparing methods it is essential to separate three questions that are
routinely conflated. A method that answers one does **not** answer the others.

| Question | What it asks | Canonical method family |
|---|---|---|
| **Model reliance** | How much does *this trained model* depend on a feature? | Permutation importance, SHAP, LIME |
| **Feature importance to the task** | How much predictive signal does a feature carry, for *any* well-trained model? | Retraining-based ablation / hold-out |
| **Local explanation** | Why did the model make *this one* prediction? | LIME, SHAP, Integrated Gradients, saliency |

Two further axes cut across all methods:

- **Global vs. local** — a single number per feature for the whole model
  (global) vs. an attribution per individual prediction (local).
- **Model-agnostic vs. model-specific** — treats the model as a black box
  (only needs `predict`) vs. exploits internals (gradients, tree structure).

The literature on *why these distinctions matter* — and why a single "importance"
number is often ill-posed — is best entered through Molnar's *Interpretable
Machine Learning* book [Molnar 2022] and the model-reliance formalism of
[Fisher, Rudin & Dominici 2019].

---

## 2. The methods

### 2.1 Ablation / hold-out retraining (leave-one-feature-out, LOCO)

**Idea.** Remove a feature (or group), **retrain** the model from scratch, and
measure the change in held-out performance. "Leave-one-covariate-out" (LOCO) is
the formal name; grouped ablations remove a whole subsystem (e.g. the elevation
MLP, as in the report's ablation study).

**Answers:** feature importance *to the task* (model reliance is re-optimised
each time). This is the gold standard for "is this feature necessary," because
the model is given the chance to compensate via other features.

**Cost:** one full training run per feature/group — the most expensive method.

**Key property:** it is the only family that accounts for the model *adapting*
to the feature's absence. All other methods below fix the trained weights.

**References:** [Lei et al. 2018] (LOCO inference, with confidence intervals);
[Molnar 2022, ch. on feature importance]; ablation studies are standard practice
in deep learning (e.g. [Meyes et al. 2019] survey of ablation in neural nets).

### 2.2 Permutation feature importance (PFI)

**Idea.** Shuffle one feature's values across the dataset (breaking its
association with the target) and measure the drop in performance on the *fixed,
already-trained* model. No retraining.

**Answers:** model reliance, globally.

**Cost:** cheap — one (or a few, for averaging) forward pass(es) per feature.

**Major caveat — correlated features.** Naive permutation evaluates the model on
**off-manifold / extrapolated** inputs (it breaks realistic feature
correlations), which can both under- and over-state importance. Conditional and
grouped variants fix this. This is the single most important pitfall for spatial
/ geophysical data where neighbouring channels (e.g. temperature, geopotential,
elevation) are strongly correlated.

**References:** original idea [Breiman 2001] (random forests); model-agnostic
formalisation as *model reliance* [Fisher, Rudin & Dominici 2019]; correlation
pitfalls and the extrapolation problem [Hooker, Mentch & Zhou 2021]; conditional
variable importance [Strobl et al. 2008]; the global SHAP-based alternative SAGE
[Covert, Lundberg & Lee 2020].

### 2.3 LIME (Local Interpretable Model-agnostic Explanations)

**Idea.** To explain one prediction, sample perturbed points *around* it, weight
them by proximity, and fit a simple interpretable surrogate (usually sparse
linear regression). The surrogate's coefficients are the local feature
attributions.

**Answers:** local explanation; model-agnostic (black box).

**Cost:** moderate per explanation (hundreds–thousands of model queries each);
no global summary without aggregating many local explanations.

**Caveats:** results depend heavily on the perturbation/neighbourhood kernel and
the choice of "interpretable" feature representation; **instability** — repeated
runs can give different explanations; can be adversarially manipulated.

**References:** [Ribeiro, Singh & Guestrin 2016] (original KDD paper);
instability analyses [Alvarez-Melis & Jaakkola 2018]; adversarial fragility of
LIME/SHAP [Slack et al. 2020].

### 2.4 SHAP (SHapley Additive exPlanations)

**Idea.** Attribute a prediction to features using **Shapley values** from
cooperative game theory: the unique attribution satisfying *local accuracy
(efficiency), missingness, and consistency*. Each feature's value is its average
marginal contribution over all feature coalitions.

**Variants (pick by model access):**
- **KernelSHAP** — model-agnostic, black-box; approximates Shapley values via a
  weighted-linear-regression sampling scheme (a principled, unified
  generalisation of LIME).
- **TreeSHAP** — exact and fast for tree ensembles.
- **DeepSHAP / GradientSHAP / DeepLIFT** — fast approximations for neural nets
  using backprop (most relevant to a ConvCNP).
- **SAGE** — Shapley values for *global* feature importance (extends SHAP from
  local to global, with an information-theoretic objective).

**Answers:** local explanation natively; global via aggregation (or SAGE);
agnostic or model-specific depending on variant.

**Cost:** exact Shapley is exponential in #features; approximations are
moderate-to-high. Background/reference distribution choice matters a lot and,
like PFI, the standard interventional sampling can probe off-manifold inputs.

**References:** unifying framework [Lundberg & Lee 2017]; TreeSHAP and global
tree attributions [Lundberg et al. 2020, *Nature Machine Intelligence*]; DeepLIFT
[Shrikumar, Greenside & Kundaje 2017]; SAGE global importance [Covert, Lundberg
& Lee 2020]; correlation/off-manifold concerns shared with PFI [Hooker, Mentch &
Zhou 2021]; adversarial fragility [Slack et al. 2020]; the "true to the model vs.
true to the data" distinction [Chen et al. 2020].

### 2.5 Gradient / backprop attribution (for differentiable models)

**Idea.** Use gradients of the output w.r.t. inputs to attribute importance —
directly applicable to the ConvCNP since it is differentiable end-to-end.

- **Vanilla saliency** — magnitude of the input gradient [Simonyan et al. 2013].
- **Integrated Gradients** — integrate gradients along a path from a baseline to
  the input; satisfies sensitivity + implementation-invariance axioms
  [Sundararajan, Taly & Yan 2017].
- **DeepLIFT** — propagate contributions relative to a reference activation
  [Shrikumar et al. 2017].
- **Grad-CAM** — class/output-discriminative spatial heatmaps from conv feature
  maps; well suited to visualising *where* on the grid a CNN attends
  [Selvaraju et al. 2017].
- **SmoothGrad** — average gradients over noised inputs to denoise saliency
  [Smilkov et al. 2017].

**Answers:** local explanation; model-specific (needs gradients).

**Cost:** very cheap (one to a few backward passes).

**Major caveat:** some saliency methods fail basic **sanity checks** — their
output is insensitive to randomising model weights or labels, i.e. they can look
plausible while explaining nothing. Always run the [Adebayo et al. 2018] tests.

### 2.6 Occlusion / perturbation maps

**Idea.** Mask or perturb a region/channel of the input and measure the change
in output; sweep across the input to build an importance map. Conceptually a
spatial, model-agnostic cousin of permutation importance.

**Answers:** local-to-regional; model-agnostic.

**Cost:** moderate–high (one forward pass per occluded region).

**References:** [Zeiler & Fergus 2014] (occlusion sensitivity for CNNs);
meaningful-perturbation masks [Fong & Vedaldi 2017].

### 2.7 Dependence / effect plots (PDP, ICE, ALE)

**Idea.** Visualise *how* the prediction changes as a feature varies (the shape
of the effect, not just a magnitude).

- **Partial Dependence Plot (PDP)** — marginal effect of a feature averaged over
  the others [Friedman 2001]. Misleading under correlation.
- **Individual Conditional Expectation (ICE)** — one PDP curve per instance,
  exposing heterogeneity [Goldstein et al. 2015].
- **Accumulated Local Effects (ALE)** — correlation-robust alternative to PDP;
  the recommended default when features are correlated [Apley & Zhu 2020].

**Answers:** global effect *shape*; model-agnostic. Complementary to importance
methods — they explain direction/shape, not a single ranking.

---

## 3. Side-by-side comparison

| Method | Global / Local | Model access | Retraining? | Cost | Faithfulness¹ | Handles correlated features | Gives effect *direction/shape* | Theoretical guarantees | Best for |
|---|---|---|---|---|---|---|---|---|---|
| **Ablation / LOCO** (§2.1) | Global (per feature/group) | Black box | **Yes** (per feature) | **Very high** | **Highest** (model re-adapts) | Yes (model recompensates) | No (magnitude only) | Statistical inference variants [Lei 2018] | "Is this feature/subsystem necessary?" — the report's ablation study |
| **Permutation importance** (§2.2) | Global | Black box | No | Low | Medium–high | **No** (naive) / Yes (conditional/grouped) | No | Model-reliance theory [Fisher 2019] | Cheap global ranking on a fixed model |
| **LIME** (§2.3) | Local | Black box | No | Moderate | Low–medium (unstable) | Weak | Local linear sign | None (heuristic surrogate) | Quick single-prediction sanity check |
| **KernelSHAP** (§2.4) | Local (→global by agg.) | Black box | No | High | Medium–high | Partial (depends on background) | Local sign + magnitude | **Shapley axioms** | Principled, consistent attributions, any model |
| **TreeSHAP** (§2.4) | Local + global | Tree-specific | No | Low (exact) | High (for trees) | Better (path-dependent) | Yes | Shapley axioms | Tree ensembles only |
| **Deep/GradientSHAP, DeepLIFT** (§2.4–2.5) | Local | NN gradients | No | Low | Medium–high | Partial | Yes | Approx. Shapley / DeepLIFT axioms | Fast attributions for the ConvCNP |
| **SAGE** (§2.4) | Global | Black box | No | Moderate–high | High | Partial | No | Shapley (global) | Global SHAP-style importance |
| **Integrated Gradients** (§2.5) | Local | NN gradients | No | Low | Medium–high | Partial | Yes | Sensitivity + impl.-invariance axioms | Differentiable models, axiomatic local attr. |
| **Saliency / SmoothGrad** (§2.5) | Local | NN gradients | No | Very low | **Low–medium** (fails some sanity checks) | No | Sign | None / weak | Fast visual hints; verify with [Adebayo 2018] |
| **Grad-CAM** (§2.5) | Local (spatial) | CNN internals | No | Very low | Medium | n/a (spatial) | Spatial heatmap | None | *Where* on the grid the CNN attends |
| **Occlusion / perturbation** (§2.6) | Local–regional | Black box | No | Moderate–high | Medium–high | Partial | No | None | Spatial importance maps, model-agnostic |
| **PDP / ICE** (§2.7) | Global / per-instance | Black box | No | Low–moderate | Medium (PDP misleads if correlated) | **No** (PDP) | **Yes** | None | Effect *shape* of a feature |
| **ALE** (§2.7) | Global | Black box | No | Low–moderate | Medium–high | **Yes** | **Yes** | None | Correlation-robust effect shape |

¹ *Faithfulness* = how well the attribution reflects the model's actual
computation. It is itself an active research area with formal metrics
(insertion/deletion, infidelity, sensitivity-n); see [Hooker et al. 2019]
(ROAR), [Yeh et al. 2019] (infidelity/sensitivity), [Adebayo et al. 2018]
(sanity checks).

---

## 4. Decision guide

- **"Does feature X actually matter to my system?"** → **Ablation / LOCO**
  (retrain). Most faithful, most expensive. This is what your report's ablation
  study does, and why it is the strongest evidence about the elevation MLP.
- **"Cheap global ranking on the already-trained model?"** → **Permutation
  importance**, but use a **conditional or grouped** variant because ERA5
  channels + elevation are correlated [Hooker, Mentch & Zhou 2021].
- **"Consistent attributions with theory, any model?"** → **SHAP** (KernelSHAP
  for black box; Deep/GradientSHAP for the ConvCNP).
- **"Why this one prediction, fast?"** → **LIME** (quick, but cross-check —
  unstable) or **Integrated Gradients** (axiomatic, cheap for a differentiable
  model).
- **"Where on the spatial grid does the CNN look?"** → **Grad-CAM** or
  **occlusion maps**.
- **"What is the *shape* of the effect (e.g. lapse-rate vs. elevation)?"** →
  **ALE** (preferred over PDP under correlation).

---

## 5. Additional methods worth knowing (beyond the core families)

The methods in §2 are the ones most people reach for first. The following are
genuinely missing from that list and several are *more* appropriate for this
project's setting (correlated geophysical inputs, a spatial CNN, a probabilistic
output).

### 5.1 Variance-based global sensitivity analysis (the climate-community standard)

These decompose the **variance of the output** into contributions from each
input and their interactions. This is the dominant paradigm in earth-system /
climate-model sensitivity analysis, and arguably the most natural fit for a
geophysical model — yet it is largely separate from the ML-interpretability
literature in §2.

- **Sobol indices** — first-order ($S_i$, main effect) and total-order ($S_{Ti}$,
  including all interactions) variance shares. Interaction-aware and global
  [Sobol 2001; Saltelli et al. 2010].
- **Morris elementary effects (screening)** — cheap one-at-a-time screening to
  rank/triage many inputs before a full Sobol analysis [Morris 1991;
  Campolongo et al. 2007].
- **FAST / eFAST** — Fourier-based estimation of sensitivity indices
  [Cukier et al. 1973; Saltelli et al. 1999].
- **Shapley effects** — Shapley-value attribution of *output variance*; unlike
  Sobol, it splits interaction and correlation effects fairly across inputs, so
  it is the recommended variance-based method **under correlated inputs** (e.g.
  ERA5 channels + elevation) [Owen 2014; Owen & Prieur 2017; Iooss & Prieur 2019].

**Why add these here:** they give interaction-aware, global, correlation-robust
importance — exactly the regime where naive permutation/PDP/SHAP struggle (§5
pitfalls). Tooling: Python `SALib`.

### 5.2 Controlled / statistically-rigorous variable selection

- **Model-X knockoffs** — construct "knockoff" copies of features that preserve
  the joint input distribution but are conditionally independent of the target,
  giving variable selection with **finite-sample false-discovery-rate control**.
  The most rigorous answer to "which features are *provably* relevant"
  [Candès et al. 2018; Barber & Candès 2015].
- **Conditional Predictive Impact (CPI)** — a model-agnostic, retraining-free
  test of *conditional* importance with valid inference, addressing the
  correlated-feature failure of marginal permutation [Watson & Wright 2021].
- **Conditional randomisation / leave-one-covariate-in tests** — hypothesis-test
  framing of feature relevance [Lei et al. 2018 (LOCO, §2.1); Candès et al. 2018].

### 5.3 Embedded / wrapper feature selection

Importance that falls out of the *fitting* process rather than a post-hoc probe:

- **L1 / Lasso and group-Lasso** sparsity — zeroed coefficients ⇒ unselected
  features; group-Lasso can prune whole channel/feature groups [Tibshirani 1996;
  Yuan & Lin 2006].
- **Recursive Feature Elimination (RFE)** — iteratively drop the weakest feature
  and refit [Guyon et al. 2002].
- **Learnable feature gates / attention** — concrete-dropout or stochastic-gate
  feature selection trained jointly with the network (e.g. STG)
  [Yamada et al. 2020]; attention weights as a (cautiously interpreted)
  importance signal.

### 5.4 Information-theoretic importance

- **Mutual information / conditional MI** and **mRMR** (minimum-redundancy
  maximum-relevance) rank features by (conditional) dependence with the target,
  model-free [Peng, Long & Ding 2005; Kraskov et al. 2004].

### 5.5 More CNN / deep-attribution methods

Complements to §2.5 that were not listed:

- **Layer-wise Relevance Propagation (LRP)** — conservation-based relevance
  redistribution through the network [Bach et al. 2015].
- **Gradient × Input** and **Expected Gradients** (a.k.a. GradientSHAP with a
  distribution of baselines) [Shrikumar et al. 2017; Erion et al. 2021].
- **RISE** — black-box saliency via randomised input masking and Monte-Carlo
  averaging; no gradients needed [Petsiuk et al. 2018].
- **Prediction Difference Analysis** — marginalise out input regions to measure
  their effect on the output, a probabilistic occlusion variant well suited to
  spatial inputs [Zintgraf et al. 2017].

### 5.6 Concept-, counterfactual- and rule-based explanations

A different explanatory *form* from per-feature magnitudes:

- **TCAV (Testing with Concept Activation Vectors)** — quantify sensitivity to
  human-defined *concepts* (e.g. "valley", "high TPI") rather than raw features
  [Kim et al. 2018]; **Concept Bottleneck Models** build this in by design
  [Koh et al. 2020].
- **Counterfactual explanations** — the smallest input change that flips/shifts
  the prediction ("how much warmer must the grid be to change the output?")
  [Wachter et al. 2017]; **DiCE** for diverse counterfactuals [Mothilal et al. 2020].
- **Anchors** — high-precision IF–THEN rules that locally "anchor" a prediction
  [Ribeiro et al. 2018].

### 5.7 Causal feature importance

The methods above are mostly associational. When the question is genuinely
causal:

- **Causal Shapley values** — Shapley attribution that respects a causal graph
  among inputs, separating direct from indirect effects [Heskes et al. 2020].
- **Asymmetric Shapley values** — incorporate known causal ordering
  [Frye et al. 2020].

### 5.8 Global surrogate & functional-decomposition models

- **Global surrogate models** — fit an interpretable model (decision tree,
  GAM/EBM) to the black box's predictions and read importance off the surrogate
  [Molnar 2022]; **Explainable Boosting Machines** as a glass-box reference
  model [Nori et al. 2019].
- **Functional ANOVA (fANOVA)** — decompose the prediction function into main
  effects and interactions; used both for importance and for hyperparameter
  importance [Hooker 2007; Hutter et al. 2014].

### 5.9 Adjacent: training-data attribution (different axis)

Not feature importance, but often the real question behind "what drives this
prediction": **influence functions** attribute a prediction to *training
examples* rather than features [Koh & Liang 2017]. Worth knowing so it is not
confused with feature attribution.

---

## 6. Pitfalls that apply across methods

1. **Correlated features** break naive permutation, PDP, and interventional
   SHAP/PFI by forcing the model onto off-manifold inputs
   [Hooker, Mentch & Zhou 2021; Apley & Zhu 2020].
2. **"True to the model" vs. "true to the data."** Interventional vs.
   conditional attributions answer different questions; neither is universally
   correct [Chen et al. 2020].
3. **Unfaithful explanations look plausible.** Run sanity checks (weight/label
   randomisation) and faithfulness metrics before trusting any saliency map
   [Adebayo et al. 2018; Hooker et al. 2019].
4. **Adversarial fragility.** LIME and SHAP explanations can be manipulated
   [Slack et al. 2020]; treat single explanations as evidence, not proof.
5. **Importance ≠ direction.** A high-importance feature could help or hurt a
   given prediction — pair magnitude methods with effect-shape methods
   (ALE/ICE).
6. **Grouped vs. individual.** For subsystems (an MLP head, a channel block),
   ablate/attribute the *group*; per-feature numbers can be misleading when
   features act jointly.

---

## 7. Note for an uncertainty-aware model (ConvCNP/GNP)

The methods above were designed for **point** predictors. A ConvCNP outputs a
*distribution* (μ, σ). Two practical consequences:

- Decide **what output you are attributing**: the mean μ, the predictive σ
  (uncertainty), or the log-likelihood/CRPS of the truth. Feature impact on σ
  can differ entirely from impact on μ — relevant given the report's
  overconfidence finding.
- Most off-the-shelf SHAP/IG tooling attributes a scalar output; wrap the model
  to expose μ, σ, or a score function separately and attribute each.
- The ablation study generalises cleanly here: just measure the *probabilistic*
  metric (CRPS, NLL, calibration) rather than MAE when you drop a feature.

This connects directly to the GNP follow-up noted in `00_CHRIS_NOTEBOOK.md`:
joint-Gaussian decoders change *what* σ means, so re-running feature-impact
analysis on the uncertainty head is a natural pairing.

---

## 8. References

> Verify DOIs/IDs on click-through before formal citation; arXiv IDs given where
> the work is most accessible there.

**Foundational / surveys**
- Molnar, C. (2022). *Interpretable Machine Learning* (2nd ed.).
  https://christophm.github.io/interpretable-ml-book/
- Fisher, A., Rudin, C., & Dominici, F. (2019). *All Models are Wrong, but Many
  are Useful: Learning a Variable's Importance by Studying an Entire Class of
  Prediction Models Simultaneously.* JMLR 20(177). arXiv:1801.01489
- Breiman, L. (2001). *Random Forests.* Machine Learning 45(1).
  (origin of permutation importance)

**Ablation / hold-out**
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R., & Wasserman, L. (2018).
  *Distribution-Free Predictive Inference for Regression* (LOCO). JASA 113(523).
- Meyes, R., et al. (2019). *Ablation Studies in Artificial Neural Networks.*
  arXiv:1901.08644

**Permutation / global importance**
- Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T., & Zeileis, A. (2008).
  *Conditional Variable Importance for Random Forests.* BMC Bioinformatics 9:307.
- Hooker, G., Mentch, L., & Zhou, S. (2021). *Unrestricted Permutation forces
  Extrapolation: Variable Importance Requires at Least One More Model.*
  Statistics and Computing 31(82). arXiv:1905.03151
- Covert, I., Lundberg, S., & Lee, S.-I. (2020). *Understanding Global Feature
  Contributions With Additive Importance Measures* (SAGE). NeurIPS. arXiv:2004.00668

**LIME**
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?":
  Explaining the Predictions of Any Classifier.* KDD. arXiv:1602.04938
- Alvarez-Melis, D., & Jaakkola, T. (2018). *On the Robustness of
  Interpretability Methods.* arXiv:1806.08049

**SHAP**
- Lundberg, S., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model
  Predictions.* NeurIPS. arXiv:1705.07874
- Lundberg, S., et al. (2020). *From Local Explanations to Global Understanding
  with Explainable AI for Trees.* Nature Machine Intelligence 2(1). (TreeSHAP)
- Chen, H., Janizek, J. D., Lundberg, S., & Lee, S.-I. (2020). *True to the Model
  or True to the Data?* arXiv:2006.16234

**Gradient / backprop attribution**
- Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). *Deep Inside Convolutional
  Networks: Visualising Image Classification Models and Saliency Maps.*
  arXiv:1312.6034
- Sundararajan, M., Taly, A., & Yan, Q. (2017). *Axiomatic Attribution for Deep
  Networks* (Integrated Gradients). ICML. arXiv:1703.01365
- Shrikumar, A., Greenside, P., & Kundaje, A. (2017). *Learning Important
  Features Through Propagating Activation Differences* (DeepLIFT). ICML.
  arXiv:1704.02685
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization.* ICCV. arXiv:1610.02391
- Smilkov, D., et al. (2017). *SmoothGrad: removing noise by adding noise.*
  arXiv:1706.03825

**Occlusion / perturbation**
- Zeiler, M. D., & Fergus, R. (2014). *Visualizing and Understanding
  Convolutional Networks.* ECCV. arXiv:1311.2901
- Fong, R., & Vedaldi, A. (2017). *Interpretable Explanations of Black Boxes by
  Meaningful Perturbation.* ICCV. arXiv:1704.03296

**Dependence / effect plots**
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting
  Machine* (PDP). Annals of Statistics 29(5).
- Goldstein, A., Kapelner, A., Bleich, J., & Pitkin, E. (2015). *Peeking Inside
  the Black Box: Visualizing Statistical Learning with ICE Plots.* JCGS 24(1).
  arXiv:1309.6392
- Apley, D. W., & Zhu, J. (2020). *Visualizing the Effects of Predictor Variables
  in Black Box Supervised Learning Models* (ALE). JRSS-B 82(4). arXiv:1612.08468

**Evaluating the explanations themselves**
- Adebayo, J., et al. (2018). *Sanity Checks for Saliency Maps.* NeurIPS.
  arXiv:1810.03292
- Hooker, S., Erhan, D., Kindermans, P.-J., & Kim, B. (2019). *A Benchmark for
  Interpretability Methods in Deep Neural Networks* (ROAR). NeurIPS. arXiv:1806.10758
- Yeh, C.-K., et al. (2019). *On the (In)fidelity and Sensitivity of
  Explanations.* NeurIPS. arXiv:1901.09392
- Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). *Fooling
  LIME and SHAP: Adversarial Attacks on Post-hoc Explanation Methods.*
  AAAI/AIES. arXiv:1911.02508
