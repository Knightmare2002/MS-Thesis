# Direction A: methodological specification and claim audit

This specification precedes implementation. It interprets section 4.1 of the supplied research-gap report and preserves its observation likelihood, notation and class-specific tempering. It corrects assumptions that the report had stated too strongly. No bridge performance, physical resolution limit or publication-level novelty is established by software tests.

## Research question and scope

Does a class-dependent model of unresolved observations improve held-out 3D damage mapping and calibration over count/agreement rules, maximum fusion, geometric filtering and generic weighting, especially under heterogeneous resolution? The deliverable is an experimental implementation, not a certified structural-condition assessment.

The contribution is the observation model, not projection, point-cloud segmentation, Platt calibration, Bayesian updating or the general concept of probability of detection. Work on cracks without a geometric footprint supports an image-to-surface approach rather than assuming that a point-only network can recover absent geometric information ([Benz and Rodehorst, 2025](https://openaccess.thecvf.com/content/WACV2025/papers/Benz_CrackStructures_and_CrackEnsembles_The_Power_of_Multi-View_for_2.5D_Crack_WACV_2025_paper.pdf)).

## Model and derivation

Let $$z_{ic}$$ be binary damage membership, independently represented for each class (not a mutually exclusive softmax). Let $$p_{ivc}$$ be the sampled, calibrated detector score. Under a reference distribution with prevalence $$\rho_c$$, exact calibration of this score implies
$$
\Lambda_c(p)=f_1(p)/f_0(p)
=\frac{p}{1-p}\frac{1-\rho_c}{\rho_c}.
$$
Platt fitting only approximates this condition. Applying the calibration to a different dataset, resized maps, interpolated scores or an average of separately calibrated models does not prove it remains calibrated. The implementation therefore uses a single Platt-calibrated model as the primary experiment; ensemble likelihood fusion is not silently treated as calibrated.

Introduce a latent resolving event $$R_{ivc}$$ with probability
$$
\pi_c(g)=\operatorname{sigmoid}((\log g_c^{50}-\log g)/\beta_c),\quad \beta_c>0.
$$
Assume $$p\mid z=1,g\sim\pi_c(g)f_1+(1-\pi_c(g))f_0$$, and $$p\mid z=0,g\sim f_0$$. Marginalization gives the report's exact mixture likelihood ratio:
$$
\widetilde\lambda_{ivc}=\log[(1-\pi_c(g))+\pi_c(g)\exp(\lambda_{ivc})],
\quad \lambda_{ivc}=\operatorname{logit}p_{ivc}-\operatorname{logit}\rho_c .
$$
The posterior score is
$$
\widehat s_{ic}=\operatorname{sigmoid}\{\operatorname{logit}\rho_c+
\alpha_c\sum_v\widetilde\lambda_{ivc}\},\quad 0<\alpha_c\leq1.
$$
The optional report weight is fixed to $$\omega_{iv}=1$$. No unvalidated blur or quality weighting is added to the primary method. A different target prior can be specified explicitly, but must be frozen using calibration data, never inferred from T.

This model censors positive AND negative evidence as $$\pi\to0$$; positive evidence retains its sign for $$0<\pi\leq1$$. Negative means $$p<\rho_c$$, not necessarily a missed thresholded detection: a score below the detector threshold but above $$\rho_c$$ still contributes positive likelihood evidence. The user's claim must be understood in this likelihood sense, not as unconditional preservation of every positive and suppression of every sub-threshold score.

The correction is not simply a weight: for $$0<\pi<1$$, Jensen's inequality gives
$$\log(1-\pi+\pi e^\lambda)\geq\pi\lambda$$, and the negative evidence saturates at $$\log(1-\pi)$$ as $$\lambda\to-\infty$$. This is a falsifiable distinction from weighted log odds, which is therefore included as an ablation.

The mixture assumes stable background and resolved-positive score distributions across geometry. It may fail under shadows, specularities, blur, saturation, wide out-of-distribution cracks or registration errors. It cannot create a detection when every view is uninformative. It provides a conditional generative explanation of false-negative dilution, not a theorem that max fusion is universally optimal.

An additional identifiability risk is that pooled DACL10K calibration includes heterogeneous defect scales. Its positive score distribution is not automatically the fully resolved $$f_1$$ assumed by the mixture. Applying another resolution mixture can double-count degradation. The proposed likelihood must therefore be treated as a working observation model until resolved-view calibration/conditional score distributions and held-out 3D NLL support it. An empirical positive result, without these checks, would justify a useful scoring rule but not an exact physical posterior interpretation.

## Geometry and units

Use source-frame unit normals and source-frame camera rays:
$$
\mu_{iv}=\frac{|n_i^\top(C_v-P_i)|}{\|n_i\|\|C_v-P_i\|},\qquad
g_{iv}=\frac{Z_{iv}s}{f_v}\frac{\sigma_v}{\sqrt{\max(\mu_{iv},\mu_{\min})}} .
$$
Here $$s$$ converts chunk distances to source distances; $$\sigma_v=W/W_m$$ is measured from the probability map, not guessed from a CLI default. A declared additional inference scale is allowed and recorded. Never count the same resize twice. The existing sliding-window inference operates on native-resolution images; its storage downsample is already included in $$W/W_m$$.

This is the report's scalar pinhole footprint approximation, not an exact lens-distortion Jacobian or a physical crack-width bound. The $1$ factor is the geometric mean of the assumed anisotropic footprint. Grazing incidence is regularized with a declared floor; zero/invalid normals must not silently become strong negative evidence. Anisotropic crack-direction correction and optional quality weights remain outside the primary experiment. $1$ has source units per map pixel; it is not millimetres unless scale has been independently established.

## Identification: a necessary correction to the report

The report's leave-view-out hit frequency is NOT generally the latent $1$. At a frozen threshold $1$, let $1$ and $1$. The mixture implies
$$
q_c(g)=P(p\geq t_c\mid z=1,g)=b_c+(a_c-b_c)\pi_c(g).
$$
Only if $1$ does $1$. Fitting recall directly as $1$ otherwise confounds detector errors with non-resolvability. The implementation fits this Bernoulli mixture with externally fixed, documented $1$; it never jointly fits all four parameters from an unanchored curve. DACL10K validation counts can supply reference rates, but their transfer to fully resolved bridge observations is an explicit assumption, not measured truth.

Pseudo-positive points use the k most resolutive OTHER cameras that unanimously exceed a predeclared class threshold, and must be finer than the held-out candidate by a predeclared ratio. Candidate outcomes are never part of their own reference. Optional camera groups allow leave-group-out, preferable for near-duplicate frames. Fitting uses C_fit only, and does not read its labels. Equal total contribution per pseudo-point and per region prevents a densely reconstructed or heavily photographed region dominating the fit.

The fitted quantity remains a weakly supervised, detector-conditioned resolvability proxy. Leave-view-out removes direct self-inclusion, not correlated errors or selection bias toward easy positives. Report support, failures, empirical hit rates, anchor assumptions, g range and optimizer boundary diagnostics. Require independent comparison with resolution-degraded DACL10K predictions and, when available, labels on C. Non-identifiable/one-sided curves must fail closed rather than produce default scientific parameters.

## Tempering and leakage

Fit one $1$ by area-weighted Bernoulli NLL on C_cal, disjoint from C_fit and T; require both labels per class. This is empirical overconfidence regularization, not a pairwise-correlation estimator. With no labels, $1$ is an explicitly untempered experiment, not an estimated correction. Evidence of overconfidence in semantic mapping motivates evaluation, but does not validate this particular correction ([GLFS](https://arxiv.org/abs/2311.10018)).

All classes, camera groups, top-k, thresholds, masks, spatial splits, buffers, GSD bins and seeds are frozen before T. Splits are region-level with a user-defined physical buffer checked on supplied coordinates. Area weights must describe actual surface support, not arbitrary point counts. T labels are only consumed by the final evaluation or clearly labelled oracle. Fitting on unlabelled T would instead be transductive and must be a separately declared experiment.

## Experimental design and thesis adequacy

The thesis is viable as a controlled hypothesis test, even if the proposed method does not outperform max. Minimum evidence is: analytical invariants; geometry/projection integration tests; known-generative-model recovery; misspecified-model stress tests; image-level degradation with the frozen detector; and independent held-out 3D annotations. Synthetic tests validate implementation and conditional behaviour only.

Use Metashape for the user's annotation workflow. An independently annotated, spatially buffered multi-region reference must include negative surfaces and hard confounders, not only predicted damage. Two annotators and adjudication on a subset are recommended. No annotation count guarantees adequate power: report available independent regions and uncertainty rather than claiming millions of points are independent samples.

Primary comparators: mean, max, angle-range, distance-range, distance-weighted, untempered naive Bayes, tempered uncensored Bayes, untempered censored Bayes and tempered censored Bayes. Include generic $1$-weighted mean and $1$-weighted log-odds to test whether the likelihood structure itself matters. Geometric baseline conventions follow Table 2 where possible, with explicit tie/no-support handling and unsigned-normal adaptation; do not claim bit-for-bit reproduction of a separate implementation ([CrackStructures](https://openaccess.thecvf.com/content/WACV2025/papers/Benz_CrackStructures_and_CrackEnsembles_The_Power_of_Multi-View_for_2.5D_Crack_WACV_2025_paper.pdf)).

Evaluate per-class area-weighted AP (step-integrated PR, not trapezoidal PR), F1, IoU, NLL and Brier; report coverage and compare on identical valid support. Preserve unknown labels as -1. For cracks, use independently supplied, uniformly sampled 3D medial axes for IoU(tolerance), not tolerant IoU on a dense positive point cloud. The reference benchmark samples axes at 1 mm and evaluates tolerances 0.5/1/2/4/8 cm ([Benz and Rodehorst](https://openaccess.thecvf.com/content/WACV2025/papers/Benz_CrackStructures_and_CrackEnsembles_The_Power_of_Multi-View_for_2.5D_Crack_WACV_2025_paper.pdf)).

Use paired region/block bootstrap for method differences, report stratified performance versus GSD/incidence with predeclared bins, and repeat nested global-camera subsets at 25%, 50%, 100% with fixed seeds. Freeze fitted parameters for cardinality stress testing; refitting per subset is a different experiment. A label-assisted per-point view oracle is diagnostic only, not a deployable baseline or a single physical camera for the whole region.

## Novelty audit and CACAIE assessment

The exact defensible claim is the user's distinction between informative non-detection and detail-limited non-detection via a class-dependent observation likelihood. The following evidence narrows, but cannot certify, novelty.

| Argomento trattato | Link (URL) | A cosa serve/Perché è utile |
|---|---|---|
| Count/agreement and unassessed surfaces | [Yamane et al.](https://doi.org/10.1111/mice.12971) | Direct bridge precedent: multi-image integration already records damage, low-damage and unresolved/unobserved areas; those outcomes alone are not novel. |
| Geometric heuristics and max fusion | [Benz and Rodehorst](https://openaccess.thecvf.com/content/WACV2025/papers/Benz_CrackStructures_and_CrackEnsembles_The_Power_of_Multi-View_for_2.5D_Crack_WACV_2025_paper.pdf) | Strong required comparator; already explains false-negative dilution for a particular detector and acquisition regime. |
| Learned per-view quality weights | [DeepChoice](https://isprs-archives.copernicus.org/articles/XLIX-B2-2026/641/2026/) | Learned weighting from angle, range and image-quality cues already exists; novelty cannot be “geometry-aware weighting.” |
| 3D probability calibration | [GLFS](https://arxiv.org/abs/2311.10018) | Bayesian overconfidence and learned fusion/calibration already studied; tempering alone is not the contribution. |
| View-sensitive statistical defect fusion in NDE | [Wilcox et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7735296/) | Uses view/defect-dependent expected response and background distributions; the general idea of sensitivity-aware likelihood fusion is prior art. Modality is ultrasound, not calibrated image segmentation on 3D surfaces. |
| Missed-detection likelihood conditional on visibility | [Multi-camera Bayesian tracking](https://arxiv.org/html/2001.04118v4) | Explicit view-dependent detection probability and missed-detection factor already distinguish weak visibility from negative evidence. Novelty must be narrower than this general principle. |
| Bayesian multi-frame crack detection | [Chen et al., 2017](https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.12256) | CACAIE already published Bayesian frame fusion for low-contrast metallic cracks; the accessible abstract does not establish a class/GSD-conditioned mixture. |
| Recent rule-based multi-view inspection | [Region-aware industrial inspection](https://arxiv.org/html/2610.29398) | Deterministic per-frame acceptance and frame-count aggregation, not the proposed latent-resolvability likelihood. Different domain; illustrative comparator, not a replicated benchmark. |
| Recent direct multi-class competitor | [Zhang et al., bibliographic listing](https://dblp.org/pid/07/314-41.html) | Full text of DOI 10.1016/j.aei.2026.104862 was inaccessible during this audit; its fusion equations remain an unresolved prior-art check. |

This is a plausible methodological contribution and suitable experimental thesis question. A CACAIE submission requires a meaningful, reproducible engineering benefit over strong baselines, independent generalization evidence and a resolved prior-art search. A single bridge, self-labelled reference, positive-only regions or software-only results would not substantiate the intended claim. Multiple acquisition regimes/structures and model backbones are strongly recommended as an author judgement, not asserted journal acceptance criteria.

The earlier report's “medium-high novelty” is not an established finding. The Wang & Liang DOI 10.1016/j.cacaie.2026.100236 could not be verified/read in this audit, and must not be used as confirmed evidence either for or against novelty. Journal scope/acceptance criteria were not recovered from an authoritative current aims-and-scope page. No acceptance probability or guaranteed novelty is claimed.

## Decision register and preregistration requirements

- **Likelihood clipping:** `p_clip=1e-4` bounds per-view log odds and prevents infinities from stored zero/one scores. It is a numerical regularizer, not an estimated sensor property; evaluate sensitivity on C, especially for extremely rare classes. Metric NLL separately uses `1e-7`, identically for every method.
- **Curve constraints:** g50 is restricted to the observed fitting range and beta to `[0.05,5]`. These broad operational bounds prevent extrapolated transitions and degenerate steps; boundary optima fail rather than become reported parameters. They do not establish physical resolution limits or a universal plausible class width. Changing bounds is a declared C-only sensitivity experiment.
- **Pseudo-reference defaults:** k=2 distinct groups, a 1.25 resolution ratio and reference threshold at least 0.9 favor precision/support conservatively without claiming optimality. Grouping must be based on acquisition geometry/time, not outcomes. Freeze alternatives for sensitivity before T.
- **Calibrated comparisons:** uncensored Bayes and generic pi-weighted log odds each receive their own C_cal NLL-optimal alpha. All deployable methods receive their own C_cal F1 threshold on the same fixed grid. This separates the proposed likelihood structure from a privileged calibration budget.
- **Statistical uncertainty:** fitting can bootstrap regions for g50/beta conditional on fixed anchors/selection. Evaluation bootstraps paired regions for method differences. Neither captures transferred-anchor uncertainty; repeat a prespecified anchor sensitivity grid and report instability. DACL degradation exports per-image counts for image-level uncertainty analysis rather than treating pixels as independent trials.
- **Completion and integrity:** projection culling legitimately omits empty shards and leaves some chunks at earlier batch indices. Completion is checked through the global committed batch frontier and a sealed hash inventory, not an erroneous requirement that every chunk receives every camera.
- **Primary acceptance criterion:** predeclare class-specific AP and NLL/Brier comparisons against max and independently tempered baselines, including coverage; use F1/IoU at frozen thresholds as complementary measures. Avoid declaring success from a favorable single class/bin/seed selected after T. Report null and adverse results.
- **Scope boundaries:** arbitrary quality weights, anisotropic crack-direction terms, automatic GT/area/medial-axis extraction and cross-run transfer are not implemented as unvalidated shortcuts. The present experimental chain is within-run spatially disjoint validation; external-structure transfer requires an explicit extension to the strict provenance protocol.

For a thesis, a complete, honestly negative test of these assumptions remains a valid result. For a paper, the narrow claim should be framed as a class/resolution-conditioned score-likelihood model for image-to-surface structural-damage mapping, with anchored identification and demonstrated benefit, not the invention of informative versus uninformative missed detections in general.
