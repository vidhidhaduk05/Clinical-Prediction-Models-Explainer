# Clinical Concepts Explainer Site

## Current status

This is a partially implemented static website for Vidhi Dhaduk's interactive medical-statistics explainer. It is inspired by the teaching approach of [MLU Explain Linear Regression](https://mlu-explain.github.io/linear-regression/) but is an original design and curriculum.

The source is in `/workspace/sites/clinical-concepts`.

Current files:

- `dist/index.html` — shell with sticky header, lesson sidebar, main content mount point, metadata, private/no-index hints, and a custom SVG favicon.
- `dist/style.css` — complete responsive visual system: typography, sidebar, lesson cards, interactive labs, tables, code blocks, charts, responsive mobile navigation, focus states, and reduced-motion support.
- `dist/math.js` — browser/module-safe statistics helper functions and toy data.
- `.openai/hosting.json` — static Sites configuration using `dist` as the published directory.

The current deployment project has been registered privately as:

- Project ID: `appgprj_6ab1c957c580819192ce82c925480ad3`
- Expected private URL: `https://clinical-concepts-vidhi.blond-box-0629.chatgpt.site`

No production deployment has been completed yet.

## Important unfinished item

`dist/app.js` has **not** been written yet. The `index.html` imports it, so the page will currently show the shell but will not render lessons until `app.js` is created.

## Required next step

Create `dist/app.js` as a single ES module importing from `./math.js`, then run it through a local preview or browser QA before publishing.

The implementation should be data-driven. Render the lesson navigation and main content from a `lessons` array. Use the URL hash for navigation, for example `#linear`, `#csv`, `#metrics`, `#validation`, `#logistic`, `#calibration`, `#missing`, `#splines`, `#ordinal`, `#survival`, `#penalization`, `#interpretability`, `#papers`, and `#sources`.

## Curriculum that must be implemented

### Start here

1. **Start with a CSV** — Medical example: the NIHSS/infarct-volume dataset used in the prior lesson. Walk through `pd.read_csv`, shape, columns, data types, and missingness. Explain each line of the Python code in plain language.
2. **Linear regression** — Question: can infarct volume help predict NIHSS? Use `outcome = intercept + slope × predictor + residual`; include an interactive scatter plot and the teaching-session anchors: `NIHSS ~ volume` R² about 0.335; repeated cross-validation R² about 0.077; MAE about 20,021 mm³; RMSE about 50,093 mm³; mean signed error about 12,900 mm³. Explain apparent versus out-of-sample R².
3. **Regression metrics** — Interactive predictions and residuals. Explain R², mean absolute error, root mean squared error, and mean signed error with a weather-forecast analogy.

### Clinical prediction fundamentals

4. **Association, prediction, causation** — Use a three-column comparison with venous outflow and poor 90-day modified Rankin Scale outcome.
5. **Binary outcomes and logistic regression** — Use mRS 3–6 or death by 90 days. Explain probability, odds, log-odds, logistic curve, odds ratio, and adjusted risk. Include an odds-to-risk mini converter; never describe odds ratio as percent risk increase.
6. **Ordinal outcomes and proportional odds** — Use the full 0–6 modified Rankin Scale. Explain the common odds ratio and cut-specific logistic checks.
7. **Time-to-event / survival analysis** — Explain time, event, censoring, Kaplan-Meier, Cox model, hazard ratio, proportional-hazards assumption, and the two-timepoint trajectory caveat.

### Data problems and flexible modeling

8. **Missing data and multiple imputation** — Explain MCAR, MAR, MNAR, MICE, predictive mean matching, Rubin's rules, and why imputation belongs inside validation folds. Contrast inference imputation (outcome included) with prediction-fold imputation from the supplied favourable-CTP protocol (outcome excluded from validation-fold imputation).
9. **Restricted cubic splines** — Explain why a straight line can be too rigid, knot slider/curve plot, NWU or age example, 4/5 knots, degrees of freedom, likelihood-ratio comparison, and avoiding extrapolation.
10. **Residual phenotyping / greater-than-expected progression** — Explain observed 24-hour increment minus expected increment from admission NWU, resample-safe derivation, and the serial-measurement reparameterization caveat.
11. **Interactions and marginal standardization** — Use admission-NWU × excess-progression. Include an interactive adjusted absolute risk difference.
12. **Feature engineering and correlation** — Explain nominal versus ordinal predictors, one-hot encoding, clinically motivated features, transformations, and correlated CTP measurements. Use ASPECTS, NIHSS, glucose, rCBF, Tmax, mismatch ratio, and COVES/CVO examples.

### Validation and performance

13. **Repeated k-fold cross-validation** — Animate five folds and repeated partitions. Explain why n=131 with 19 events means roughly 3–4 events per test fold and unstable AUC. Explain the anti-leakage rule.
14. **Full-pipeline bootstrap and out-of-bag evaluation** — Show bootstrap sampling, out-of-bag patients, refitting all pipeline steps, and paired model comparisons with percentile intervals.
15. **AUC / discrimination** — Interactive ROC threshold slider. Explain ranking, sensitivity, specificity, and why AUC is not calibration. Manuscript anchors: residual perfusion variation AUC about 0.65; 16-variable death AUC about 0.84; greater-than-expected progression model AUC about 0.55.
16. **Calibration and Brier score** — Interactive calibration plot; explain calibration-in-the-large, slope, and Brier. Include calibration slope 0.91 and Brier score 0.107 where appropriate.
17. **Decision curve analysis** — Interactive threshold slider with model / treat-all / treat-none net benefit. Explain clinical usefulness.
18. **Continuous-outcome comparison** — Explain out-of-bag R², ANCOVA, mixed models, baseline adjustment, residual diagnostics, and two-timepoint difference-in-differences.

### Model stability and reporting

19. **Penalization** — Compare ridge, lasso, and elastic net with coefficient shrinkage. Explain EPV as parameter count and why penalization beats stepwise selection when events are limited or predictors correlate.
20. **Machine-learning benchmarking** — Compare penalized regression, random forest, and gradient boosting conceptually. Explain nested cross-validation and paired comparison.
21. **Interpretability** — Explain permutation importance, SHAP, PDP, and ALE. Emphasize held-out/out-of-bag computation and that attribution is not causality.
22. **Model updating and transportability** — Explain no update, intercept update, recalibration slope, selective refitting, and full refitting using multi-center stroke validation.
23. **TRIPOD+AI and decision trace** — Build a small checkbox checklist based on the supplied 27-item guidance. Explain recording alternatives, rationale, thresholds, seeds, software versions, and model choices.
24. **Anti-patterns** — Interactive “spot the leak” cards: impute before split, stepwise selection, apparent AUC, OR-as-risk, post-treatment predictor in an admission model, SHAP on training data, and censoring death in competing risks.

## Manuscript map page

Create a `#papers` page with four cards:

1. **Greater-than-Expected NWU Progression After Thrombectomy** — residual phenotype, restricted cubic spline, linear mixed model, ANCOVA, bootstrap OOB, MICE inside resampling, logistic regression, marginal standardization, interaction, Brier, calibration, decision curve. Anchor: n=626 edema analysis, n=600 with 90-day mRS, death AUC 0.84, progression AUC 0.55, adjusted mortality risk difference 20.0 percentage points in high-admission-NWU patients.
2. **Integrating Cerebral Venous Outflow With CT Perfusion** — linear recalibration, repeated stratified 5-fold CV ×20, R², signed error, MAE, RMSE, AUC, calibration, bootstrap 2,000 replicates, MICE 25 imputations, site-held-out validation. Anchor cohorts: 527 successfully reperfused, 131 unsuccessful-reperfusion, 505 and 128 with functional outcome.
3. **Beyond Favorable CT Perfusion** — proportional-odds model, logistic secondary model, splines, proportional-odds assumption, MICE m=20, repeated stratified five-fold CV ×10, three imputations per repetition, 1,000 patient-level bootstrap comparisons, AUC, calibration, Brier, sensitivity analyses. Anchors: n=151, 90 poor outcomes, VO cOR 0.64 per point, residual perfusion variation AUC 0.65, clinical model AUC 0.85.
4. **Striatal Localization of Hemorrhagic Risk After Thrombectomy** — logistic regression, burden adjustments, joint models, conditional probabilities, phi coefficient, Benjamini-Hochberg false-discovery-rate correction, MICE and Rubin pooling, optimism correction, likelihood-ratio test, spline linearity test, interaction.

## Required implementation patterns

Use the exported helpers in `math.js` whenever possible: `metrics`, `ols`, `predict`, `repeatedCV`, `auc`, `confusion`, `brier`, `netBenefit`, `rubin`, `bh`, `bootstrapIndices`, `km`, `rcs`, and `fitBasis`. Use SVG for dependency-free charts. Keep the site static and self-contained: no patient data, uploads, analytics, or external runtime APIs. Preserve `meta robots=noindex,nofollow` and private Sites access.

## Validation checklist

1. `node --check dist/app.js`.
2. Run a numerical smoke test covering ordinary least squares, R², MAE, RMSE, signed error, AUC, Brier, net benefit, Rubin pooling, Benjamini-Hochberg, bootstrap OOB sets, repeated cross-validation, and Kaplan-Meier.
3. Start managed preview with `sites-preview start /workspace/sites/clinical-concepts` if available. Managed instructions say this preview is internal and has no user-facing handoff.
4. Browser QA at `http://terminal.local:4173/` if preview works: confirm first screen, navigation, sliders, mobile menu, keyboard focus, no console errors, and no horizontal overflow.
5. Publish only after lessons render. Reuse the existing private project ID and source-repository credential; do not create a second Site.

## Citation/source notes

Content was extracted from the supplied DOCX files and the supplied `clinical-prediction-modeling` skill. Methodological framing follows the skill's chapters and templates. External background sources consulted during planning included TRIPOD+AI, Riley sample-size guidance, and official scikit-learn documentation on cross-validation, metrics, calibration, pipelines, and leakage. Do not copy the reference MLU Explain site's text or code; use it only as a pedagogy reference.

## Current numerical helper smoke-test result

The existing `math.js` passed 13 numerical assertions covering ordinary least squares, perfect and tied AUC, Brier score, Benjamini-Hochberg rejection, Rubin variance, bootstrap OOB membership, repeated cross-validation, and Kaplan-Meier endpoint behavior. This does not replace final browser QA.
