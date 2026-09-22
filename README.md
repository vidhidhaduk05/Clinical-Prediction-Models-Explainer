# Clinical Prediction Models Explainer

> **A Visual Broadsheet on Medical Statistics, Prediction Modeling & Stroke Recalibration**  
> Inspired by [MLU-Explain](https://mlu-explain.github.io/) and [21st.dev](https://21st.dev), built for medical students, clinical researchers, and data scientists.

🌐 **Live Demo:** [https://vidhidhaduk05.github.io/Clinical-Prediction-Models-Explainer/](https://vidhidhaduk05.github.io/Clinical-Prediction-Models-Explainer/)

---

## Overview

Clinical Prediction Models Explainer is an interactive, browser-native educational platform designed to make advanced biostatistics and clinical prediction modeling intuitive, visual, and grounded in real-world acute stroke research.

Built entirely without heavy frameworks (pure vanilla JavaScript and curated modern CSS), every lesson features:
- **Everyday Life Analogies** (e.g., the Emergency Department census board, the custom tailor's ruler, the weather forecast penalty).
- **Interactive Visual Labs** (real-time SVG scatter plots, dynamic sliders, ROC/AUC curve morphing, calibration curve simulations, decision curves).
- **Mathematical Formula Cards** with term-by-term clinical and algebraic breakdowns ($Y, X, \beta_0, \beta_1, \epsilon$, Odds Ratios, Proportional Odds, Kaplan-Meier & Cox hazards, Brier Score, and Vickers' Net Benefit).
- **Stroke Manuscript Grounding** based on four contemporary acute ischemic stroke cohorts.

---

## The 24-Lesson Curriculum

### 1. Start Here
- **Lesson 1: Start with a CSV** — Rows, columns, feature types, and inspection of acute stroke patient data.
- **Lesson 2: Linear Regression** — Modeling continuous stroke outcomes (Final Infarct Volume, CVO score, baseline core).
- **Lesson 3: Regression Metrics** — Quantifying prediction errors with MAE, RMSE, Mean Signed Error, and cross-validated $R^2$.

### 2. Clinical Prediction Fundamentals
- **Lesson 4: Association, Prediction, Causation** — The three distinct clinical questions and why prediction models must not guide causal interventions without counterfactuals.
- **Lesson 5: Binary Outcomes & Logistic Regression** — Sigmoid S-curves, odds ratios vs. relative risks, and modeling 90-day functional recovery.
- **Lesson 6: Ordinal Outcomes & Proportional Odds** — The modified Rankin Scale (mRS 0–6), the harm of "dichotomania", and common odds ratios.
- **Lesson 7: Time-to-Event & Survival Analysis** — Right-censoring, Kaplan-Meier curves, and competing risks (Fine-Gray subdistribution models).

### 3. Data Problems & Flexible Modeling
- **Lesson 8: Missing Data & Multiple Imputation** — MCAR, MAR, MNAR mechanisms, and why complete-case analysis introduces fatal clinical selection bias.
- **Lesson 9: Restricted Cubic Splines** — Relaxing straight-line assumptions and placing knots at inflection points.
- **Lesson 10: Residual Phenotyping** — Discovering biological subtypes via greater-than-expected Net Water Uptake (NWU) progression.
- **Lesson 11: Interactions & Marginal Standardization** — Effect modification between collateral robustness and reperfusion status.
- **Lesson 12: Feature Engineering & Correlation** — Managing collinear perfusion parameters (CBF, CBV, MTT, Tmax).

### 4. Validation & Performance
- **Lesson 13: Repeated k-Fold Cross-Validation** — Preventing data leakage and fold instability in small cohorts.
- **Lesson 14: Bootstrap & Out-of-Bag Evaluation** — Resampling patient cohorts and optimism correction.
- **Lesson 15: AUC & Discrimination** — Concordance index ($C$-statistic), ranking patients, and why high AUC does not imply clinical utility.
- **Lesson 16: Calibration & Brier Score** — Evaluating truth-in-advertising for clinical probabilities (intercept $\alpha = 0$, slope $\beta = 1.0$).
- **Lesson 17: Decision Curve Analysis (DCA)** — Vickers' Net Benefit across clinical decision thresholds ($p_t$).
- **Lesson 18: Continuous Outcomes Comparison** — Out-of-bag $R^2$ and ANCOVA baseline adjustments.

### 5. Model Stability & Reporting
- **Lesson 19: Penalization (Ridge, Lasso, Elastic Net)** — Shrinkage, parameter budgeting, and why automated stepwise selection is discredited.
- **Lesson 20: Machine Learning Benchmarking** — Fair paired model comparisons (Random Forests, Gradient Boosting vs. Logistic Regression).
- **Lesson 21: Interpretability & Explainability** — SHAP values, Partial Dependence Plots, and avoiding causal illusions.
- **Lesson 22: Model Updating & Transportability** — From intercept recalibration to full refitting across external stroke centers.
- **Lesson 23: TRIPOD+AI & Decision Trace** — The 27-item reporting checklist and auditable decision tracing.
- **Lesson 24: Anti-Patterns Catalog** — An interactive spot-the-leak challenge highlighting common clinical modeling pitfalls.

---

## Real-World Clinical Evidence

The explainer is structured around findings from four acute ischemic stroke studies:
1. **CVO CTP Recalibration Study**: Evaluating Cortical Venous Outflow as an imaging buffer against final infarct expansion and documenting the shrinkage of apparent $R^2 = 0.335$ to cross-validated out-of-fold $R^2 = 0.077$.
2. **ASPECTS Territory & Hemorrhagic Transformation Study**: Investigating regional topographic ischemic damage and secondary hemorrhagic transformation risk.
3. **Fast Responders in Favourable CTP Profile Study**: Characterizing patient subgroups who achieve dramatic clinical recovery following endovascular thrombectomy.
4. **Greater than Expected NWU Progression**: Identifying patients with accelerated blood-brain barrier failure and malignant ionic edema via residual phenotyping.

---

## Architecture & Tech Stack

- **Zero External Dependencies**: Pure client-side JavaScript (ES6+), semantic HTML5, and bespoke CSS.
- **21st.dev & MLU-Explain Design Principles**:
  - Centered single-column reading container with generous breathing room (`max-width: 860px`).
  - Warm broadsheet color palette with HSL-tailored tokens (`#fbf8f3`, `#252422`, `#bd4d24`).
  - High-performance, responsive SVG data visualizations and interactive control sliders.
  - Dedicated formula cards with term-by-term algebraic and clinical annotations.
  - Smooth keyboard arrow navigation (`←` / `→`) and mobile-responsive drawer navigation.
- **Deployment**: Configured for instant deployment via GitHub Pages and GitHub Actions.

---

## Local Development

To run locally:
```bash
# Clone the repository
git clone https://github.com/vidhidhaduk05/Clinical-Prediction-Models-Explainer.git
cd Clinical-Prediction-Models-Explainer

# Start a local static file server (Python)
python -m http.server 8000

# Or using Node
npx serve .
```
Then open `http://localhost:8000` in your web browser.

---

## License & Citation

Published for open-access medical education and statistical transparency.
When utilizing these interactive modules or curriculum structures in educational lectures or research, please cite the respective acute stroke publications referenced in the curriculum.
