# Pedagogical Audit — Clinical Concepts Explainer

## 1. Executive Summary
This audit reviews the current 24-lesson clinical prediction modeling explainer from the perspective of our primary learner persona: **a medical student, neurology resident, or clinical investigator who is clinically astute but has zero prior coding experience in Python, pandas, or mathematical statistics**.

While the current platform features impressive aesthetic breadth and responsive interactive SVG graphs, its instructional architecture previously suffered from **academic cognitive overload**: introducing equations before intuition, using statistical jargon before plain English, presenting code without line-by-line token breakdowns, and employing interactive graphs as static displays rather than discovery engines.

---

## 2. Strengths of the Current Platform
1. **Visual Identity**: Warm, restrained broadsheet styling (`#fbf8f3`, `#252422`, `#bd4d24`) providing high contrast, readability, and a scholarly feel without becoming a generic corporate dashboard.
2. **Interactive SVGs**: Fully custom, responsive SVG graphs for linear regression, ROC/AUC, calibration, and survival curves with zero heavy framework bloat.
3. **Comprehensive Breadth**: Covers the entire translation pipeline from admission CSVs to restricted cubic splines, multiple imputation, cross-validation, and TRIPOD+AI guidelines.
4. **Clinical Relevance**: Grounded in acute ischemic stroke, endovascular thrombectomy, perfusion imaging, and functional recovery outcomes.

---

## 3. Friction Points & Identified Gaps

### Friction Point 1: Premature Equations (Formulas Before Intuition)
- **Previous Pattern**: Lesson 2 and Lesson 3 presented algebraic formulas like $Y = \beta_0 + \beta_1 X + \epsilon$ and $\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$ early in the text before the learner understood what a prediction error or slope physically represented.
- **Remediation**: Reorder according to the 17-step sequence: Clinical Question $\to$ Guessing $\to$ Visual Manipulation $\to$ Plain-English Explanation $\to$ Formal Term Name $\to$ Equation & Symbol Breakdown.

### Friction Point 2: The "Pandas Wall" for Non-Programmers
- **Previous Pattern**: Code blocks dumped multiple lines of Python (`df = pd.read_csv(...)`, `df.shape`, `df.isna().sum()`) without explaining what `import`, `pd`, `.read_csv`, or `DataFrame` actually mean.
- **Remediation**: Build an interactive token-by-token code explainer (`renderTokenCodeBlock`). Explain every word:
  - `import`: Load a specialized tool into our workspace.
  - `pandas`: The gold-standard library for working with tabular data (rows and columns).
  - `as pd`: A universally accepted nickname to keep our code short.
  - `df`: Short for "DataFrame", Python's term for a spreadsheet.
  - `.read_csv()`: The function that opens our comma-separated file and loads it into memory.

### Friction Point 3: Predictor vs. Outcome Ambiguity ($X$ vs. $Y$)
- **Previous Pattern**: The terms "independent variable" and "dependent variable" were recited as textbook definitions.
- **Remediation**: Replace with the direct clinical question: *"What information do we know on arrival ($X$, Candidate Predictor) versus what are we trying to forecast ($Y$, Outcome Target)?"* Provide an interactive click/drag widget where learners classify variables into $X$ or $Y$.

### Friction Point 4: Disconnected Residuals and Metrics
- **Previous Pattern**: MAE, RMSE, Mean Signed Error, and $R^2$ were presented in separate subsections as distinct mathematical formulas.
- **Remediation**: Unify them visually. Show vertical error lines (residuals) on 4 stroke patients. Then demonstrate:
  - Absolute error bar lengths $\to$ MAE.
  - Squaring large error bars $\to$ RMSE.
  - Signed error bars canceling out $\to$ Mean Signed Error (explaining why MSE = 0 can hide terrible predictions).
  - Residual scatter compared to raw variance $\to R^2$ (and destroying the myth that $R^2 = 0.30$ means "30% accurate").

### Friction Point 5: Cross-Validation as Prose rather than Action
- **Previous Pattern**: Described $k$-fold cross-validation in text and static diagrams.
- **Remediation**: Build a dynamic fold stepper where 10 patient cards are partitioned into 5 folds. The learner steps through Fold 1 to Fold 5 and witnesses Patient A placed in the test fold, predicted by a model fitted strictly on Patients B–J.

---

## 4. Prioritized Refactor Roadmap
1. **Foundation Prototype 1**: Lesson 1 (CSV, pandas token-by-token, cohort inspection, target leakage).
2. **Foundation Prototype 2**: Lesson 2 (Linear Regression, $X$ vs. $Y$ discovery, slope/intercept visual slider, residual error bars).
3. **Foundation Prototype 3**: Lesson 3 (Regression Metrics, unified residual deconstruction, common traps).
4. **Validation Milestone**: Lesson 13 (Repeated 5-Fold Cross-Validation, out-of-fold predictions, optimism prevention).
