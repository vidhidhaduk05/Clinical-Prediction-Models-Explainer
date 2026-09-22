# Implementation Checklist — Pedagogical Refactor

## Phase 1: Governance & Planning
- [x] Create `PEDAGOGY_AUDIT.md` (Audit friction points, gap analysis)
- [x] Create `LEARNING_ARCHITECTURE.md` (17-step discovery sequence, concept graph)
- [x] Create `STATE.md` (System state and component inventory)
- [x] Create `TODO.md` (Track implementation tasks)
- [x] Create `DECISIONS.md` (Architectural decisions and JEV methodology)

## Phase 2: Design & UI Component Infrastructure (`style.css` & `app.js`)
- [x] Implement CSS styles for `.predict-reveal` cards and interactive state
- [x] Implement CSS styles for `.token-code` blocks and interactive token tooltips
- [x] Implement CSS styles for `.why-care` and `.common-trap` callouts
- [x] Implement CSS styles for interactive $X$ vs. $Y$ variable sorting container
- [x] Implement CSS styles for 5-Fold Cross-Validation step cards

## Phase 3: Core Prototype Lessons Refactor
- [x] **Lesson 1 (Start with a CSV)**:
  - [x] Opening clinical question: Acute ischemic stroke triage
  - [x] Everyday analogy: Emergency Department census board
  - [x] Token-by-token Python/pandas breakdown (`import`, `pandas`, `as pd`, `read_csv`, `df.shape`, `df.dtypes`, `df.isna().sum()`)
  - [x] Interactive patient cohort table lab with patient row inspection and missing value highlight
  - [x] Common trap: Data leakage (post-admission complication predicting admission fate)
  - [x] Knowledge check with rationale
  - [x] Narrative bridge to Lesson 2
- [x] **Lesson 2 (Linear Regression)**:
  - [x] Clinical question: Does admission NIHSS predict final infarct volume?
  - [x] Interactive $X$ vs. $Y$ discovery before equation
  - [x] Scatter plot lab: Dots before lines, learner manipulates slope $\beta_1$ and intercept $\beta_0$
  - [x] Drag patient above/below regression line to see residual error bar
  - [x] Equation & symbol breakdown for $Y = \beta_0 + \beta_1 X + \epsilon$
  - [x] "Why do I care?" box: Clinical consequence of slope
  - [x] Knowledge check
  - [x] Narrative bridge to Lesson 3
- [x] **Lesson 3 (Regression Metrics)**:
  - [x] Visual error bars on 4 stroke patients
  - [x] Connect residuals to MAE (absolute lengths)
  - [x] Connect residuals to RMSE (squaring penalizes large mistakes)
  - [x] Connect residuals to Mean Signed Error (bias cancellation trap)
  - [x] Visual $R^2$ variance comparison (destroy "30% accurate" myth)
  - [x] Common trap: $R^2$ is not accuracy
  - [x] Knowledge check
  - [x] Narrative bridge to Lesson 4
- [x] **Lesson 13 (Repeated 5-Fold Cross-Validation)**:
  - [x] Exam analogy: Studying past tests vs. unseen final exam
  - [x] Interactive 5-fold cross-validation stepper with 10 patient cards
  - [x] Out-of-fold prediction revelation (Patient A evaluated only on unseen model)
  - [x] Why apparent $R^2$ shrinks from $0.335$ to $0.077$ without leakage
  - [x] Common trap: Preprocessing before splitting leaks data
  - [x] Knowledge check

## Phase 4: Verification, Synchronization & Deployment
- [x] Run syntax check: `node --check app.js`
- [x] Run headless test across all 24 lesson routes to ensure zero regressions
- [x] Sync changes to `dist/`
- [ ] Git commit and push to `main` and `gh-pages`
- [ ] Verify live GitHub Pages deployment
