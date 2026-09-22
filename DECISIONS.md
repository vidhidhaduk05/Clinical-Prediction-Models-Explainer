# Architectural Decisions Record (ADR) — Clinical Concepts Explainer

## Decision 1: Pedagogical Refactor over Technical Rewrite
- **Context**: The existing application is fast, stable, and completely framework-free, rendering in vanilla JavaScript and CSS. Rebuilding with React, Next.js, or Vue would introduce build complexity, hydrations quirks, and potential regressions without improving instructional clarity.
- **Decision**: Retain the vanilla JS/HTML/CSS architecture. Enhance the pedagogy within `app.js` using reusable micro-templates and interactive SVG state handlers.
- **Consequences**: Zero framework dependencies, instant loading on GitHub Pages, maximum longevity and readability.

## Decision 2: The 17-Step Pedagogical Discovery Sequence
- **Context**: Medical learners were encountering abstract formulas ($R^2$, $\beta_1$, logits) before understanding what physical entity they were measuring.
- **Decision**: Adopt the 17-step discovery sequence from `NEXT_AGENT_HANDOFF_CLINICAL_PREDICTION_EXPLAINER.md`. Every concept starts with a clinical question, guess-before-reveal, visual interaction, plain English explanation, and medical application before introducing algebraic symbols.
- **Consequences**: Dramatically increases conceptual retention while preventing statistical intimidation.

## Decision 3: Token-by-Token Python/pandas Breakdown
- **Context**: Clinical trainees frequently feel alienated by lines like `df = pd.read_csv("file.csv")` or `df.isna().sum()`.
- **Decision**: Provide interactive visual breakdown cards for every single token in the code. Never assume a clinician knows what an alias, method, or attribute is.
- **Consequences**: Clinicians gain genuine programming literacy without needing a separate CS course.

## Decision 4: JEV Method Advisor & Statistical Integrity
- **Context**: Statistical compromises (e.g., calling $R^2$ "accuracy", or treating predictive association as causal effect) undermine clinical credibility.
- **Decision**: Apply JEV method advisor principles: JEV proposes; human/engineer decides. Ensure rigorous distinctions:
  - Predictive association $\neq$ causal effect.
  - $R^2 = 0.30$ does NOT mean "30% accuracy".
  - Out-of-fold cross-validated $R^2$ evaluates generalizability, not in-sample fit.
  - Full-pipeline resampling is strictly maintained to prevent data leakage.
- **Consequences**: Uncompromised scientific integrity suitable for top medical journals and academic teaching.
