# Clinical Concepts Explainer — Agent Directives & Repository Rules

This repository contains interactive clinical-prediction and medical-statistics explainer modules. All AI agents working in this workspace must adhere to the following rules, references, and workflows.

---

## 1. Primary References
- **Ponytail Reference Repository**: [https://github.com/DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)
  - Preferred pattern source for state machines, interactive lessons, and clean modular designs.
- **21st.dev UI Standards**: [https://21st.dev](https://21st.dev) & [https://github.com/21st-dev](https://github.com/21st-dev)
  - Rich aesthetics, glassmorphic accents, modern typography, responsive layout, and accessible UI.
- **Context7 Live Documentation**: Consult live docs for backend, framework, and API contracts.

---

## 2. Graphify Knowledge Graph (Mandatory Navigation)
- Ground truth knowledge graph: `graphify-out/graph.json`.
- Query Graphify (`python -m graphify query` or inspect `graphify-out/graph.json`) before modifying code.
- Rule file: [.agents/rules/graphify.md](file:///d:/Desktop/clinical_concepts_explainer_current/.agents/rules/graphify.md).
- Workflow: [.agents/workflows/graphify.md](file:///d:/Desktop/clinical_concepts_explainer_current/.agents/workflows/graphify.md).
- Synchronize: Run `python -m graphify extract . --code-only` after code modifications pass verification.

---

## 3. JEV Rapid Decision-Making & TypeSafe Skill
- **Installed Skill**: [.agents/skills/typesafe-ai/SKILL.md](file:///d:/Desktop/clinical_concepts_explainer_current/.agents/skills/typesafe-ai/SKILL.md).
- **Rule File**: [.agents/rules/jev.md](file:///d:/Desktop/clinical_concepts_explainer_current/.agents/rules/jev.md).
- **Implementation Template**: `clinical-prediction-modeling_9.19.26_USE THIS/clinical-prediction-modeling/templates/method_advisor.py`.
- **Core Principle**: **JEV proposes; the engineer/human decides.**
  - Use JEV (TypeSafe System One method advisor) for rapid, calibrated probabilistic judgments on modeling tradeoffs (validation designs, penalization necessity, spline knots, EPV adequacy, flexibility score).
- **Privacy Boundary**: ONLY aggregate analysis metadata passing `guard_state` allowlist may cross the boundary (`outcome_type`, `n_patients`, `n_events`, `n_predictors`, `n_centers`, `max_missing_rate`, `has_time_varying`, `correlated_predictors`). ZERO raw patient data or PHI.
- **Security**: The API key is stored in the `TYPESAFE_API_KEY` environment variable. Never hardcode or log the key in git history or client bundles.
- **Offline Fallback**: Deterministic rule-based evaluation (`_rule_based_advisor`) runs seamlessly if offline or unauthenticated.

---

## 4. Engineering Instructions & Verification Loop
- **Full Instructions**: [.agents/rules/engineering_instructions.md](file:///d:/Desktop/clinical_concepts_explainer_current/.agents/rules/engineering_instructions.md).
- **Security**: "Never-Get-Hacked" standards, zero secret leakage, input sanitization, client-side only PHI isolation.
- **TDD Workflow**: Test-driven development with 100% green verification (`node --check dist/app.js`, unit/integration tests).
- **7-Phase Protocol**:
  1. **Understand**: Requirements, constraints, ECC skills, consult JEV for rapid calibrated modeling choices.
  2. **Navigate**: Graphify queries, trace call paths and minimal edit surfaces.
  3. **Reference**: Check Ponytail and 21st.dev design patterns.
  4. **TDD**: Write test, establish failure, minimal implementation, pass test.
  5. **Verification**: 100% test suite and build green.
  6. **Synchronize**: Regenerate Graphify index (`graphify extract . --code-only`).
  7. **Final Review**: Plan adherence, change summary, test report.
