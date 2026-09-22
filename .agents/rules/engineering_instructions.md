---
trigger: always_on
description: Antigravity Engineering Instructions including Ponytail reference, Graphify-first navigation, ECC skills, TDD, and 7-phase execution protocol.
---
# Antigravity Engineering Instructions
## 1. Mandatory Repository Reference — Ponytail
- **Ponytail Repository**: https://github.com/DietrichGebert/ponytail
- Use as the preferred reference implementation and pattern source when relevant.
- Adapt patterns to current codebase architecture; do not blindly copy code.
- Current project specifications and tests remain authoritative.
## 2. Navigation First — Graphify Is Mandatory
- Ground truth knowledge graph: `graphify-out/graph.json`.
- Before modifying any code, always query Graphify (`python -m graphify query` or `graphify-out/graph.json`) first.
- Trace dependencies, callers, callees, data flow, and integration boundaries.
- Never begin implementation by blindly grepping the repository.
## 3. Mandatory Design Standard — 21st.dev
- **Reference**: https://21st.dev & https://github.com/21st-dev
- Use 21st.dev component patterns, design tokens, micro-animations, modern glassmorphic accents, and curated accessible UI building blocks.
- Avoid basic, generic, or unstyled UI. Ensure rich aesthetics, responsive layouts, crisp typography, and high visual craft.
## 4. Backend Architecture & Documentation — Context7
- **Reference**: Context7 live documentation & backend patterns (`backend-patterns`, `documentation-lookup`, `api-design`).
- When writing backend services, API routes, data stores, or external integrations, leverage up-to-date documentation and robust patterns from Context7 rather than stale training data.
- Enforce strict typing, schema validation, rate-limiting, error boundaries, and connection timeouts.
## 5. Security & Hardening — "Never-Get-Hacked" Standards
- **Reference**: Never-Get-Hacked security guidelines (`security-review`, `accidental-data-loss-prevention`).
- Zero secret leakage: Never commit or hardcode API keys, tokens, or credentials in client bundles or git history.
- Clinical Data Privacy & PHI Protection:
  - Zero unconsented data egress: Process sensitive data client-side/in-browser where required.
  - Client-side storage of user keys must be strictly isolated to user's `localStorage` and never proxied through unauthenticated remote endpoints.
- Input sanitization & injection defense: sanitize all user inputs, validate file uploads, prevent XSS in markdown rendering, and enforce safe JSON parsing.
- Principle of least privilege: strictly scope permissions, network calls, and local storage access.
## 6. Rapid Decision-Making — JEV Method Advisor
- **Reference**: JEV (TypeSafe System One method advisor) in `templates/method_advisor.py`.
- Use JEV to accelerate modeling, validation design, and architecture decisions with typed, calibrated probabilistic proposals.
- **Boundary**: JEV proposes; human/engineer decides. It never autonomously sets statistical thresholds or overrides computed results.
- **Privacy**: Pass state through `guard_state` allowlist only. Transmit strictly aggregate metadata (`outcome_type`, `n_patients`, `n_events`, `n_predictors`, `n_centers`, `max_missing_rate`, `has_time_varying`, `correlated_predictors`). ZERO patient data.
- **Credentials**: Reads `TYPESAFE_API_KEY` from environment. Never commit or log API keys.
- **Offline Fallback**: Automatically reverts to deterministic rule-based evaluation (`_rule_based_advisor`) if offline or unauthenticated.
- **Trace Logging**: Record proposals, calibrated probabilities, and chosen rationale in the decision trace.
## 7. ECC Skills Mandatory
- Load and apply relevant ECC skills (`orch-add-feature`, `tdd-workflow`, `backend-patterns`/`frontend-patterns`, `security-review`, `verification-loop`) before coding.
## 8. TDD Mandatory
- Understand existing behavior -> Write/modify tests -> Establish failure -> Implement minimal change -> Green -> Refactor -> Broad verification.
## 9. Implementation Rules
- Read implementation plan completely, trace via Graphify, consult JEV for rapid design tradeoffs, check Ponytail patterns, make smallest correct implementation, preserve APIs and compatibility.
## 10. Verification Loop
- Unit/integration tests -> Full test suite (`pytest`, `npm run build`, linting/typechecks) -> 100% green before declaring completion.
## 11. Graphify Synchronization Mandatory
- Regenerate index after all code changes and passing verification: `python -m graphify extract . --code-only`.
## 12. Required Execution Order (7 Phases)
1. **Understand**: Plan, constraints, ECC skills, and consult JEV method advisor for rapid calibrated decisions.
2. **Navigate**: Graphify query, trace call paths, minimal edit surface.
3. **Reference**: Inspect Ponytail, adapt relevant patterns.
4. **TDD**: Write test, establish failure, minimal implementation, pass test.
5. **Verification**: Full suite & build green (100%).
6. **Synchronize**: Regenerate Graphify graph.
7. **Final Review**: Plan adherence, change summary, test report.
