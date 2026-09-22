---
trigger: always_on
description: Use JEV (TypeSafe System One method advisor) for rapid, calibrated clinical prediction modeling and architectural decisions.
---
# JEV Rapid Decision-Making Protocol (TypeSafe Method Advisor)

## 1. Role & Objective
- Use **JEV** (`templates/method_advisor.py` / `TypeSafeClient`) to accelerate modeling, validation, and design decisions with typed, calibrated probabilistic proposals.
- Leverages calibrated System One intelligence to evaluate modeling tradeoffs (validation design, penalization, spline knots, EPV adequacy, flexibility score) rapidly.

## 2. Core Operational Boundary
- **JEV proposes; the engineer/human decides.**
- JEV proposals provide rapid prior recommendations and probabilities, but never autonomously decide scientific thresholds, replace empirical data, or substitute for computed statistical validation (such as bootstrap confidence intervals or likelihood-ratio tests).

## 3. Strict Metadata-Only Privacy Boundary (`guard_state`)
- **Zero PHI / Raw Data Egress**: Under no circumstances may patient-level records, individual identifiers, or uncurated tables be sent to JEV.
- Only aggregate analysis metadata passing the strict allowlist in `guard_state` is permitted:
  - `outcome_type` ("binary" | "survival" | "continuous")
  - `n_patients` (integer)
  - `n_events` (integer)
  - `n_predictors` (parameter count, not variable count)
  - `n_centers` (integer)
  - `max_missing_rate` (largest per-variable missing fraction)
  - `has_time_varying` (boolean)
  - `correlated_predictors` (boolean)
- Any attempt to pass unapproved keys or non-numeric metadata raises a `PrivacyViolation`.

## 4. Key Management & "Never-Get-Hacked" Standards
- The API key is read strictly from the `TYPESAFE_API_KEY` environment variable.
- Never hardcode, commit, display, or log the API key in git repositories, documentation, or client-side bundles.

## 5. Offline Resiliency & Fallback
- If `TYPESAFE_API_KEY` is absent or an API/network error occurs, fall back seamlessly to the deterministic `_rule_based_advisor`.
- The rule-based engine computes exact EPV rules and sample-size heuristics (`source="rule_based"`), ensuring offline development is never blocked.

## 6. Decision Trace Logging
- Every call to `advise(state, trace)` must log a `jev_advisory` entry in the decision trace containing:
  - Proposal values (`validation_design`, `penalization_needed`, `spline_knots`, `epv_adequate`)
  - Probability distributions and confidence weights
  - Advisory source (`"jev"` or `"rule_based"`)
  - Final human/agent decision and rationale
