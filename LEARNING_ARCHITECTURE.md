# Learning Architecture — Clinical Concepts Explainer

## 1. Primary Pedagogical Model: The 17-Step Core Discovery Sequence

Every lesson and concept follows this non-negotiable progression:

```text
 1. CLINICAL QUESTION           ("Does admission stroke severity predict final tissue death?")
        ↓
 2. WHY PROBLEM EXISTS          (Why clinical examination alone is insufficient)
        ↓
 3. GUESS BEFORE REVEAL         (Ask learner to predict relationship or sign)
        ↓
 4. INTERACTIVE VISUAL LAB      (Manipulate sliders, dots, knots, or folds)
        ↓
 5. PLAIN-ENGLISH EXPLANATION   (Explain what just moved in everyday language)
        ↓
 6. INTRODUCE FORMAL TERM       ("This vertical distance is called the residual")
        ↓
 7. MEDICAL APPLICATION         (Acute ischemic stroke, thrombectomy, perfusion CT)
        ↓
 8. EQUATION & SYMBOL BREAKDOWN (Term-by-term annotation of every symbol)
        ↓
 9. PYTHON CODE BLOCK           (Reproducible clinical code snippet)
        ↓
10. TOKEN-BY-TOKEN BREAKDOWN    (Granular line-by-line explanation for non-coders)
        ↓
11. SAMPLE OUTPUT DISPLAY       (Terminal or DataFrame printout)
        ↓
12. PLAIN-ENGLISH OUTPUT INTERPRETATION ("Our model underpredicted tissue death by 18 mL")
        ↓
13. "WHY DO I CARE?" CALLOUT    (Clinical implications for patient outcomes)
        ↓
14. COMMON TRAP / MISCONCEPTION (Prevent dangerous clinical or statistical errors)
        ↓
15. KNOWLEDGE CHECK QUIZ        (Immediate active recall with explanatory feedback)
        ↓
16. "GO DEEPER" EXPANSION       (Optional mathematical rigor for advanced learners)
        ↓
17. NARRATIVE BRIDGE            (Why the next lesson is logically necessary)
```

---

## 2. Concept Dependency Graph

```mermaid
graph TD
    CSV[Lesson 1: Start with a CSV] --> LinReg[Lesson 2: Linear Regression]
    LinReg --> Metrics[Lesson 3: Regression Metrics]
    Metrics --> Causation[Lesson 4: Association vs. Prediction vs. Causation]
    Causation --> Logistic[Lesson 5: Binary Outcomes & Logistic Regression]
    Logistic --> Ordinal[Lesson 6: Ordinal Outcomes & mRS]
    Ordinal --> Survival[Lesson 7: Time-to-Event & Survival Analysis]
    
    CSV --> Missing[Lesson 8: Missing Data & Imputation]
    LinReg --> Splines[Lesson 9: Restricted Cubic Splines]
    LinReg --> Pheno[Lesson 10: Residual Phenotyping]
    
    Metrics --> CV[Lesson 13: Repeated 5-Fold Cross-Validation]
    CV --> Boot[Lesson 14: Bootstrap & Optimism Correction]
    Logistic --> AUC[Lesson 15: AUC & Discrimination]
    Logistic --> Calib[Lesson 16: Calibration & Brier Score]
    Calib --> DCA[Lesson 17: Decision Curve Analysis]
    
    Splines --> Penal[Lesson 19: Penalization & Shrinkage]
    CV --> ML[Lesson 20: Machine Learning Benchmarking]
    ML --> Interp[Lesson 21: Interpretability & SHAP]
    DCA --> TRIPOD[Lesson 23: TRIPOD+AI Reporting]
    TRIPOD --> Anti[Lesson 24: Anti-Patterns Catalog]
```

---

## 3. Teaching Modes: Learn Mode vs. Live Demo Mode

| Feature | **Learn Mode (Default)** | **Live Demo Mode (Presentation)** |
| :--- | :--- | :--- |
| **Target Audience** | Solo medical student / researcher learning from zero | Clinician presenting findings on rounds / screenshare |
| **Content Depth** | Everyday analogies, guess-before-reveal, code tokens, quizzes | Research question, clean code, output, one-sentence takeaway |
| **Pacing** | Self-directed, interactive discovery, micro-steps | High-level, streamlined, speaker notes enabled |
| **State Storage** | `localStorage` tracks quiz completions and widget progress | Zero distraction, presentation-ready layout |

---

## 4. Reusable Pedagogical Component Blueprints

1. **`PredictBeforeReveal`**:
   - Presents a multiple-choice hypothesis prompt.
   - Disables options upon selection and transitions smoothly into the explanation with a colored feedback pill.
2. **`TokenCodeBlock`**:
   - Syntax-highlighted code container.
   - Clicking or hovering over tokens (e.g., `import`, `pd`, `read_csv`) reveals a tooltip/card explaining what that specific token does.
3. **`WhyCareBox`**:
   - Editorial card with an anchor icon and callout emphasizing why the statistical metric directly alters clinical prognosis or triage.
4. **`CommonTrapCard`**:
   - Two-column comparative card: "⚠️ Common Misconception" vs. "✅ Correct Clinical Understanding".
5. **`KnowledgeCheck`**:
   - Dynamic assessment with instant evaluation, randomized distractors, and clinical rationale.
