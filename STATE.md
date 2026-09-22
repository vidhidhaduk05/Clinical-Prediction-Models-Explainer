# System State — Clinical Prediction Models Explainer

## 1. Environment & Architecture
- **Runtime**: Pure client-side static web application (HTML5, Vanilla CSS, ES6+ JavaScript, SVG, KaTeX).
- **Hosting / Deploy**: GitHub Pages (`https://vidhidhaduk05.github.io/Clinical-Prediction-Models-Explainer/`).
- **Dependencies**: KaTeX (CDN: `katex.min.js`, `katex.min.css`) for display & inline LaTeX formulas.
- **Repository Root**: `d:\Desktop\clinical_concepts_explainer_current`
- **Dist Mirror**: `d:\Desktop\clinical_concepts_explainer_current\dist`

---

## 2. Component Inventory
- `index.html`: Broadsheet header, side drawer navigation, main reading column, KaTeX CDN links.
- `style.css`: 21st.dev design tokens, typography, broadsheet newspaper layout, interactive lab containers, quiz cards.
- `app.js`: Core router, state store, lesson content templates, SVG renderers, interactive lab handlers.
- `math.js`: Pure mathematical and biostatistical computation engine (OLS, RCS, AUC, DCA, repeated CV, bootstrap).

---

## 3. Current Working Capabilities
- ✅ All 24 routes navigable via hash routing (`#csv`, `#linear`, `#metrics`, ..., `#antipatterns`).
- ✅ KaTeX math formulas render correctly with proper display and inline delimiters.
- ✅ Unpublished manuscripts and private patient references completely eliminated.
- ✅ GitHub Pages deployment fully operational.
