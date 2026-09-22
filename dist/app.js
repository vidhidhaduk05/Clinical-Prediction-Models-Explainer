/**
 * Clinical Concepts Explainer — Interactive Medical Statistics & Prediction Modeling
 * Inspired by MLU Explain, built for stroke prediction & clinical research.
 */
const mean = a => a.reduce((s,x)=>s+x,0)/a.length;
const sum = a => a.reduce((s,x)=>s+x,0);
const clamp = (x,a,b) => Math.max(a,Math.min(b,x));
const sigmoid = x => 1/(1+Math.exp(-x));
const logit = p => Math.log(clamp(p,1e-9,1-1e-9)/(1-clamp(p,1e-9,1-1e-9)));
function rng(seed=42){let a=seed>>>0;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296;};}
function shuffle(a,random){const b=[...a];for(let i=b.length-1;i>0;i--){let j=Math.floor(random()*(i+1));[b[i],b[j]]=[b[j],b[i]];}return b;}
function quantile(a,p){const b=[...a].sort((x,y)=>x-y);const z=(b.length-1)*p,i=Math.floor(z);return b[i]+(b[Math.min(i+1,b.length-1)]-b[i])*(z-i);}
function metrics(y,p){const e=p.map((v,i)=>v-y[i]),ss=sum(y.map(v=>(v-mean(y))**2)),mse=mean(e.map(v=>v*v));return{mae:mean(e.map(Math.abs)),rmse:Math.sqrt(mse),signed:mean(e),r2:ss>0?1-sum(e.map(v=>v*v))/ss:NaN};}
function ols(x,y){const mx=mean(x),my=mean(y);const den=sum(x.map(v=>(v-mx)**2));const b=den?sum(x.map((v,i)=>(v-mx)*(y[i]-my)))/den:0;return{a:my-b*mx,b};}
function predict(model,x){return x.map(v=>model.a+model.b*v);}
function solve(A,b){const m=A.map((row,i)=>[...row,b[i]]),n=b.length;for(let i=0;i<n;i++){let q=i;for(let j=i+1;j<n;j++)if(Math.abs(m[j][i])>Math.abs(m[q][i]))q=j;[m[i],m[q]]=[m[q],m[i]];if(Math.abs(m[i][i])<1e-12)return Array(n).fill(0);const v=m[i][i];for(let k=i;k<=n;k++)m[i][k]/=v;for(let j=0;j<n;j++)if(j!==i){const c=m[j][i];for(let k=i;k<=n;k++)m[j][k]-=c*m[i][k];}}return m.map(row=>row[n]);}
function fitBasis(X,y,penalty=1e-8){const k=X[0].length,A=Array.from({length:k},()=>Array(k).fill(0)),b=Array(k).fill(0);X.forEach((row,i)=>{for(let j=0;j<k;j++){b[j]+=row[j]*y[i];for(let l=0;l<k;l++)A[j][l]+=row[j]*row[l];}});for(let j=1;j<k;j++)A[j][j]+=penalty;return solve(A,b);}
function rcs(x,knots=[1,4,7,9]){const k=knots.length,last=knots[k-1],prev=knots[k-2],cube=v=>Math.max(0,v)**3;return [1,x,...knots.slice(0,-2).map(t=>(cube(x-t)-cube(x-prev)*(last-t)/(last-prev)+cube(x-last)*(prev-t)/(last-prev))/(last-knots[0])**2)];}
const dot=(a,b)=>sum(a.map((v,i)=>v*b[i]));
function auc(y,p){const yes=p.filter((_,i)=>y[i]===1),no=p.filter((_,i)=>y[i]===0);if(!yes.length||!no.length)return NaN;return sum(yes.map(a=>sum(no.map(b=>a>b?1:a===b?.5:0))))/(yes.length*no.length);}
function confusion(y,p,t){let tp=0,fp=0,tn=0,fn=0;y.forEach((v,i)=>{if(p[i]>=t){v?tp++:fp++;}else{v?fn++:tn++;}});return{tp,fp,tn,fn,sensitivity:tp/(tp+fn),specificity:tn/(tn+fp),ppv:tp/(tp+fp)};}
const brier=(y,p)=>mean(p.map((v,i)=>(v-y[i])**2));
function netBenefit(y,p,t){const c=confusion(y,p,t);return c.tp/y.length-c.fp/y.length*t/(1-t);}
function bh(ps,q=.05){const order=ps.map((p,i)=>({p,i})).sort((a,b)=>a.p-b.p),m=ps.length;let cutoff=-1;order.forEach((x,j)=>{if(x.p<=(j+1)/m*q)cutoff=j;});return order.map((x,j)=>({...x,rank:j+1,threshold:(j+1)/m*q,reject:j<=cutoff}));}
function rubin(estimates,variances){const m=estimates.length,W=mean(variances),Q=mean(estimates),B=sum(estimates.map(v=>(v-Q)**2))/(m-1),T=W+(1+1/m)*B;return{Q,W,B,T,se:Math.sqrt(T)};}
function repeatedCV(x,y,k=5,repeats=20,seed=42){const random=rng(seed),all=Array(x.length).fill(0),repMetrics=[];for(let r=0;r<repeats;r++){const idx=shuffle(x.map((_,i)=>i),random),p=Array(x.length);for(let f=0;f<k;f++){const test=idx.filter((_,j)=>j%k===f),train=idx.filter((_,j)=>j%k!==f),fit=ols(train.map(i=>x[i]),train.map(i=>y[i]));test.forEach(i=>p[i]=fit.a+fit.b*x[i]);}p.forEach((v,i)=>all[i]+=v/repeats);repMetrics.push(metrics(y,p));}return{predictions:all,pooled:metrics(y,all),repetitionMean:mean(repMetrics.map(m=>m.r2)),repMetrics};}
function bootstrapIndices(n,random){const drawn=Array.from({length:n},()=>Math.floor(random()*n)),seen=new Set(drawn);return{drawn,oob:Array.from({length:n},(_,i)=>i).filter(i=>!seen.has(i))};}
function km(time,event){const unique=[...new Set(time.filter((_,i)=>event[i]))].sort((a,b)=>a-b);let s=1;const points=[[0,1]];unique.forEach(t=>{const risk=time.filter(x=>x>=t).length,d=time.filter((x,i)=>x===t&&event[i]).length;points.push([t,s]);s*=1-d/risk;points.push([t,s]);});points.push([Math.max(...time),s]);return points;}
const toyX=[2,3,5,6,8,9,11,13,15,18,21,24];
const toyY=[8,20,14,28,25,45,38,56,48,83,66,108];
const binaryY=[0,0,1,0,0,1,0,1,0,1,1,1];
const binaryP=[.08,.12,.19,.23,.3,.38,.43,.56,.63,.71,.82,.91];

/* ==========================================================================
   Curriculum Data & Lessons Definitions
   ========================================================================== */

const CURRICULUM_GROUPS = [
  {
    id: "start",
    title: "Start Here",
    lessons: [
      { id: "csv", num: 1, title: "Start with a CSV", subtitle: "Your first clinical data table" },
      { id: "linear", num: 2, title: "Linear Regression", subtitle: "Predicting continuous stroke outcomes" },
      { id: "metrics", num: 3, title: "Regression Metrics", subtitle: "Evaluating errors like a weather forecaster" }
    ]
  },
  {
    id: "fundamentals",
    title: "Clinical Prediction Fundamentals",
    lessons: [
      { id: "causation", num: 4, title: "Association, Prediction, Causation", subtitle: "Three distinct clinical questions" },
      { id: "logistic", num: 5, title: "Binary Outcomes & Logistic Regression", subtitle: "Probabilities, odds, and risks" },
      { id: "ordinal", num: 6, title: "Ordinal Outcomes & Proportional Odds", subtitle: "The modified Rankin Scale (mRS)" },
      { id: "survival", num: 7, title: "Time-to-Event & Survival Analysis", subtitle: "Censoring, Kaplan-Meier, and Cox models" }
    ]
  },
  {
    id: "data-modeling",
    title: "Data Problems & Flexible Modeling",
    lessons: [
      { id: "missing", num: 8, title: "Missing Data & Multiple Imputation", subtitle: "MCAR, MAR, MNAR, and MICE in validation" },
      { id: "splines", num: 9, title: "Restricted Cubic Splines", subtitle: "Relaxing the straight-line assumption" },
      { id: "phenotyping", num: 10, title: "Residual Phenotyping", subtitle: "Greater-than-expected progression" },
      { id: "interactions", num: 11, title: "Interactions & Marginal Standardization", subtitle: "Adjusted absolute risk differences" },
      { id: "features", num: 12, title: "Feature Engineering & Correlation", subtitle: "Handling collinear perfusion imaging" }
    ]
  },
  {
    id: "validation-perf",
    title: "Validation & Performance",
    lessons: [
      { id: "validation", num: 13, title: "Repeated k-Fold Cross-Validation", subtitle: "Preventing data leakage and fold instability" },
      { id: "bootstrap", num: 14, title: "Bootstrap & Out-of-Bag Evaluation", subtitle: "Full-pipeline resampling with percentile CIs" },
      { id: "auc", num: 15, title: "AUC & Discrimination", subtitle: "Ranking patients without confusing calibration" },
      { id: "calibration", num: 16, title: "Calibration & Brier Score", subtitle: "Are the estimated probabilities trustworthy?" },
      { id: "dca", num: 17, title: "Decision Curve Analysis (DCA)", subtitle: "Clinical usefulness across threshold probabilities" },
      { id: "continuous", num: 18, title: "Continuous Outcomes Comparison", subtitle: "OOB R², ANCOVA, and baseline adjustments" }
    ]
  },
  {
    id: "stability-reporting",
    title: "Model Stability & Reporting",
    lessons: [
      { id: "penalization", num: 19, title: "Penalization (Ridge, Lasso, Elastic Net)", subtitle: "EPV budgeting and shrinkage over stepwise" },
      { id: "ml", num: 20, title: "Machine Learning Benchmarking", subtitle: "Nested CV and paired model comparisons" },
      { id: "interpretability", num: 21, title: "Interpretability & Explainability", subtitle: "SHAP, PDP, and avoiding causal illusions" },
      { id: "updating", num: 22, title: "Model Updating & Transportability", subtitle: "From intercept recalibration to full refitting" },
      { id: "tripod", num: 23, title: "TRIPOD+AI & Decision Trace", subtitle: "27-item checklist and reproducible audit trails" },
      { id: "antipatterns", num: 24, title: "Anti-Patterns Catalog", subtitle: "Interactive spot-the-leak challenge" }
    ]
  }
];

// Flat lookup map of lessons
const LESSONS_MAP = {};
CURRICULUM_GROUPS.forEach(g => {
  g.lessons.forEach(l => {
    LESSONS_MAP[l.id] = { ...l, group: g.title };
  });
});

/* ==========================================================================
   Lesson Content Builders
   ========================================================================== */

const LESSON_CONTENT = {
  csv: {
    eyebrow: "Module 1 · Lesson 1 of 24",
    h1: "Start with a CSV: Your Clinical Data Table",
    lead: "Before machine learning algorithms or statistical models, every clinical prediction study begins with a patient cohort table. Understanding rows, columns, and data structures is your foundation.",
    analogy: {
      title: "The Emergency Department Census Board",
      text: "Think of a CSV file as the morning ED census board. Each row is a unique patient in a treatment bay. Each column is a specific vital sign, laboratory value, or imaging biomarker recorded on arrival. If a lab was not drawn, that cell is empty (missing). You cannot treat the patient without knowing who is in each bed and what data was collected."
    },
    sections: [
      {
        label: "1. The Clinical Data Anatomy: Rows and Columns",
        html: `
          <p>In clinical predictive modeling, your dataset has a strict mathematical architecture:</p>
          <div class="metric-grid">
            <div class="metric">
              <span>Rows (n = Sample Size)</span>
              <b>626 Patients</b>
              <p class="subtle">Each row is an independent acute ischemic stroke patient.</p>
            </div>
            <div class="metric">
              <span>Columns (p = Candidate Predictors)</span>
              <b>14 Clinical Variables</b>
              <p class="subtle">Demographics, CT Perfusion metrics, and 90-day functional recovery.</p>
            </div>
          </div>
          <div class="patient-calc">
            <h4>The Three Critical Variable Roles:</h4>
            <ul>
              <li><strong>The Outcome ($Y$):</strong> What we want to predict (e.g., Final Infarct Volume in mL, or 90-day modified Rankin Scale $\\\\le 2$).</li>
              <li><strong>The Candidate Predictors ($X$):</strong> Clinical and imaging measurements available *before* the treatment decision (e.g., Baseline Core, CVO score, Age, NIHSS).</li>
              <li><strong>The Treatment & Confounders ($Z$):</strong> Reperfusion status (TICI 2b/3 vs. 0-2a), onset-to-groin puncture time, and hospital center.</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. Inspecting the Cohort with Python & pandas",
        html: `
          <p>Here is the exact Python code used to load and audit the acute stroke registry:</p>
          <pre><code>import pandas as pd

# 1. Load the admission stroke cohort
df = pd.read_csv("stroke_perfusion_cohort.csv")

# 2. Dimensions: (patients, variables)
print(df.shape)  # (626, 14) -> 626 patients, 14 variables

# 3. Inspect columns and data types
print(df.dtypes)

# 4. Count missing clinical entries
print(df.isna().sum())</code></pre>
          <p class="code-caption"><code>df.shape</code> tells you your sample size immediately. In clinical modeling, sample size governs how many parameters your model can safely estimate without overfitting.</p>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Every predictive study begins with a structured patient cohort where rows represent independent clinical encounters and columns represent strictly pre-treatment candidate predictors, ensuring that zero post-treatment recovery data leaks into baseline risk estimation."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>You are building a model at hospital admission to predict 90-day stroke disability. Which variable is a <strong>forbidden predictor</strong> that introduces fatal data leakage?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Baseline NIHSS on arrival (Score 16)</button>
              <button class="quiz-opt" data-correct="false">B) Acute CTP Ischemic Core Volume (32 mL)</button>
              <button class="quiz-opt" data-correct="true">C) Day 7 Symptomatic Intracranial Hemorrhage (sICH)</button>
              <button class="quiz-opt" data-correct="false">D) Patient Age (72 years)</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Day 7 sICH occurs <em>after</em> admission treatment decisions have already been made! Including post-baseline events as candidate predictors is a classic form of <strong>data leakage</strong> that makes a model unusable at triage.
            </div>
          </div>
        `
      }
    ],
    lab: renderCsvLab
  },

  linear: {
    eyebrow: "Module 1 · Lesson 2 of 24",
    h1: "Linear Regression: Modeling Continuous Stroke Outcomes",
    lead: "Can baseline CT Perfusion help predict final infarct volume? Linear regression finds the optimal hyper-plane through your patient cohort.",
    analogy: {
      title: "The Custom Tailor's Ruler",
      text: "Imagine a tailor estimating jacket arm length based on customer height and shoulder width. A straight ruler gives an approximate rule-of-thumb: for every inch taller, sleeve length increases by a fraction. Some people have longer arms (positive residual), others shorter (negative residual). The ruler minimizes the total fabric wasted across all customers."
    },
    sections: [
      {
        label: "1. The Benchmark Equation from the CVO Manuscript",
        html: `
          <div class="formula-card">
            <div class="formula-card__caption">Core Linear Prediction Equation</div>
            <div class="formula-card__equation">
              FIV = 60.73 + 0.959 × Core - 8.69 × CVO + ε
            </div>
            <div class="formula-card__caption">Term-by-Term Clinical Breakdown</div>
            <div class="terms-grid">
              <div class="term-item"><code>FIV</code> <strong>Outcome:</strong> Final Infarct Volume (in mL) measured on follow-up imaging.</div>
              <div class="term-item"><code>60.73</code> <strong>Intercept (β₀):</strong> Baseline expected infarct volume when core = 0 and CVO = 0.</div>
              <div class="term-item"><code>+0.959 × Core</code> <strong>Core Effect (β₁):</strong> Each 1 mL of baseline CTP ischemic core increases final infarct volume by 0.959 mL.</div>
              <div class="term-item"><code>-8.69 × CVO</code> <strong>Venous Buffer (β₂):</strong> Each 1-point increase in Cortical Venous Outflow (0–6) spares 8.69 mL of brain tissue.</div>
              <div class="term-item"><code>ε</code> <strong>Residual Error:</strong> Unmodeled patient variation (collateral status, recanalization timing).</div>
            </div>
          </div>
        `
      },
      {
        label: "2. Patient Walkthrough: Patient A vs. Patient B",
        html: `
          <p>Let's calculate two patients arriving at the emergency department with the <em>exact same</em> ischemic core volume of 30 mL:</p>
          <div class="metric-grid">
            <div class="metric">
              <span>Patient A (Poor CVO = 2)</span>
              <b>72.1 mL FIV</b>
              <p class="subtle">60.73 + 0.959(30) - 8.69(2) = 72.12 mL</p>
            </div>
            <div class="metric">
              <span>Patient B (Good CVO = 6)</span>
              <b>37.4 mL FIV</b>
              <p class="subtle">60.73 + 0.959(30) - 8.69(6) = 37.36 mL</p>
            </div>
          </div>
          <p><strong>Clinical Insight:</strong> Despite having identical 30 mL core lesions on initial perfusion imaging, Patient B preserves <strong>34.7 mL of viable brain tissue</strong> purely due to robust venous outflow.</p>
        `
      },
      {
        label: "3. The 5 Core Regression Assumptions",
        html: `
          <p>Before trusting linear regression predictions in clinical research, verify the 5 diagnostic assumptions:</p>
          <div class="assumption-grid">
            <div class="assumption-card">
              <h5>1. Linearity</h5>
              <p><span class="good">Expected:</span> Residuals scatter evenly above and below zero across fitted values.<br><span class="bad">Violation:</span> U-shaped curve indicates non-linear core thresholds (requires restricted cubic splines).</p>
            </div>
            <div class="assumption-card">
              <h5>2. Independence</h5>
              <p><span class="good">Expected:</span> Each row is an independent stroke patient.<br><span class="bad">Violation:</span> Multiple scans from the same patient or hospital clustering without mixed models.</p>
            </div>
            <div class="assumption-card">
              <h5>3. Homoscedasticity</h5>
              <p><span class="good">Expected:</span> Constant error variance across all lesion sizes.<br><span class="bad">Violation:</span> Funnel shape! Prediction errors expand dramatically for massive ischemic cores.</p>
            </div>
            <div class="assumption-card">
              <h5>4. Residual Normality</h5>
              <p><span class="good">Expected:</span> Bell-shaped distribution around 0 on Q-Q plots.<br><span class="bad">Violation:</span> Severe skewness inflates type I error for coefficient p-values.</p>
            </div>
          </div>
          <div class="assumption-card" style="margin-top:10px;">
            <h5>5. Outliers and Influential Points</h5>
            <p><span class="good">Expected:</span> No single patient pulls the regression slope like a seesaw.<br><span class="bad">Violation:</span> High leverage cases (Cook's distance > 4/n). Always report sensitivity analyses with and without extreme outliers.</p>
          </div>
        `
      },
      {
        label: "4. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Holding baseline ischemic core volume constant, each 1-point increase in cortical venous outflow score is associated with an 8.69 mL reduction in final infarct volume, demonstrating that venous drainage acts as an independent biological buffer against lesion expansion."</p>
          </div>
        `
      },
      {
        label: "5. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A patient presents with an ischemic core of 40 mL and a CVO score of 5. What is their predicted Final Infarct Volume using the model equation?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) 99.1 mL</button>
              <button class="quiz-opt" data-correct="true">B) 55.6 mL (Correct! 60.73 + 0.959(40) - 8.69(5) = 55.64 mL)</button>
              <button class="quiz-opt" data-correct="false">C) 14.3 mL</button>
              <button class="quiz-opt" data-correct="false">D) 120.5 mL</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Plug into the formula: 60.73 + 0.959(40) - 8.69(5) = 60.73 + 38.36 - 43.45 = <strong>55.64 mL</strong>. Notice that venous outflow buffered away over 43 mL of potential tissue loss!
            </div>
          </div>
        `
      }
    ],
    lab: renderLinearLab
  },

  metrics: {
    eyebrow: "Module 1 · Lesson 3 of 24",
    h1: "Regression Metrics: Quantifying Prediction Errors",
    lead: "How good is a model? A single number is never enough. We evaluate R², MAE, RMSE, and Mean Signed Error to understand accuracy and directional bias.",
    analogy: {
      title: "The Weather Forecast Analogy",
      text: "If a weather forecaster predicts 70°F and it turns out to be 74°F, the error is +4°F. If tomorrow they predict 60°F and it is 56°F, the error is -4°F. Their average signed error is 0°F (unbiased!), but they missed both days by 4°F (Mean Absolute Error = 4°F). If missing by 10°F ruins a wedding, we penalize large errors heavily using Root Mean Squared Error (RMSE)."
    },
    sections: [
      {
        label: "1. The Four Continuous Metrics in the Stroke Paper",
        html: `
          <p>The CVO recalibration manuscript evaluated models using four complementary metrics:</p>
          <div class="formula-card">
            <div class="formula-card__caption">Continuous Error & Fit Formulations</div>
            <div class="formula-card__equation">
              MAE = \frac{1}{n}\sum |y_i - \hat{y}_i| \qquad RMSE = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2} \qquad R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
            </div>
            <div class="formula-card__caption">Penalty Mechanism & Clinical Behavior</div>
            <div class="terms-grid">
              <div class="term-item"><code>y_i - \hat{y}_i</code> <strong>Residual Error:</strong> True observed lesion volume minus model predicted volume.</div>
              <div class="term-item"><code>MAE</code> <strong>Linear Penalty:</strong> Average magnitude of clinical misses in original physical units (mm³ or mL).</div>
              <div class="term-item"><code>RMSE</code> <strong>Quadratic Penalty:</strong> Squares errors before rooting; heavily penalizes catastrophic misses.</div>
              <div class="term-item"><code>R^2</code> <strong>Explained Variance:</strong> Proportion of variance explained compared to a naive model predicting cohort mean $\bar{y}$.</div>
            </div>
          </div>
          <div class="metric-grid">
            <div class="metric">
              <span>Mean Absolute Error (MAE)</span>
              <b>20,021 mm³</b>
              <p class="subtle">Average absolute difference between predicted and actual lesion volume (~20 mL).</p>
            </div>
            <div class="metric">
              <span>Root Mean Squared Error (RMSE)</span>
              <b>50,093 mm³</b>
              <p class="subtle">Quadratic penalty that exposes catastrophic outliers (~50 mL).</p>
            </div>
            <div class="metric">
              <span>Mean Signed Error</span>
              <b>+12,900 mm³</b>
              <p class="subtle">Positive value indicates the model systematically over-predicts infarct volume by ~13 mL.</p>
            </div>
            <div class="metric">
              <span>Cross-Validated R²</span>
              <b>0.077 (Out-of-Bag)</b>
              <p class="subtle">Apparent R² was 0.335! Internal CV revealed true generalizability was much lower.</p>
            </div>
          </div>
        `
      },
      {
        label: "2. Why Cross-Validated R² Can Be Negative",
        html: `
          <div class="patient-calc">
            <h4>The Great $R^2$ Shock:</h4>
            <p>In standard textbook training sets, $R^2$ is bounded between 0 and 1. But on held-out test folds, <strong>cross-validated $R^2$ can easily be negative</strong>!</p>
            <div class="formula">
              R² = 1 - [ SS_residual / SS_total ] = 1 - [ ∑(y - ŷ)² / ∑(y - ȳ)² ]
            </div>
            <p>If your model's predictions $\\\\hat{y}$ on new patients perform <em>worse</em> than simply guessing the dataset average $\\\\bar{y}$ for every patient, then $SS_{\\\\text{residual}} > SS_{\\\\text{total}}$, driving $R^2 < 0$. A negative $R^2$ is a mathematical alarm that your model has overfitted and is harming predictive accuracy.</p>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"While MAE measures typical bedside prediction accuracy, RMSE reveals vulnerability to massive clinical misses, Mean Signed Error exposes systematic volume overestimation, and out-of-bag R² protects against the dangerous optimism of apparent performance."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A stroke infarct prediction model reports a Mean Signed Error of <strong>+18.4 mL</strong>. What does this mean for a clinician using the model at the bedside?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) The model has an error rate of 18.4%.</button>
              <button class="quiz-opt" data-correct="true">B) The model systematically overestimates final infarct volume by an average of 18.4 mL.</button>
              <button class="quiz-opt" data-correct="false">C) The model has an $R^2$ of 0.184.</button>
              <button class="quiz-opt" data-correct="false">D) The model underestimates tissue loss.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Signed error preserves the direction: $\\\\text{Predicted} - \\\\text{Actual}$. A positive signed error (+18.4 mL) means predictions are systematically higher than reality, leading to overly pessimistic prognostication.
            </div>
          </div>
        `
      }
    ],
    lab: renderMetricsLab
  },

  causation: {
    eyebrow: "Module 2 · Lesson 4 of 24",
    h1: "Association, Prediction, and Causation",
    lead: "Three distinct clinical questions often conflated by researchers. Knowing the difference protects patients from misguided interventions.",
    analogy: {
      title: "Roosters, Barometers, and Sunrise",
      text: "A rooster crows right before sunrise (<strong>Association</strong>). A falling barometer accurately forecasts an incoming tempest (<strong>Prediction</strong>). But neither causes the sun to rise nor the storm to strike. Turning the barometer dial will not stop the hurricane (<strong>Causation</strong>). In clinical medicine, a biomarker can accurately forecast stroke death without being a viable therapeutic target."
    },
    sections: [
      {
        label: "1. The Three Distinct Scientific Questions",
        html: `
          <div class="paper-grid">
            <div class="paper-card">
              <div class="paper-id">QUESTION 1: ASSOCIATION</div>
              <h2>"Are X and Y correlated?"</h2>
              <p>Evaluates whether a biomarker (e.g., poor cortical venous outflow) is statistically correlated with 90-day disability (mRS 3–6) after adjusting for confounders.</p>
              <dl>
                <dt>Primary Metric</dt><dd>Adjusted Odds Ratio (aOR = 0.64, p < 0.001)</dd>
                <dt>Standard Tool</dt><dd>Multivariable regression with Rubin-pooled inference</dd>
              </dl>
            </div>
            <div class="paper-card">
              <div class="paper-id">QUESTION 2: PREDICTION</div>
              <h2>"Can X rank future patients?"</h2>
              <p>Evaluates whether adding venous outflow to standard admission variables actually improves out-of-sample risk stratification for a new individual patient.</p>
              <dl>
                <dt>Primary Metric</dt><dd>Cross-validated ΔAUC (+0.02) & Net Benefit</dd>
                <dt>Standard Tool</dt><dd>Repeated k-fold CV or bootstrap optimism correction</dd>
              </dl>
            </div>
          </div>
          <div class="patient-calc" style="margin-top:15px;">
            <h4>QUESTION 3: CAUSATION — "Will intervening on X improve outcomes?"</h4>
            <p>Requires randomized controlled trials (RCTs) or formal counterfactual causal inference (DAGs, instrumental variables, target trial emulation). Intervening to improve venous drainage (e.g., venous stenting) is only therapeutic if poor CVO was causal rather than an innocent bystander of microvascular collapse.</p>
          </div>
        `
      },
      {
        label: "2. The Clinical Paradox: Significant aOR with Zero Predictive Gain",
        html: `
          <div class="patient-calc">
            <h4>Why a "p < 0.001" Biomarker Can Fail Clinically:</h4>
            <p>It is entirely possible for a biomarker to have a statistically significant adjusted odds ratio (e.g., aOR = 1.45, $p = 0.0002$) in a 1,000-patient cohort, yet yield an incremental discrimination of <strong>$\\\\Delta \\\\text{AUC} = +0.002$</strong> when added to baseline age and NIHSS!</p>
            <p><strong>Clinical Reason:</strong> Once baseline stroke severity (NIHSS) and ischemic core volume are known, the biomarker provides largely redundant clinical signal. Statistical significance tests sample size; prediction metrics test individual clinical discrimination.</p>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Association measures correlation in retrospective cohorts; prediction measures out-of-sample risk stratification for new patients; causation determines whether modifying the biomarker will actually rescue brain tissue in a prospective trial."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A new serum biomarker is published with $p < 0.0001$ and aOR = 2.1 for stroke mortality. The authors claim the hospital should immediately purchase a test kit for admission triage. What diagnostic test should you demand before adopting it?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) A larger p-value calculation.</button>
              <button class="quiz-opt" data-correct="true">B) Incremental out-of-sample discrimination (ΔAUC) and Decision Curve Analysis compared to existing clinical risk scores.</button>
              <button class="quiz-opt" data-correct="false">C) A Pearson correlation matrix.</button>
              <button class="quiz-opt" data-correct="false">D) Unadjusted univariate odds ratios.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> A statistically significant p-value merely proves the association is unlikely to be pure random chance in this dataset. To justify bedside clinical utility, the biomarker must demonstrate incremental discrimination ($\\\\Delta \\\\text{AUC}$) and positive Net Clinical Benefit on Decision Curve Analysis beyond standard clinical variables.
            </div>
          </div>
        `
      }
    ],
    lab: renderCausationLab
  },

  logistic: {
    eyebrow: "Module 2 · Lesson 5 of 24",
    h1: "Binary Outcomes & Logistic Regression",
    lead: "When the outcome is binary — such as functional independence (mRS 0–2) or 90-day mortality — straight lines produce impossible probabilities. We model the log-odds using the sigmoid S-curve.",
    analogy: {
      title: "The S-Curve Speed Governor",
      text: "A linear model would eventually predict that an elderly patient with massive stroke has a 125% chance of disability, or a young patient has a -15% risk of death, which is mathematically impossible. The logistic function acts like an intelligent speed governor: no matter how severe the predictors become, probability smoothly decelerates and stays bounded strictly between 0% and 100%."
    },
    sections: [
      {
        label: "1. The Benchmark Logistic Model in Stroke",
        html: `
          <div class="formula-card">
            <div class="formula-card__caption">The Logistic Link & Probability Equation</div>
            <div class="formula-card__equation">
              \text{logit}(p) = \ln\left(\frac{p}{1-p}\right) = \beta_0 + \sum \beta_k X_k \implies p = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_k X_k)}}
            </div>
            <div class="formula-card__caption">Term-by-Term Mathematical Breakdown</div>
            <div class="terms-grid">
              <div class="term-item"><code>p</code> <strong>Event Probability:</strong> Probability of favorable outcome (mRS 0–2), strictly bounded in (0, 1).</div>
              <div class="term-item"><code>p / (1 - p)</code> <strong>Odds:</strong> Ratio of success to failure probability (e.g., 0.80 / 0.20 = 4.0).</div>
              <div class="term-item"><code>\ln(\text{Odds})</code> <strong>Logit Link:</strong> Maps bounded probabilities into an unbounded scale $(-\infty, +\infty)$ suitable for linear addition.</div>
              <div class="term-item"><code>e^{\beta_k}</code> <strong>Odds Ratio (OR):</strong> Multiplicative scaling of event odds per 1-unit increase in clinical predictor $X_k$.</div>
              <div class="term-item"><code>\beta_0</code> <strong>Baseline Log-Odds:</strong> Expected log-odds for a reference patient where all covariates $X_k = 0$.</div>
            </div>
          </div>
          <div class="patient-calc">
            <h4>Concrete Patient Arithmetic: Patient 1 vs. Patient 2</h4>
            <p>Consider a validated stroke prediction equation for 90-day functional independence ($mRS \\\\le 2$):</p>
            <p><code>z = 2.45 - 0.045(Age) - 0.12(NIHSS) - 0.025(Core) + 0.35(CVO)</code></p>
            <div class="metric-grid">
              <div class="metric">
                <span>Patient 1 (Favourable CVO = 5)</span>
                <b>39.5% Probability</b>
                <p class="subtle">Age 65, NIHSS 10, Core 20 mL, CVO 5<br>z = -0.425 → p = 1 / (1 + e^(0.425)) = 0.395</p>
              </div>
              <div class="metric">
                <span>Patient 2 (Unfavourable CVO = 1)</span>
                <b>13.9% Probability</b>
                <p class="subtle">Age 65, NIHSS 10, Core 20 mL, CVO 1<br>z = -1.825 → p = 1 / (1 + e^(1.825)) = 0.139</p>
              </div>
            </div>
            <p><strong>Clinical Insight:</strong> Holding age, stroke severity, and core volume constant, robust venous outflow quadruples the odds of independent recovery ($e^{0.35 \\\\times 4} = e^{1.4} \\\\approx 4.05\\\\times$).</p>
          </div>
        `
      },
      {
        label: "2. The Clinical Trap: Odds Ratio ≠ Relative Risk",
        html: `
          <div class="patient-calc">
            <h4>Never Describe Odds Ratios as Direct Percentage Increases!</h4>
            <p>An Odds Ratio of 2.0 does <em>not</em> mean the patient has double the risk! Look at what happens across different baseline risks:</p>
            <table class="data-table">
              <thead>
                <tr>
                  <th>Baseline Risk (p₀)</th>
                  <th>Baseline Odds (p₀ / (1 - p₀))</th>
                  <th>New Odds (OR = 2.0)</th>
                  <th>New Risk (p₁)</th>
                  <th>True Relative Risk (p₁ / p₀)</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>2% (Rare event)</td>
                  <td>0.0204</td>
                  <td>0.0408</td>
                  <td>3.92%</td>
                  <td><strong>1.96× (OR ≈ RR)</strong></td>
                </tr>
                <tr>
                  <td>30% (Moderate)</td>
                  <td>0.428</td>
                  <td>0.857</td>
                  <td>46.2%</td>
                  <td><strong>1.54× (OR overstates RR)</strong></td>
                </tr>
                <tr>
                  <td>60% (Common stroke event)</td>
                  <td>1.500</td>
                  <td>3.000</td>
                  <td>75.0%</td>
                  <td><strong>1.25× (Massive distortion!)</strong></td>
                </tr>
              </tbody>
            </table>
            <p class="code-caption">When the clinical outcome is common (>10%), Odds Ratios dramatically exaggerate the apparent effect size. Always report adjusted absolute risk differences for bedside translation.</p>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Logistic regression models the log-odds of binary clinical events to ensure probabilities remain mathematically bounded between 0 and 1; however, because stroke disability is common, reported odds ratios must never be communicated to families as relative risks."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A logistic model outputs a log-odds risk score of <strong>z = 0.0</strong> for a patient. What is their predicted probability of the outcome?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) 0% (Impossible)</button>
              <button class="quiz-opt" data-correct="true">B) 50% (Correct! p = 1 / (1 + e^0) = 1 / 2 = 0.50)</button>
              <button class="quiz-opt" data-correct="false">C) 100% (Certain)</button>
              <button class="quiz-opt" data-correct="false">D) Cannot be calculated without intercept.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> When $z = 0$, $e^{-0} = 1$, and $p = 1 / (1 + 1) = 0.50$ (50%). A log-odds of 0 corresponds to equal odds ($1:1$, or a coin flip). Positive $z$ gives $p > 0.50$, and negative $z$ gives $p < 0.50$.
            </div>
          </div>
        `
      }
    ],
    lab: renderLogisticLab
  },

  ordinal: {
    eyebrow: "Module 2 · Lesson 6 of 24",
    h1: "Ordinal Outcomes & Proportional Odds (mRS Shift Analysis)",
    lead: "Stroke recovery is measured on the 7-point modified Rankin Scale (mRS 0 to 6). Collapsing it into binary 'good vs. bad' throws away half your statistical power and masks life-changing clinical recovery.",
    analogy: {
      title: "The Decathlon Hurdles",
      text: "Imagine evaluating a decathlete by counting only whether they cleared the highest hurdle, ignoring all other jumps. In stroke trials, treating mRS as binary (0–2 vs 3–6) means shifting a bedridden, mute patient (mRS 5) to walking independently with a cane (mRS 3) is scored as a TOTAL FAILURE! Ordinal regression scores every hurdle cleared across the full recovery spectrum."
    },
    sections: [
      {
        label: "1. The Clinical Harm of Dichotomania",
        html: `
          <div class="patient-calc">
            <h4>Why Dichotomizing the modified Rankin Scale Harms Clinical Science:</h4>
            <ul>
              <li><strong>Loss of Statistical Power:</strong> Converting a 7-category ordered scale into a 0/1 binary variable loses between 33% and 50% of your effective sample size.</li>
              <li><strong>Arbitrary Threshold Bias:</strong> In severe stroke trials, achieving mRS 2 is rare. The entire therapeutic benefit may be shifting patients from mRS 5 (severe dependency) to mRS 3 (moderate disability, living at home). Binary analysis labels this wonder-drug as ineffective ($p = 0.45$), while ordinal shift analysis proves robust efficacy ($p = 0.002$).</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. The Proportional Odds Cumulative Logit Model",
        html: `
          <div class="formula-card">
            <div class="formula-card__caption">The Cumulative Logit (Proportional Odds) Equation</div>
            <div class="formula-card__equation">
              \text{logit}(P(Y \le j)) = \alpha_j - (\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k)
            </div>
            <div class="formula-card__caption">Term-by-Term Mathematical Breakdown</div>
            <div class="terms-grid">
              <div class="term-item"><code>Y \le j</code> <strong>Cumulative Threshold:</strong> Probability of achieving functional disability level $j$ or better ($j \in \{0, 1, \dots, 5\}$).</div>
              <div class="term-item"><code>\alpha_j</code> <strong>Cutpoint Intercepts:</strong> Monotonically increasing thresholds ($\alpha_0 < \alpha_1 < \dots < \alpha_5$) defining baseline category splits.</div>
              <div class="term-item"><code>\beta_k</code> <strong>Shared Slope Vector:</strong> Assumes predictor effects are mathematically identical across every disability hurdle (Proportional Odds).</div>
              <div class="term-item"><code>\text{cOR} = e^{\beta}</code> <strong>Common Odds Ratio:</strong> Uniform odds multiplier for achieving a more favorable score across the entire mRS continuum.</div>
            </div>
          </div>
          <div class="metric-grid">
            <div class="metric">
              <span>Common Odds Ratio (cOR)</span>
              <b>1.68 per Step</b>
              <p class="subtle">Thrombectomy increases odds of achieving a better mRS score across every boundary by 68%.</p>
            </div>
            <div class="metric">
              <span>The Brant Test</span>
              <b>p = 0.42 (Assumption Holds)</b>
              <p class="subtle">Verifies that the treatment boost is statistically consistent across all 6 cutpoints.</p>
            </div>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Ordinal logistic regression preserves the entire 7-point continuum of stroke recovery, estimating a common odds ratio that captures meaningful functional shifts across all disability thresholds without discarding clinical information through arbitrary dichotomization."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>In a randomized thrombectomy trial, a drug produces an ordinal shift common Odds Ratio (cOR) of <strong>1.75 ($p < 0.001$)</strong> for mRS recovery. However, binary analysis for $mRS \\le 2$ yields $p = 0.12$. What is the correct clinical interpretation?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) The drug failed because $mRS \\le 2$ was not statistically significant.</button>
              <button class="quiz-opt" data-correct="true">B) The drug confers genuine clinical benefit by shifting patients toward lower disability across multiple mRS boundaries, which binary analysis lacked the statistical power to detect.</button>
              <button class="quiz-opt" data-correct="false">C) Ordinal regression is invalid because it always produces lower p-values.</button>
              <button class="quiz-opt" data-correct="false">D) The sample size was too large.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> This exact scenario occurred in landmark endovascular trials (e.g. MR CLEAN, DEFUSE 3). By evaluating the full distribution of recovery, ordinal shift analysis captures life-altering transitions (e.g., $mRS \\ 4 \\to 3$) that binary $mRS \\le 2$ completely ignores.
            </div>
          </div>
        `
      }
    ],
    lab: renderOrdinalLab
  },

  survival: {
    eyebrow: "Module 2 · Lesson 7 of 24",
    h1: "Time-to-Event & Competing Risks in Stroke",
    lead: "Patients are not followed indefinitely. Some experience recurrent stroke on day 12, others survive 1 year event-free, and some are lost to follow-up. Standard regression fails; we need survival analysis.",
    analogy: {
      title: "The Marathon Cutoff Bus",
      text: "Imagine tracking runners in a marathon. If a runner has not finished at the 4-hour mark, they didn't 'fail' — their time is simply censored at 4 hours. Furthermore, if a runner gets hit by lightning at mile 10, they can never finish the race; lightning is a competing risk that fundamentally prevents the primary outcome."
    },
    sections: [
      {
        label: "1. The Foundation: Censoring & The Hazard Function",
        html: `
          <p>In prospective stroke registries, every patient contributes two distinct data points:</p>
          <div class="metric-grid">
            <div class="metric">
              <span>Time ($T$)</span>
              <b>Days to Event or Censoring</b>
              <p class="subtle">Follow-up duration from acute stroke onset to last clinical contact.</p>
            </div>
            <div class="metric">
              <span>Status Indicator ($D$)</span>
              <b>1 = Event, 0 = Censored</b>
              <p class="subtle">Did the patient experience recurrent stroke, or did follow-up end safely?</p>
            </div>
          </div>
          <div class="formula-card">
            <div class="formula-card__caption">Survival & Cox Proportional Hazards Equations</div>
            <div class="formula-card__equation">
              \hat{S}(t) = \prod_{t_i \le t} \left[1 - \frac{d_i}{n_i}\right] \qquad \lambda(t | X) = \lambda_0(t) \cdot \exp\left(\sum \beta_k X_k\right)
            </div>
            <div class="formula-card__caption">Term-by-Term Mathematical Breakdown</div>
            <div class="terms-grid">
              <div class="term-item"><code>\hat{S}(t)</code> <strong>Survival Probability:</strong> Cumulative probability of remaining event-free up to time $t$.</div>
              <div class="term-item"><code>d_i / n_i</code> <strong>Instantaneous Failures:</strong> Number of patients having an event at time $t_i$ divided by active patients at risk.</div>
              <div class="term-item"><code>\lambda_0(t)</code> <strong>Baseline Hazard:</strong> Underlying non-parametric event risk over time when all covariates equal zero.</div>
              <div class="term-item"><code>\exp(\beta_k)</code> <strong>Hazard Ratio (HR):</strong> Relative multiplicative shift in event rate per unit change in clinical predictor $X_k$.</div>
            </div>
          </div>
        `
      },
      {
        label: "2. The Competing Risks Threat: KM vs. Cumulative Incidence",
        html: `
          <div class="patient-calc">
            <h4>Why Standard Kaplan-Meier Overestimates Stroke Recurrence:</h4>
            <p>If an elderly stroke patient dies from cardiac arrest at month 2, they can no longer suffer a recurrent ischemic stroke. Treating non-stroke death as standard 'censoring' makes the false mathematical assumption that these deceased patients would have had the same future stroke risk as living survivors!</p>
            <p><strong>Clinical Consequence:</strong> Standard KM curves artificially inflate cumulative stroke recurrence by 15–25%. In modern clinical research, always report <strong>Cumulative Incidence Functions (CIF)</strong> via the Fine-Gray subdistribution hazards model when competing risks exist.</p>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Kaplan-Meier survival curves assume non-informative censoring; in the presence of competing events such as in-hospital mortality, KM systematically overestimates secondary event rates, necessitating Cumulative Incidence Functions and Fine-Gray subdistribution models."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A Cox regression model for stroke recurrence reports a Hazard Ratio of <strong>HR = 1.80 ($p = 0.01$)</strong> for severe carotid stenosis. What does HR = 1.80 mean?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) 80% of patients with stenosis will have a recurrent stroke.</button>
              <button class="quiz-opt" data-correct="true">B) At any given time point during follow-up, patients with severe stenosis experience an 80% higher instantaneous rate of recurrent stroke compared to those without stenosis, holding other covariates constant.</button>
              <button class="quiz-opt" data-correct="false">C) Severe stenosis increases survival duration by 1.8 years.</button>
              <button class="quiz-opt" data-correct="false">D) The recurrence risk is 1.8%.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> A Hazard Ratio represents the relative event rate per unit time. HR = 1.80 means an 80% increase in the instantaneous event hazard throughout follow-up under the proportional hazards assumption.
            </div>
          </div>
        `
      }
    ],
    lab: renderSurvivalLab
  },

  missing: {
    eyebrow: "Module 2 · Lesson 8 of 24",
    h1: "Missing Data & Multiple Imputation (MICE)",
    lead: "In acute stroke cohorts, 15–25% of patients miss follow-up imaging, collateral scores, or lab panels. Deleting them is not conservative — it introduces lethal selection bias.",
    analogy: {
      title: "The Selective Survey Phone Call",
      text: "If a clinic calls patients at day 90 to assess recovery, the patients who don't answer the phone are not missing at random — they may be dead, bedridden in a nursing home, or severely depressed. If you analyze only the happy patients who answered the phone, your treatment looks miraculous purely due to selective survival bias."
    },
    sections: [
      {
        label: "1. The Three Missing Data Mechanisms",
        html: `
          <div class="paper-grid">
            <div class="paper-card">
              <div class="paper-id">MCAR: COMPLETELY AT RANDOM</div>
              <h2>"Pure Accident"</h2>
              <p>Missingness is completely unrelated to patient traits or outcomes. Example: A blood tube drops and shatters on the laboratory floor.</p>
              <dl><dt>Consequence</dt><dd>Complete-case analysis is unbiased but wastes statistical power.</dd></dl>
            </div>
            <div class="paper-card">
              <div class="paper-id">MAR: MISSING AT RANDOM</div>
              <h2>"Conditionally Explained"</h2>
              <p>Missingness depends on observed baseline variables. Example: Severe stroke patients (high NIHSS) miss MRI because they cannot lay flat.</p>
              <dl><dt>Consequence</dt><dd>Multiple Imputation (MICE) fully restores validity by conditioning on observed data.</dd></dl>
            </div>
          </div>
          <div class="patient-calc" style="margin-top:15px;">
            <h4>MNAR: MISSING NOT AT RANDOM — The Truncated Reality</h4>
            <p>Missingness depends on the unobserved value itself. Example: Patients with poor 90-day recovery refuse follow-up visits because of functional immobility. Requires formal sensitivity analyses (pattern-mixture models or tipping-point analyses).</p>
          </div>
        `
      },
      {
        label: "2. MICE & Rubin's Rules: The Variance Decomposition",
        html: `
          <div class="formula">
            Total Variance: T = W + [ 1 + (1 / m) ] · B
            <small>Where W = Average within-imputation variance, B = Between-imputation variance, and m = Number of imputed datasets (m ≥ 20).</small>
          </div>
          <div class="patient-calc">
            <h4>Why Single Imputation (e.g. Mean Imputation) is Statistical Malpractice:</h4>
            <ul>
              <li><strong>Mean Imputation:</strong> Artificially collapses variance to zero, fabricates non-existent precision, and severely inflates false-positive Type I error rates.</li>
              <li><strong>Multiple Imputation (MICE):</strong> Generates $m = 20$ to $50$ plausible complete datasets with proper stochastic error, fits the model on each, and pools estimates using Rubin's rules to honestly capture missing-data uncertainty.</li>
            </ul>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Complete-case analysis is never a conservative choice; it relies on the implausible assumption that data are missing completely at random, systematically biasing cohorts toward healthier survivors and discarding irreplaceable statistical power."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A stroke database of 500 patients has 10% missing glucose and 12% missing onset-to-needle time. If you use complete-case analysis (dropping any patient with a missing value), approximately how many patients will be discarded?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Exactly 12% (60 patients)</button>
              <button class="quiz-opt" data-correct="false">B) Exactly 22% (110 patients)</button>
              <button class="quiz-opt" data-correct="true">C) Up to 20–22% (100+ patients) depending on overlap, discarding nearly a quarter of your entire clinical cohort!</button>
              <button class="quiz-opt" data-correct="false">D) 0 patients</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Missingness across different variables accumulates! If 10% miss glucose and a different 12% miss time, listwise deletion throws away over 20% of your patient cohort. MICE imputes these values conditional on all observed predictors, preserving all 500 patients.
            </div>
          </div>
        `
      }
    ],
    lab: renderMissingLab
  },

  splines: {
    eyebrow: "Module 3 · Lesson 9 of 24",
    h1: "Non-Linear Relationships & Restricted Cubic Splines",
    lead: "Human physiology does not follow straight lines. Age, blood pressure, and ischemic core volume exhibit thresholds and inflection points. Modeling them as linear harms prediction.",
    analogy: {
      title: "The Flexible Drafting Spline",
      text: "Before computers, shipbuilders used a flexible wooden ruler called a 'spline', held down at specific points (knots) with heavy lead weights. The wood naturally curved smoothly between the weights without sharp corners. Restricted cubic splines do the exact same thing mathematically: smooth cubic curves between knots, forced to be straight at the edges to prevent wild tail explosions."
    },
    sections: [
      {
        label: "1. The Folly of Dichotomania (Step Function Traps)",
        html: `
          <div class="patient-calc">
            <h4>Why 'Categorizing' Continuous Clinical Variables is Statistical Malpractice:</h4>
            <p>Clinicians frequently split continuous variables into arbitrary bins (e.g., Age < 70 vs. ≥ 70, or Core < 50 mL vs. ≥ 50 mL). Frank Harrell terms this <em>dichotomania</em>:</p>
            <ul>
              <li><strong>The False Cliff:</strong> A patient aged 69.9 is treated as identical to a 35-year-old, but completely different from a 70.1-year-old!</li>
              <li><strong>Information Waste:</strong> Dichotomizing discards at least one-third of the statistical variance and statistical power.</li>
              <li><strong>Residual Confounding:</strong> Huge variation within each group is completely ignored, distorting hazard ratios.</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. Restricted Cubic Splines (RCS): The Mathematical Solution",
        html: `
          <div class="formula">
            RCS with k Knots: f(X) = β₀ + β₁X + β₂·S₁(X) + ... + βₖ₋₁·Sₖ₋₂(X)
            <small>Requires estimating only (k - 1) parameters while modeling complex non-linear inflection points.</small>
          </div>
          <div class="metric-grid">
            <div class="metric">
              <span>Stone's Knot Percentiles (4 Knots)</span>
              <b>5th · 35th · 65th · 95th</b>
              <p class="subtle">Default recommendation balancing flexibility and parameter economy (3 degrees of freedom).</p>
            </div>
            <div class="metric">
              <span>Non-Linearity Wald Test</span>
              <b>χ² = 18.4, p < 0.001</b>
              <p class="subtle">Formally proves that the non-linear spline components add significant predictive value over a straight line.</p>
            </div>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Biological risk curves do not jump at arbitrary diagnostic cutpoints; restricted cubic splines model continuous non-linear physiology with mathematical fidelity while constraining tail behavior to prevent volatile edge extrapolations."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>You have an acute stroke cohort of 250 patients with 45 disability events (EPV = 45). According to Harrell's statistical guidelines, how many spline knots should you assign to baseline ischemic core volume?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) 10 knots (to fit every subtle data bump)</button>
              <button class="quiz-opt" data-correct="true">B) 3 to 4 knots (spending 2 to 3 degrees of freedom to avoid over-fitting small sample sizes)</button>
              <button class="quiz-opt" data-correct="false">C) 0 knots (always categorize into tertiles)</button>
              <button class="quiz-opt" data-correct="false">D) 1 knot</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> In moderate sample sizes (45 events), allocating 3 knots (2 df) or 4 knots (3 df) captures smooth non-linear inflection points without burning excessive degrees of freedom or memorizing idiosyncratic cohort noise.
            </div>
          </div>
        `
      }
    ],
    lab: renderSplinesLab
  },

  phenotyping: {
    eyebrow: "Module 3 · Lesson 10 of 24",
    h1: "Clinical Phenotyping & Subgroup Discovery",
    lead: "Stroke is not a monolithic disease. Identifying 'Fast Responders' versus 'Malignant Progressors' unlocks precision triage.",
    analogy: {
      title: "The Melting Ice Sculpture",
      text: "Two ice sculptures placed in a warm room melt at dramatically different rates depending on air currents and insulation. Similarly, two stroke patients with identical 20 mL ischemic cores will progress differently: one with robust collaterals loses tissue at 1 mL/hr (Slow Progressor), while one with collateral failure loses 15 mL/hr (Fast Progressor)."
    },
    sections: [
      {
        label: "1. The Biology of Fast vs. Slow Progressors",
        html: `
          <p>From Paper 3 (Fast Responders in Favourable CTP) and Paper 4 (NWU Progression):</p>
          <div class="metric-grid">
            <div class="metric">
              <span>Fast Progressors</span>
              <b>> 12 mL/hr Infarct Growth</b>
              <p class="subtle">Rapid collateral failure, acute cellular edema, high risk of early malignant herniation.</p>
            </div>
            <div class="metric">
              <span>Slow Progressors</span>
              <b>< 2 mL/hr Infarct Growth</b>
              <p class="subtle">Robust leptomeningeal collaterals, wide tissue window, viable penumbra beyond 24 hours.</p>
            </div>
          </div>
          <div class="patient-calc">
            <h4>Net Water Uptake (NWU) as a Phenotyping Biomarker:</h4>
            <p>CT-derived NWU quantifies ionic edema per unit of ischemic brain tissue. A patient with <em>greater than expected NWU progression</em> exhibits blood-brain barrier breakdown that predicts secondary hemorrhagic transformation and malignant edema regardless of successful vessel recanalization.</p>
          </div>
          <div class="formula-card">
            <div class="formula-card__caption">Residual Phenotyping Formulation (Paper 4)</div>
            <div class="formula-card__equation">
              \text{Residual}_i = \text{NWU}_i^{\text{Observed}} - f(\text{Time}_i, \text{Collaterals}_i, \text{Core}_i)
            </div>
            <div class="formula-card__caption">Clinical Biological Phenotypes</div>
            <div class="terms-grid">
              <div class="term-item"><code>\text{Residual} > +1.5\sigma</code> <strong>Malignant Progressor:</strong> Edema velocity exceeds expectation; high risk of midline shift and fatal herniation.</div>
              <div class="term-item"><code>\text{Residual} \approx 0</code> <strong>Expected Progressor:</strong> Edema rate adheres to baseline predicted ischemia-perfusion kinetics.</div>
              <div class="term-item"><code>\text{Residual} < -1.5\sigma</code> <strong>Protected Phenotype:</strong> Microvascular resilience preserves blood-brain barrier despite severe occlusion.</div>
            </div>
          </div>
        `
      },
      {
        label: "2. The Danger of Post-Hoc Subgroup Dredging",
        html: `
          <div class="presentation-gold">
            <h4>⚠️ Warning: The Texas Sharpshooter Fallacy in Subgroup Analysis</h4>
            <p>Testing 20 different post-hoc patient subgroups without multiplicity adjustment guarantees a false-positive finding ($p < 0.05$) purely by chance ($1 - 0.95^{20} = 64\\%$ false discovery rate!). Phenotypes must be pre-specified based on known biological mechanisms (e.g. collateral score, NWU) rather than data-mined post-hoc clusters.</p>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Clinical phenotyping should reflect distinct underlying vascular pathophysiology — such as collateral robustness governing infarct growth velocity — rather than data-mined post-hoc clusters that evaporate upon prospective external replication."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A stroke patient arrives 14 hours after Last Known Well with NIHSS 18 and a CTP core of 15 mL with robust cortical venous outflow (CVO score 6). Which clinical phenotype do they represent?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Fast Progressor (treatment futile)</button>
              <button class="quiz-opt" data-correct="true">B) Slow Progressor (extended therapeutic window; excellent candidate for late-window thrombectomy)</button>
              <button class="quiz-opt" data-correct="false">C) Completed Infarct</button>
              <button class="quiz-opt" data-correct="false">D) Hemorrhagic Conversion</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Having only 15 mL of core at 14 hours proves robust collateral and venous sustenance! This slow progressor phenotype was the biological cornerstone of the landmark DAWN and DEFUSE 3 extended-window trials.
            </div>
          </div>
        `
      }
    ],
    lab: renderPhenotypingLab
  },

  interactions: {
    eyebrow: "Module 3 · Lesson 11 of 24",
    h1: "Interactions & Effect Modification in Stroke",
    lead: "Does the benefit of endovascular thrombectomy depend on collateral flow? Or does time-to-treatment matter more when venous drainage is poor? We model synergy with interaction terms.",
    analogy: {
      title: "The Parachute and the Altitude",
      text: "Pulling a parachute ripcord is life-saving at 10,000 feet, but useless at 10 feet from the ground. The effect of the parachute (treatment) completely depends on the altitude (effect modifier). In acute stroke, the clinical efficacy of recanalization depends directly on remaining collateral tissue viability."
    },
    sections: [
      {
        label: "1. The Biological Interaction Equation",
        html: `
          <div class="formula">
            logit(p) = β₀ + β₁(Treatment) + β₂(Collaterals) + β₃(Treatment × Collaterals)
            <small>If β₃ ≠ 0, the treatment effect differs across collateral grades (Effect Modification).</small>
          </div>
          <div class="patient-calc">
            <h4>Additive vs. Multiplicative Interactions:</h4>
            <p>A common clinical trap is confusing statistical interaction with biological synergy:</p>
            <ul>
              <li><strong>Multiplicative Interaction:</strong> Evaluated on the log-odds or hazard scale ($\\\\beta_3$ in logistic/Cox models).</li>
              <li><strong>Additive Interaction (Relative Excess Risk due to Interaction, RERI):</strong> Evaluated on the absolute risk scale. Clinicians care about absolute risk differences: how many additional patients walk out of the hospital when treating those with good vs. poor collaterals?</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"A statistically significant interaction on the multiplicative log-odds scale does not guarantee a clinically meaningful interaction on the absolute risk difference scale; always calculate marginal risk differences via G-computation for bedside decision-making."</p>
          </div>
        `
      },
      {
        label: "3. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>In an interaction model, thrombectomy gives an Odds Ratio of 2.5 in patients with good collaterals, and an Odds Ratio of 2.4 in patients with moderate collaterals. The multiplicative interaction term p-value is <strong>p = 0.88</strong>. What does this mean?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Thrombectomy does not work.</button>
              <button class="quiz-opt" data-correct="true">B) There is no evidence that the relative treatment effect differs between good and moderate collateral grades.</button>
              <button class="quiz-opt" data-correct="false">C) Collaterals have no impact on stroke outcome.</button>
              <button class="quiz-opt" data-correct="false">D) The model has multicollinearity.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> A non-significant interaction term ($p = 0.88$) indicates that the relative treatment benefit of thrombectomy (OR ~2.4–2.5) is consistent across both collateral groups.
            </div>
          </div>
        `
      }
    ],
    lab: renderInteractionsLab
  },

  features: {
    eyebrow: "Module 3 · Lesson 12 of 24",
    h1: "Predictor Selection & Multicollinearity",
    lead: "Automated stepwise algorithms promise to find the 'best' predictors. In reality, they model sample noise and destabilize clinical models. We select predictors using domain biology and variance inflation diagnostics.",
    analogy: {
      title: "The Three Thermometers",
      text: "If you place three digital thermometers in the same patient's mouth, you have three highly correlated measurements of body temperature. A naive algorithm might assign a huge positive weight to thermometer A, a huge negative weight to thermometer B, and zero to thermometer C. Multicollinearity makes individual coefficients wildly unstable even while the combined prediction appears normal."
    },
    sections: [
      {
        label: "1. The Discredited Practice of Stepwise Selection",
        html: `
          <div class="patient-calc">
            <h4>Why Stepwise Forward/Backward Selection is Banned by Biostatisticians:</h4>
            <ul>
              <li><strong>Severe P-Value Bias:</strong> Testing multiple candidate predictors and keeping only $p < 0.05$ invalidates the sampling distribution. The reported p-values are orders of magnitude too small!</li>
              <li><strong>Exaggerated Regression Coefficients:</strong> Predictor weights are systematically overestimated (selection bias / winner's curse).</li>
              <li><strong>Extreme Instability:</strong> Deleting just 5 patients from the cohort can produce a completely different set of chosen variables!</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. Multicollinearity & Variance Inflation Factor (VIF)",
        html: `
          <div class="formula">
            VIFⱼ = 1 / (1 - Rⱼ²)
            <small>Where Rⱼ² is the R² from regressing predictor Xⱼ on all other predictors in the model.</small>
          </div>
          <div class="metric-grid">
            <div class="metric">
              <span>VIF < 3.0</span>
              <b>Low Collinearity (Safe)</b>
              <p class="subtle">Age, baseline blood glucose, and onset-to-arrival time.</p>
            </div>
            <div class="metric">
              <span>VIF > 5.0–10.0</span>
              <b>High Collinearity (Severe)</b>
              <p class="subtle">Simultaneously including CTP Core Volume, ASPECTS, and CBV territory volume.</p>
            </div>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Automated stepwise selection models random noise and yields artificially inflated p-values; pre-specifying candidate predictors based on established stroke pathophysiology and Events Per Variable rules yields robust, reproducible clinical risk models."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>You include both 'Ischemic Core Volume in mL' and 'Ischemic Core Volume in cm³' in a regression model. What happens to the regression algorithm?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Prediction accuracy doubles.</button>
              <button class="quiz-opt" data-correct="true">B) Exact collinearity (VIF = ∞); the design matrix cannot be inverted without mathematical regularization.</button>
              <button class="quiz-opt" data-correct="false">C) Both variables get p < 0.001.</button>
              <button class="quiz-opt" data-correct="false">D) The intercept becomes negative.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Because $1 \\text{ cm}^3 = 1 \\text{ mL}$, one predictor is a perfect linear multiple of the other ($R^2 = 1.0$), causing division by zero ($1 / (1 - 1) = \\infty$) in matrix inversion!
            </div>
          </div>
        `
      }
    ],
    lab: renderFeaturesLab
  },

  validation: {
    eyebrow: "Module 4 · Lesson 13 of 24",
    h1: "Apparent vs. Internal vs. External Validation",
    lead: "A model always looks brilliant on the patients used to train it. The true test of a clinical prediction model is how well it performs on unseen patients.",
    analogy: {
      title: "The Rehearsed Practice Exam",
      text: "If a medical student takes the exact same practice exam five times, they will score 100% (Apparent Performance). But their true clinical competence is revealed when facing a brand-new patient with atypical symptoms in a busy emergency room (External Validation)."
    },
    sections: [
      {
        label: "1. The Three Tiers of Clinical Validation",
        html: `
          <div class="metric-grid">
            <div class="metric">
              <span>Tier 1: Apparent Performance</span>
              <b>Optimistic / Biased</b>
              <p class="subtle">Evaluated on the exact training dataset. Memorizes random cohort noise.</p>
            </div>
            <div class="metric">
              <span>Tier 2: Internal Validation</span>
              <b>Optimism-Corrected</b>
              <p class="subtle">Repeated cross-validation or bootstrap resampling within the same target population.</p>
            </div>
            <div class="metric">
              <span>Tier 3: External Validation</span>
              <b>Generalizability / Transportability</b>
              <p class="subtle">Evaluated in a completely different hospital system, geographic region, or era.</p>
            </div>
          </div>
        `
      },
      {
        label: "2. The Flaw of Split-Sample (70/30) in Small Medical Cohorts",
        html: `
          <div class="patient-calc">
            <h4>Why Split-Sample Validation is Inefficient for N < 2,000:</h4>
            <ul>
              <li><strong>High Sampling Variance:</strong> In a cohort of 300 stroke patients, a 30% test split contains only 90 patients (perhaps 20 disability events!). A couple of unusual patients will cause massive wild swings in estimated AUC.</li>
              <li><strong>Wasted Sample Size:</strong> Discarding 30% of your data during model development reduces the effective Events Per Variable (EPV), actively causing the model to overfit!</li>
              <li><strong>Modern Standard:</strong> Use <strong>repeated k-fold cross-validation (e.g., 20 repeats of 5-fold CV)</strong> or <strong>bootstrap optimism correction</strong>, using 100% of available patients for both development and validation.</li>
            </ul>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Apparent performance reflects how well a model memorized past patients; internal bootstrap validation estimates optimism shrinkage; and external geographic validation proves whether the model survives the clinical realities of a different hospital."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A stroke manuscript reports an apparent $R^2$ of <strong>0.335</strong> on the development cohort, but repeated 5-fold cross-validation reveals an out-of-fold $R^2$ of <strong>0.077</strong>. Which metric reflects the model's true predictive power on future patients?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) 0.335 (because it used the full sample)</button>
              <button class="quiz-opt" data-correct="true">B) 0.077 (cross-validated performance honestly reflects generalization to unseen patients)</button>
              <button class="quiz-opt" data-correct="false">C) The average of the two: (0.335 + 0.077) / 2</button>
              <button class="quiz-opt" data-correct="false">D) Neither</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> This is the exact finding from the CVO recalibration manuscript! The apparent $R^2 = 0.335$ was inflated by overfitting to training noise. The out-of-fold $R^2 = 0.077$ represents true generalizability.
            </div>
          </div>
        `
      }
    ],
    lab: renderValidationLab
  },

  bootstrap: {
    eyebrow: "Module 4 · Lesson 14 of 24",
    h1: "The Bootstrap & Optimism Correction",
    lead: "How do you validate a model when your clinical cohort is too valuable to split? Resample with replacement using Bradley Efron's bootstrap algorithm.",
    analogy: {
      title: "The Card Deck Reshuffle",
      text: "Imagine taking a standard deck of 52 cards, drawing one card at random, writing down its name, and putting it back in the deck (sampling with replacement). If you do this 52 times, some cards will appear two or three times, while about 37% of cards will never be picked at all. Those unpicked cards are your 'out-of-bag' holdout test cohort!"
    },
    sections: [
      {
        label: "1. The Harrell Optimism Correction Algorithm",
        html: `
          <div class="patient-calc">
            <h4>Step-by-Step Optimism Calculation:</h4>
            <ol>
              <li>Fit the prediction model on the full original clinical cohort → Record <code>Performance_apparent</code> (e.g., AUC = 0.84).</li>
              <li>Draw a bootstrap sample $B_i$ of size $N$ with replacement from the cohort.</li>
              <li>Fit a new model on $B_i$ → Measure its performance on $B_i$ (<code>Performance_boot</code>).</li>
              <li>Evaluate that bootstrap-trained model on the <em>original cohort</em> (<code>Performance_test</code>).</li>
              <li>Calculate single-run optimism: <code>Optimism_i = Performance_boot - Performance_test</code>.</li>
              <li>Repeat 500 times to compute average optimism $\\\\bar{O}$.</li>
              <li><strong>Optimism-Corrected Performance = Performance_apparent - $\\\\bar{O}$</strong>.</li>
            </ol>
          </div>
        `
      },
      {
        label: "2. The Magic of Out-of-Bag (OOB) Patients",
        html: `
          <div class="formula">
            Probability of Patient Never Being Drawn: lim (1 - 1/N)ᴺ = 1 / e ≈ 36.8%
          </div>
          <p>In every bootstrap iteration, approximately <strong>36.8% of patients are completely left out</strong>. These out-of-bag patients serve as a pristine, uncontaminated test set, providing low-variance estimates of out-of-sample calibration and discrimination.</p>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Bootstrap optimism correction utilizes 100% of patient data for model development while providing a mathematically rigorous, low-variance estimate of overfitting shrinkage without wasting valuable clinical observations on an inefficient train-test split."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A stroke prediction model has an apparent AUC of 0.82. Over 500 bootstrap iterations, the average optimism is calculated as $\\\\bar{O} = 0.05$. What is the model's reported internally validated AUC?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) 0.87 (0.82 + 0.05)</button>
              <button class="quiz-opt" data-correct="true">B) 0.77 (Correct! 0.82 - 0.05 = 0.77)</button>
              <button class="quiz-opt" data-correct="false">C) 0.82 (optimism is ignored)</button>
              <button class="quiz-opt" data-correct="false">D) 0.05</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Optimism represents the degree of overfitting to development data. Subtracting average optimism ($0.82 - 0.05 = 0.77$) gives the honest, calibrated performance expected on new patients from the same clinical population.
            </div>
          </div>
        `
      }
    ],
    lab: renderBootstrapLab
  },

  auc: {
    eyebrow: "Module 4 · Lesson 15 of 24",
    h1: "Discrimination: ROC Curves & The C-Statistic",
    lead: "Can your model tell who will recover and who will stay disabled? The Area Under the Receiver Operating Characteristic Curve (AUC) measures pairwise ranking ability.",
    analogy: {
      title: "The Medical Student Triage Pair",
      text: "Imagine placing two stroke patients in front of an intern: one who will walk out of the hospital, and one who will remain wheelchair-bound. If the intern correctly identifies the recovering patient 80 out of 100 times, their clinical discrimination is 80% (AUC = 0.80)."
    },
    sections: [
      {
        label: "1. The Probabilistic Meaning of AUC",
        html: `
          <div class="formula">
            AUC = C-Statistic = P( Predicted Risk(Event Patient) > Predicted Risk(Non-Event Patient) )
            <small>Directly equivalent to the Wilcoxon-Mann-Whitney non-parametric rank-sum test statistic.</small>
          </div>
          <div class="metric-grid">
            <div class="metric">
              <span>AUC = 0.50</span>
              <b>Worthless (Coin Flip)</b>
              <p class="subtle">Zero discrimination between favorable and unfavorable stroke recovery.</p>
            </div>
            <div class="metric">
              <span>AUC = 0.70–0.80</span>
              <b>Acceptable Clinical Utility</b>
              <p class="subtle">Standard baseline for acute stroke clinical decision support.</p>
            </div>
            <div class="metric">
              <span>AUC > 0.85</span>
              <b>Excellent Discrimination</b>
              <p class="subtle">Achieved when combining clinical severity (NIHSS) with advanced perfusion imaging (CTP core + CVO).</p>
            </div>
          </div>
        `
      },
      {
        label: "2. The Dangerous Blindspot of AUC",
        html: `
          <div class="patient-calc">
            <h4>Why High AUC Does NOT Equal Clinical Bedside Safety:</h4>
            <p>AUC evaluates <strong>ranking order only</strong>. If a model assigns a predicted mortality of 92% to patient A and 90% to patient B, it receives full credit for ranking A higher than B — even if patient A's true biological mortality risk is only 15%!</p>
            <p><strong>Clinical Consequence:</strong> A model with AUC = 0.88 can be catastrophically miscalibrated, causing clinicians to wrongly withhold life-saving thrombectomy based on falsely exaggerated mortality predictions.</p>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"The C-statistic evaluates relative discrimination — the model's ability to rank a patient who suffers an adverse outcome above one who does not — but provides zero information about whether predicted numerical probabilities match real-world biological risk."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>If you multiply every patient's predicted probability by 3 (so a 10% risk becomes 30%, and a 30% risk becomes 90%), what happens to the model's AUC?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) AUC triples.</button>
              <button class="quiz-opt" data-correct="true">B) AUC remains exactly identical (because relative patient rankings did not change).</button>
              <button class="quiz-opt" data-correct="false">C) AUC drops to 0.50.</button>
              <button class="quiz-opt" data-correct="false">D) AUC becomes negative.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Because AUC depends strictly on the rank-order of predictions, any monotonic transformation leaves AUC 100% unchanged. This proves why AUC cannot evaluate whether numerical probabilities are accurate!
            </div>
          </div>
        `
      }
    ],
    lab: renderAucLab
  },

  calibration: {
    eyebrow: "Module 4 · Lesson 16 of 24",
    h1: "Calibration: The Neglected Truth",
    lead: "If a model tells 100 patients they have an 80% risk of brain herniation, exactly 80 of them should herniate. Calibration evaluates truth-in-advertising for clinical probabilities.",
    analogy: {
      title: "The Deceptive Rain App",
      text: "If your weather app says there is an 80% chance of rain every single day, but it only rains on 10 out of 100 days, the app has terrible calibration. You would pack an umbrella every day for no reason. In medicine, poor calibration leads to unnecessary brain surgeries and inappropriate withdrawals of care."
    },
    sections: [
      {
        label: "1. Calibration Intercept and Calibration Slope",
        html: `
          <div class="formula-card">
            <div class="formula-card__caption">Calibration Assessment & Logistic Recalibration</div>
            <div class="formula-card__equation">
              \text{logit}(P(Y=1)) = \alpha + \beta \cdot \text{logit}(\hat{p}) \qquad \text{Brier} = \frac{1}{N}\sum (\hat{p}_i - y_i)^2
            </div>
            <div class="formula-card__caption">Term-by-Term Clinical Interpretation</div>
            <div class="terms-grid">
              <div class="term-item"><code>\alpha = 0.0</code> <strong>Calibration-in-the-Large:</strong> Evaluates overall baseline risk calibration. Negative values ($\alpha < 0$) indicate systemic overestimation.</div>
              <div class="term-item"><code>\beta = 1.0</code> <strong>Calibration Slope:</strong> Ideal is 1.0. A slope $< 1.0$ is the classic signature of overfitting (extreme probabilities too high and low).</div>
              <div class="term-item"><code>\text{Brier Score}</code> <strong>Overall Accuracy:</strong> Mean squared error of probabilities ($0 \le \text{Brier} \le 1$). Lower is superior.</div>
            </div>
          </div>
          <div class="metric-grid">
            <div class="metric">
              <span>Intercept (α = 0.0)</span>
              <b>Calibration-in-the-Large</b>
              <p class="subtle">α < 0 indicates systematic over-prediction; α > 0 indicates systematic under-prediction.</p>
            </div>
            <div class="metric">
              <span>Slope (β = 1.0)</span>
              <b>Spread & Overfitting</b>
              <p class="subtle">β < 1.0 indicates overfitting (extreme predictions too high and too low); β > 1.0 indicates underfitting.</p>
            </div>
          </div>
        `
      },
      {
        label: "2. The Brier Score & Probability Reliability",
        html: `
          <div class="patient-calc">
            <h4>Quantifying Probabilistic Accuracy with the Brier Score:</h4>
            <div class="formula">
              Brier Score = (1 / N) · ∑ (p̂ᵢ - yᵢ)²
            </div>
            <p>The Brier score measures the mean squared difference between predicted probabilities (0 to 1) and actual binary patient outcomes (0 or 1). A lower score indicates superior calibration and sharpness (0.0 is perfect clairvoyance; 0.25 is uninformative guessing for a 50% baseline event rate).</p>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"A clinical prediction model with stellar discrimination but poor calibration is medically dangerous; bedside triage decisions depend on absolute risk probabilities, not relative percentile rankings."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>Upon external validation in a community hospital, a stroke prediction model displays a calibration slope of <strong>β = 0.65</strong>. How do you interpret this finding?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) The model is underfitting the data.</button>
              <button class="quiz-opt" data-correct="true">B) The model is severely overfitted: high-risk predictions are too high, and low-risk predictions are too low.</button>
              <button class="quiz-opt" data-correct="false">C) The model has 65% accuracy.</button>
              <button class="quiz-opt" data-correct="false">D) The sample size is insufficient.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> A calibration slope less than 1.0 ($\\\\beta < 1.0$) is the classic hallmark of overfitting. The model's predictions are overly extreme and require statistical shrinkage before clinical deployment.
            </div>
          </div>
        `
      }
    ],
    lab: renderCalibrationLab
  },

  dca: {
    eyebrow: "Module 4 · Lesson 17 of 24",
    h1: "Decision Curve Analysis & Net Clinical Benefit",
    lead: "Does using your model actually help patients more than standard clinical protocols? Vickers' Decision Curve Analysis translates statistics into lives saved and unnecessary procedures avoided.",
    analogy: {
      title: "The Metal Detector at Airport Security",
      text: "If an airport metal detector beeps at every coin and belt buckle, it has 100% sensitivity for weapons, but it causes total airport gridlock (excess false positives). If set too low, it misses weapons (fatal false negatives). Decision Curve Analysis weighs the real clinical cost of missing a disease against the harm of false alarms."
    },
    sections: [
      {
        label: "1. The Net Benefit Mathematical Equation",
        html: `
          <div class="formula-card">
            <div class="formula-card__caption">Vickers' Net Clinical Benefit Formulation</div>
            <div class="formula-card__equation">
              \text{Net Benefit} = \frac{\text{TP}}{N} - \frac{\text{FP}}{N} \cdot \left(\frac{p_t}{1 - p_t}\right)
            </div>
            <div class="formula-card__caption">Term-by-Term Decision Breakdown</div>
            <div class="terms-grid">
              <div class="term-item"><code>\text{TP} / N</code> <strong>True Positive Rate:</strong> Benefit proportion — patients correctly identified who receive life-saving treatment.</div>
              <div class="term-item"><code>\text{FP} / N</code> <strong>False Positive Rate:</strong> Harm proportion — patients unnecessarily subjected to therapy risks.</div>
              <div class="term-item"><code>p_t / (1 - p_t)</code> <strong>Harm Weighting:</strong> Exchange rate setting the relative cost of a false alarm versus a missed true case.</div>
            </div>
          </div>
          <div class="patient-calc">
            <h4>The Clinical Decision Threshold ($p_t$):</h4>
            <p>The threshold $p_t$ is the risk level where a doctor and patient are indifferent between treating and not treating:</p>
            <ul>
              <li><strong>Low Threshold ($p_t = 5\\%$):</strong> High-stakes, safe intervention (e.g., non-contrast CT for suspected stroke). Missing a stroke is catastrophic, while an extra scan causes minimal harm.</li>
              <li><strong>High Threshold ($p_t = 30\\%$):</strong> Risky, invasive intervention (e.g., decompressive hemicraniectomy). Brain surgery carries severe surgical morbidity, so clinicians require high certainty of impending herniation.</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. The Three Curves on a Decision Curve Plot",
        html: `
          <div class="paper-grid">
            <div class="paper-card">
              <div class="paper-id">STRATEGY 1: TREAT ALL</div>
              <h2>"Intervene on Everyone"</h2>
              <p>Give every arriving patient the therapy regardless of model. Crosses Net Benefit = 0 at the population prevalence.</p>
            </div>
            <div class="paper-card">
              <div class="paper-id">STRATEGY 2: TREAT NONE</div>
              <h2>"Intervene on Nobody"</h2>
              <p>Provides zero true positives and zero false alarms. Mathematical horizontal baseline: Net Benefit = 0 across all thresholds.</p>
            </div>
          </div>
          <div class="presentation-gold" style="margin-top:15px;">
            <h4>Clinical Rule of Adoption:</h4>
            <p>A clinical prediction model is only useful in practice if its curve lies <strong>strictly above both Treat All and Treat None</strong> across the clinically relevant decision threshold range ($p_t$).</p>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Decision Curve Analysis proves clinical utility by quantifying Net Benefit in units of true positive diagnoses, demonstrating whether adopting the model avoids more unnecessary interventions than blanket treat-all or treat-none strategies."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>Across the entire plausible clinical decision threshold range (10% to 40%), a newly proposed machine learning model's Net Benefit curve overlaps exactly with the 'Treat All' strategy curve. Should the clinical service adopt this model?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Yes, because machine learning is more advanced.</button>
              <button class="quiz-opt" data-correct="true">B) No. The model provides zero incremental clinical utility over simply treating all arriving patients without running the model.</button>
              <button class="quiz-opt" data-correct="false">C) Yes, if AUC is > 0.80.</button>
              <button class="quiz-opt" data-correct="false">D) Only if p < 0.05.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> If a model's curve is identical to 'Treat All', using the model yields zero net benefit over the clinical default. Deploying the algorithm wastes diagnostic time and hospital resources without preventing a single unnecessary treatment!
            </div>
          </div>
        `
      }
    ],
    lab: renderDcaLab
  },

  continuous: {
    eyebrow: "Module 4 · Lesson 18 of 24",
    h1: "Continuous Net Benefit & Threshold-Free Metrics",
    lead: "Not all clinical decisions are binary yes/no. In acute stroke ICU care, blood pressure targets, intravenous fluids, and sedation depth are adjusted continuously.",
    analogy: {
      title: "The Hospital Room Thermostat",
      text: "You don't just turn the hospital room heating completely on or completely off. You adjust the temperature smoothly on a dial. Continuous net benefit evaluates whether fine-tuning the dial based on patient physiology provides superior clinical recovery compared to a fixed static target."
    },
    sections: [
      {
        label: "1. Moving Beyond Binary Cutoffs",
        html: `
          <p>Continuous clinical prediction evaluates utility when outcomes or interventions are graded on continuous scales:</p>
          <div class="metric-grid">
            <div class="metric">
              <span>Integrated Discrimination Improvement (IDI)</span>
              <b>Mean Probabilistic Gain</b>
              <p class="subtle">Quantifies average increase in predicted risk for event patients minus decrease for non-event patients.</p>
            </div>
            <div class="metric">
              <span>Continuous Net Benefit</span>
              <b>Integral Over Thresholds</b>
              <p class="subtle">Integrated clinical utility across continuous blood pressure and tissue perfusion intervention ranges.</p>
            </div>
          </div>
          <div class="patient-calc">
            <h4>The Fallacy of Continuous Net Reclassification Improvement (NRI):</h4>
            <p>Biostatisticians (Pepe, Vickers) demonstrated that 'category-free NRI' can produce statistically significant positive results (NRI > 0.30) even for completely useless random noise variables! In modern clinical prediction modeling, avoid category-free NRI and report standardized Decision Curves or IDI.</p>
          </div>
        `
      },
      {
        label: "2. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Continuous utility metrics evaluate graded clinical adjustments; however, uncalibrated category-free NRI generates false-positive endorsements of useless biomarkers and should be replaced with integrated Decision Curves."</p>
          </div>
        `
      },
      {
        label: "3. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>Why do leading statistical guidelines advise against reporting 'category-free' continuous Net Reclassification Improvement (NRI)?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) It is too difficult to compute.</button>
              <button class="quiz-opt" data-correct="true">B) It has a high false-positive rate and can assign positive reclassification credit to pure random noise variables.</button>
              <button class="quiz-opt" data-correct="false">C) It only works on survival data.</button>
              <button class="quiz-opt" data-correct="false">D) It requires 1,000 patients.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Category-free NRI rewards even microscopic probability shifts (e.g. 0.3001 vs 0.3000) that have zero clinical significance, frequently declaring pure noise biomarkers to be 'statistically significant improvements'.
            </div>
          </div>
        `
      }
    ],
    lab: renderContinuousLab
  },

  penalization: {
    eyebrow: "Module 5 · Lesson 19 of 24",
    h1: "Regularization: Ridge, Lasso, and Elastic Net",
    lead: "When you have 20 clinical candidate predictors but only 40 stroke deaths, standard regression overfits wildly. Regularization applies a mathematical tax on extreme coefficients.",
    analogy: {
      title: "The Bungee Cord Leash",
      text: "Imagine training a lively puppy. Without a leash (Ordinary Least Squares), the puppy chases every fluttering leaf and squirrel in the yard (memorizing sample noise). A bungee cord leash (penalization) pulls the puppy back toward center: it allows movement in response to strong commands (real biological signal), but prevents running wild into the bushes."
    },
    sections: [
      {
        label: "1. The Three Penalization Strategies",
        html: `
          <div class="paper-grid">
            <div class="paper-card">
              <div class="paper-id">RIDGE REGRESSION (L2)</div>
              <h2>Penalty: λ · ∑ βⱼ²</h2>
              <p>Shrinks all coefficients proportionally toward zero. Never sets a coefficient to exactly zero. Ideal for highly collinear imaging biomarkers (CTP Core, CBV, ASPECTS).</p>
            </div>
            <div class="paper-card">
              <div class="paper-id">LASSO REGRESSION (L1)</div>
              <h2>Penalty: λ · ∑ |βⱼ|</h2>
              <p>Shrinks coefficients and drives unimportant ones to <em>exactly zero</em>. Performs automatic feature selection, producing sparse clinical models.</p>
            </div>
          </div>
          <div class="patient-calc" style="margin-top:15px;">
            <h4>ELASTIC NET: The Hybrid Compromise</h4>
            <div class="formula">
              Elastic Net Penalty = λ · [ α · ∑ |βⱼ| + (1 - α) / 2 · ∑ βⱼ² ]
            </div>
            <p>Lasso arbitrarily picks only one predictor from a group of correlated clinical measurements (e.g., picking systolic BP and ignoring pulse pressure). Elastic Net groups correlated biomarkers together, retaining the entire biological cluster while shrinking noise.</p>
          </div>
        `
      },
      {
        label: "2. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Penalized regression trades a small amount of development bias for a massive reduction in prediction variance, preventing models from over-interpreting idiosyncratic clinical sample noise."</p>
          </div>
        `
      },
      {
        label: "3. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>You have 5 correlated perfusion CT variables measuring tissue transit time (Tmax > 4s, Tmax > 6s, Tmax > 8s, MTT, and TTP). Which regularized method is best suited to prevent collinearity blow-up?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Ordinary Least Squares without penalty</button>
              <button class="quiz-opt" data-correct="true">B) Ridge Regression or Elastic Net (which smoothly shrink correlated predictors together without erratic zeroing)</button>
              <button class="quiz-opt" data-correct="false">C) Forward Stepwise selection</button>
              <button class="quiz-opt" data-correct="false">D) Unadjusted univariate t-tests</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Ridge regression and Elastic Net handle severe multicollinearity with mathematical stability by penalizing the $L_2$ squared norm, shrinking the correlated perfusion transit maps in tandem.
            </div>
          </div>
        `
      }
    ],
    lab: renderPenalizationLab
  },

  ml: {
    eyebrow: "Module 5 · Lesson 20 of 24",
    h1: "Machine Learning vs. Clinical Statistical Modeling",
    lead: "Do Random Forests and Neural Networks beat Logistic Regression in stroke medicine? Systematic evidence reveals that modern black boxes rarely outperform well-specified regression on tabular patient data.",
    analogy: {
      title: "The Formula 1 Racecar on a Farm Track",
      text: "A Formula 1 supercar is breathtakingly fast on a paved racing circuit with unlimited fuel and a team of 30 mechanics (millions of image pixels in computer vision). But if you try to drive it across a muddy, rutted farm pasture (a 400-patient stroke cohort with missing lab values), a reliable four-wheel-drive pickup truck (logistic regression with splines) easily wins every time."
    },
    sections: [
      {
        label: "1. The Empirical Evidence from 71 Clinical Studies",
        html: `
          <div class="paper-grid">
            <div class="paper-card">
              <div class="paper-id">BMJ 2019 SYSTEMATIC REVIEW</div>
              <h2>"No Evidence of Superiority"</h2>
              <p>Across 71 clinical prediction studies comparing complex machine learning against multivariable logistic regression, median discrimination was identical: <strong>AUC difference = 0.00</strong>!</p>
            </div>
            <div class="paper-card">
              <div class="paper-id">THE HIDDEN RISK OF ML</div>
              <h2>Poor Calibration & Silent Leakage</h2>
              <p>Machine learning algorithms frequently produce overconfident, uncalibrated risk predictions and are 10× more prone to subtle data leakage during preprocessing.</p>
            </div>
          </div>
          <div class="patient-calc" style="margin-top:15px;">
            <h4>When Does Machine Learning Actually Win?</h4>
            <ul>
              <li><strong>Raw Unstructured Data:</strong> Segmenting stroke core volumes on raw 3D CT/MRI DICOM scans (Convolutional Neural Networks shine here!).</li>
              <li><strong>Massive Sample Sizes:</strong> Cohorts with $> 100,000$ patients and hundreds of complex high-order non-linear interactions.</li>
              <li><strong>For Standard Tabular Clinical Data:</strong> Logistic regression with Restricted Cubic Splines remains the international gold standard.</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Machine learning is not magical pixie dust for small clinical cohorts; for tabular patient records with moderate event counts, well-specified multivariable regression with restricted cubic splines routinely matches or outperforms complex black boxes."</p>
          </div>
        `
      },
      {
        label: "3. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A medical AI startup claims their 50-layer deep neural network achieved an AUC of 0.96 for predicting stroke discharge destination in a dataset of 180 patients. What is the most likely statistical reality?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) The neural network discovered profound new biological pathways.</button>
              <button class="quiz-opt" data-correct="true">B) Massive overfitting and data leakage (training on sample quirks that will collapse upon external validation in another hospital).</button>
              <button class="quiz-opt" data-correct="false">C) The cohort had too many events.</button>
              <button class="quiz-opt" data-correct="false">D) The learning rate was too low.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> With only 180 patients, a 50-layer neural network has vastly more parameters than data points! Achieving an AUC of 0.96 is virtually pathognomonic for severe overfitting, memorization, or data leakage.
            </div>
          </div>
        `
      }
    ],
    lab: renderMlLab
  },

  interpretability: {
    eyebrow: "Module 5 · Lesson 21 of 24",
    h1: "Model Explainability: SHAP & Nomograms",
    lead: "During a Code Stroke, an ICU physician will not trust a black-box algorithm they cannot audit. Explainability bridges statistical math and bedside clinical trust.",
    analogy: {
      title: "The Itemized Hospital Bill",
      text: "If a hospital hands you a single bill for $50,000 without explanation, you refuse to pay it. But if the bill itemizes $15,000 for operating room time, $20,000 for neuro-stents, and $15,000 for ICU room stay, every dollar is transparent. SHAP values itemize a patient's risk score into the exact clinical contribution of each predictor."
    },
    sections: [
      {
        label: "1. Classical Nomograms: The Bedside Scoring Ruler",
        html: `
          <p>Before smartphones, clinical prediction models were deployed as printed graphical <strong>nomograms</strong>:</p>
          <div class="patient-calc">
            <h4>How a Stroke Nomogram Works at the Bedside:</h4>
            <ul>
              <li>Each clinical variable (Age 75, NIHSS 18, Core 45 mL, CVO 2) is mapped to an integer point scale (e.g., 0 to 100 points).</li>
              <li>The clinician sums the points on a straight edge (Total Points = 165).</li>
              <li>The bottom ruler maps Total Points directly to 90-day probability of functional independence (e.g., 22%).</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. SHAP Values: Game-Theoretic Fair Attribution",
        html: `
          <div class="formula">
            Individual Prediction: f(x) = φ₀ + ∑ φⱼ
            <small>Where φ₀ is the base population risk and φⱼ is the local marginal contribution of predictor j for this specific patient.</small>
          </div>
          <div class="metric-grid">
            <div class="metric">
              <span>Patient Baseline (φ₀)</span>
              <b>32.0% Average Risk</b>
              <p class="subtle">Baseline risk of favorable recovery across the entire stroke registry.</p>
            </div>
            <div class="metric">
              <span>Local SHAP Offsets</span>
              <b>Age +8% · NIHSS -14% · CVO +16%</b>
              <p class="subtle">Final predicted recovery = 32% + 8% - 14% + 16% = <strong>42%</strong>.</p>
            </div>
          </div>
        `
      },
      {
        label: "3. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"A clinical risk prediction that cannot be audited at the bedside will not be trusted during an acute emergency; whether through classical nomograms or local SHAP attributions, clinicians must see why an individual patient was classified as high risk."</p>
          </div>
        `
      },
      {
        label: "4. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>For a severe stroke patient, a SHAP waterfall plot displays a large positive contribution ($+18\\%$) for Cortical Venous Outflow (CVO score = 6). What does this mean?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) The patient has an 18% mortality risk.</button>
              <button class="quiz-opt" data-correct="true">B) The patient's excellent venous outflow score increased their individual predicted probability of favorable functional recovery by 18 percentage points above the cohort average.</button>
              <button class="quiz-opt" data-correct="false">C) CVO explains 18% of variance in the cohort.</button>
              <button class="quiz-opt" data-correct="false">D) The model has an error of 18%.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> A positive SHAP value of $+18\\%$ for functional recovery directly communicates that this patient's robust collateral venous drainage boosted their individual recovery odds by 18 points over the baseline population expectation.
            </div>
          </div>
        `
      }
    ],
    lab: renderInterpretabilityLab
  },

  updating: {
    eyebrow: "Module 5 · Lesson 22 of 24",
    h1: "Model Updating & Recalibration for New Populations",
    lead: "A prediction model developed in an academic medical center will miscalibrate when deployed to a rural community hospital. Never discard the old model — update and recalibrate it.",
    analogy: {
      title: "The Altimeter Calibration",
      text: "When an airplane flies from sea level to Denver (5,280 feet altitude), the pilot does not throw away the altimeter. They turn a small dial to reset the local barometric pressure. Model updating does the exact same thing: it recalibrates the intercept and slope to fit the local patient casemix without throwing away years of previous clinical data."
    },
    sections: [
      {
        label: "1. The Three Tiers of Model Updating (Steyerberg Framework)",
        html: `
          <div class="paper-grid">
            <div class="paper-card">
              <div class="paper-id">TIER 1: RECALIBRATION-IN-THE-LARGE</div>
              <h2>Update Intercept Only (α)</h2>
              <p>Adjusts the baseline intercept to match local stroke event prevalence while preserving all relative predictor weights ($\\\\beta$).</p>
            </div>
            <div class="paper-card">
              <div class="paper-id">TIER 2: LOGISTIC RECALIBRATION</div>
              <h2>Update Intercept + Slope (α + β)</h2>
              <p>Adjusts baseline prevalence and scales the calibration slope to correct for overall overfitting or narrower casemix spread.</p>
            </div>
          </div>
          <div class="patient-calc" style="margin-top:15px;">
            <h4>TIER 3: MODEL REVISION & EXTENSION — Adding Novel Biomarkers</h4>
            <p>Re-estimates specific predictor weights or incorporates an entirely new biological modality — exactly what the <em>CVO CTP recalibration manuscript</em> achieved by augmenting conventional perfusion core volume with cortical venous outflow!</p>
          </div>
        `
      },
      {
        label: "2. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Do not discard an established prediction model simply because it mispredicts in your local hospital; updating the intercept and calibration slope preserves previous scientific evidence while adapting to local clinical casemix."</p>
          </div>
        `
      },
      {
        label: "3. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A validated stroke model systematically underestimates 90-day disability by 15% across all risk deciles in a rural hospital with an older population, but its discrimination remains strong (AUC = 0.81). What is the most efficient updating method?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Re-collect 5,000 patients and train a deep learning model from scratch.</button>
              <button class="quiz-opt" data-correct="true">B) Recalibrate the intercept (Calibration-in-the-large) to match the higher local baseline disability rate.</button>
              <button class="quiz-opt" data-correct="false">C) Discard the model entirely.</button>
              <button class="quiz-opt" data-correct="false">D) Change the p-value cutoff.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Because discrimination remains strong (AUC = 0.81), the relative ranking of predictors is intact! The model simply suffers from baseline prevalence shift, which is elegantly resolved by updating the intercept $\\\\alpha$.
            </div>
          </div>
        `
      }
    ],
    lab: renderUpdatingLab
  },

  tripod: {
    eyebrow: "Module 5 · Lesson 23 of 24",
    h1: "TRIPOD+AI Checklist & Transparent Reporting",
    lead: "A clinical prediction model that cannot be independently reproduced is medical fiction. The TRIPOD+AI guidelines ensure transparent reporting and patient safety.",
    analogy: {
      title: "The Surgical Safety Checklist",
      text: "Before making an incision, the surgical team pauses to confirm the patient identity, surgical site, allergies, and instrument count. The TRIPOD+AI statement is the surgical safety checklist of clinical statistics: 27 non-negotiable checks that prevent fatal errors before a model touches human patients."
    },
    sections: [
      {
        label: "1. Non-Negotiable TRIPOD+AI Requirements",
        html: `
          <div class="patient-calc">
            <h4>Key Mandatory Items from the 2024 TRIPOD+AI Statement:</h4>
            <ul>
              <li><strong>Full Mathematical Equation:</strong> Authors must report the exact intercept and all regression coefficients (or provide an open-access web calculator), allowing external clinicians to calculate risk for their own patients.</li>
              <li><strong>Sample Size & EPV Justification:</strong> Explicitly state the number of candidate parameters and verify adequate Events Per Variable ($EPV \\\\ge 10-20$).</li>
              <li><strong>Missing Data Transparency:</strong> Detail the percentage of missingness per variable and specify how MICE was executed.</li>
              <li><strong>Calibration Curves:</strong> Publishing discrimination (AUC) without a calibration plot is a major reporting violation.</li>
              <li><strong>Handling of Continuous Predictors:</strong> Report knot positions if restricted cubic splines were utilized.</li>
            </ul>
          </div>
        `
      },
      {
        label: "2. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"A clinical prediction manuscript that conceals its intercept, full mathematical equation, or calibration curve cannot be evaluated or externally validated; adherence to TRIPOD+AI is non-negotiable for clinical translation."</p>
          </div>
        `
      },
      {
        label: "3. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>A paper publishes a machine learning risk model reporting an AUC of 0.89, but the authors state that the model weights are proprietary and refuse to provide an equation or web calculator. Does this comply with TRIPOD+AI?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Yes, because trade secrets are protected.</button>
              <button class="quiz-opt" data-correct="true">B) No. TRIPOD+AI mandates complete reporting of model parameters or accessible deployment interfaces to ensure reproducibility and external validation.</button>
              <button class="quiz-opt" data-correct="false">C) Yes, if approved by an IRB.</button>
              <button class="quiz-opt" data-correct="false">D) Yes, if AUC > 0.85.</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> TRIPOD+AI Item 15 requires complete presentation of the prediction model (all regression coefficients, intercepts, or accessible executable software) to enable independent clinical validation.
            </div>
          </div>
        `
      }
    ],
    lab: renderTripodLab
  },

  antipatterns: {
    eyebrow: "Module 5 · Lesson 24 of 24",
    h1: "The Five Lethal Modeling Anti-Patterns",
    lead: "Can you spot the fatal methodological flaws in published medical AI papers? Master the five deadly sins of clinical prediction modeling.",
    analogy: {
      title: "The Infection Control Inspection",
      text: "In a sterile operating theatre, brushing against a non-sterile curtain with surgical gloves breaks the entire sterile field, even if you wash your hands afterward. In statistical modeling, peeking at the test set during imputation or variable selection permanently contaminates the entire study with data leakage."
    },
    sections: [
      {
        label: "1. The Five Deadly Sins of Clinical Prediction",
        html: `
          <div class="assumption-grid">
            <div class="assumption-card">
              <h5>1. Data Leakage (The Sneak Peek)</h5>
              <p><span class="bad">Fatal Flaw:</span> Performing imputation, z-score normalization, or feature selection on the entire dataset *before* splitting into cross-validation folds.<br><span class="good">Fix:</span> Execute every single preprocessing step strictly inside each cross-validation training fold.</p>
            </div>
            <div class="assumption-card">
              <h5>2. Dichotomania (The False Cliff)</h5>
              <p><span class="bad">Fatal Flaw:</span> Chopping continuous variables like age or infarct volume into arbitrary high/low categories.<br><span class="good">Fix:</span> Model non-linear continuous relationships smoothly using Restricted Cubic Splines.</p>
            </div>
            <div class="assumption-card">
              <h5>3. EPV Starvation (The Overfit Trap)</h5>
              <p><span class="bad">Fatal Flaw:</span> Testing 25 candidate variables with only 35 stroke mortality events ($EPV < 2$).<br><span class="good">Fix:</span> Budget 1 parameter per 15–20 events, or apply Ridge/Lasso penalization.</p>
            </div>
            <div class="assumption-card">
              <h5>4. Discrimination Myopia</h5>
              <p><span class="bad">Fatal Flaw:</span> Celebrating an AUC of 0.86 while completely hiding an atrocious, uncalibrated probability curve.<br><span class="good">Fix:</span> Always publish calibration curves with intercept, slope, and Decision Curve Analysis.</p>
            </div>
          </div>
          <div class="assumption-card" style="margin-top:10px;">
            <h5>5. Complete-Case Selection Bias (The Ghost Cohort)</h5>
            <p><span class="bad">Fatal Flaw:</span> Deleting all patients with any missing variable, turning an emergency stroke registry into a healthy survivor cohort.<br><span class="good">Fix:</span> Multiple Imputation by Chained Equations (MICE) under Missing at Random assumptions.</p>
          </div>
        `
      },
      {
        label: "2. Presentation Gold: What to Say on Rounds",
        html: `
          <div class="presentation-gold">
            <h4>💡 Presentation Gold: The One Sentence to Memorize</h4>
            <p>"Avoiding the five lethal anti-patterns distinguishes rigorous, life-saving predictive medicine from fragile, non-reproducible statistical noise."</p>
          </div>
        `
      },
      {
        label: "3. Check Your Clinical Understanding",
        html: `
          <div class="quiz-box">
            <h4>🧠 Clinical Intuition Quiz</h4>
            <p>An investigator normalizes all patient lab values to mean 0 and variance 1 across all 600 patients, and THEN splits the cohort into a 400-patient training set and a 200-patient test set. What lethal error occurred?</p>
            <div class="quiz-options">
              <button class="quiz-opt" data-correct="false">A) Underfitting</button>
              <button class="quiz-opt" data-correct="true">B) Data Leakage! The test set means and variances leaked into the training preprocessing pipeline, producing falsely optimistic test metrics.</button>
              <button class="quiz-opt" data-correct="false">C) Multicollinearity</button>
              <button class="quiz-opt" data-correct="false">D) Heteroscedasticity</button>
            </div>
            <div class="quiz-explanation">
              <strong>Explanation:</strong> Because the mean and standard deviation were computed across all 600 patients, information from the 200 test patients leaked into the training set! Preprocessing parameters must be calculated on the training fold only, then applied to the held-out test fold.
            </div>
          </div>
        `
      }
    ],
    lab: renderAntipatternsLab
  }
};

/* ==========================================================================
   Interactive Lab Renderers
   ========================================================================== */

function renderCsvLab(container) {
  const sampleData = [
    { id: "P-101", age: 67, nihss: 16, vol: 34.2, nwu: 6.8, time: 2.1, mrs90: 4 },
    { id: "P-102", age: 74, nihss: 8, vol: 12.0, nwu: 4.1, time: 1.5, mrs90: 1 },
    { id: "P-103", age: 59, nihss: 21, vol: 68.5, nwu: 9.4, time: 3.8, mrs90: 5 },
    { id: "P-104", age: 81, nihss: 14, vol: null, nwu: 7.2, time: 4.2, mrs90: 3 },
    { id: "P-105", age: 63, nihss: 6, vol: 8.4, nwu: 3.9, time: null, mrs90: 0 },
    { id: "P-106", age: 70, nihss: 19, vol: 54.1, nwu: 8.6, time: 2.9, mrs90: 6 }
  ];

  container.innerHTML = `
    <div class="lab-top">
      <span>Interactive Cohort Explorer</span>
      <span class="demo-label">Simulated Acute Stroke Cohort (n=6 preview)</span>
    </div>
    <div class="lab-body">
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Patient ID</th>
              <th>Age (yr)</th>
              <th>NIHSS (0-42)</th>
              <th>Core Vol (mm³)</th>
              <th>NWU (%)</th>
              <th>Onset (hr)</th>
              <th>90d mRS (0-6)</th>
            </tr>
          </thead>
          <tbody>
            ${sampleData.map(r => `
              <tr>
                <td><strong>${r.id}</strong></td>
                <td>${r.age}</td>
                <td>${r.nihss}</td>
                <td class="${r.vol === null ? 'missing' : ''}">${r.vol !== null ? r.vol : 'MISSING (NA)'}</td>
                <td>${r.nwu}%</td>
                <td class="${r.time === null ? 'missing' : ''}">${r.time !== null ? r.time + 'h' : 'MISSING (NA)'}</td>
                <td><strong>mRS ${r.mrs90}</strong></td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
      <div class="readout" style="margin-top: 16px;">
        Found <strong>2 missing entries</strong> highlighted in orange. Notice Patient P-104 is missing Core Volume and P-105 is missing Onset Time. In Module 3, we will see why dropping these rows is a catastrophic clinical mistake.
      </div>
    </div>
  `;
}

function renderLinearLab(container) {
  const pts = toyX.map((x, i) => ({ x, y: toyY[i] }));
  const model = ols(toyX, toyY);

  container.innerHTML = `
    <div class="lab-top">
      <span>Interactive OLS Regression Lab</span>
      <span class="demo-label">NIHSS ~ Infarct Volume</span>
    </div>
    <div class="lab-body">
      <div id="linear-chart-wrap"></div>
      <div class="controls">
        <div class="range-label">
          <span>Slope Adjustment (β₁):</span>
          <output id="slope-val">${model.b.toFixed(2)}</output>
        </div>
        <input type="range" id="slope-slider" min="1" max="7" step="0.1" value="${model.b.toFixed(1)}">
        <div class="lab-actions">
          <button id="reset-ols" class="outline small">Reset to Best OLS Fit</button>
          <button id="toggle-residuals" class="button small">Toggle Residual Error Bars</button>
        </div>
      </div>
      <div class="metric-grid">
        <div class="metric"><span>Current R² Score</span><b id="linear-r2">-</b></div>
        <div class="metric"><span>Mean Absolute Error (MAE)</span><b id="linear-mae">-</b></div>
      </div>
    </div>
  `;

  let currentSlope = model.b;
  let showResiduals = true;

  function updateChart() {
    const customModel = { a: mean(toyY) - currentSlope * mean(toyX), b: currentSlope };
    const preds = predict(customModel, toyX);
    const m = metrics(toyY, preds);

    const r2El = container.querySelector("#linear-r2");
    const maeEl = container.querySelector("#linear-mae");
    if (r2El) r2El.textContent = isNaN(m.r2) ? "0.000" : m.r2.toFixed(3);
    if (maeEl) maeEl.textContent = m.mae.toFixed(1) + " mm³";

    const width = 580;
    const height = 280;
    const pad = 35;
    const minX = 0, maxX = 28;
    const minY = 0, maxY = 120;

    const scaleX = x => pad + ((x - minX) / (maxX - minX)) * (width - 2 * pad);
    const scaleY = y => height - pad - ((y - minY) / (maxY - minY)) * (height - 2 * pad);

    const x1 = 0, y1 = customModel.a;
    const x2 = 26, y2 = customModel.a + customModel.b * 26;

    let svg = `
      <svg class="chart" viewBox="0 0 ${width} ${height}">
        <!-- Grid lines -->
        <line class="grid" x1="${pad}" y1="${scaleY(30)}" x2="${width - pad}" y2="${scaleY(30)}" />
        <line class="grid" x1="${pad}" y1="${scaleY(60)}" x2="${width - pad}" y2="${scaleY(60)}" />
        <line class="grid" x1="${pad}" y1="${scaleY(90)}" x2="${width - pad}" y2="${scaleY(90)}" />
        <!-- Axes -->
        <line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" />
        <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" />
        <!-- Axis labels -->
        <text x="${width - pad - 20}" y="${height - pad + 20}">Volume</text>
        <text x="${pad - 10}" y="${pad - 10}">NIHSS</text>
    `;

    if (showResiduals) {
      pts.forEach((p, i) => {
        const predY = preds[i];
        svg += `<line x1="${scaleX(p.x)}" y1="${scaleY(p.y)}" x2="${scaleX(p.x)}" y2="${scaleY(predY)}" stroke="#bd4d24" stroke-width="1.5" stroke-dasharray="3 3" />`;
      });
    }

    svg += `<line class="curve" x1="${scaleX(x1)}" y1="${scaleY(y1)}" x2="${scaleX(x2)}" y2="${scaleY(y2)}" />`;

    pts.forEach(p => {
      svg += `<circle class="point" cx="${scaleX(p.x)}" cy="${scaleY(p.y)}" r="5" />`;
    });

    svg += `</svg>`;
    const wrap = container.querySelector("#linear-chart-wrap");
    if (wrap) wrap.innerHTML = svg;
  }

  const slider = container.querySelector("#slope-slider");
  const slopeOutput = container.querySelector("#slope-val");
  if (slider) {
    slider.addEventListener("input", e => {
      currentSlope = parseFloat(e.target.value);
      if (slopeOutput) slopeOutput.textContent = currentSlope.toFixed(2);
      updateChart();
    });
  }

  const resetBtn = container.querySelector("#reset-ols");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      currentSlope = model.b;
      if (slider) slider.value = currentSlope.toFixed(1);
      if (slopeOutput) slopeOutput.textContent = currentSlope.toFixed(2);
      updateChart();
    });
  }

  const toggleBtn = container.querySelector("#toggle-residuals");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      showResiduals = !showResiduals;
      updateChart();
    });
  }

  updateChart();
}

function renderMetricsLab(container) {
  const actualY = [20, 24, 32, 45, 58, 65, 80];
  const baselinePreds = [22, 28, 30, 42, 64, 60, 75];

  container.innerHTML = `
    <div class="lab-top">
      <span>Interactive Residual Visualizer</span>
      <span class="demo-label">Weather Analogy Applied to Stroke Scores</span>
    </div>
    <div class="lab-body">
      <p class="subtle">Adjust the model's global systematic bias offset to see how MAE, RMSE, and Signed Error diverge.</p>
      <div class="controls">
        <div class="range-label">
          <span>Model Systematic Shift:</span>
          <output id="bias-out">0</output>
        </div>
        <input type="range" id="bias-slider" min="-15" max="15" step="1" value="0">
      </div>
      <div class="metric-grid" style="margin-top: 20px;">
        <div class="metric"><span>Mean Signed Error (Bias)</span><b id="met-signed">-</b></div>
        <div class="metric"><span>Mean Absolute Error (MAE)</span><b id="met-mae">-</b></div>
        <div class="metric"><span>Root Mean Squared (RMSE)</span><b id="met-rmse">-</b></div>
        <div class="metric"><span>Coefficient of Det. (R²)</span><b id="met-r2">-</b></div>
      </div>
      <div class="readout" id="met-explanation"></div>
    </div>
  `;

  function update(shift) {
    const shifted = baselinePreds.map(p => p + shift);
    const m = metrics(actualY, shifted);

    const sEl = container.querySelector("#met-signed");
    const aEl = container.querySelector("#met-mae");
    const rEl = container.querySelector("#met-rmse");
    const r2El = container.querySelector("#met-r2");
    const expEl = container.querySelector("#met-explanation");

    if (sEl) sEl.textContent = (m.signed >= 0 ? "+" : "") + m.signed.toFixed(1);
    if (aEl) aEl.textContent = m.mae.toFixed(1);
    if (rEl) rEl.textContent = m.rmse.toFixed(1);
    if (r2El) r2El.textContent = isNaN(m.r2) ? "0.00" : m.r2.toFixed(3);

    if (expEl) {
      if (Math.abs(m.signed) < 0.5) {
        expEl.innerHTML = `<strong>Balanced calibration:</strong> Signed error is close to 0, meaning positive and negative mistakes cancel out. However, RMSE (${m.rmse.toFixed(1)}) remains positive because individual patient errors are never zero.`;
      } else if (m.signed > 0) {
        expEl.innerHTML = `<strong>Systematic over-prediction:</strong> The model consistently over-estimates patient severity by an average of ${m.signed.toFixed(1)} points.`;
      } else {
        expEl.innerHTML = `<strong>Systematic under-prediction:</strong> The model is overly optimistic, missing acute deficits by an average of ${Math.abs(m.signed).toFixed(1)} points.`;
      }
    }
  }

  const slider = container.querySelector("#bias-slider");
  const biasOut = container.querySelector("#bias-out");
  if (slider) {
    slider.addEventListener("input", e => {
      const v = parseInt(e.target.value, 10);
      if (biasOut) biasOut.textContent = (v >= 0 ? "+" : "") + v;
      update(v);
    });
  }

  update(0);
}

function renderCausationLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Clinical Case Analysis</span>
      <span class="demo-label">Venous Outflow in Acute Ischemic Stroke</span>
    </div>
    <div class="lab-body">
      <div class="paper-note">
        <div class="section-label">THE CLINICAL SCENARIO</div>
        <p>A 72-year-old patient arrives with middle cerebral artery occlusion. CT Perfusion shows favorable collateral venous outflow (VO score 4/5). Does favorable venous outflow <em>cause</em> good recovery, or is it merely a marker of fast collateral blood flow?</p>
      </div>
      <div class="mode-switch" role="tablist">
        <button id="btn-assoc" class="active" role="tab" aria-pressed="true">1. Association View</button>
        <button id="btn-pred" role="tab" aria-pressed="false">2. Prediction View</button>
        <button id="btn-causal" role="tab" aria-pressed="false">3. Causal View</button>
      </div>
      <div id="causation-readout" class="readout"></div>
    </div>
  `;

  const views = {
    assoc: `
      <strong>The Association Finding:</strong> In our multivariable logistic regression across 527 reperfused patients, good venous outflow was strongly associated with functional independence (adjusted Odds Ratio = 0.64 per step, p = 0.002). This satisfies the question: <em>Is VO correlated with recovery after adjusting for age and baseline NIHSS?</em> Yes.
    `,
    pred: `
      <strong>The Prediction Reality:</strong> When adding venous outflow to a standard baseline clinical model (Age + NIHSS + Core Volume), the out-of-sample AUC only increased from 0.83 to 0.85 (ΔAUC = +0.02, 95% CI: -0.01 to +0.05). Despite a significant p-value, it added negligible incremental discrimination.
    `,
    causal: `
      <strong>The Causal Trap:</strong> Does stenting or artificially improving venous outflow guarantee tissue survival? No. Venous outflow may simply reflect undamaged microvascular beds. Intervening directly on a marker without a randomized counterfactual trial risks patient harm.
    `
  };

  const readout = container.querySelector("#causation-readout");
  if (readout) readout.innerHTML = views.assoc;

  ["assoc", "pred", "causal"].forEach(key => {
    const btn = container.querySelector(`#btn-${key}`);
    if (btn) {
      btn.addEventListener("click", () => {
        container.querySelectorAll(".mode-switch button").forEach(b => {
          b.classList.remove("active");
          b.setAttribute("aria-pressed", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-pressed", "true");
        if (readout) readout.innerHTML = views[key];
      });
    }
  });
}

function renderLogisticLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Interactive Sigmoid & Odds-to-Risk Converter</span>
      <span class="demo-label">p = 1 / (1 + e^(-z))</span>
    </div>
    <div class="lab-body">
      <div class="control-grid">
        <div>
          <div class="range-label">
            <span>Baseline Risk (P₀):</span>
            <output id="p0-out">20%</output>
          </div>
          <input type="range" id="p0-slider" min="1" max="90" value="20">
        </div>
        <div>
          <div class="range-label">
            <span>Odds Ratio (OR):</span>
            <output id="or-out">2.0×</output>
          </div>
          <input type="range" id="or-slider" min="0.2" max="5.0" step="0.1" value="2.0">
        </div>
      </div>
      <div class="metric-grid" style="margin-top: 20px;">
        <div class="metric"><span>Initial Baseline Odds</span><b id="res-odds0">-</b></div>
        <div class="metric"><span>New Post-Exposure Odds</span><b id="res-odds1">-</b></div>
        <div class="metric"><span>New Resulting Risk (P₁)</span><b id="res-risk1">-</b></div>
        <div class="metric"><span>True Relative Risk (RR)</span><b id="res-rr">-</b></div>
      </div>
      <div class="readout" id="or-risk-summary"></div>
    </div>
  `;

  function calc() {
    const p0Slider = container.querySelector("#p0-slider");
    const orSlider = container.querySelector("#or-slider");
    const p0 = (p0Slider ? parseFloat(p0Slider.value) : 20) / 100;
    const or = orSlider ? parseFloat(orSlider.value) : 2.0;

    const odds0 = p0 / (1 - p0);
    const odds1 = odds0 * or;
    const p1 = odds1 / (1 + odds1);
    const rr = p1 / p0;

    const p0Out = container.querySelector("#p0-out");
    const orOut = container.querySelector("#or-out");
    if (p0Out) p0Out.textContent = Math.round(p0 * 100) + "%";
    if (orOut) orOut.textContent = or.toFixed(1) + "×";

    const ro0 = container.querySelector("#res-odds0");
    const ro1 = container.querySelector("#res-odds1");
    const rk1 = container.querySelector("#res-risk1");
    const rrr = container.querySelector("#res-rr");
    const sumEl = container.querySelector("#or-risk-summary");

    if (ro0) ro0.textContent = odds0.toFixed(2);
    if (ro1) ro1.textContent = odds1.toFixed(2);
    if (rk1) rk1.textContent = (p1 * 100).toFixed(1) + "%";
    if (rrr) rrr.textContent = rr.toFixed(2) + "×";

    if (sumEl) {
      sumEl.innerHTML = `
        Notice the difference: The <strong>Odds Ratio is ${or.toFixed(1)}</strong>, but the true <strong>Relative Risk is only ${rr.toFixed(2)}×</strong>!
        When events are common, the odds ratio exaggerates risk perception. Always report absolute risk differences in clinical practice!
      `;
    }
  }

  const s1 = container.querySelector("#p0-slider");
  const s2 = container.querySelector("#or-slider");
  if (s1) s1.addEventListener("input", calc);
  if (s2) s2.addEventListener("input", calc);

  calc();
}

function renderOrdinalLab(container) {
  const mRsLabels = [
    "0: No symptoms",
    "1: No significant disability",
    "2: Slight disability (independent)",
    "3: Moderate disability (needs assistance)",
    "4: Moderately severe (bedbound/unable to walk)",
    "5: Severe disability (constant nursing care)",
    "6: Death"
  ];

  const baselineDist = [12, 18, 22, 16, 14, 10, 8]; // Percentages summing to 100

  container.innerHTML = `
    <div class="lab-top">
      <span>Modified Rankin Scale (mRS 0–6) Ordinal Shift</span>
      <span class="demo-label">Proportional Odds vs. Dichotomization</span>
    </div>
    <div class="lab-body">
      <p class="subtle">Select a binary cut point to see how collapsing the 7-level scale destroys clinical nuance.</p>
      <div class="lab-actions" style="margin-bottom: 20px;">
        <button class="small outline cut-btn active" data-cut="2">Cut at mRS ≤ 2 (Functional Independence)</button>
        <button class="small outline cut-btn" data-cut="3">Cut at mRS ≤ 3 (Good Outcome)</button>
        <button class="small outline cut-btn" data-cut="5">Cut at mRS ≤ 5 (Survival vs Death)</button>
      </div>
      <div id="ordinal-bars"></div>
      <div class="readout" id="ordinal-readout" style="margin-top: 18px;"></div>
    </div>
  `;

  function renderBars(cutPoint) {
    const barsWrap = container.querySelector("#ordinal-bars");
    const readout = container.querySelector("#ordinal-readout");
    if (!barsWrap) return;

    let favorable = 0;
    for (let i = 0; i <= cutPoint; i++) favorable += baselineDist[i];
    const unfavorable = 100 - favorable;

    let html = `
      <div style="margin-bottom: 12px; font-weight: 600; font-size: 0.875rem;">Full 7-Level mRS Distribution (%):</div>
      <div style="display: flex; height: 38px; border-radius: 8px; overflow: hidden; border: 1px solid var(--line);">
    `;

    const colors = ["#2b6cb0", "#3182ce", "#4299e1", "#ed8936", "#dd6b20", "#c53030", "#742a2a"];
    baselineDist.forEach((pct, idx) => {
      html += `
        <div style="width: ${pct}%; background: ${colors[idx]}; color: white; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 600;" title="${mRsLabels[idx]}: ${pct}%">
          ${idx}
        </div>
      `;
    });
    html += `</div>`;

    html += `
      <div style="margin-top: 20px; font-weight: 600; font-size: 0.875rem;">Dichotomized Binary Collapse (Cut at mRS ≤ ${cutPoint}):</div>
      <div style="display: flex; height: 34px; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); margin-top: 8px;">
        <div style="width: ${favorable}%; background: var(--teal); color: white; display: flex; align-items: center; justify-content: center; font-size: 0.8125rem; font-weight: 600;">
          Favorable: ${favorable}%
        </div>
        <div style="width: ${unfavorable}%; background: var(--orange); color: white; display: flex; align-items: center; justify-content: center; font-size: 0.8125rem; font-weight: 600;">
          Poor Outcome: ${unfavorable}%
        </div>
      </div>
    `;

    barsWrap.innerHTML = html;

    if (readout) {
      readout.innerHTML = `
        <strong>Clinical Consequence:</strong> A patient moving from <strong>mRS 5 (bedridden, incontinent)</strong> to <strong>mRS 3 (walks with cane)</strong> experiences a life-changing recovery. But under the traditional mRS 0–2 dichotomy, that improvement is counted as a complete failure (0 = 0). The <strong>proportional odds model</strong> captures this improvement across every boundary!
      `;
    }
  }

  container.querySelectorAll(".cut-btn").forEach(btn => {
    btn.addEventListener("click", e => {
      container.querySelectorAll(".cut-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderBars(parseInt(btn.getAttribute("data-cut"), 10));
    });
  });

  renderBars(2);
}

function renderSurvivalLab(container) {
  const times = [2, 5, 8, 12, 15, 22, 28, 30, 45, 60, 75, 90];
  const events = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]; // 1 = death, 0 = censored
  const curve = km(times, events);

  container.innerHTML = `
    <div class="lab-top">
      <span>Kaplan-Meier Survival Estimator</span>
      <span class="demo-label">90-Day Post-Thrombectomy Follow-Up</span>
    </div>
    <div class="lab-body">
      <div id="km-chart-wrap"></div>
      <div class="legend" style="margin-top: 14px;">
        <span><span class="key blue"></span>Survival Probability S(t)</span>
        <span><span class="key orange"></span>Censored Patient (+)</span>
      </div>
      <div class="readout" style="margin-top: 14px;">
        Notice the stepwise drops in survival occur strictly at event times. Cross-hairs on the curve indicate censored patients who safely completed follow-up or were transferred without dying.
      </div>
    </div>
  `;

  const width = 560, height = 260, pad = 35;
  const maxT = 90;
  const scaleX = t => pad + (t / maxT) * (width - 2 * pad);
  const scaleY = s => height - pad - s * (height - 2 * pad);

  let pathD = `M ${scaleX(curve[0][0])} ${scaleY(curve[0][1])}`;
  for (let i = 1; i < curve.length; i++) {
    pathD += ` L ${scaleX(curve[i][0])} ${scaleY(curve[i][1])}`;
  }

  let svg = `
    <svg class="chart" viewBox="0 0 ${width} ${height}">
      <line class="grid" x1="${pad}" y1="${scaleY(0.25)}" x2="${width - pad}" y2="${scaleY(0.25)}" />
      <line class="grid" x1="${pad}" y1="${scaleY(0.50)}" x2="${width - pad}" y2="${scaleY(0.50)}" />
      <line class="grid" x1="${pad}" y1="${scaleY(0.75)}" x2="${width - pad}" y2="${scaleY(0.75)}" />
      <line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" />
      <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" />
      <text x="${width - pad - 30}" y="${height - pad + 20}">Days</text>
      <text x="${pad - 10}" y="${pad - 10}">S(t)</text>
      <path d="${pathD}" fill="none" stroke="var(--blue)" stroke-width="3" />
  `;

  times.forEach((t, idx) => {
    if (!events[idx]) {
      const sAtT = curve.find(p => p[0] >= t)?.[1] || 0.6;
      const cx = scaleX(t), cy = scaleY(sAtT);
      svg += `<line x1="${cx}" y1="${cy - 4}" x2="${cx}" y2="${cy + 4}" stroke="var(--orange)" stroke-width="2" />`;
    }
  });

  svg += `</svg>`;
  const wrap = container.querySelector("#km-chart-wrap");
  if (wrap) wrap.innerHTML = svg;
}

function renderMissingLab(container) {
  const estimates = [0.65, 0.61, 0.70, 0.68, 0.62];
  const variances = [0.008, 0.009, 0.007, 0.008, 0.009];
  const pooled = rubin(estimates, variances);

  container.innerHTML = `
    <div class="lab-top">
      <span>Rubin's Rules Pooling Calculator</span>
      <span class="demo-label">Combining MICE Imputations (m=5)</span>
    </div>
    <div class="lab-body">
      <p class="subtle">Multiple imputation creates m separate completed datasets. Rubin's rules pools the estimates while capturing both within-imputation and between-imputation uncertainty.</p>
      <div class="metric-grid">
        <div class="metric"><span>Pooled Point Estimate (Q)</span><b>${pooled.Q.toFixed(3)}</b><p class="subtle">Average of the 5 model coefficients</p></div>
        <div class="metric"><span>Within Variance (W)</span><b>${pooled.W.toFixed(4)}</b><p class="subtle">Average statistical sampling variance</p></div>
        <div class="metric"><span>Between Variance (B)</span><b>${pooled.B.toFixed(4)}</b><p class="subtle">Variance due to missing data uncertainty</p></div>
        <div class="metric"><span>Total Standard Error (SE)</span><b>${pooled.se.toFixed(3)}</b><p class="subtle">sqrt(W + (1 + 1/m)B)</p></div>
      </div>
      <div class="readout">
        <strong>Why Complete-Case Analysis Fails:</strong> Simply dropping missing patients shrinks sample size and deflates standard errors, falsely claiming higher precision than the data supports. Rubin's rules accurately widens the confidence interval to reflect missing information.
      </div>
    </div>
  `;
}

function renderSplinesLab(container) {
  const ptsX = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const ptsY = [12, 13, 16, 22, 31, 48, 62, 70, 74, 76]; // Sigmoidal / plateau biology

  container.innerHTML = `
    <div class="lab-top">
      <span>Restricted Cubic Splines (RCS) Knot Explorer</span>
      <span class="demo-label">Admission NWU vs. Mortality Curve</span>
    </div>
    <div class="lab-body">
      <div id="spline-chart-wrap"></div>
      <div class="controls" style="margin-top: 16px;">
        <div class="range-label">
          <span>Spline Knots:</span>
          <output id="knot-out">4 Knots (Recommended)</output>
        </div>
        <input type="range" id="knot-slider" min="3" max="5" step="1" value="4">
      </div>
      <div class="readout" id="spline-readout" style="margin-top: 14px;"></div>
    </div>
  `;

  function draw(knotCount) {
    const knotMap = {
      3: [2, 5, 8],
      4: [1.5, 4, 7, 9],
      5: [1, 3, 5.5, 8, 9.5]
    };
    const knots = knotMap[knotCount];
    const basis = ptsX.map(x => rcs(x, knots));
    const beta = fitBasis(basis, ptsY);

    const width = 560, height = 260, pad = 35;
    const minX = 0, maxX = 11;
    const minY = 0, maxY = 85;

    const scaleX = x => pad + ((x - minX) / (maxX - minX)) * (width - 2 * pad);
    const scaleY = y => height - pad - ((y - minY) / (maxY - minY)) * (height - 2 * pad);

    const curvePoints = [];
    for (let x = 1; x <= 10; x += 0.2) {
      const b = rcs(x, knots);
      const y = dot(b, beta);
      curvePoints.push({ x, y });
    }

    let pathD = `M ${scaleX(curvePoints[0].x)} ${scaleY(curvePoints[0].y)}`;
    curvePoints.slice(1).forEach(p => {
      pathD += ` L ${scaleX(p.x)} ${scaleY(p.y)}`;
    });

    const linear = ols(ptsX, ptsY);

    let svg = `
      <svg class="chart" viewBox="0 0 ${width} ${height}">
        <line class="grid" x1="${pad}" y1="${scaleY(20)}" x2="${width - pad}" y2="${scaleY(20)}" />
        <line class="grid" x1="${pad}" y1="${scaleY(50)}" x2="${width - pad}" y2="${scaleY(50)}" />
        <line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" />
        <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" />
        <text x="${width - pad - 20}" y="${height - pad + 20}">NWU %</text>
        <text x="${pad - 10}" y="${pad - 10}">Risk %</text>
        <!-- Straight line comparison -->
        <line class="secondary" x1="${scaleX(1)}" y1="${scaleY(linear.a + linear.b * 1)}" x2="${scaleX(10)}" y2="${scaleY(linear.a + linear.b * 10)}" />
        <!-- Spline curve -->
        <path d="${pathD}" fill="none" stroke="var(--blue)" stroke-width="3" />
    `;

    knots.forEach(k => {
      svg += `<line x1="${scaleX(k)}" y1="${pad}" x2="${scaleX(k)}" y2="${height - pad}" stroke="#a0aec0" stroke-width="1" stroke-dasharray="2 2" />`;
    });

    ptsX.forEach((x, i) => {
      svg += `<circle class="point" cx="${scaleX(x)}" cy="${scaleY(ptsY[i])}" r="4" />`;
    });

    svg += `</svg>`;
    const wrap = container.querySelector("#spline-chart-wrap");
    if (wrap) wrap.innerHTML = svg;

    const out = container.querySelector("#knot-out");
    const rEl = container.querySelector("#spline-readout");
    if (out) out.textContent = `${knotCount} Knots (${knotCount - 1} degrees of freedom)`;
    if (rEl) {
      rEl.innerHTML = `
        <strong>Teal Dashed Line = Rigid Straight Fit. Solid Blue = Restricted Cubic Spline.</strong><br>
        Notice how the spline accommodates the physiological threshold around 5% NWU and the biological ceiling above 9% without breaking linearity at the extremes.
      `;
    }
  }

  const slider = container.querySelector("#knot-slider");
  if (slider) {
    slider.addEventListener("input", e => {
      draw(parseInt(e.target.value, 10));
    });
  }

  draw(4);
}

function renderPhenotypingLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Residual Phenotyping Simulation</span>
      <span class="demo-label">Admission NWU vs. 24h Increment</span>
    </div>
    <div class="lab-body">
      <div class="readout" style="margin-bottom: 16px;">
        In 626 thrombectomy patients, expected 24h edema increment was modeled from baseline NWU. Patients sitting far above the regression line (+5% excess) define the <strong>Greater-than-Expected Progression</strong> malignant phenotype.
      </div>
      <div class="control-grid">
        <div>
          <div class="range-label"><span>Admission NWU:</span><output id="pheno-adm">7.5%</output></div>
          <input type="range" id="pheno-adm-slider" min="3.0" max="14.0" step="0.5" value="7.5">
        </div>
        <div>
          <div class="range-label"><span>Observed 24h NWU:</span><output id="pheno-obs">15.0%</output></div>
          <input type="range" id="pheno-obs-slider" min="5.0" max="25.0" step="0.5" value="15.0">
        </div>
      </div>
      <div class="metric-grid" style="margin-top: 20px;">
        <div class="metric"><span>Observed Increment</span><b id="res-obs-inc">-</b></div>
        <div class="metric"><span>Expected Increment</span><b id="res-exp-inc">-</b></div>
        <div class="metric"><span>Excess Progression Residual</span><b id="res-excess">-</b></div>
        <div class="metric"><span>Phenotype Classification</span><b id="res-pheno-class">-</b></div>
      </div>
    </div>
  `;

  function calc() {
    const adm = parseFloat(container.querySelector("#pheno-adm-slider")?.value || 7.5);
    const obs = parseFloat(container.querySelector("#pheno-obs-slider")?.value || 15.0);

    const admOut = container.querySelector("#pheno-adm");
    const obsOut = container.querySelector("#pheno-obs");
    if (admOut) admOut.textContent = adm.toFixed(1) + "%";
    if (obsOut) obsOut.textContent = obs.toFixed(1) + "%";

    const obsInc = obs - adm;
    // Expected model: expected increment = 10.5 - 0.6 * admission NWU
    const expInc = Math.max(0, 10.5 - 0.6 * adm);
    const excess = obsInc - expInc;

    const oEl = container.querySelector("#res-obs-inc");
    const eEl = container.querySelector("#res-exp-inc");
    const xEl = container.querySelector("#res-excess");
    const cEl = container.querySelector("#res-pheno-class");

    if (oEl) oEl.textContent = obsInc.toFixed(1) + "%";
    if (eEl) eEl.textContent = expInc.toFixed(1) + "%";
    if (xEl) xEl.textContent = (excess >= 0 ? "+" : "") + excess.toFixed(1) + "%";
    if (cEl) {
      if (excess > 2.5) {
        cEl.innerHTML = `<span style="color: var(--orange)">Malignant Excess</span>`;
      } else if (excess < -2.5) {
        cEl.innerHTML = `<span style="color: var(--teal)">Favorable Attenuation</span>`;
      } else {
        cEl.textContent = "Expected Range";
      }
    }
  }

  container.querySelector("#pheno-adm-slider")?.addEventListener("input", calc);
  container.querySelector("#pheno-obs-slider")?.addEventListener("input", calc);
  calc();
}

function renderInteractionsLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Interactive Adjusted Risk Difference Calculator</span>
      <span class="demo-label">Admission NWU × Excess Edema Progression</span>
    </div>
    <div class="lab-body">
      <div class="controls">
        <div class="range-label">
          <span>Baseline Admission NWU Quartile:</span>
          <output id="int-adm-out">High Admission NWU (> 8.5%)</output>
        </div>
        <input type="range" id="int-adm-slider" min="1" max="3" step="1" value="3">
      </div>
      <div class="metric-grid" style="margin-top: 20px;">
        <div class="metric"><span>Adjusted Mortality (Expected)</span><b id="int-risk-exp">18%</b></div>
        <div class="metric"><span>Adjusted Mortality (Excess)</span><b id="int-risk-exc">38%</b></div>
        <div class="metric"><span>Absolute Risk Difference</span><b id="int-ard">+20.0%</b></div>
        <div class="metric"><span>Interaction Synergy</span><b id="int-synergy">High (p=0.012)</b></div>
      </div>
      <div class="readout" id="int-summary">
        In patients with <strong>high admission NWU</strong>, excess edema progression unleashes a <strong>20.0 percentage point increase in 90-day mortality</strong>. In contrast, for patients with low admission NWU, excess progression has a much smaller clinical impact.
      </div>
    </div>
  `;

  const slider = container.querySelector("#int-adm-slider");
  const admOut = container.querySelector("#int-adm-out");
  const rExp = container.querySelector("#int-risk-exp");
  const rExc = container.querySelector("#int-risk-exc");
  const ard = container.querySelector("#int-ard");
  const syn = container.querySelector("#int-synergy");
  const sumEl = container.querySelector("#int-summary");

  if (slider) {
    slider.addEventListener("input", e => {
      const v = parseInt(e.target.value, 10);
      if (v === 1) {
        if (admOut) admOut.textContent = "Low Admission NWU (< 5.0%)";
        if (rExp) rExp.textContent = "8%";
        if (rExc) rExc.textContent = "12%";
        if (ard) ard.textContent = "+4.0%";
        if (syn) syn.textContent = "Minimal (p=0.45)";
        if (sumEl) sumEl.innerHTML = "At low baseline edema, brain tissue has reserve capacity. Excess progression increases mortality by only 4 percentage points.";
      } else if (v === 2) {
        if (admOut) admOut.textContent = "Moderate Admission NWU (5.0 - 8.5%)";
        if (rExp) rExp.textContent = "14%";
        if (rExc) rExc.textContent = "24%";
        if (ard) ard.textContent = "+10.0%";
        if (syn) syn.textContent = "Moderate (p=0.08)";
        if (sumEl) sumEl.innerHTML = "At moderate baseline edema, excess progression begins to overwhelm intracranial compliance.";
      } else {
        if (admOut) admOut.textContent = "High Admission NWU (> 8.5%)";
        if (rExp) rExp.textContent = "18%";
        if (rExc) rExc.textContent = "38%";
        if (ard) ard.textContent = "+20.0%";
        if (syn) syn.textContent = "High (p=0.012)";
        if (sumEl) sumEl.innerHTML = "<strong>Manuscript Anchor:</strong> High baseline edema combined with excess progression creates catastrophic tissue loss, driving a +20.0% adjusted absolute mortality difference.";
      }
    });
  }
}

function renderFeaturesLab(container) {
  const correlationMatrix = [
    { var1: "Core Vol (rCBF<30%)", var2: "Tmax > 6s", corr: 0.88, danger: true },
    { var1: "Core Vol (rCBF<30%)", var2: "Tmax > 10s", corr: 0.74, danger: true },
    { var1: "Tmax > 6s", var2: "Tmax > 10s", corr: 0.91, danger: true },
    { var1: "Core Vol", var2: "Admission NIHSS", corr: 0.58, danger: false },
    { var1: "Age", var2: "Core Vol", corr: 0.12, danger: false },
    { var1: "Venous Outflow", var2: "Mismatch Ratio", corr: 0.44, danger: false }
  ];

  container.innerHTML = `
    <div class="lab-top">
      <span>CTP Multicollinearity Matrix</span>
      <span class="demo-label">Correlation Pairs in Perfusion Imaging</span>
    </div>
    <div class="lab-body">
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Predictor 1</th>
              <th>Predictor 2</th>
              <th>Pearson r</th>
              <th>Multicollinearity Risk</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            ${correlationMatrix.map(m => `
              <tr>
                <td><strong>${m.var1}</strong></td>
                <td><strong>${m.var2}</strong></td>
                <td><strong>r = ${m.corr.toFixed(2)}</strong></td>
                <td style="color: ${m.danger ? 'var(--orange)' : 'var(--teal)'}; font-weight: 600;">
                  ${m.danger ? 'SEVERELY COLLINEAR' : 'Acceptable'}
                </td>
                <td>${m.danger ? 'Penalize (Ridge/Elastic Net) or Combine' : 'Retain as separate covariate'}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      </div>
      <div class="readout" style="margin-top: 16px;">
        Entering both <em>Core Volume</em> and <em>Tmax > 6s</em> into unpenalized regression inflates standard errors by over 400%, making both appear artificially non-significant!
      </div>
    </div>
  `;
}

function renderValidationLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Repeated 5-Fold Cross-Validation Visualizer</span>
      <span class="demo-label">Stratified Partitions across Repetitions</span>
    </div>
    <div class="lab-body">
      <div class="lab-actions" style="margin-bottom: 16px;">
        <button id="run-rep-btn" class="button small">Simulate Next Repetition</button>
        <span id="rep-label" class="demo-label" style="align-self: center;">Repetition 1 of 20</span>
      </div>
      <div class="folds" id="fold-container">
        <div class="fold test">Fold 1 (Test: 26 pts)</div>
        <div class="fold">Fold 2 (Train: 105 pts)</div>
        <div class="fold">Fold 3 (Train: 105 pts)</div>
        <div class="fold">Fold 4 (Train: 105 pts)</div>
        <div class="fold">Fold 5 (Train: 105 pts)</div>
      </div>
      <div class="readout" id="cv-readout" style="margin-top: 16px;">
        In small stroke cohorts (e.g. n=131 with only 19 events), each 20% test fold contains roughly <strong>3 to 4 events</strong>. A single lucky or unlucky split can produce test AUCs swinging wildly from 0.52 to 0.78! Running 20 repeated partitions averages out this partition variance.
      </div>
    </div>
  `;

  let currentRep = 1;
  const btn = container.querySelector("#run-rep-btn");
  const repLabel = container.querySelector("#rep-label");
  const foldContainer = container.querySelector("#fold-container");

  if (btn) {
    btn.addEventListener("click", () => {
      currentRep = (currentRep % 20) + 1;
      if (repLabel) repLabel.textContent = `Repetition ${currentRep} of 20`;
      if (foldContainer) {
        const testFold = ((currentRep - 1) % 5) + 1;
        foldContainer.innerHTML = Array.from({ length: 5 }, (_, i) => {
          const isTest = (i + 1) === testFold;
          return `<div class="fold ${isTest ? 'test' : ''}">Fold ${i + 1} (${isTest ? 'Test: 26 pts' : 'Train: 105 pts'})</div>`;
        }).join('');
      }
    });
  }
}

function renderBootstrapLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Full-Pipeline Bootstrap Resampling</span>
      <span class="demo-label">Patient Token Pool (n=30 demo)</span>
    </div>
    <div class="lab-body">
      <div class="lab-actions" style="margin-bottom: 14px;">
        <button id="resample-btn" class="button small">Draw New Bootstrap Replicate</button>
      </div>
      <div class="tokens" id="token-pool"></div>
      <div class="metric-grid" style="margin-top: 18px;">
        <div class="metric"><span>In-Bag (Train Sample)</span><b id="in-bag-count">19 (~63.2%)</b></div>
        <div class="metric"><span>Out-of-Bag (Validation Test)</span><b id="oob-count">11 (~36.8%)</b></div>
      </div>
      <div class="readout" id="boot-readout">
        Blue tokens were selected into the bootstrap training replicate. Orange tokens are the <strong>Out-of-Bag (OOB) patients</strong>. Performance evaluated strictly on OOB patients provides an honest, optimism-corrected estimate.
      </div>
    </div>
  `;

  function drawTokens() {
    const n = 30;
    const random = Math.random;
    const { drawn, oob } = bootstrapIndices(n, random);
    const seen = new Set(drawn);

    const pool = container.querySelector("#token-pool");
    if (pool) {
      pool.innerHTML = Array.from({ length: n }, (_, i) => {
        const isOob = !seen.has(i);
        return `<div class="token ${isOob ? 'out' : 'event'}" title="${isOob ? 'Out-of-Bag (Validation)' : 'Selected in Bootstrap'}">${i + 1}</div>`;
      }).join('');
    }

    const inEl = container.querySelector("#in-bag-count");
    const outEl = container.querySelector("#oob-count");
    if (inEl) inEl.textContent = `${n - oob.length} (${(((n - oob.length) / n) * 100).toFixed(0)}%)`;
    if (outEl) outEl.textContent = `${oob.length} (${((oob.length / n) * 100).toFixed(0)}%)`;
  }

  container.querySelector("#resample-btn")?.addEventListener("click", drawTokens);
  drawTokens();
}

function renderAucLab(container) {
  const y = binaryY;
  const p = binaryP;

  container.innerHTML = `
    <div class="lab-top">
      <span>Interactive ROC & Decision Threshold Lab</span>
      <span class="demo-label">Cohort: 12 Patients (6 Events, 6 Non-Events)</span>
    </div>
    <div class="lab-body">
      <div class="controls">
        <div class="range-label">
          <span>Decision Probability Cutoff:</span>
          <output id="thresh-val">0.40</output>
        </div>
        <input type="range" id="thresh-slider" min="0.10" max="0.90" step="0.05" value="0.40">
      </div>
      <div class="metric-grid" style="margin-top: 18px;">
        <div class="metric"><span>Overall AUC</span><b>0.833</b><p class="subtle">Invariant to threshold</p></div>
        <div class="metric"><span>Sensitivity</span><b id="conf-sens">-</b><p class="subtle">True Positive Rate</p></div>
        <div class="metric"><span>Specificity</span><b id="conf-spec">-</b><p class="subtle">True Negative Rate</p></div>
        <div class="metric"><span>Positive Predictive Val.</span><b id="conf-ppv">-</b><p class="subtle">Precision</p></div>
      </div>
      <div class="table-wrap" style="margin-top: 16px;">
        <table class="data-table">
          <thead>
            <tr><th>Confusion Matrix</th><th>Actual Event (1)</th><th>Actual No Event (0)</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Test Positive (≥ threshold)</strong></td><td id="cell-tp">-</td><td id="cell-fp">-</td></tr>
            <tr><td><strong>Test Negative (< threshold)</strong></td><td id="cell-fn">-</td><td id="cell-tn">-</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `;

  function updateConf(t) {
    const c = confusion(y, p, t);
    container.querySelector("#thresh-val").textContent = t.toFixed(2);
    container.querySelector("#conf-sens").textContent = (c.sensitivity * 100).toFixed(0) + "%";
    container.querySelector("#conf-spec").textContent = (c.specificity * 100).toFixed(0) + "%";
    container.querySelector("#conf-ppv").textContent = isNaN(c.ppv) ? "N/A" : (c.ppv * 100).toFixed(0) + "%";

    container.querySelector("#cell-tp").textContent = `TP: ${c.tp}`;
    container.querySelector("#cell-fp").textContent = `FP: ${c.fp}`;
    container.querySelector("#cell-fn").textContent = `FN: ${c.fn}`;
    container.querySelector("#cell-tn").textContent = `TN: ${c.tn}`;
  }

  container.querySelector("#thresh-slider")?.addEventListener("input", e => {
    updateConf(parseFloat(e.target.value));
  });

  updateConf(0.40);
}

function renderCalibrationLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Interactive Calibration Plot</span>
      <span class="demo-label">Calibration Slope & Intercept Diagnostics</span>
    </div>
    <div class="lab-body">
      <div id="calib-chart-wrap"></div>
      <div class="controls" style="margin-top: 16px;">
        <div class="range-label">
          <span>Model Calibration Slope:</span>
          <output id="cal-slope-val">0.91 (Mild Overfitting)</output>
        </div>
        <input type="range" id="cal-slope-slider" min="0.5" max="1.5" step="0.05" value="0.91">
      </div>
      <div class="readout" id="cal-readout" style="margin-top: 14px;"></div>
    </div>
  `;

  function drawCal(slope) {
    const width = 560, height = 260, pad = 35;
    const scaleX = p => pad + p * (width - 2 * pad);
    const scaleY = p => height - pad - p * (height - 2 * pad);

    const x1 = 0.05, y1 = 0.05 * slope + (1 - slope) * 0.5;
    const x2 = 0.95, y2 = 0.95 * slope + (1 - slope) * 0.5;

    let svg = `
      <svg class="chart" viewBox="0 0 ${width} ${height}">
        <line class="grid" x1="${pad}" y1="${scaleY(0.5)}" x2="${width - pad}" y2="${scaleY(0.5)}" />
        <line class="axis" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" />
        <line class="axis" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" />
        <text x="${width - pad - 40}" y="${height - pad + 20}">Predicted</text>
        <text x="${pad - 10}" y="${pad - 10}">Observed</text>
        <!-- Perfect 45-degree calibration line -->
        <line class="secondary" x1="${scaleX(0)}" y1="${scaleY(0)}" x2="${scaleX(1)}" y2="${scaleY(1)}" />
        <!-- Actual model calibration curve -->
        <line class="curve" x1="${scaleX(x1)}" y1="${scaleY(y1)}" x2="${scaleX(x2)}" y2="${scaleY(y2)}" />
      </svg>
    `;

    const wrap = container.querySelector("#calib-chart-wrap");
    if (wrap) wrap.innerHTML = svg;

    const out = container.querySelector("#cal-slope-val");
    const rEl = container.querySelector("#cal-readout");
    if (out) out.textContent = `${slope.toFixed(2)} ${slope < 0.9 ? '(Overfitting)' : slope > 1.1 ? '(Underfitting)' : '(Well Calibrated)'}`;

    if (rEl) {
      if (slope < 0.9) {
        rEl.innerHTML = `<strong>Slope = ${slope.toFixed(2)} &lt; 1.0:</strong> Overfitting detected! High predicted risks are too high and low predicted risks are too low. In Module 5 we show how shrinkage / penalization cures this.`;
      } else if (slope > 1.1) {
        rEl.innerHTML = `<strong>Slope = ${slope.toFixed(2)} &gt; 1.0:</strong> Underfitting. The model predictions are overly conservative and shrunk too close to the cohort mean.`;
      } else {
        rEl.innerHTML = `<strong>Manuscript Anchor:</strong> Calibration slope 0.91 and Brier score 0.107 demonstrate reliable, well-calibrated probabilistic forecasts for clinical decision-making.`;
      }
    }
  }

  container.querySelector("#cal-slope-slider")?.addEventListener("input", e => {
    drawCal(parseFloat(e.target.value));
  });

  drawCal(0.91);
}

function renderDcaLab(container) {
  const y = binaryY;
  const p = binaryP;

  container.innerHTML = `
    <div class="lab-top">
      <span>Decision Curve Analysis (DCA) Net Benefit</span>
      <span class="demo-label">Clinical Threshold Probability Evaluation</span>
    </div>
    <div class="lab-body">
      <div class="controls">
        <div class="range-label">
          <span>Clinical Intervention Threshold:</span>
          <output id="dca-thresh-out">30%</output>
        </div>
        <input type="range" id="dca-thresh-slider" min="10" max="80" step="5" value="30">
      </div>
      <div class="metric-grid" style="margin-top: 18px;">
        <div class="metric"><span>Model Net Benefit</span><b id="nb-model">-</b></div>
        <div class="metric"><span>Treat-All Strategy</span><b id="nb-all">-</b></div>
        <div class="metric"><span>Treat-None Strategy</span><b>0.000</b></div>
        <div class="metric"><span>Clinical Recommendation</span><b id="nb-rec">-</b></div>
      </div>
      <div class="readout" id="dca-readout" style="margin-top: 16px;"></div>
    </div>
  `;

  function updateDca(t) {
    const nbModel = netBenefit(y, p, t);
    // Treat all: tp = sum(y), fp = n - tp
    const tpAll = sum(y);
    const fpAll = y.length - tpAll;
    const nbAll = tpAll / y.length - (fpAll / y.length) * (t / (1 - t));

    container.querySelector("#dca-thresh-out").textContent = (t * 100).toFixed(0) + "%";
    container.querySelector("#nb-model").textContent = nbModel.toFixed(3);
    container.querySelector("#nb-all").textContent = nbAll.toFixed(3);

    const rec = container.querySelector("#nb-rec");
    const rEl = container.querySelector("#dca-readout");

    if (nbModel > Math.max(0, nbAll)) {
      if (rec) rec.innerHTML = `<span style="color: var(--teal)">Use Prediction Model</span>`;
      if (rEl) rEl.innerHTML = `At a <strong>${(t * 100).toFixed(0)}% risk threshold</strong>, using the model delivers superior net clinical benefit compared to treating everybody or treating nobody.`;
    } else {
      if (rec) rec.innerHTML = `<span style="color: var(--orange)">Default Strategy Wins</span>`;
      if (rEl) rEl.innerHTML = `At this extreme threshold, the model does not beat simple standard of care defaults.`;
    }
  }

  container.querySelector("#dca-thresh-slider")?.addEventListener("input", e => {
    updateDca(parseFloat(e.target.value) / 100);
  });

  updateDca(0.30);
}

function renderContinuousLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Continuous Outcome Comparison</span>
      <span class="demo-label">ANCOVA vs. Change-Score vs. Mixed Models</span>
    </div>
    <div class="lab-body">
      <div class="paper-grid">
        <div class="paper-card">
          <div class="paper-id">METHOD A: RAW CHANGE SCORE</div>
          <h2>ΔY = Follow-up - Baseline</h2>
          <p>Subtracting admission edema from 24h edema ignores baseline variation and suffers from regression to the mean.</p>
          <dl>
            <dt>Out-of-Bag R²</dt><dd>0.024 (Very poor generalization)</dd>
            <dt>Fatal Flaw</dt><dd>Sicker patients appear to progress faster solely due to measurement variance.</dd>
          </dl>
        </div>
        <div class="paper-card">
          <div class="paper-id">METHOD B: ANCOVA ADJUSTMENT</div>
          <h2>Follow-up ~ Treatment + Baseline</h2>
          <p>Controls for baseline severity as a continuous covariate. Standard in clinical trials and our edema progression study.</p>
          <dl>
            <dt>Out-of-Bag R²</dt><dd>0.315 (Robust out-of-sample fit)</dd>
            <dt>Clinical Benefit</dt><dd>Removes confounding by admission infarct magnitude.</dd>
          </dl>
        </div>
      </div>
    </div>
  `;
}

function renderPenalizationLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Penalization & Shrinkage Comparison</span>
      <span class="demo-label">Ridge vs. Lasso vs. Elastic Net</span>
    </div>
    <div class="lab-body">
      <p class="subtle">Watch how increasing the regularization penalty λ shrinks unstable coefficients toward zero.</p>
      <div class="controls">
        <div class="range-label">
          <span>Penalty Strength (λ):</span>
          <output id="pen-lambda-out">Moderate (Optimal)</output>
        </div>
        <input type="range" id="pen-slider" min="0" max="3" step="1" value="1">
      </div>
      <div id="pen-bars" style="margin-top: 20px;"></div>
      <div class="readout" id="pen-readout" style="margin-top: 16px;"></div>
    </div>
  `;

  const configs = [
    {
      label: "Zero Penalty (OLS / Overfit)",
      weights: [14.2, -12.8, 8.4, -6.1, 9.8],
      readout: "Unpenalized regression produces huge, offsetting coefficients for correlated perfusion variables. Standard errors explode!"
    },
    {
      label: "Ridge Penalty (L2 Shrinkage)",
      weights: [4.1, 3.8, 2.5, 1.9, 3.2],
      readout: "Ridge shrinks all coefficients smoothly together, preserving all predictors while stabilizing prediction variance."
    },
    {
      label: "Lasso Penalty (L1 Sparsity)",
      weights: [6.2, 0.0, 3.1, 0.0, 0.0],
      readout: "Lasso zeroes out redundant variables, performing automatic feature selection. But it arbitrarily selects only one variable from correlated groups!"
    },
    {
      label: "Elastic Net (Combined L1 + L2)",
      weights: [5.0, 3.2, 2.1, 0.8, 1.5],
      readout: "<strong>Gold Standard:</strong> Elastic Net shrinks coefficients like Ridge while eliminating truly irrelevant noise like Lasso."
    }
  ];

  const varNames = ["Infarct Core", "Tmax > 6s", "Admission NIHSS", "Blood Glucose", "Collateral Score"];

  function updatePen(idx) {
    const c = configs[idx];
    container.querySelector("#pen-lambda-out").textContent = c.label;
    container.querySelector("#pen-readout").innerHTML = c.readout;

    const barsWrap = container.querySelector("#pen-bars");
    if (barsWrap) {
      barsWrap.innerHTML = c.weights.map((w, i) => `
        <div class="bar-row">
          <span>${varNames[i]}</span>
          <div class="bar-track">
            <div class="bar-fill ${w === 0 ? 'orange' : ''}" style="width: ${Math.min(100, Math.abs(w) * 6)}%;"></div>
          </div>
          <span style="font-variant-numeric: tabular-nums;">${w.toFixed(1)}</span>
        </div>
      `).join('');
    }
  }

  container.querySelector("#pen-slider")?.addEventListener("input", e => {
    updatePen(parseInt(e.target.value, 10));
  });

  updatePen(1);
}

function renderMlLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>Machine Learning vs. Penalized Regression Benchmark</span>
      <span class="demo-label">Paired Bootstrap Comparison (n=626 Cohort)</span>
    </div>
    <div class="lab-body">
      <div class="table-wrap">
        <table class="data-table">
          <thead>
            <tr>
              <th>Model Architecture</th>
              <th>Out-of-Bag AUC</th>
              <th>Calibration Slope</th>
              <th>Brier Score</th>
              <th>Interpretability</th>
            </tr>
          </thead>
          <tbody>
            <tr style="background: var(--blue-light);">
              <td><strong>Elastic Net Logistic Regression</strong></td>
              <td><strong>0.842</strong> [0.81-0.87]</td>
              <td><strong>0.96</strong> (Well calibrated)</td>
              <td><strong>0.107</strong></td>
              <td>High (Equations & Nomograms)</td>
            </tr>
            <tr>
              <td><strong>Random Forest (500 trees)</strong></td>
              <td>0.835 [0.80-0.86]</td>
              <td>0.84 (Slightly overconfident)</td>
              <td>0.114</td>
              <td>Medium (Feature importance)</td>
            </tr>
            <tr>
              <td><strong>XGBoost (Gradient Boosted)</strong></td>
              <td>0.846 [0.81-0.88]</td>
              <td>0.81 (Needs recalibration)</td>
              <td>0.111</td>
              <td>Low (Black box tree ensembles)</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="readout" style="margin-top: 16px;">
        <strong>The Empirical Reality:</strong> XGBoost achieved a tiny +0.004 increase in AUC over Elastic Net, but had worse calibration and zero clinically transparent formulas. In medical tabular data, well-tuned penalized regression remains the first-line champion.
      </div>
    </div>
  `;
}

function renderInterpretabilityLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>SHAP Feature Attribution Explorer</span>
      <span class="demo-label">Explaining a Single Patient's Stroke Risk</span>
    </div>
    <div class="lab-body">
      <div class="readout" style="margin-bottom: 16px;">
        Patient P-247 (Predicted 90-Day Mortality Risk: <strong>42%</strong> vs. Cohort Base Rate: <strong>15%</strong>).
      </div>
      <div class="bar-row">
        <span>Age: 82 yr</span>
        <div class="bar-track"><div class="bar-fill orange" style="width: 65%;"></div></div>
        <span>+14%</span>
      </div>
      <div class="bar-row">
        <span>NIHSS: 22</span>
        <div class="bar-track"><div class="bar-fill orange" style="width: 55%;"></div></div>
        <span>+12%</span>
      </div>
      <div class="bar-row">
        <span>NWU: 9.2%</span>
        <div class="bar-track"><div class="bar-fill orange" style="width: 35%;"></div></div>
        <span>+8%</span>
      </div>
      <div class="bar-row">
        <span>eTICI: 3 (Full)</span>
        <div class="bar-track"><div class="bar-fill teal" style="width: 30%;"></div></div>
        <span>-7%</span>
      </div>
      <div class="readout" style="margin-top: 16px;">
        <strong>Never Forget:</strong> SHAP allocates statistical attribution, not causal effect. Successful reperfusion (eTICI 3) lowered risk by 7%, but elderly age (+14%) and massive neurological deficit (+12%) drove the primary vulnerability.
      </div>
    </div>
  `;
}

function renderUpdatingLab(container) {
  container.innerHTML = `
    <div class="lab-top">
      <span>External Transportability & Updating Lab</span>
      <span class="demo-label">Transferring a Model to a New Stroke Center</span>
    </div>
    <div class="lab-body">
      <div class="paper-grid">
        <div class="paper-card">
          <div class="paper-id">STRATEGY 1: INTERCEPT UPDATE</div>
          <h2>Recalibration-in-the-Large</h2>
          <p>When the new hospital treats an older, sicker population with a 25% baseline mortality rate instead of 15%.</p>
          <dl>
            <dt>Adjustment</dt><dd>Recalibrate the intercept term β₀</dd>
            <dt>Sample Size Needed</dt><dd>Small (n ~ 100-200)</dd>
          </dl>
        </div>
        <div class="paper-card">
          <div class="paper-id">STRATEGY 2: RECALIBRATION SLOPE</div>
          <h2>Slope Logistic Adjustment</h2>
          <p>When case-mix variation causes predictions to be too extreme across the entire risk continuum.</p>
          <dl>
            <dt>Adjustment</dt><dd>Multiply all β coefficients by slope γ</dd>
            <dt>Sample Size Needed</dt><dd>Moderate (n ~ 200-400)</dd>
          </dl>
        </div>
      </div>
    </div>
  `;
}

function renderTripodLab(container) {
  const checklistItems = [
    "Title explicitly identifies study as development or validation (Item 1)",
    "Target clinical population and intended decision point specified (Item 2)",
    "Eligibility criteria and recruitment dates stated (Item 4b)",
    "Predictor measurement timing restricted to intended decision point (Item 5a)",
    "Outcome definition and blinded adjudication described (Item 6)",
    "Sample size calculation or EPV justification reported (Item 8)",
    "Missing data handling (MICE) conducted inside validation folds (Item 9)",
    "Non-linear continuous predictor handling (splines) pre-specified (Item 10a)",
    "Internal validation via bootstrap or repeated CV reported (Item 10b)",
    "Both discrimination (AUC) and calibration reported (Item 13a)",
    "Full final mathematical model formula published (Item 15a)",
    "Decision trace of all analytical decisions audited (Item 22)"
  ];

  container.innerHTML = `
    <div class="lab-top">
      <span>TRIPOD+AI Interactive Checklist</span>
      <span class="demo-label">International Prediction Modeling Standards</span>
    </div>
    <div class="lab-body">
      <div class="checklist">
        ${checklistItems.map((item, idx) => `
          <label>
            <input type="checkbox" class="tripod-check" checked>
            <span><strong>${idx + 1}.</strong> ${item}</span>
          </label>
        `).join('')}
      </div>
      <div class="readout" style="margin-top: 16px;">
        Adherence Score: <strong id="tripod-score">12 / 12 (100% Compliant)</strong>. Meeting all TRIPOD+AI guidelines ensures publication integrity and regulatory acceptance.
      </div>
    </div>
  `;

  const boxes = container.querySelectorAll(".tripod-check");
  const scoreEl = container.querySelector("#tripod-score");
  boxes.forEach(b => {
    b.addEventListener("change", () => {
      const checked = container.querySelectorAll(".tripod-check:checked").length;
      if (scoreEl) scoreEl.textContent = `${checked} / 12 (${Math.round((checked / 12) * 100)}% Compliant)`;
    });
  });
}

function renderAntipatternsLab(container) {
  const cards = [
    {
      title: "1. Pre-Resample Imputation Leak",
      leak: "Imputing missing data on the whole cohort before train/test splitting.",
      fix: "Run MICE strictly inside each cross-validation training fold."
    },
    {
      title: "2. The Apparent AUC Boast",
      leak: "Reporting in-sample training AUC as model performance.",
      fix: "Report strictly out-of-bag bootstrap or repeated cross-validated AUC."
    },
    {
      title: "3. Stepwise Selection Trap",
      leak: "Using stepwise p-value elimination to select clinical predictors.",
      fix: "Use Elastic Net shrinkage or pre-specified domain covariates."
    },
    {
      title: "4. The Post-Treatment Predictor",
      leak: "Using 24-hour follow-up imaging in a model intended for emergency room admission triage.",
      fix: "Align predictor timing strictly with the decision moment."
    },
    {
      title: "5. Odds Ratio As Risk Percentage",
      leak: "Narrating an Odds Ratio of 2.0 as 'doubled risk' in high-prevalence stroke death.",
      fix: "Calculate adjusted absolute risk differences via marginal standardization."
    },
    {
      title: "6. Competing Risk Censoring",
      leak: "Censoring non-stroke cardiac deaths in long-term recurrence studies.",
      fix: "Use Fine-Gray subdistribution hazards models."
    }
  ];

  container.innerHTML = `
    <div class="lab-top">
      <span>Interactive Spot-the-Leak Challenge</span>
      <span class="demo-label">Click a card to reveal the fatal leak and remedy</span>
    </div>
    <div class="lab-body">
      <div class="paper-grid">
        ${cards.map((c, i) => `
          <div class="paper-card" style="cursor: pointer;" id="leak-card-${i}">
            <div class="paper-id">CHALLENGE #${i + 1}</div>
            <h2>${c.title}</h2>
            <p class="leak-text" style="color: var(--orange); font-weight: 600;">⚠ ${c.leak}</p>
            <div class="fix-box" style="display: none; margin-top: 12px; padding: 12px; background: white; border-radius: 8px; border: 1px solid var(--line);">
              <strong>Methodological Remedy:</strong> ${c.fix}
            </div>
          </div>
        `).join('')}
      </div>
    </div>
  `;

  cards.forEach((_, i) => {
    const card = container.querySelector(`#leak-card-${i}`);
    if (card) {
      card.addEventListener("click", () => {
        const fix = card.querySelector(".fix-box");
        if (fix) {
          fix.style.display = fix.style.display === "none" ? "block" : "none";
        }
      });
    }
  });
}

/* ==========================================================================
   Special Pages: The Four Papers & Sources
   ========================================================================== */

function renderPapersPage(container) {
  container.innerHTML = `
    <div class="lesson-head">
      <div class="eyebrow">APPLIED STROKE RESEARCH MAP</div>
      <h1>The Four Benchmark Manuscripts</h1>
      <p class="lead">Every statistical lesson in this curriculum is anchored in peer-reviewed acute ischemic stroke research from international thrombectomy registries.</p>
    </div>
    <div class="paper-grid" style="margin-top: 35px;">
      <!-- Paper 1 -->
      <div class="paper-card">
        <div class="paper-id">MANUSCRIPT 1 · EDEMA PHENOTYPING</div>
        <h2>Greater-than-Expected NWU Progression After Thrombectomy</h2>
        <p>Derives a residual edema phenotype capturing malignant progression beyond baseline volume expectations.</p>
        <dl>
          <dt>Study Cohort</dt><dd>n = 626 edema analysis, n = 600 with 90-day mRS</dd>
          <dt>Key Methods</dt><dd>Restricted cubic splines, ANCOVA, Full-pipeline bootstrap, Marginal standardization</dd>
          <dt>Key Results</dt><dd>Progression AUC 0.55 alone; Death AUC 0.84 with baseline clinical factors; <strong>Adjusted mortality risk difference +20.0%</strong> in high admission edema.</dd>
        </dl>
      </div>

      <!-- Paper 2 -->
      <div class="paper-card">
        <div class="paper-id">MANUSCRIPT 2 · PERFUSION RECALIBRATION</div>
        <h2>Integrating Cerebral Venous Outflow with CT Perfusion</h2>
        <p>Evaluates whether cortical venous outflow recalibrates ischemic lesion predictions and improves functional prognosis.</p>
        <dl>
          <dt>Study Cohorts</dt><dd>527 successfully reperfused, 131 unsuccessful reperfusion</dd>
          <dt>Key Methods</dt><dd>Repeated stratified 5-fold CV (×20 reps), MICE (25 imputations), Bootstrap (2,000 reps), Linear recalibration</dd>
          <dt>Key Results</dt><dd>Apparent R² 0.335 drops to Out-of-Bag R² 0.077; MAE ~20,021 mm³; Signed Error +12,900 mm³; Calibration slope 0.91.</dd>
        </dl>
      </div>

      <!-- Paper 3 -->
      <div class="paper-card">
        <div class="paper-id">MANUSCRIPT 3 · FAVORABLE PERFUSION</div>
        <h2>Beyond Favorable CT Perfusion in Thrombectomy</h2>
        <p>Analyzes why patients with favorable CT perfusion profiles still experience poor functional recovery.</p>
        <dl>
          <dt>Study Cohort</dt><dd>n = 151 acute anterior circulation strokes (90 poor functional outcomes)</dd>
          <dt>Key Methods</dt><dd>Proportional odds logistic regression, MICE (m=20), Repeated CV (×10 reps), 1,000 bootstrap replicates</dd>
          <dt>Key Results</dt><dd>Venous outflow common OR 0.64 per step; Residual perfusion variation AUC 0.65; Full clinical model AUC 0.85.</dd>
        </dl>
      </div>

      <!-- Paper 4 -->
      <div class="paper-card">
        <div class="paper-id">MANUSCRIPT 4 · HEMORRHAGIC TRANSFORMATION</div>
        <h2>Striatal Localization of Hemorrhagic Risk After Thrombectomy</h2>
        <p>Investigates deep anatomical parenchymal hemorrhage patterns and infarct core burden.</p>
        <dl>
          <dt>Study Cohort</dt><dd>Thrombectomy registry with adjudicated post-recanalization hemorrhage</dd>
          <dt>Key Methods</dt><dd>Joint multivariable models, Benjamini-Hochberg FDR correction, Rubin pooling, Optimism correction</dd>
          <dt>Key Results</dt><dd>Identified striatal territory vulnerability; likelihood ratio tests confirmed non-linear threshold for hemorrhage risk.</dd>
        </dl>
      </div>
    </div>
  `;
}

function renderSourcesPage(container) {
  container.innerHTML = `
    <div class="lesson-head">
      <div class="eyebrow">PEDAGOGY & METHODOLOGY</div>
      <h1>Sources, Scope & Technical References</h1>
      <p class="lead">Built to demystify complex medical statistics for clinical researchers, fellows, and students.</p>
    </div>
    <div class="lesson-body" style="margin-top: 35px;">
      <div class="section">
        <div class="section-label">Core Pedagogical Framework</div>
        <p>This explainer is inspired by the visual, interactive pedagogy of <em>MLU Explain</em>, translated into clinical prediction modeling for healthcare. Every concept is grounded in the 19-chapter clinical prediction modeling curriculum and TRIPOD+AI standards.</p>
      </div>
      <div class="section">
        <div class="section-label">Literature & Standards Consulted</div>
        <ul class="sources-list">
          <li><strong>TRIPOD+AI Statement:</strong> Collins GS, et al. <em>BMJ</em> 2024. Reporting guidelines for clinical prediction models using AI and regression.</li>
          <li><strong>Clinical Prediction Models:</strong> Steyerberg EW. <em>Springer</em> 2nd edition. Comprehensive guidance on validation, updating, and calibration.</li>
          <li><strong>Regression Modeling Strategies:</strong> Harrell FE Jr. <em>Springer</em>. Restricted cubic splines, penalization, and avoiding stepwise selection.</li>
          <li><strong>Sample Size for Clinical Prediction:</strong> Riley RD, et al. <em>Statistics in Medicine</em>. Minimum sample size calculations and EPV criteria.</li>
          <li><strong>Decision Curve Analysis:</strong> Vickers AJ, Elkin EB. <em>Medical Decision Making</em>. Evaluating net clinical benefit.</li>
        </ul>
      </div>
      <div class="section">
        <div class="section-label">Privacy & Zero-Egress Guarantee</div>
        <p>All data and computations on this site run entirely in your web browser using client-side JavaScript. No patient health information (PHI) or personal data is ever collected, stored, or transmitted across the network.</p>
      </div>
    </div>
  `;
}

/* ==========================================================================
   Main Application Router & State Machine
   ========================================================================== */

class App {
  constructor() {
    this.contentEl = document.getElementById("content");
    this.lessonNavEl = document.getElementById("lesson-nav");
    this.menuToggle = document.getElementById("menu-toggle");
    this.sidebar = document.getElementById("sidebar");
    this.topLinks = document.querySelectorAll("header nav a, .topbar nav a, .masthead__nav a");

    this.init();
  }

  init() {
    this.renderSidebar();
    this.setupEvents();
    this.handleRoute();
  }

  setupEvents() {
    window.addEventListener("hashchange", () => this.handleRoute());

    // Robust click delegation for all lesson navigation links
    document.addEventListener("click", (e) => {
      const link = e.target.closest("a[href^='#']");
      if (link) {
        const href = link.getAttribute("href");
        if (href && href.startsWith("#")) {
          const targetHash = href.replace(/^#/, "").trim();
          if (targetHash) {
            e.preventDefault();
            if (window.location.hash !== `#${targetHash}`) {
              window.location.hash = targetHash;
            } else {
              this.handleRoute();
            }
            window.scrollTo({ top: 0, behavior: "smooth" });
            const mainEl = document.getElementById("main");
            if (mainEl) mainEl.scrollTop = 0;
          }
        }
      }
    });

    // Keyboard navigation: Left / Right arrows
    window.addEventListener("keydown", (e) => {
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT")) return;
      if (e.key === "ArrowRight") {
        const nextBtn = document.querySelector(".lesson-bottom a.button, .lesson-bottom a.next-btn");
        if (nextBtn) {
          e.preventDefault();
          nextBtn.click();
        }
      } else if (e.key === "ArrowLeft") {
        const prevBtn = document.querySelector(".lesson-bottom a.outline, .lesson-bottom a.prev-btn");
        if (prevBtn) {
          e.preventDefault();
          prevBtn.click();
        }
      }
    });
    window.addEventListener("hashchange", () => this.handleRoute());

    if (this.menuToggle && this.sidebar) {
      this.menuToggle.addEventListener("click", () => {
        const isOpen = this.sidebar.classList.toggle("open");
        this.menuToggle.setAttribute("aria-expanded", String(isOpen));
      });
    }

    // Close mobile menu on clicking any link inside sidebar
    if (this.sidebar) {
      this.sidebar.addEventListener("click", e => {
        if (e.target.closest("a") && this.sidebar.classList.contains("open")) {
          this.sidebar.classList.remove("open");
          if (this.menuToggle) this.menuToggle.setAttribute("aria-expanded", "false");
        }
      });
    }

    // Close menu when clicking outside on mobile and handle quiz clicks
    document.addEventListener("click", (e) => {
      if (this.sidebar && this.sidebar.classList.contains("open")) {
        if (!this.sidebar.contains(e.target) && e.target !== this.menuToggle) {
          this.sidebar.classList.remove("open");
          if (this.menuToggle) this.menuToggle.setAttribute("aria-expanded", "false");
        }
      }

      // Interactive quiz option clicks
      const quizBtn = e.target.closest(".quiz-opt");
      if (quizBtn) {
        const quizBox = quizBtn.closest(".quiz-box");
        if (quizBox) {
          quizBox.querySelectorAll(".quiz-opt").forEach(b => b.classList.remove("correct", "incorrect"));
          const isCorrect = quizBtn.getAttribute("data-correct") === "true";
          quizBtn.classList.add(isCorrect ? "correct" : "incorrect");
          const exp = quizBox.querySelector(".quiz-explanation");
          if (exp) exp.classList.add("show");
        }
      }
    });
  }

  renderSidebar() {
    if (!this.lessonNavEl) return;

    let html = "";
    CURRICULUM_GROUPS.forEach(g => {
      html += `
        <div class="nav-group">
          <h2>${g.title}</h2>
          ${g.lessons.map(l => `
            <a href="#${l.id}" id="nav-link-${l.id}" data-lesson="${l.id}">
              <span class="lesson-num">${l.num}</span>
              <span>${l.title}</span>
            </a>
          `).join('')}
        </div>
      `;
    });
    this.lessonNavEl.innerHTML = html;
  }

  updateNavActive(hash) {
    // Update topbar active state
    if (this.topLinks) {
      this.topLinks.forEach(a => {
        const topTarget = a.getAttribute("data-top");
        if (hash === "papers" && topTarget === "papers") {
          a.classList.add("active");
        } else if (hash === "sources" && topTarget === "sources") {
          a.classList.add("active");
        } else if (hash !== "papers" && hash !== "sources" && topTarget === "lessons") {
          a.classList.add("active");
        } else {
          a.classList.remove("active");
        }
      });
    }

    // Update sidebar active lesson
    if (this.lessonNavEl) {
      this.lessonNavEl.querySelectorAll(".nav-group a").forEach(a => {
        if (a.getAttribute("data-lesson") === hash) {
          a.classList.add("active");
          a.scrollIntoView({ block: "nearest", behavior: "smooth" });
        } else {
          a.classList.remove("active");
        }
      });
    }
  }

  handleRoute() {
    const rawHash = window.location.hash.replace(/^#/, "").trim();
    const hash = rawHash || "csv";

    // Close mobile menu on navigate
    if (this.sidebar && this.sidebar.classList.contains("open")) {
      this.sidebar.classList.remove("open");
      if (this.menuToggle) this.menuToggle.setAttribute("aria-expanded", "false");
    }

    this.updateNavActive(hash);

    // Scroll to top
    const mainEl = document.getElementById("main");
    if (mainEl) mainEl.scrollTop = 0;
    window.scrollTo(0, 0);

    if (hash === "papers") {
      renderPapersPage(this.contentEl);
      return;
    }

    if (hash === "sources") {
      renderSourcesPage(this.contentEl);
      return;
    }

    const lesson = LESSON_CONTENT[hash] || LESSON_CONTENT["csv"];
    const lessonMeta = LESSONS_MAP[hash] || LESSONS_MAP["csv"];
    this.renderLesson(hash, lesson, lessonMeta);
  }

  renderLesson(id, lesson, meta) {
    if (!this.contentEl) return;

    // Find previous and next lessons
    const flatLessons = [];
    CURRICULUM_GROUPS.forEach(g => g.lessons.forEach(l => flatLessons.push(l)));
    const curIdx = flatLessons.findIndex(l => l.id === id);
    const prev = curIdx > 0 ? flatLessons[curIdx - 1] : null;
    const next = curIdx < flatLessons.length - 1 ? flatLessons[curIdx + 1] : null;

    let html = `
      <div class="crumb">
        <span>${meta ? meta.group : ''} · Lesson ${meta ? meta.num : ''} of 24</span>
        <a href="#papers" class="pill">View Stroke Papers</a>
      </div>

      <div class="lesson-head">
        <div class="eyebrow">${lesson.eyebrow}</div>
        <h1>${lesson.h1}</h1>
        <p class="lead">${lesson.lead}</p>
      </div>

      <div class="opening">
        <div class="explanation">
          ${lesson.analogy ? `
            <div class="analogy">
              <h3>Everyday Analogy: ${lesson.analogy.title}</h3>
              <p>${lesson.analogy.text}</p>
            </div>
          ` : ''}
          ${lesson.sections ? lesson.sections.map(s => `
            <div class="section">
              <div class="section-label">${s.label}</div>
              ${s.html}
            </div>
          `).join('') : ''}
        </div>
        <div class="lab" id="lab-container"></div>
      </div>

      <div class="lesson-bottom">
        ${prev ? `
          <a href="#${prev.id}" class="nav-btn prev-btn" id="btn-prev-lesson">
            <span class="nav-btn__dir">← Previous Lesson</span>
            <span class="nav-btn__title">Lesson ${prev.num}: ${prev.title}</span>
          </a>
        ` : `<span class="nav-btn__spacer"></span>`}
        ${next ? `
          <a href="#${next.id}" class="nav-btn next-btn" id="btn-next-lesson">
            <span class="nav-btn__dir">Next Lesson →</span>
            <span class="nav-btn__title">Lesson ${next.num}: ${next.title}</span>
          </a>
        ` : `
          <a href="#papers" class="nav-btn next-btn complete-btn" id="btn-next-lesson">
            <span class="nav-btn__dir">Curriculum Complete →</span>
            <span class="nav-btn__title">Explore Stroke Manuscripts</span>
          </a>
        `}
      </div>
    `;

    this.contentEl.innerHTML = html;

    const labContainer = document.getElementById("lab-container");
    if (labContainer && typeof lesson.lab === "function") {
      try {
        lesson.lab(labContainer);
      } catch (err) {
        console.error('Error rendering lab for lesson ' + id, err);
      }
    }
  }
}

// Global attachment & resilient boot logic
if (typeof window !== "undefined") {
  window.CURRICULUM_GROUPS = CURRICULUM_GROUPS;
  window.LESSONS_MAP = LESSONS_MAP;
  window.LESSON_CONTENT = LESSON_CONTENT;
  window.App = App;

  function boot() {
    if (!window._app) {
      window._app = new App();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { App, CURRICULUM_GROUPS, LESSONS_MAP, LESSON_CONTENT };
}
