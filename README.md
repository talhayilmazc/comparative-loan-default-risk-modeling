# Comparative Loan Default Risk Modeling  
### Supervised Learning Benchmarking under Financial Risk Constraints

---

## 1. Objective

This project presents a comparative supervised learning study for loan default prediction using structured financial data.  

The objective is not merely predictive accuracy, but disciplined modeling under realistic financial risk conditions, including:

- Target leakage prevention
- Class imbalance handling
- Cross-validation benchmarking
- Decision threshold optimization

---

## 2. Problem Context

Loan default prediction is a core problem in credit risk modeling.  

In real-world financial systems, models must:

- Avoid information leakage from post-outcome variables  
- Maintain stability across cross-validation folds  
- Balance precision and recall depending on institutional risk tolerance  

This study frames default prediction as a structured modeling and decision problem rather than a leaderboard-style experiment.

---

## 3. Data Preprocessing & Leakage Control

Several post-outcome and recovery-related variables were removed to prevent target leakage, including:

- Recovery amounts
- Settlement-related features
- Payment-related post-event fields

Missing values were handled using:

- Median imputation (numerical)
- Most-frequent imputation (categorical)

Categorical variables were encoded via one-hot encoding using a `ColumnTransformer` pipeline.

---

## 4. Modeling Approach

Three supervised learning models were benchmarked:

- Logistic Regression (interpretable linear baseline)
- Random Forest (bagging ensemble)
- XGBoost (gradient boosting ensemble)

Stratified train/test split was used to preserve class distribution.

Evaluation metrics:

- ROC-AUC
- 5-fold Cross-Validation ROC-AUC
- Precision / Recall trade-off analysis

Observed performance:

- Logistic Regression: ~0.96 ROC-AUC
- Random Forest: ~0.95 ROC-AUC
- XGBoost: ~0.96 ROC-AUC

---

## 5. Threshold Optimization

Rather than using the default 0.5 probability threshold, multiple thresholds were evaluated to balance:

- Precision
- Recall
- F1-score

This demonstrates how deployment decisions differ from pure model training metrics in financial institutions.

---

## 6. Model Interpretation

Coefficient analysis (Logistic Regression) and feature importance exploration provide insights into the drivers of loan default risk.

Key predictive signals include:

- Credit score related features
- Debt-to-income metrics
- Account history characteristics

This reinforces financial plausibility and interpretability.

---

## 7. Key Contributions

- Structured comparative modeling framework
- Explicit leakage prevention
- Cross-validated benchmarking
- Threshold-based deployment perspective
- Financially grounded interpretation

---

## 8. Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt

Run notebook:

python -m jupyter lab
9. Academic Framing

This project emphasizes:

Applied statistical learning in financial risk modeling

Robust model comparison under controlled evaluation

Deployment-aware threshold tuning

Interpretability considerations in credit risk systems

It is structured as a research-aware applied modeling study rather than a performance-only experiment.
