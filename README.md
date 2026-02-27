\# Comparative Loan Default Risk Modeling

\### Comparative Supervised Learning Study (Logistic Regression vs Random Forest vs XGBoost)



\## Objective

Build and evaluate supervised learning models for loan default prediction using structured financial features under realistic conditions (missing values, class imbalance, and leakage prevention).



\## Dataset

Lending Club dataset (sampled and cleaned subset used for modeling).

> Note: Raw dataset files are not included in this repository.



\## Methodology

\- Removed high-missing variables and high-cardinality identifiers

\- Prevented target leakage by dropping post-outcome variables (recoveries/payment/settlement-related fields)

\- Stratified train/test split

\- Preprocessing pipeline using `ColumnTransformer`:

&nbsp; - Median imputation + scaling for numeric features

&nbsp; - Most-frequent imputation + one-hot encoding for categorical features

\- Benchmarked models:

&nbsp; - Logistic Regression (baseline)

&nbsp; - Random Forest (bagging ensemble)

&nbsp; - XGBoost (boosting ensemble)

\- Evaluated using ROC-AUC and 5-fold cross-validation

\- Performed threshold optimization to analyze precision–recall trade-offs



\## Results (ROC-AUC)

\- Logistic Regression: ~0.957

\- Random Forest: ~0.955

\- XGBoost: ~0.960



\## Notebook

\- `notebooks/01\_loan\_default\_analysis.ipynb`



\## Reproducibility

Create and activate a virtual environment, then install dependencies:



```bash

pip install -r requirements.txt

