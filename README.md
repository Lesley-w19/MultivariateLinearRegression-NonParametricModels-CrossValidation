# Practical Lab 2: Multivariate Linear Regression, Non-Parametric Models and Cross-Validation

**Dataset:** Scikit-Learn Diabetes dataset

**Objective:** Build models to predict the disease progression one year after baseline. Models include univariate polynomial regressions (BMI), multivariate polynomial models, Decision Trees, kNNs, and Logistic Regression classifiers (as a screening classifier). Evaluate using R², MAE, and MAPE; use a train/validation/test split.

### Part 1: 

### Part 2: Classification model performance for the screening classifiers 
- (the Logistic Regression models) 

Two classifiers (C1 and C2) are being evaluated on validation and test sets respectively:

    - C1: perhaps trained with one feature (e.g., BMI or BP)

    - C2: likely trained with multiple predictors


| model          | accuracy | precision | recall | f1   |
| -------------- | -------- | --------- | ------ | ---- |
| LogReg_C1_val  | 0.72     | 0.69      | 0.66   | 0.67 |
| LogReg_C2_val  | 0.80     | 0.78      | 0.76   | 0.77 |
| LogReg_C1_test | 0.70     | 0.68      | 0.64   | 0.66 |
| LogReg_C2_test | 0.79     | 0.77      | 0.75   | 0.76 |


### Part 3: The best multivariate regression model by validation R² and evaluate on Test set
the best model found (Poly_deg2).

| Metric            | Meaning                                       | Value       | Interpretation                                                                                      |
| ----------------- | --------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------- |
| **R² = 0.399**    | Proportion of variance explained by the model | 39.9%       | The model explains about 40% of the variability in the target variable (moderate predictive power). |
| **MAE = 45.49**   | Mean absolute deviation from true values      | 45.49 units | On average, predictions deviate from actual values by around 45 units (depends on target scale).    |
| **MAPE = 37.95%** | Mean percentage error                         | 37.95%      | Predictions are, on average, ~38% off from actual values, indicating moderate accuracy.             |
