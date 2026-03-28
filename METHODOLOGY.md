# Methodology - Credit Risk Scorecard

## Overview

This project builds a credit risk scorecard from scratch using the German Credit Dataset. The goal is to predict the probability that a loan applicant will default, and translate that probability into an interpretable 0–1000 credit score. The approach follows industry-standard credit scoring practice: structured data pipeline, logistic regression as the core model, and log-odds scaling to produce the final score.

---

## 1. Data

**Source:** UCI Machine Learning Repository - [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

**Shape:** 1,000 applicants × 20 features + 1 binary target

**Target variable:** `credit_risk` - 1 = bad (default), 0 = good

**Class balance:** approximately 70% good, 30% bad - moderately imbalanced but within acceptable range for logistic regression without resampling.

The raw data was loaded from `german.data.txt`, parsed into a structured DataFrame, and stored in a local SQLite database (`creditrisk.db`) for reproducibility.

---

## 2. Exploratory Data Analysis

Key findings from EDA:

- Several features showed strong univariate separation between good and bad applicants, particularly account status, credit history, loan duration, and credit amount.
- No missing values were present in the dataset.
- Some numeric features (e.g. credit amount, age) had right-skewed distributions, common in financial data.
- Categorical features were encoded for modelling in the feature engineering phase.

---

## 3. Feature Engineering

The dataset contains a mix of categorical and numeric features. Preprocessing steps applied:

- Categorical features were one-hot encoded.
- All features were standardised using `sklearn`'s `StandardScaler` (zero mean, unit variance) prior to model fitting - this is important for logistic regression to ensure coefficients are on a comparable scale.
- The data was split 80/20 into train (800 obs) and test (200 obs) sets using a fixed random seed for reproducibility.

---

## 4. Modelling

Two models were trained and compared on the test set:

| Model               | AUC    | Gini   | KS     |
|---------------------|--------|--------|--------|
| Logistic Regression | 0.8255 | 0.6510 | 0.6000 |
| XGBoost             | 0.8090 | 0.6181 | 0.5286 |

**Logistic Regression was selected as the final model** for three reasons:

- It outperformed XGBoost on all three metrics without any hyperparameter tuning.
- It is the industry standard for credit scoring - interpretable, auditable, and regulatorily defensible.
- The dataset is relatively small and well-structured, which favours simpler linear models over complex ensemble methods.

XGBoost's underperformance here is expected: the dataset size (~1,000 observations) and structured features do not give gradient boosting the conditions in which it typically excels (large, high-dimensional, noisy data).

**Evaluation metrics:**

- **AUC (Area Under the ROC Curve):** Measures overall ranking ability - the probability that the model ranks a bad applicant lower than a good one.
- **Gini:** Derived as `2 × AUC − 1`. A common credit industry metric; values above 0.5 are considered strong.
- **KS (Kolmogorov-Smirnov):** Measures maximum separation between the cumulative distributions of good and bad applicants. KS of 0.60 is considered very good in credit scoring.

---

## 5. Score Scaling

The logistic regression model outputs a probability of default P(bad) for each applicant. This is converted to a 0–1000 credit score using log-odds scaling, a standard technique in the credit industry.

**Scaling formula:**

```
log_odds = log( P(bad) / P(good) )
Score    = Offset − Factor × log_odds
```

**Scaling parameters:**

| Parameter  | Value | Rationale |
|------------|-------|-----------|
| PDO        | 50    | Points to Double Odds - controls score spread |
| Base Score | 500   | Score assigned at 1:1 odds (50% bad rate) |
| Base Odds  | 1     | Good:Bad odds at the base score |

These give:

```
Factor = PDO / ln(2) = 72.13
Offset = Base Score − Factor × ln(Base Odds) = 500
```

Scores are clipped to [0, 1000] and rounded to integers. A higher score means lower credit risk.

---

## 6. Results

**Score validation on the test set:**

| Metric | Value |
|--------|-------|
| AUC    | 0.826 |
| Gini   | 0.652 |
| KS     | 0.600 |

The score metrics are essentially identical to the raw model metrics, confirming the scaling function preserved discriminatory power.

**Score band analysis:**

| Band    | Count | Bads | Bad Rate | % Population |
|---------|-------|------|----------|--------------|
| 700+    | 13    | 0    | 0.0%     | 6.5%         |
| 650–699 | 11    | 1    | 9.1%     | 5.5%         |
| 600–649 | 22    | 3    | 13.6%    | 11.0%        |
| 550–599 | 40    | 4    | 10.0%    | 20.0%        |
| 500–549 | 33    | 3    | 9.1%     | 16.5%        |
| 450–499 | 28    | 11   | 39.3%    | 14.0%        |
| 400–449 | 23    | 16   | 69.6%    | 11.5%        |
| 300–399 | 28    | 21   | 75.0%    | 14.0%        |
| <300    | 2     | 1    | 50.0%    | 1.0%         |

The bad rate decreases consistently as the score increases - this monotonic behaviour is the key property of a well-calibrated scorecard. Applicants scoring 700+ had a 0% bad rate in the test set, while those in the 300–399 band had a 75% bad rate.

The slight non-monotonicity between the 550–599 (10.0%) and 600–649 (13.6%) bands is a minor anomaly, likely attributable to small sample sizes in those bands (22–40 observations).

---

## 7. Limitations and Next Steps

**Limitations:**

- The dataset has only 1,000 observations - results may not generalise to larger, real-world portfolios.
- The model uses StandardScaler rather than WoE (Weight of Evidence) binning. A WoE-based approach would improve interpretability and allow a true points-per-bin scorecard table.
- No cross-validation was performed - a k-fold validation would give more reliable estimates of out-of-sample performance.
- The model has not been calibrated - predicted probabilities may not be well-calibrated even if ranking performance is strong.

**Potential next steps:**

- Implement WoE binning and rebuild as a traditional scorecard with a points table per characteristic.
- Add k-fold cross-validation to assess stability.
- Apply Platt scaling or isotonic regression for probability calibration.
- Build a simple scoring interface (e.g. a Streamlit app) to score new applicants interactively.
