# Model Card — Credit Risk Scorecard

**Version:** 1.0
**Date:** March 2026
**Author:** Dani Baumgarten
**Status:** Research / Portfolio Project

---

## Model Overview

This project implements a credit risk scorecard using the German Credit Dataset (UCI Machine Learning Repository). The goal is to predict the probability that a loan applicant will default, using a Logistic Regression baseline and an XGBoost challenger model. The project is designed to reflect real-world practices in regulated financial environments, including Weight of Evidence (WoE) feature encoding, Information Value (IV) feature selection, and SHAP-based explainability.

---

## Intended Use

**Primary use:** Classify loan applicants as good (low risk) or bad (high risk) credit.
**Intended users:** Risk analysts, credit underwriters, data scientists in financial services.
**Out-of-scope use:** This model should not be used for production lending decisions without further validation, bias auditing, and regulatory review.

---

## Data

| Property | Detail |
|---|---|
| Source | UCI ML Repository — Statlog (German Credit) |
| Size | 1,000 applicants, 21 features |
| Target | Binary: 0 = Good Credit, 1 = Bad Credit |
| Class distribution | 70% Good (700), 30% Bad (300) |
| Missing values | None |
| Storage | SQLite (`creditrisk.db`) |

### Key Features (by Information Value)

| Feature | IV | Strength |
|---|---|---|
| checking_account | 0.666 | Very Strong |
| credit_history | 0.293 | Strong |
| savings_account | 0.196 | Medium |
| purpose | 0.169 | Medium |
| property | 0.113 | Medium |
| employment | 0.086 | Weak |

Features with IV < 0.02 (`job`, `telephone`) were excluded from modelling.

---

## Pre-processing

- **Categorical encoding:** Weight of Evidence (WoE) — standard in credit scoring, produces interpretable, monotonic features aligned with regulatory expectations (Basel III, GDPR Article 22).
- **Scaling:** StandardScaler applied to all features before Logistic Regression.
- **Class imbalance:** Handled via `class_weight='balanced'` in Logistic Regression and `scale_pos_weight=700/300` in XGBoost.

---

## Model Performance

| Metric | Logistic Regression | XGBoost |
|---|---|---|
| AUC | 0.8255 | 0.8090 |
| Gini | 0.6510 | 0.6181 |
| KS Statistic | 0.6000 | 0.5286 |

**Selected model: Logistic Regression.** Despite XGBoost being a more complex model, Logistic Regression outperformed it on all three regulated metrics. This is consistent with credit scoring literature — simpler, interpretable models often generalise better on structured financial data, and are preferred in regulated environments due to their explainability.

### Metric Interpretation

- **Gini > 0.5** is considered strong in credit risk. At 0.651, this scorecard meets a production-grade threshold.
- **KS > 0.4** indicates strong separation between good and bad applicants. At 0.600, the model discriminates well across the score distribution.
- **AUC of 0.826** means the model correctly ranks a random bad applicant above a random good one 82.6% of the time.

---

## Explainability

SHAP (SHapley Additive exPlanations) was used to explain both global and individual-level predictions.

- **Global:** `checking_account` dominates — applicants with overdrawn accounts have significantly higher default probability. `purpose`, `savings_account`, and `installment_rate` are the next strongest drivers.
- **Local:** Waterfall plots provide individual-level explanations, supporting compliance with GDPR Article 22 (right to explanation for automated decisions).

---

## Limitations

- The dataset is historical (1994, West Germany) and may not reflect current credit behaviour or modern applicant demographics.
- The dataset is relatively small (1,000 rows). Performance may vary on larger, more diverse populations.
- No fairness audit has been conducted. Features such as `personal_status` and `foreign_worker` may carry demographic correlations that require scrutiny before any real-world deployment.
- The model has not been tested for Population Stability Index (PSI) over time — production models require ongoing monitoring for data drift.

---

## Ethical Considerations

Credit scoring models have real-world consequences for individuals' access to financial services. Before any production deployment, this model would require:

- A formal bias and fairness audit across protected characteristics
- Independent model validation by a risk function
- Regulatory review (e.g., Consumer Credit Act, GDPR, EBA Guidelines on Internal Governance)
- A defined model monitoring and review cadence

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Data Storage | SQLite via SQLAlchemy |
| Modelling | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualisation | matplotlib, seaborn |
| Environment | VS Code + Jupyter Notebooks |
