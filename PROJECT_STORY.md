# Credit Risk Scorecard - Project Write-Up

## The Goal

Every time someone applies for a loan, a bank faces the same fundamental question: will this person pay it back?

Answering that question well is worth billions. Approve too many risky borrowers and losses pile up. Reject too many creditworthy applicants and you leave money on the table. The solution the financial industry settled on decades ago is the credit scorecard - a model that assigns every applicant a single number reflecting their likelihood of default. The higher the score, the safer the bet.

This project builds one from scratch. Starting from raw applicant data and ending with a live scoring tool, the goal was to replicate the kind of model used in real credit decisions - transparent, statistically rigorous, and explainable to anyone who asks why they got the score they did.

---

## The Data

The dataset comes from the UCI Machine Learning Repository and contains records for 1,000 loan applicants from a German bank. Each applicant is described by 20 features - a mix of financial information, personal details, and loan characteristics - and labelled as either a good or bad credit risk based on their repayment history.

The features cover things like:

- How much money is in their checking and savings accounts
- The purpose and amount of the loan
- Their employment history and length
- Whether they own property, rent, or live for free
- Their age, number of dependants, and personal status
- Their credit history with this and other institutions

The target split was 70% good borrowers, 30% bad - a moderately imbalanced dataset, but well within the range where standard modelling techniques work reliably without needing any special corrections.

---

## Exploratory Data Analysis

Before building any model, the data needs to be understood. This phase - exploratory data analysis, or EDA - is where assumptions get challenged and surprises surface.

**No missing values.** The dataset was unusually clean. Every one of the 1,000 rows was complete, with no gaps to fill or impute. In real-world credit data this is rare, so it was a meaningful head start.

**Categorical codes.** The raw data stored categorical values as cryptic codes - things like "A11", "A32", "A143". The first task was decoding these into human-readable labels (for example, A11 = "checking account below 0 DM", A32 = "existing credits paid back on time"). This made the data interpretable and analysis meaningful.

**Outliers in financial features.** Credit amount and loan duration showed right-skewed distributions - a small number of applicants were applying for very large loans over very long terms. This is typical in financial data and not a data quality issue; it reflects genuine variation in borrower behaviour. These values were retained rather than removed.

**Strong separation in key features.** Several features showed clear differences between good and bad borrowers at the univariate level - meaning they appeared useful for prediction even before any modelling. Checking account status was the strongest signal: applicants with no checking account or a healthy balance defaulted far less often than those overdrawn. Credit history and loan purpose also showed strong patterns.

**Class imbalance was manageable.** With 30% bad borrowers, the dataset was imbalanced but not severely. A model trained on this data would not be overwhelmed by the majority class, so no resampling techniques (such as SMOTE) were needed.

---

## Feature Engineering

Raw data rarely goes straight into a model. This step transforms it into a form the algorithm can use effectively.

**Weight of Evidence (WoE) encoding.** For the categorical features, a technique called Weight of Evidence encoding was applied. Rather than converting categories into arbitrary numbers (which can mislead a model), WoE replaces each category with a value that directly reflects how much riskier or safer that group is compared to the overall population. A checking account with a balance over 200 DM gets a positive WoE value; an overdrawn account gets a negative one. This makes the features directly interpretable in terms of credit risk.

**Information Value for feature selection.** Each feature was scored using Information Value (IV), which measures how predictive it is of the target. Features with very low IV - specifically "job type" and "telephone ownership" - were dropped. Keeping weak predictors adds noise without improving accuracy.

**Standardisation.** The remaining numeric features were standardised using a technique called StandardScaler, which centres each feature around zero and gives it a consistent scale. This is important for logistic regression, which is sensitive to features being on very different scales (for example, age in years versus loan amount in thousands of Deutsche Marks).

After feature engineering, the dataset was split 80/20 into training data (800 applicants) and test data (200 applicants). The model was trained on the first group and evaluated on the second - a group it had never seen before.

---

## Choosing the Model

Two models were trained and compared head-to-head.

**Logistic Regression** is one of the oldest and most battle-tested algorithms in machine learning. It estimates the probability of an event (default, in this case) as a function of the input features. The output is a number between 0 and 1 - the predicted probability of default.

**XGBoost** is a modern, high-performance algorithm that builds hundreds of decision trees in sequence, each one correcting the errors of the last. It has won countless data science competitions and is widely considered one of the best off-the-shelf algorithms for structured data.

The results:

| Model               | AUC   | Gini  | KS    |
|---------------------|-------|-------|-------|
| Logistic Regression | 0.826 | 0.651 | 0.600 |
| XGBoost             | 0.809 | 0.618 | 0.529 |

Logistic Regression won on every single metric - and by a meaningful margin.

This might seem counterintuitive. How does the simpler, older algorithm beat the powerful modern one? A few reasons:

- **Dataset size.** XGBoost thrives on large, messy, high-dimensional datasets. With only 1,000 clean, well-structured records, there simply is not enough complexity for it to exploit. Logistic regression is better suited to this kind of problem.
- **Feature engineering quality.** The WoE encoding essentially pre-processed the data into a near-linear form. Logistic regression is designed to exploit exactly this kind of structure. By the time the features were engineered, the problem had been made easy for LR and offered little room for XGBoost to improve.
- **Overfitting risk.** XGBoost with default settings on a small dataset can overfit - learning patterns that happen to exist in the training data but do not generalise. Logistic regression is more restrained by nature.

Beyond performance, logistic regression has a critical practical advantage in credit scoring: it is explainable. Every coefficient in the model directly corresponds to a feature's influence on the probability of default. This makes it auditable, regulatorily defensible, and - crucially - possible to explain to an applicant why their application was declined. XGBoost offers none of this transparency.

---

## What the Metrics Mean

Three metrics were used to evaluate model performance, each capturing something different.

**AUC (Area Under the ROC Curve)** measures the model's ability to rank applicants correctly - specifically, the probability that a randomly chosen bad applicant receives a lower (riskier) score than a randomly chosen good one. An AUC of 0.826 means that in 82.6% of such comparisons, the model gets it right. Random guessing would give 0.5.

**Gini coefficient** is derived directly from AUC (Gini = 2 x AUC - 1) and is the standard reporting metric in the credit industry. A Gini above 0.5 is considered strong; above 0.6 is very strong. This model achieved 0.651.

**KS (Kolmogorov-Smirnov) statistic** measures the maximum separation between the score distributions of good and bad borrowers. A KS of 0.60 means the model creates a very clear gap between the two populations - exactly what a lender needs to make confident decisions.

---

## The Credit Score

The model outputs a probability of default for each applicant - a number between 0% and 100%. But a probability is not a particularly useful thing to hand to a loan officer or a customer. What banks use instead is a score.

The probability was converted to a 0-1000 credit score using a technique called log-odds scaling, the same methodology used in commercial credit scoring systems worldwide. The formula ensures that the score has a consistent, interpretable meaning: every 50-point drop in score doubles the odds that the applicant will default.

A score of 500 corresponds to equal odds (50/50). Scores above 500 indicate increasingly creditworthy applicants; scores below indicate increasing risk.

The final validation confirmed the score works exactly as intended:

| Score Band | Bad Rate |
|------------|----------|
| 700+       | 0%       |
| 600-699    | 10-14%   |
| 500-599    | 9-10%    |
| 400-499    | 40-70%   |
| Below 400  | 70%+     |

Every applicant who scored above 700 in the test set repaid their loan. Three quarters of those who scored below 400 defaulted. The score is doing exactly what it was designed to do.

---

## The Live Scoring App

The final step was building a live scoring tool - a web application where any applicant's details can be entered and a credit score generated in real time.

The app was built using Streamlit, an open-source Python framework designed for turning data science models into interactive tools. It takes the same 18 features used to train the model, applies the same preprocessing pipeline, and returns a score on a 0-1000 gauge along with the associated risk band and estimated default rate.

This transforms the model from a static research artefact into something a lender could actually use. It also makes the project tangible to anyone who is not a data scientist - you can hand someone a laptop, describe an applicant, and watch the score move in real time as the details change.

---

## Tools and Technologies

- **Python** - all analysis, modelling, and app development
- **Pandas and NumPy** - data manipulation and numerical computation
- **Scikit-learn** - model training, preprocessing, and evaluation
- **XGBoost** - gradient boosting comparison model
- **Matplotlib and Seaborn** - data visualisation
- **Plotly** - interactive gauge in the scoring app
- **Streamlit** - live scoring web application
- **SQLite and SQLAlchemy** - structured data storage
- **GitHub** - version control and project hosting
