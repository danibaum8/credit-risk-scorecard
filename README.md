# Credit Risk Scorecard

A end-to-end credit risk scoring model built on the German Credit Dataset. The project takes raw applicant data through exploratory analysis, feature engineering, model selection, and scoring - producing a 0–1000 credit score where a higher score means lower default risk.

---

## Results

| Metric | Score |
|--------|-------|
| AUC    | 0.826 |
| Gini   | 0.652 |
| KS     | 0.600 |

Score band performance (test set):

| Band     | Bad Rate | Population |
|----------|----------|------------|
| 700+     | 0.0%     | 6.5%       |
| 650–699  | 9.1%     | 5.5%       |
| 600–649  | 13.6%    | 11.0%      |
| 550–599  | 10.0%    | 20.0%      |
| 500–549  | 9.1%     | 16.5%      |
| 450–499  | 39.3%    | 14.0%      |
| 400–449  | 69.6%    | 11.5%      |
| 300–399  | 75.0%    | 14.0%      |
| <300     | 50.0%    | 1.0%       |

---

## Project Structure

```
credit-risk-scorecard/
│
├── data/
│   ├── german.data.txt          # Raw dataset
│   └── creditrisk.db            # SQLite database
│
├── 01_data_loading.ipynb        # Load raw data, store to SQLite
├── 02_eda.ipynb                 # Exploratory data analysis
├── 03_feature_engineering.ipynb # Feature prep and scaling
├── 04_modelling.ipynb           # LR vs XGBoost comparison
├── 05_scorecard.ipynb           # Score scaling and validation
│
├── model_artifacts.pkl          # Saved model and data objects
├── scored_applicants.csv        # Full dataset with credit scores
└── score_validation.png         # Validation charts
```

---

## Quickstart

**1. Clone the repo and set up the environment**

```bash
git clone https://github.com/your-username/credit-risk-scorecard.git
cd credit-risk-scorecard
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Run notebooks in order**

Open each notebook in VS Code (or Jupyter) and run top to bottom:

```
01_data_loading.ipynb
02_eda.ipynb
03_feature_engineering.ipynb
04_modelling.ipynb
05_scorecard.ipynb
```

> `05_scorecard.ipynb` depends on `model_artifacts.pkl` saved at the end of `04_modelling.ipynb`.

---

## Dependencies

- Python 3.11
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- sqlalchemy
- scipy

---

## Dataset

The [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) from the UCI Machine Learning Repository. 1,000 applicants, 20 features, binary target (1 = bad credit risk, 0 = good).

---

## Methodology

See [METHODOLOGY.md](METHODOLOGY.md) for a full write-up of the approach, modelling decisions, and results.
