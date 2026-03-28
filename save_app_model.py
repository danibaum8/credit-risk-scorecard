"""
Run this script once from the project root to generate model/inference.pkl.
This file is committed to git and used by the Streamlit app.

Usage:
    python save_app_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Load LR model from existing artifacts ────────────────────────────────────
with open('model_artifacts.pkl', 'rb') as f:
    arts = pickle.load(f)
lr = arts['lr_model']
print("LR model loaded.")

# ── Load raw data and apply label mappings ───────────────────────────────────
engine = create_engine('sqlite:///data/creditrisk.db')
df = pd.read_sql('SELECT * FROM german_credit', engine)
df['target'] = df['target'].map({1: 0, 2: 1})

MAPPINGS = {
    'checking_account'  : {'A11': '<0 DM', 'A12': '0-200 DM', 'A13': '>200 DM', 'A14': 'no account'},
    'credit_history'    : {'A30': 'no credits', 'A31': 'all paid', 'A32': 'existing paid',
                           'A33': 'delay in past', 'A34': 'critical account'},
    'savings_account'   : {'A61': '<100 DM', 'A62': '100-500 DM', 'A63': '500-1000 DM',
                           'A64': '>1000 DM', 'A65': 'no savings'},
    'employment'        : {'A71': 'unemployed', 'A72': '<1 yr', 'A73': '1-4 yrs',
                           'A74': '4-7 yrs', 'A75': '>7 yrs'},
    'personal_status'   : {'A91': 'male divorced', 'A92': 'female divorced', 'A93': 'male single',
                           'A94': 'male married', 'A95': 'female single'},
    'other_debtors'     : {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'},
    'property'          : {'A121': 'real estate', 'A122': 'life insurance',
                           'A123': 'car', 'A124': 'no property'},
    'other_installments': {'A141': 'bank', 'A142': 'stores', 'A143': 'none'},
    'housing'           : {'A151': 'rent', 'A152': 'own', 'A153': 'free'},
    'job'               : {'A171': 'unskilled non-resident', 'A172': 'unskilled resident',
                           'A173': 'skilled', 'A174': 'highly skilled'},
    'telephone'         : {'A191': 'no', 'A192': 'yes'},
    'foreign_worker'    : {'A201': 'yes', 'A202': 'no'},
    'purpose'           : {'A40': 'car new', 'A41': 'car used', 'A42': 'furniture',
                           'A43': 'radio/tv', 'A44': 'appliances', 'A45': 'repairs',
                           'A46': 'education', 'A47': 'vacation', 'A48': 'retraining',
                           'A49': 'business', 'A410': 'other'},
}

for col, mapping in MAPPINGS.items():
    df[col] = df[col].map(mapping)

# ── Compute WoE maps ─────────────────────────────────────────────────────────
CAT_COLS = [
    'checking_account', 'credit_history', 'purpose', 'savings_account',
    'employment', 'personal_status', 'other_debtors', 'property',
    'other_installments', 'housing', 'foreign_worker',
]

def calculate_woe(df, feature, target):
    total_good = (df[target] == 0).sum()
    total_bad  = (df[target] == 1).sum()
    stats = df.groupby(feature)[target].agg(
        bad  = lambda x: (x == 1).sum(),
        good = lambda x: (x == 0).sum()
    ).reset_index()
    stats['bad_rate']  = (stats['bad']  / total_bad).replace(0, 1e-4)
    stats['good_rate'] = (stats['good'] / total_good).replace(0, 1e-4)
    stats['woe'] = np.log(stats['good_rate'] / stats['bad_rate'])
    return stats.set_index(feature)['woe'].to_dict()

woe_maps = {col: calculate_woe(df, col, 'target') for col in CAT_COLS}
print(f"WoE maps computed for {len(woe_maps)} features.")

# ── Apply WoE and prepare model-ready dataset ────────────────────────────────
df_woe = df.copy()
for col in CAT_COLS:
    df_woe[col] = df_woe[col].map(woe_maps[col])

COLS_TO_DROP = ['job', 'telephone', 'target']
X = df_woe.drop(columns=COLS_TO_DROP)
y = df_woe['target']

# ── Refit StandardScaler on same train split ─────────────────────────────────
X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
print(f"StandardScaler fitted on {len(X_train)} training rows.")
print(f"Feature order ({len(X.columns)}): {list(X.columns)}")

# ── Save inference bundle ────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
bundle = {
    'lr'            : lr,
    'scaler'        : scaler,
    'woe_maps'      : woe_maps,
    'feature_names' : list(X.columns),
    'mappings'      : MAPPINGS,
    'pdo'           : 50,
    'base_score'    : 500,
    'base_odds'     : 1,
}
with open('model/inference.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("Saved: model/inference.pkl")
