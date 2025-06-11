# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: m2m
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, GridSearchCV
import xgboost as xgb

from my_functions import load_data
# -

(train, test, comp,
y_train_miss_first, X_train,
y_test_miss_first, X_test,
y_comp_miss_first, X_comp) = load_data()

xgbparams = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth": [3, 5, 7, 9, 12],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
    "n_estimators": [100, 200, 300, 500, 1000],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9],
    "subsample": [0.6, 0.8, 1.0],  # Add subsample parameter
    "reg_alpha": [0, 0.1, 0.5, 1],  # Add L1 regularization
    "reg_lambda": [0, 0.1, 0.5, 1]  # Add L2 regularization
}

# +
xgb_cl = xgb.XGBClassifier(
    random_state=87937)

xgb_rs = RandomizedSearchCV(
    xgb_cl,
    n_iter=100,
    param_distributions = xgbparams,
    cv=TimeSeriesSplit(n_splits=5),
    scoring=['roc_auc', 'accuracy', 'f1'],
    refit='roc_auc',
    n_jobs=8,
    verbose=1)
# -

xgb_rs.fit(X_train, y_train_miss_first)

xgb_rs.best_estimator_
print(xgb_rs.best_params_)
xgb_rs.best_score_
