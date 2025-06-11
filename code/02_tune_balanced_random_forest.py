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
from imblearn.ensemble import BalancedRandomForestClassifier

from my_functions import load_data
# -

(train, test, comp,
y_train_miss_first, X_train,
y_test_miss_first, X_test,
y_comp_miss_first, X_comp) = load_data()

brfparams = {
    'n_estimators':[100,400,700,1000],
    'max_depth': [10, 40, 70, 100, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]}

brf = BalancedRandomForestClassifier(
    random_state=87937,
    replacement=True,
    sampling_strategy="all")

brf_random = GridSearchCV(
    brf,
    param_grid = brfparams,
    #n_iter=100,  # Number of iterations
    cv=TimeSeriesSplit(n_splits=5),
    scoring=['roc_auc', 'accuracy', 'f1'],
    refit='roc_auc',
    n_jobs=-1,
    verbose=1,
    #random_state=87937,
    return_train_score=True
)

# Fit the randomized search
brf_random.fit(X_train, y_train_miss_first)
print(brf_random.best_params_)
brf_random.best_estimator_

brf_random.best_score_
