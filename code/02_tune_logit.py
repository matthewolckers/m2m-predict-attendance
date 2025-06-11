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
from sklearn.linear_model import LogisticRegression

from my_functions import load_data
# -

(train, test, comp,
y_train_miss_first, X_train,
y_test_miss_first, X_test,
y_comp_miss_first, X_comp) = load_data()

logregparams = {
    "penalty": ['l1', 'l2', 'elasticnet', None],
    "max_iter": [100, 200, 500, 1000],
    "C": np.logspace(-4, 4, 20),
    "solver": ["lbfgs", "liblinear", "saga"],
    "l1_ratio": np.linspace(0, 1, 10)  # Only used when penalty='elasticnet'
}

# +
logreg = LogisticRegression(random_state=87937)

logreg_rs = RandomizedSearchCV(
    logreg,
    n_iter=200,  # Increased number of iterations
    param_distributions=logregparams,
    cv=TimeSeriesSplit(n_splits=5),
    scoring=['roc_auc', 'accuracy', 'f1'],  # Multiple scoring metrics
    refit='roc_auc',  # Specify which metric to use for selecting the best model
    n_jobs=8,
    verbose=1,
    return_train_score=True
)
# -

logreg_rs.fit(X_train, y_train_miss_first)
print(logreg_rs.best_params_)
logreg_rs.best_estimator_
