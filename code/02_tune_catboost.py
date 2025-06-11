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
import catboost as cat

from my_functions import load_data
# -

(train, test, comp,
y_train_miss_first, X_train,
y_test_miss_first, X_test,
y_comp_miss_first, X_comp) = load_data()

catparams={"learning_rate"    : [0.05, 0.15, 0.25] ,
 "max_depth"        : [ 4, 7, 10],
 "min_child_samples" : [ 1, 3, 5, 7 ],
 "l2_leaf_reg"            : [ 0.0, 0.2 , 0.4 ],
 "colsample_bylevel" : [ 0.3, 0.5 , 0.7 ] }

cat_cl = cat.CatBoostClassifier(random_state=87937, verbose=False)

cat_grid = GridSearchCV(
    cat_cl,
    param_grid = catparams,
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
cat_grid.fit(X_train, y_train_miss_first)
print(cat_grid.best_params_)
cat_grid.best_estimator_

cat_grid.best_score_
