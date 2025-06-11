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
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import roc_auc_score
import shap

from my_functions import load_data
# -

(train, test, comp,
y_train_miss_first, X_train, 
y_test_miss_first, X_test, 
y_comp_miss_first, X_comp) = load_data()

# I will have to manually copy across the parameters for each model. A better approach would be to export the parameters from the tuning notebook and import them here.
#

# +
xgb_cl = XGBClassifier(
    subsample=0.6,
    reg_lambda=1,
    reg_alpha=0.5,
    n_estimators=500, 
    min_child_weight=1, 
    max_depth=5, 
    learning_rate=0.01, 
    gamma=0.1, 
    colsample_bytree=0.9,
    random_state=87937)

brf = BalancedRandomForestClassifier(
    random_state=87937, 
    replacement=True, 
    sampling_strategy="all",
    n_estimators=400, 
    max_depth=10, 
    min_samples_split=2, 
    min_samples_leaf=4, 
    bootstrap=True)

logreg = LogisticRegression(
    random_state=87937,
    penalty='l2',
    max_iter=1000,
    C=0.0018329807108324356,
    solver='lbfgs')

cat_cl = CatBoostClassifier(
    random_state=87937, 
    verbose=False,
    colsample_bylevel=0.3,
    l2_leaf_reg=0.2,
    learning_rate=0.05,
    max_depth=4,
    min_child_samples=1)

lgbm_cl = LGBMClassifier(
    random_state=87937, 
    verbose=-1,
    colsample_bytree=1,
    learning_rate=0.05,
    max_depth=4,
    n_estimators=200,
    num_leaves=10,
    reg_lambda=0.2)
# -

xgb_cl.fit(X_train, y_train_miss_first)

brf.fit(X_train, y_train_miss_first)

logreg.fit(X_train, y_train_miss_first)

cat_cl.fit(X_train, y_train_miss_first)

lgbm_cl.fit(X_train, y_train_miss_first)

test["MissFirst_Prediction_xgb"] = xgb_cl.predict(X_test)
test["MissFirst_Prediction_prob_xgb"] = xgb_cl.predict_proba(X_test)[:,1]
test["MissFirst_Prediction_brf"] = brf.predict(X_test)
test["MissFirst_Prediction_prob_brf"] = brf.predict_proba(X_test)[:,1]
test["MissFirst_Prediction_logreg"] = logreg.predict(X_test)
test["MissFirst_Prediction_prob_logreg"] = logreg.predict_proba(X_test)[:,1]
test["MissFirst_Prediction_cat"] = cat_cl.predict(X_test)
test["MissFirst_Prediction_prob_cat"] = cat_cl.predict_proba(X_test)[:,1]
test["MissFirst_Prediction_lgbm"] = lgbm_cl.predict(X_test)
test["MissFirst_Prediction_prob_lgbm"] = lgbm_cl.predict_proba(X_test)[:,1]


def print_model_performance(model_name, y_true, y_pred, y_pred_prob):
    '''
    Prints the classification report and ROC AUC score for a given model.
    '''
    print(f"{model_name}")
    print(classification_report_imbalanced(y_true, y_pred))
    print(f"ROC AUC score is {roc_auc_score(y_true, y_pred_prob):.4f}")
    print("-" * 85)


# +
models = {
    "XGBoost": ("MissFirst_Prediction_xgb", "MissFirst_Prediction_prob_xgb"),
    "Balanced Random Forest": ("MissFirst_Prediction_brf", "MissFirst_Prediction_prob_brf"),
    "Logistic Regression": ("MissFirst_Prediction_logreg", "MissFirst_Prediction_prob_logreg"),
    "CatBoost": ("MissFirst_Prediction_cat", "MissFirst_Prediction_prob_cat"),
    "LightGBM": ("MissFirst_Prediction_lgbm", "MissFirst_Prediction_prob_lgbm")
}

for model_name, (pred_col, prob_col) in models.items():
    print_model_performance(model_name, y_test_miss_first, test[pred_col], test[prob_col])

# +
# explainer = shap.TreeExplainer(brf)
# shap_values = explainer.shap_values(X_test)

# +
# shap.summary_plot(shap_values, X_test, plot_type="bar")

# +
# shap.summary_plot(shap_values, X_test)
# -

# Make predictions using the Balanced Random Forest model
comp["MissFirst_Prediction_brf"] = brf.predict(X_comp)
comp["MissFirst_Prediction_prob_brf"] = brf.predict_proba(X_comp)[:,1]

print_model_performance(
    "BRF",
    y_comp_miss_first, 
    comp["MissFirst_Prediction_brf"], 
    comp["MissFirst_Prediction_prob_brf"])

comp.to_csv("../data/comp_predictions.csv", index=False)
