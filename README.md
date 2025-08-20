# M2M Predict client attendance

This repository contains replicaiton code for predicting client attendance in the mothers2mothers (m2m). The workflow includes data preparation, model training and tuning, model selection, comparison with mentor mother predictions, and visualization of results.

## Workflow Overview

1. **Data Preparation**
   - `01_create_train_validation_test_datasets.py`: Loads raw data, cleans and engineers features, and creates train, test, and comparison datasets for modeling.

2. **Model Training and Hyperparameter Tuning**
   - `02_tune_balanced_random_forest.py`: Tunes a Balanced Random Forest classifier using grid search and time series cross-validation.
   - `02_tune_catboost.py`: Tunes a CatBoost classifier using grid search.
   - `02_tune_lgbm.py`: Tunes a LightGBM classifier using grid search.
   - `02_tune_xgboost.py`: Tunes an XGBoost classifier using randomized search.
   - `02_tune_logit.py`: Tunes a logistic regression model using randomized search.

3. **Model Selection and Evaluation**
   - `03_pick_best_model.py`: Loads tuned models with selected hyperparameters, fits them, evaluates performance on the test set, and generates predictions for the comparison set. Also includes SHAP analysis for model interpretability.

4. **Comparison with Mentor Mother Predictions**
   - `04_compare_accuracy_mentors_vs_machine.py`: Compares the accuracy of machine learning predictions with mentor mother survey responses, performs statistical tests, and analyzes prediction sets.

5. **Visualization and Results**
   - `05_figures_and_results.py`: Generates descriptive statistics and visualizations, including time trends in missed appointments across data sources.

## Usage

1. **Prepare Data**
   - Run `01_create_train_validation_test_datasets.py` to generate processed datasets in the `data/` folder.

2. **Tune Models**
   - Run the scripts in the `02_tune_*` series to perform hyperparameter tuning for each model. Adjust parameters as needed.

3. **Select and Evaluate Models**
   - Run `03_pick_best_model.py` to fit the best models, evaluate their performance, and generate predictions.

4. **Compare with Mentor Mother Predictions**
   - Run `04_compare_accuracy_mentors_vs_machine.py` to analyze and compare machine learning predictions with mentor mother survey data.

5. **Generate Figures**
   - Run `05_figures_and_results.py` to produce summary statistics and publication-ready figures.

## Dependencies

- Python 3.x
- pandas, numpy, scikit-learn, imbalanced-learn, xgboost, catboost, lightgbm, matplotlib, seaborn, statsmodels, shap

Install dependencies using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost catboost lightgbm matplotlib seaborn statsmodels shap
```

## Data

Place raw data files in the `data/` directory as expected by the scripts. Output files and figures will also be saved in this directory and in `figures/`.

## Structure

- `code/`: Python scripts for each step of the workflow.
- `data/`: Input and output datasets.
- `figures/`: Generated figures for analysis and publication.
