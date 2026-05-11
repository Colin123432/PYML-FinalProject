# Mushroom Classification Project

This project turns a single exploratory notebook into a reusable, modular classification workflow for predicting whether a mushroom is **edible** or **poisonous**.

## What is included

- reusable preprocessing and one-hot encoding
- feature-significance ranking using chi-square and Cramér's V
- feature engineering with interaction crosses
- multiple model families and experiment variants
- train/test evaluation and stratified k-fold cross-validation
- confusion matrices, feature-importance charts, SHAP summaries, and leaderboard plots
- saved predictions, metrics, and fitted pipelines

## Folder structure

```text
mushroom_models_project/
├── data/raw/mushrooms.csv
├── models/
│   ├── dummy/baseline.py
│   ├── logistic_regression/baseline.py
│   ├── logistic_regression/engineered.py
│   ├── decision_tree/baseline.py
│   ├── random_forest/baseline.py
│   ├── random_forest/split_30.py
│   ├── random_forest/top8_features.py
│   ├── xgboost/baseline.py
│   └── xgboost/top8_features.py
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   └── reports/
├── src/
│   ├── config.py
│   ├── data_io.py
│   ├── evaluation.py
│   ├── explainability.py
│   ├── experiment.py
│   ├── feature_engineering.py
│   ├── plotting.py
│   ├── preprocessing.py
│   └── significance.py
└── run_all.py
```

## Experiments included

1. `dummy_baseline` – most-frequent baseline
2. `logreg_baseline` – one-hot logistic regression
3. `logreg_engineered` – logistic regression plus engineered flags/interactions
4. `decision_tree_baseline` – shallow, interpretable decision tree
5. `random_forest_baseline` – stronger ensemble model
6. `random_forest_split_30` – split sensitivity check
7. `random_forest_top8` – uses only the top 8 original features
8. `xgboost_baseline` – gradient-boosted tree model
9. `xgboost_top8` – XGBoost using only the top 8 original features

## How to run

From the project root:

```bash
python run_all.py
```

Or run any single experiment:

```bash
python -m models.random_forest.baseline
python -m models.logistic_regression.engineered
python -m models.xgboost.top8_features
```

## Notes on preprocessing

- all predictors are categorical
- `?` is treated as missing
- missing values are imputed with the most frequent category
- categorical variables are one-hot encoded
- the target is mapped as `edible = 0`, `poisonous = 1`

## Recommended next upgrades

- add Optuna or RandomizedSearchCV tuning scripts
- export a simplified rules-based tree for business-friendly interpretation
- add calibration plots and threshold tuning
- compare against CatBoost with native categorical handling
- add unit tests for each module

## Validation and feature-count sweeps

The project now includes a separate sweep runner for testing the same model families under different validation and feature-selection settings.

Run a smaller, faster sweep:

```bash
python run_validation_sweeps.py --mode quick
```

Run the full sweep:

```bash
python run_validation_sweeps.py --mode full
```

Run only selected model families:

```bash
python run_validation_sweeps.py --mode quick --models logreg random_forest
```

The sweep runner tests:

- different stratified holdout ratios, such as 90/10, 80/20, 70/30, and 60/40 in full mode
- different stratified k-fold settings, such as 3-fold, 5-fold, and 10-fold in full mode
- different top-k feature counts, such as top 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, and all features in full mode
- engineered-feature on/off comparisons

By default, sweep experiments save metrics and summary plots but do not save every model, prediction file, and confusion matrix. To save full outputs for every sweep run, use:

```bash
python run_validation_sweeps.py --mode quick --save-full-outputs
```

You can also run the quick sweep from `run_all.py`:

```bash
python run_all.py --include-validation-sweeps
```

Sweep outputs are written to:

```text
outputs/reports/validation_split_cv_sweep.csv
outputs/reports/topk_feature_sweep.csv
outputs/reports/feature_engineering_comparison.csv
outputs/figures/validation_split_cv_sweep_f1.png
outputs/figures/topk_feature_sweep_accuracy.png
outputs/figures/feature_engineering_comparison_f1.png
```

Individual model-family sweep entry points are also available:

```bash
python -m models.logistic_regression.split_cv_sweep
python -m models.logistic_regression.topk_sweep
python -m models.decision_tree.split_cv_sweep
python -m models.decision_tree.topk_sweep
python -m models.random_forest.split_cv_sweep
python -m models.random_forest.topk_sweep
python -m models.xgboost.split_cv_sweep
python -m models.xgboost.topk_sweep
```

## Notes on interpreting sweep results

The split/k-fold sweep is meant to answer: "Do the same model types stay stable under different stratified holdout ratios and different k-fold counts?"

The top-k sweep is meant to answer: "How many of the most significant original features are needed before each model reaches near-perfect or perfect accuracy?"

The feature-engineering comparison is meant to answer: "Do the manually engineered flags and interactions improve a model compared with the original features alone?"

For a final report, the most useful plots are the top-k feature curve and the split/CV F1 curve because they show robustness rather than only a single accuracy number.
