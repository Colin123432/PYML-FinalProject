# Mushroom Classification Results Summary

## Overall findings

This dataset is highly separable. Several models reached perfect holdout and cross-validation scores:

- `logreg_engineered`
- `random_forest_baseline`
- `random_forest_split_30`
- `random_forest_top8`
- `xgboost_baseline`
- `xgboost_top8`

The simpler baselines were still extremely strong:

- `logreg_baseline`: accuracy 0.998769, F1 0.998721
- `decision_tree_baseline`: accuracy 0.998769, F1 0.998721
- `dummy_baseline`: accuracy 0.518154, F1 0.000000

## Most significant original features

Top original features by Cramér's V:

- odor: Cramér's V = 0.971
- spore-print-color: Cramér's V = 0.752
- gill-color: Cramér's V = 0.680
- ring-type: Cramér's V = 0.603
- stalk-surface-above-ring: Cramér's V = 0.588
- stalk-surface-below-ring: Cramér's V = 0.575
- gill-size: Cramér's V = 0.540
- stalk-color-above-ring: Cramér's V = 0.524
- stalk-color-below-ring: Cramér's V = 0.514
- bruises: Cramér's V = 0.501

## Why the improvements worked

- Adding engineered interaction features allowed Logistic Regression to reach perfect classification.
- Using only the top 8 original features preserved perfect performance for Random Forest and XGBoost, showing that the full 22-feature set is not necessary.
- Changing the train/test split from 80/20 to 70/30 did not hurt Random Forest performance, suggesting the model is stable on this dataset.

## Recommendation

For this dataset, the best practical model is either `random_forest_top8` or `xgboost_top8` if you want a strong but compact feature set. If you want maximum interpretability with still-excellent performance, use `logreg_engineered` or `decision_tree_baseline`.

## Key files

- `outputs/reports/leaderboard.csv`
- `outputs/reports/feature_significance.csv`
- `outputs/figures/leaderboard_f1.png`
- `outputs/figures/eda_top_feature_significance.png`
- `outputs/figures/random_forest_baseline_shap_summary.png`
- `outputs/figures/xgboost_baseline_shap_summary.png`
