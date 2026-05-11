from __future__ import annotations

import argparse
import json
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import DEFAULT_DATA_PATH, FIGURES_DIR, METRICS_DIR, REPORTS_DIR
from src.data_io import load_data, prepare_xy
from src.evaluation import leaderboard_from_metrics
from src.experiment import ExperimentConfig, run_experiment
from src.plotting import (
    save_class_distribution,
    save_leaderboard,
    save_missing_values,
    save_top_significance,
)
from src.significance import rank_features
from src.sweeps import (
    QUICK_SWEEP_PLAN,
    build_engineered_comparison_configs,
    build_split_cv_configs,
    build_topk_configs,
    run_configs,
    save_sweep_tables_and_figures,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the baseline mushroom model suite.")
    parser.add_argument(
        "--include-validation-sweeps",
        action="store_true",
        help="Also run the quick sweep suite for split ratios, k-fold counts, top-k features, and engineered features.",
    )
    parser.add_argument(
        "--save-full-sweep-outputs",
        action="store_true",
        help="When sweeps are enabled, save model/prediction/figure artifacts for each sweep experiment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data(DEFAULT_DATA_PATH)
    X, y = prepare_xy(df)

    significance = rank_features(X, y)
    significance.to_csv(REPORTS_DIR / "feature_significance.csv", index=False)
    save_class_distribution(df, FIGURES_DIR / "eda_class_distribution.png")
    save_missing_values(df, FIGURES_DIR / "eda_missing_values.png")
    save_top_significance(significance, FIGURES_DIR / "eda_top_feature_significance.png", top_n=10)

    experiments = [
        ExperimentConfig(
            experiment_name="dummy_baseline",
            estimator=DummyClassifier(strategy="most_frequent"),
            notes="Sanity-check baseline.",
        ),
        ExperimentConfig(
            experiment_name="logreg_baseline",
            estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
            notes="One-hot encoded logistic regression.",
        ),
        ExperimentConfig(
            experiment_name="logreg_engineered",
            estimator=LogisticRegression(max_iter=1000, solver="liblinear"),
            feature_engineering=True,
            notes="Adds domain-inspired flags and interaction crosses.",
        ),
        ExperimentConfig(
            experiment_name="decision_tree_baseline",
            estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
            notes="Interpretable shallow tree.",
        ),
        ExperimentConfig(
            experiment_name="random_forest_baseline",
            estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            save_shap=True,
            notes="Strong tree ensemble baseline.",
        ),
        ExperimentConfig(
            experiment_name="random_forest_split_30",
            estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            test_size=0.3,
            notes="Split sensitivity check using a 70/30 train-test split.",
        ),
        ExperimentConfig(
            experiment_name="random_forest_top8",
            estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            top_n_features=8,
            notes="Uses only the top 8 original features by Cramér's V.",
        ),
        ExperimentConfig(
            experiment_name="xgboost_baseline",
            estimator=XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=4,
            ),
            save_shap=True,
            notes="Gradient-boosted tree model.",
        ),
        ExperimentConfig(
            experiment_name="xgboost_top8",
            estimator=XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=4,
            ),
            top_n_features=8,
            notes="XGBoost with the top 8 original features only.",
        ),
        ExperimentConfig(
            experiment_name="random_forest_top7_features",
            estimator=RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            top_n_features=7,
            save_shap=True,
            notes=(
                "Random Forest using only the top 7 original features by Cramér's V. "
                "Selected after top-k sweep as the smallest tested feature subset with perfect performance."
            ),
        ),
        ExperimentConfig(
            experiment_name="decision_tree_top7_features",
            estimator=DecisionTreeClassifier(max_depth=6, random_state=42),
            top_n_features=7,
            notes=(
                "Decision Tree using only the top 7 original features by Cramér's V. "
                "Selected after top-k sweep as the smallest tested feature subset with perfect performance."
            ),
        ),
    ]

    results = []
    for experiment in experiments:
        results.append(run_experiment(experiment, csv_path=DEFAULT_DATA_PATH))

    if args.include_validation_sweeps:
        split_cv_df = run_configs(
            build_split_cv_configs(QUICK_SWEEP_PLAN, save_full_outputs=args.save_full_sweep_outputs),
            csv_path=DEFAULT_DATA_PATH,
        )
        topk_df = run_configs(
            build_topk_configs(QUICK_SWEEP_PLAN, save_full_outputs=args.save_full_sweep_outputs),
            csv_path=DEFAULT_DATA_PATH,
        )
        engineered_df = run_configs(
            build_engineered_comparison_configs(QUICK_SWEEP_PLAN, save_full_outputs=args.save_full_sweep_outputs),
            csv_path=DEFAULT_DATA_PATH,
        )
        save_sweep_tables_and_figures(split_cv_df, topk_df, engineered_df)

    leaderboard = leaderboard_from_metrics(METRICS_DIR)
    leaderboard.to_csv(REPORTS_DIR / "leaderboard.csv", index=False)
    save_leaderboard(leaderboard, FIGURES_DIR / "leaderboard_f1.png")

    summary = {
        "best_by_f1": leaderboard.iloc[0]["experiment_name"] if not leaderboard.empty else None,
        "n_experiments": int(len(leaderboard)),
        "top_features": significance.head(10)["feature"].tolist(),
        "validation_sweeps_enabled": bool(args.include_validation_sweeps),
    }
    with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(leaderboard[[
        "experiment_name", "model_class", "accuracy", "f1", "cv_accuracy_mean", "cv_f1_mean"
    ]].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
