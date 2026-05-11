from __future__ import annotations

import argparse

from src.config import DEFAULT_DATA_PATH
from src.sweeps import (
    FULL_SWEEP_PLAN,
    QUICK_SWEEP_PLAN,
    SweepPlan,
    build_engineered_comparison_configs,
    build_split_cv_configs,
    build_topk_configs,
    run_configs,
    save_sweep_tables_and_figures,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run split-ratio, k-fold, top-k feature, and engineered-feature sweeps."
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick runs a smaller sweep; full runs all planned combinations.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model keys to run: logreg decision_tree random_forest xgboost.",
    )
    parser.add_argument(
        "--save-full-outputs",
        action="store_true",
        help="Also save predictions, confusion matrices, feature-importance files, and joblib models for each sweep run.",
    )
    parser.add_argument(
        "--skip-split-cv",
        action="store_true",
        help="Skip split-ratio and k-fold sweep.",
    )
    parser.add_argument(
        "--skip-topk",
        action="store_true",
        help="Skip top-k feature-count sweep.",
    )
    parser.add_argument(
        "--skip-engineering",
        action="store_true",
        help="Skip engineered-feature on/off comparison.",
    )
    return parser.parse_args()


def with_model_override(plan: SweepPlan, model_names: list[str] | None) -> SweepPlan:
    if not model_names:
        return plan
    return SweepPlan(
        model_names=tuple(model_names),
        test_sizes=plan.test_sizes,
        cv_folds=plan.cv_folds,
        cv_repeats=plan.cv_repeats,
        top_feature_counts=plan.top_feature_counts,
        include_engineered_comparison=plan.include_engineered_comparison,
        random_state=plan.random_state,
    )


def main():
    args = parse_args()
    plan = QUICK_SWEEP_PLAN if args.mode == "quick" else FULL_SWEEP_PLAN
    plan = with_model_override(plan, args.models)

    split_cv_df = None
    topk_df = None
    engineered_df = None

    if not args.skip_split_cv:
        split_cv_configs = build_split_cv_configs(plan, save_full_outputs=args.save_full_outputs)
        split_cv_df = run_configs(split_cv_configs, csv_path=DEFAULT_DATA_PATH)

    if not args.skip_topk:
        topk_configs = build_topk_configs(plan, save_full_outputs=args.save_full_outputs)
        topk_df = run_configs(topk_configs, csv_path=DEFAULT_DATA_PATH)

    if not args.skip_engineering:
        engineered_configs = build_engineered_comparison_configs(plan, save_full_outputs=args.save_full_outputs)
        engineered_df = run_configs(engineered_configs, csv_path=DEFAULT_DATA_PATH)

    save_sweep_tables_and_figures(split_cv_df, topk_df, engineered_df)

    if topk_df is not None and not topk_df.empty:
        cols = ["experiment_name", "model_class", "top_n_features", "accuracy", "f1", "cv_accuracy_mean", "cv_f1_mean"]
        print("\nTop-k sweep summary:")
        print(topk_df[cols].sort_values(["model_class", "top_n_features"], na_position="last").to_string(index=False))

    if split_cv_df is not None and not split_cv_df.empty:
        cols = ["experiment_name", "model_class", "test_size", "cv_folds", "accuracy", "f1", "cv_accuracy_mean", "cv_f1_mean"]
        print("\nSplit/CV sweep summary:")
        print(split_cv_df[cols].sort_values(["model_class", "test_size", "cv_folds"]).to_string(index=False))


if __name__ == "__main__":
    main()
